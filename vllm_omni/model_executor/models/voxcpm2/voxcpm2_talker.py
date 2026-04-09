# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VoxCPM2 native AR talker — decomposes VoxCPM2 so that vllm handles
the base LM (MiniCPM4) natively with PagedAttention + KV cache.

Architecture:
  base_lm (MiniCPMModel) — vllm native forward, PagedAttention
  side computation:
    fsq_layer, lm_to_dit_proj, res_to_dit_proj → feat_decoder (diffusion)
    → feat_encoder → enc_to_lm_proj → curr_embed (next base_lm input)
    fusion_concat_proj → residual_lm (manual KV cache, 8 layers)
    stop_proj → stop_head (binary stop prediction)

Follows the same pattern as Qwen3TTSTalkerForConditionalGeneration.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models.minicpm import MiniCPMModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .voxcpm2_import_utils import (
    import_feat_decoder,
    import_feat_encoder,
    import_minicpm4_config,
    import_minicpm4_model,
    import_scalar_quantization_layer,
)

logger = init_logger(__name__)


class VoxCPM2TalkerForConditionalGeneration(nn.Module):
    """vllm-native VoxCPM2 talker: base_lm with PagedAttention + side
    computation (diffusion + residual_lm).

    Per-step flow:
      1. talker_mtp: diffusion(lm_hidden, res_hidden) → curr_embed
      2. forward: base_lm(curr_embed) → new hidden
      3. compute_logits: stop_head(fsq(new_hidden)) → [2] logits
      4. postprocess: fsq, residual_lm → store (lm_hidden, res_hidden)
    """

    # Weight mapping: VoxCPM2 checkpoint → vllm model
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "base_lm.": "model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        # Flags for OmniGPUModelRunner
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        # No talker_mtp — all side computation runs in forward()
        self._accumulated_patches: list[torch.Tensor] = []

        # ---- Base LM (vllm MiniCPMModel with PagedAttention) ----
        self.model = MiniCPMModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # muP scale_width for output (neutralised to 1.0 by config)
        self.scale_width = (
            self.config.hidden_size / self.config.dim_model_base
        )

        # ---- Stop predictor ----
        h = self.config.hidden_size
        self.stop_proj = nn.Linear(h, h)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(h, 2, bias=False)

        # ---- Side computation modules (from VoxCPM package) ----
        self._init_side_computation()

        # Keys that stay on GPU in model_intermediate_buffer.
        # These avoid CPU-GPU round-trips on every decode step.
        self.gpu_resident_buffer_keys: set[str] = {
            "lm_hidden",
            "res_hidden",
        }

        # Store config values used in side computation
        self._patch_size = self.config.patch_size
        self._feat_dim = self.config.feat_dim
        self._inference_timesteps = 10
        self._cfg_value = 2.0
        self._streaming_prefix_len = 4

    def _init_side_computation(self):
        """Initialize side computation modules from VoxCPM package."""
        cfg = self.config
        h = cfg.hidden_size

        # FSQ layer
        ScalarQuantizationLayer = import_scalar_quantization_layer()
        self.fsq_layer = ScalarQuantizationLayer(
            h, h,
            cfg.scalar_quantization_latent_dim,
            cfg.scalar_quantization_scale,
        )

        # Projection layers
        enc_h = cfg.encoder_config.get("hidden_dim", 1024)
        dit_h = cfg.dit_config.get("hidden_dim", 1024)
        self.enc_to_lm_proj = nn.Linear(enc_h, h)
        self.lm_to_dit_proj = nn.Linear(h, dit_h)
        self.res_to_dit_proj = nn.Linear(h, dit_h)
        self.fusion_concat_proj = nn.Linear(h * 2, h)

        # Feature encoder (VoxCPMLocEnc)
        # VoxCPM2 copies lm_config then overrides specific fields.
        # num_key_value_heads is inherited from lm_config (GQA with 2).
        MiniCPM4Config = import_minicpm4_config()
        from voxcpm.modules.minicpm4.config import RopeScalingConfig
        lm = cfg.lm_config
        num_kv_heads = lm.get("num_key_value_heads", 2)

        # Build RopeScalingConfig from raw dict
        raw_rs = lm.get("rope_scaling", {})
        if isinstance(raw_rs, dict) and raw_rs:
            rope_scaling_obj = RopeScalingConfig(**{
                k: v for k, v in raw_rs.items()
                if k in ("type", "long_factor", "short_factor",
                         "original_max_position_embeddings")
            })
        else:
            rope_scaling_obj = RopeScalingConfig(
                type="longrope",
                long_factor=[1.0] * 64,
                short_factor=[1.0] * 64,
                original_max_position_embeddings=32768,
            )
        enc_cfg = cfg.encoder_config
        encoder_minicpm_config = MiniCPM4Config(
            bos_token_id=1, eos_token_id=2,
            hidden_size=enc_cfg.get("hidden_dim", 1024),
            intermediate_size=enc_cfg.get("ffn_dim", 4096),
            num_attention_heads=enc_cfg.get("num_heads", 16),
            num_hidden_layers=enc_cfg.get("num_layers", 12),
            num_key_value_heads=num_kv_heads,
            kv_channels=enc_cfg.get("kv_channels", 128),
            rms_norm_eps=cfg.rms_norm_eps,
            max_position_embeddings=cfg.max_position_embeddings,
            vocab_size=0,
            use_mup=False,
            scale_emb=1.0,
            dim_model_base=enc_cfg.get("hidden_dim", 1024),
            scale_depth=1.0,
            rope_theta=cfg.rope_theta,
            rope_scaling=rope_scaling_obj,
        )
        VoxCPMLocEnc = import_feat_encoder()
        self.feat_encoder = VoxCPMLocEnc(
            encoder_minicpm_config, input_dim=cfg.feat_dim,
        )

        # Feature decoder (UnifiedCFM with LocDiT)
        UnifiedCFM, VoxCPMLocDiTV2, CfmConfig = import_feat_decoder()
        dit_cfg = cfg.dit_config
        decoder_minicpm_config = MiniCPM4Config(
            bos_token_id=1, eos_token_id=2,
            hidden_size=dit_cfg.get("hidden_dim", 1024),
            intermediate_size=dit_cfg.get("ffn_dim", 4096),
            num_attention_heads=dit_cfg.get("num_heads", 16),
            num_hidden_layers=dit_cfg.get("num_layers", 12),
            num_key_value_heads=num_kv_heads,
            kv_channels=dit_cfg.get("kv_channels", 128),
            rms_norm_eps=cfg.rms_norm_eps,
            max_position_embeddings=cfg.max_position_embeddings,
            vocab_size=0,
            use_mup=False,
            scale_emb=1.0,
            dim_model_base=dit_cfg.get("hidden_dim", 1024),
            scale_depth=1.0,
            rope_theta=cfg.rope_theta,
            rope_scaling=rope_scaling_obj,
        )
        cfm_raw = dit_cfg.get("cfm_config", {})
        cfm_config = CfmConfig(
            sigma_min=cfm_raw.get("sigma_min", 1e-6),
            solver=cfm_raw.get("solver", "euler"),
            t_scheduler=cfm_raw.get("t_scheduler", "log-norm"),
            inference_cfg_rate=cfm_raw.get("inference_cfg_rate", 2.0),
        )
        self.feat_decoder = UnifiedCFM(
            in_channels=cfg.feat_dim,
            cfm_params=cfm_config,
            estimator=VoxCPMLocDiTV2(
                decoder_minicpm_config, in_channels=cfg.feat_dim,
            ),
            mean_mode=dit_cfg.get("mean_mode", False),
        )
        # Keep diffusion in float32 — the 10-step Euler ODE solver
        # accumulates precision errors in bfloat16.
        self.feat_decoder = self.feat_decoder.float()

        # Residual LM (VoxCPM's MiniCPMModel with manual KV cache)
        NativeMiniCPMModel = import_minicpm4_model()
        lm_cfg = cfg.lm_config
        residual_lm_config = MiniCPM4Config(
            bos_token_id=1, eos_token_id=2,
            hidden_size=lm_cfg.get("hidden_size", h),
            intermediate_size=lm_cfg.get("intermediate_size", h * 3),
            num_attention_heads=lm_cfg.get("num_attention_heads", 16),
            num_hidden_layers=cfg.residual_lm_num_layers,
            num_key_value_heads=lm_cfg.get("num_key_value_heads", 2),
            kv_channels=lm_cfg.get("kv_channels"),
            rms_norm_eps=lm_cfg.get("rms_norm_eps", 1e-5),
            max_position_embeddings=lm_cfg.get(
                "max_position_embeddings", 32768
            ),
            vocab_size=0,
            use_mup=lm_cfg.get("use_mup", False),
            scale_emb=lm_cfg.get("scale_emb", 1.0),
            dim_model_base=lm_cfg.get("dim_model_base", 256),
            scale_depth=lm_cfg.get("scale_depth", 1.0),
            rope_theta=lm_cfg.get("rope_theta", 10000.0),
            rope_scaling=rope_scaling_obj,
            no_rope=cfg.residual_lm_no_rope,
        )
        self.residual_lm = NativeMiniCPMModel(residual_lm_config)

    # -------------------- vllm required hooks --------------------

    def embed_input_ids(
        self, input_ids: torch.Tensor, **_: Any
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """Full VoxCPM2 AR step in one forward call (nanovllm pattern).

        base_lm → FSQ → residual_lm → diffusion → feat_encoder → stop.
        All side computation runs here, not in separate preprocess/postprocess.
        """
        # --- 1. Base LM forward (vllm PagedAttention) ---
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        if isinstance(model_output, IntermediateTensors):
            return model_output
        hidden_states = model_output
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        enc_outputs = hidden_states / self.scale_width

        # --- 2. Side computation (entire pipeline in one call) ---
        side_dtype = self.fusion_concat_proj.weight.dtype
        dev = enc_outputs.device
        is_prefill = enc_outputs.shape[0] > 1

        # Only run side computation for real requests, not warmup/dummy
        has_infos = bool(getattr(self, "_current_step_infos", None))

        if is_prefill and has_infos:
            logger.info(
                "PREFILL side: enc shape=%s last_norm=%.4f last5=%s",
                enc_outputs.shape,
                enc_outputs[-1].norm().item(),
                enc_outputs[-1, :5].tolist(),
            )
            return self._forward_prefill_side(enc_outputs, dev, side_dtype)

        if not is_prefill and getattr(self, "_prev_feat_embed", None) is not None:
            return self._forward_decode_side(enc_outputs, dev, side_dtype)

        return enc_outputs

    def _forward_prefill_side(
        self, enc_outputs: torch.Tensor, dev: torch.device,
        side_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prefill side computation: FSQ + residual_lm full forward."""
        # Get masks from buffer (stored by preprocess)
        info = {}
        for d in getattr(self, "_current_step_infos", []):
            if isinstance(d, dict):
                info = d
                break

        text_mask_cpu = info.get("text_mask_cpu")
        audio_mask_cpu = info.get("audio_mask_cpu")
        combined_embed_cpu = info.get("combined_embed_cpu")

        h = enc_outputs.to(side_dtype)

        if text_mask_cpu is not None:
            text_mask = text_mask_cpu.to(dev)
            audio_mask = audio_mask_cpu.to(dev)
            mask_len = min(h.shape[0], text_mask.shape[0])
            tm = text_mask[:mask_len].unsqueeze(-1).float().to(side_dtype)
            am = audio_mask[:mask_len].unsqueeze(-1).float().to(side_dtype)
            h_slice = h[:mask_len]
            fsq_all = self.fsq_layer(h_slice)
            fsq_enc = fsq_all * am + h_slice * tm

            combined_embed = combined_embed_cpu.to(dev, dtype=side_dtype)
            feat_embed_approx = combined_embed[:mask_len] * am
        else:
            # Pure text, no audio — FSQ is identity on text positions
            fsq_enc = h
            feat_embed_approx = torch.zeros_like(h)

        # Setup and run residual_lm
        if self.residual_lm.kv_cache is None:
            self.residual_lm.setup_cache(
                1, self.config.max_length, dev, side_dtype,
            )

        residual_input = self.fusion_concat_proj(
            torch.cat([fsq_enc, feat_embed_approx], dim=-1)
        )
        res_out = self.residual_lm(
            inputs_embeds=residual_input.unsqueeze(0), is_causal=True,
        )
        res_outputs, res_kv = res_out
        self.residual_lm.kv_cache.fill_caches(res_kv)

        # Store states for decode
        lm_hidden = fsq_enc[-1, :].detach()
        res_hidden = res_outputs[0, -1, :].detach()
        p = self._patch_size
        d = self._feat_dim
        pfc = torch.zeros(p, d, device=dev, dtype=side_dtype)

        # Precompute stop logits (checked BEFORE first diffusion, like native)
        stop_logits = self.stop_head(
            self.stop_actn(self.stop_proj(lm_hidden.unsqueeze(0)))
        )
        self._precomputed_stop_logits = stop_logits.detach()

        # Run first diffusion step during prefill (like nanovllm).
        # This produces feat_0 and curr_embed_0, which becomes the
        # input to base_lm on the first decode step.
        dit_h1 = self.lm_to_dit_proj(lm_hidden.unsqueeze(0))
        dit_h2 = self.res_to_dit_proj(res_hidden.unsqueeze(0))
        dit_hidden = torch.cat([dit_h1, dit_h2], dim=-1)
        pred_feat = self.feat_decoder(
            mu=dit_hidden.float(),
            patch_size=p,
            cond=pfc.unsqueeze(0).transpose(1, 2).contiguous().float(),
            n_timesteps=self._inference_timesteps,
            cfg_value=self._cfg_value,
        ).to(side_dtype).transpose(1, 2)  # [1, P, D]

        curr_embed_enc = self.feat_encoder(pred_feat.unsqueeze(1))
        curr_embed = self.enc_to_lm_proj(curr_embed_enc).squeeze(1)

        # Store first patch and curr_embed for decode
        self._curr_embed_for_next = curr_embed.detach()
        self._curr_prefix_feat_cond = pred_feat[0].detach()
        self._last_audio_patch = pred_feat.reshape(1, -1).detach().cpu().to(
            torch.float32
        )
        # Store the prev_feat_embed for residual_lm in next decode step
        self._prev_feat_embed = curr_embed.detach()

        return enc_outputs.to(self.model.norm.weight.dtype)

    def _forward_decode_side(
        self, enc_outputs: torch.Tensor, dev: torch.device,
        side_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Decode side computation following nanovllm pattern.

        Order (matching nanovllm):
          1. FSQ current enc_outputs → lm_hidden (CURRENT step)
          2. residual_lm step with (lm_hidden, prev_feat_embed) → res_hidden
          3. diffusion(lm_hidden, res_hidden) → pred_feat
          4. feat_encoder(pred_feat) → curr_embed (for next step's base_lm)
          5. stop_head(lm_hidden)
        """
        p = self._patch_size
        d = self._feat_dim

        if not hasattr(self, "_prev_feat_embed"):
            return enc_outputs

        # 1. FSQ current enc_outputs → CURRENT lm_hidden
        h = enc_outputs.to(side_dtype)
        lm_hidden = self.fsq_layer(h[-1:]).squeeze(0)  # [H]

        # 2. Residual LM step: cat(lm_hidden, prev_feat_embed)
        #    In nanovllm: residual_inputs = fusion_concat_proj(cat(enc_outputs, feat_embeds))
        #    where feat_embeds is from the INPUT feat (= previous step's pred_feat)
        prev_fe = self._prev_feat_embed.to(side_dtype)
        if prev_fe.ndim == 1:
            prev_fe = prev_fe.unsqueeze(0)
        fusion_input = self.fusion_concat_proj(
            torch.cat([lm_hidden.unsqueeze(0), prev_fe], dim=-1)
        )  # [1, H]
        step_pos = torch.tensor(
            [self.residual_lm.kv_cache.step()], device=dev,
        )
        res_hidden = self.residual_lm.forward_step(
            fusion_input, step_pos,
        ).clone()
        if res_hidden.ndim == 1:
            res_hidden = res_hidden.unsqueeze(0)

        # 3. Diffusion with CURRENT lm_hidden + res_hidden
        dit_h1 = self.lm_to_dit_proj(lm_hidden.unsqueeze(0))
        dit_h2 = self.res_to_dit_proj(res_hidden)
        dit_hidden = torch.cat([dit_h1, dit_h2], dim=-1)

        pfc = getattr(self, "_curr_prefix_feat_cond", None)
        if pfc is None:
            pfc = torch.zeros(p, d, device=dev, dtype=side_dtype)
        pfc = pfc.to(side_dtype).unsqueeze(0)

        pred_feat = self.feat_decoder(
            mu=dit_hidden.float(),
            patch_size=p,
            cond=pfc.transpose(1, 2).contiguous().float(),
            n_timesteps=self._inference_timesteps,
            cfg_value=self._cfg_value,
        ).to(side_dtype).transpose(1, 2)  # [1, P, D]

        # 4. feat_encoder → curr_embed (input for next step's base_lm)
        curr_embed_enc = self.feat_encoder(pred_feat.unsqueeze(1))
        curr_embed = self.enc_to_lm_proj(curr_embed_enc).squeeze(1)  # [1, H]

        # 5. Store state for next step
        self._curr_prefix_feat_cond = pred_feat[0].detach()
        self._last_audio_patch = pred_feat.reshape(1, -1).detach().cpu().to(
            torch.float32
        )
        self._curr_embed_for_next = curr_embed.detach()
        self._prev_feat_embed = curr_embed.detach()

        # 6. Stop logits from CURRENT lm_hidden (like nanovllm)
        stop_logits = self.stop_head(
            self.stop_actn(self.stop_proj(lm_hidden.unsqueeze(0)))
        )
        self._precomputed_stop_logits = stop_logits.detach()

        return enc_outputs

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        """Binary stop prediction: [continue=0, stop=1].

        In native VoxCPM2, stop is checked on the PREVIOUS step's FSQ'd
        lm_hidden (before the current forward_step). We match this by
        using precomputed stop logits stored in postprocess.
        If no precomputed logits are available (first step), we compute
        from the current hidden states.
        """
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None

        # Use precomputed stop logits from previous postprocess
        precomputed = getattr(self, "_precomputed_stop_logits", None)
        if precomputed is not None:
            self._precomputed_stop_logits = None
            raw_logits = precomputed[:hidden_states.shape[0]]
            logger.info(
                "compute_logits: PRECOMPUTED stop=[%.3f, %.3f] pred=%d",
                raw_logits[0, 0].item(), raw_logits[0, 1].item(),
                raw_logits[0].argmax().item(),
            )
        else:
            # Fallback: compute from current hidden (for first prefill step)
            fsq_hidden = self.fsq_layer(hidden_states)
            raw_logits = self.stop_head(
                self.stop_actn(self.stop_proj(fsq_hidden))
            )  # [B, 2]
            logger.info(
                "compute_logits: FALLBACK hs_shape=%s stop=[%.3f, %.3f] pred=%d",
                hidden_states.shape, raw_logits[-1, 0].item(),
                raw_logits[-1, 1].item(), raw_logits[-1].argmax().item(),
            )

        # Pad to vocab_size: engine expects full-vocab logits.
        # Token 0 = continue, token 1 = stop (matched by stop_token_ids).
        bsz = raw_logits.shape[0]
        full_logits = torch.full(
            (bsz, self.config.vocab_size),
            float("-inf"),
            device=raw_logits.device,
            dtype=raw_logits.dtype,
        )
        full_logits[:, 0] = raw_logits[:, 0]  # continue
        full_logits[:, 1] = raw_logits[:, 1]  # stop
        return full_logits

    # -------------------- Omni output plumbing --------------------

    def make_omni_output(
        self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any
    ) -> OmniOutput:
        """Build OmniOutput with accumulated latent patches.

        Each decode step's talker_mtp produces a patch stored in the
        buffer as ``audio_latent_patch``. We accumulate them and emit
        the full latent as ``latent_audio_feat`` (matching what the
        latent2vae processor expects).
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []

        # Get audio patch from forward's side computation
        patch = getattr(self, "_last_audio_patch", None)
        if patch is not None:
            self._last_audio_patch = None
            logger.info("make_omni_output: got patch shape=%s norm=%.4f val[:3]=%s",
                        patch.shape, patch.norm().item(), patch[0,:3].tolist())
            self._accumulated_patches.append(patch.clone())
            # Save accumulated patches for offline analysis
            import numpy as np
            all_p = torch.cat(self._accumulated_patches, dim=0)
            np.save("/tmp/our_patches.npy", all_p.numpy())

        mm: dict[str, Any] = {}
        if patch is not None:
            # Emit single-step patch as [1, P*D] for output_processor
            # to concatenate across steps → [N, P*D].
            # latent2vae reshapes [N, P*D] → [D, N*P].
            flat = patch.detach().cpu().to(torch.float32)  # [1, P*D]

            sample_rate = 48000
            mm["latent_audio_feat"] = [flat]
            mm["sr"] = [torch.tensor(sample_rate, dtype=torch.int32)]

        return OmniOutput(
            text_hidden_states=hidden, multimodal_outputs=mm,
        )

    # -------------------- preprocess / postprocess --------------------

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged = {
                k: v for k, v in info_dict.items()
                if k != "additional_information"
            }
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        dev = input_ids.device

        if span_len > 1:
            # ---- Prefill: fix tokenization, embed, store masks ----
            # Speech API sends [BOS, text_tokens...] but native VoxCPM2
            # expects [text_tokens..., audio_start_token(101)].
            # Fix: strip BOS (token 1), append audio_start (101).
            ids = input_ids.clone()
            bos_id = self.config.bos_token_id  # 1
            audio_start = 101

            # Strip leading BOS if present
            if ids[0].item() == bos_id:
                ids = ids[1:]

            # Append audio_start_token if not present
            if ids[-1].item() != audio_start:
                ids = torch.cat([
                    ids,
                    torch.tensor([audio_start], device=dev, dtype=ids.dtype),
                ])

            new_span = ids.shape[0]
            embeds = self.embed_input_ids(ids)

            # Store masks for forward's prefill side computation
            tm = torch.ones(new_span, dtype=torch.int32, device=dev)
            am = torch.zeros(new_span, dtype=torch.int32, device=dev)
            self._current_step_infos = [{
                "text_mask_cpu": tm.cpu(),
                "audio_mask_cpu": am.cpu(),
                "combined_embed_cpu": embeds.detach().cpu(),
            }]

            # Pad or truncate embeds to match original span_len
            if new_span < span_len:
                pad = self.embed_input_ids(
                    torch.zeros(span_len - new_span, device=dev, dtype=torch.long)
                )
                embeds = torch.cat([embeds, pad], dim=0)
            elif new_span > span_len:
                embeds = embeds[:span_len]

            return input_ids, embeds, {}

        # ---- Decode: provide curr_embed from previous forward ----
        curr_embed = getattr(self, "_curr_embed_for_next", None)
        if curr_embed is not None:
            inputs_embeds = curr_embed.to(
                device=dev, dtype=torch.bfloat16
            ).reshape(1, -1)
        else:
            inputs_embeds = self.embed_input_ids(input_ids.reshape(1, 1).to(torch.long)).reshape(1, -1)

        self._current_step_infos = [{}]
        return input_ids, inputs_embeds, {}

    def _preprocess_prefill(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        span_len: int,
        dev: torch.device,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build combined text+audio embeddings for prefill.

        Also runs residual_lm prefill and stores initial states.
        """
        # Check if we have explicit text in buffer (for custom embedding).
        # The speech API passes text as prompt_token_ids, not in the buffer.
        # In that case, just use standard embed_input_ids on the actual tokens.
        text_list = info_dict.get("text")
        if not isinstance(text_list, list) or not text_list:
            # Standard path: use actual input_ids (already tokenized by API)
            embeds = self.embed_input_ids(input_ids)
            # Setup residual_lm for deferred prefill
            initial_states = self._prefill_side_computation(
                embeds,
                torch.ones(span_len, dtype=torch.int32, device=dev),  # all text
                torch.zeros(span_len, dtype=torch.int32, device=dev),  # no audio
                dev,
            )
            return input_ids, embeds, initial_states

        # Build prompt embeddings from text tokens + reference audio
        prompt_embeds_cpu = info_dict.get("voxcpm2_prompt_embeds")
        is_first_prefill = not isinstance(prompt_embeds_cpu, torch.Tensor)

        if is_first_prefill:
            prompt_embeds, text_mask, audio_mask = (
                self._build_prompt_embeds(info_dict, dev)
            )
            prompt_embeds_cpu = (
                prompt_embeds.detach().to("cpu").contiguous()
            )

            # Run residual_lm prefill and store initial states
            initial_states = self._prefill_side_computation(
                prompt_embeds, text_mask, audio_mask, dev,
            )

            info_update: dict[str, Any] = {
                "voxcpm2_prompt_embeds": prompt_embeds_cpu,
                "voxcpm2_prefill_offset": 0,
                **initial_states,
            }

            # Slice for this chunk
            take = prompt_embeds_cpu[:span_len]
            if take.shape[0] < span_len:
                pad = torch.zeros(
                    span_len - take.shape[0],
                    take.shape[-1],
                    dtype=take.dtype,
                )
                take = torch.cat([take, pad], dim=0)
            embeds = take.to(device=dev, dtype=torch.bfloat16)
            info_update["voxcpm2_prefill_offset"] = span_len

            dummy_ids = torch.zeros_like(input_ids)
            return dummy_ids, embeds, info_update

        # Subsequent prefill chunk
        offset = int(info_dict.get("voxcpm2_prefill_offset", 0))
        s = max(0, min(offset, prompt_embeds_cpu.shape[0]))
        e = max(0, min(offset + span_len, prompt_embeds_cpu.shape[0]))
        take = prompt_embeds_cpu[s:e]
        if take.shape[0] < span_len:
            pad = torch.zeros(
                span_len - take.shape[0],
                take.shape[-1],
                dtype=take.dtype,
            )
            take = torch.cat([take, pad], dim=0)
        embeds = take.to(device=dev, dtype=torch.bfloat16)
        info_update = {"voxcpm2_prefill_offset": offset + span_len}
        dummy_ids = torch.zeros_like(input_ids)
        return dummy_ids, embeds, info_update

    def _preprocess_decode(
        self,
        input_ids: torch.Tensor,
        dev: torch.device,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Decode step: provide mtp_inputs for talker_mtp.

        talker_mtp will overwrite inputs_embeds with curr_embed, so the
        initial value here is just a placeholder.
        """
        lm_hidden = info_dict.get("lm_hidden")
        res_hidden = info_dict.get("res_hidden")

        if not isinstance(lm_hidden, torch.Tensor):
            raise RuntimeError(
                "Missing lm_hidden in buffer; prefill must run first."
            )
        if not isinstance(res_hidden, torch.Tensor):
            # res_hidden may not be set yet if this is the very first
            # decode step and postprocess hasn't run. Use zeros.
            res_hidden = torch.zeros(
                self.config.hidden_size,
                device=dev, dtype=torch.bfloat16,
            )

        lm_hidden = lm_hidden.to(device=dev, dtype=torch.bfloat16)
        res_hidden = res_hidden.to(device=dev, dtype=torch.bfloat16)

        # Placeholder embedding — talker_mtp will overwrite this with
        # curr_embed from diffusion → feat_encoder.
        inputs_embeds = torch.zeros(
            1, self.config.hidden_size,
            device=dev, dtype=torch.bfloat16,
        )

        info_update: dict[str, Any] = {
            "mtp_inputs": (
                lm_hidden.reshape(1, -1),
                res_hidden.reshape(1, -1),
            ),
        }
        return input_ids, inputs_embeds, info_update

    def postprocess(
        self, hidden_states: torch.Tensor, **info: Any
    ) -> dict[str, Any]:
        """Postprocess is now minimal — forward() does all side computation."""
        if hidden_states.numel() == 0:
            return {}
        # Nothing to do — forward() already updated all states
        return {}

    def _postprocess_accumulate_prefill(
        self,
        hidden_states: torch.Tensor,
        dev: torch.device,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        """Accumulate prefill hidden states for deferred residual_lm."""
        prev_chunks = info.get("_prefill_hs_chunks")
        if not isinstance(prev_chunks, list):
            prev_chunks = []
        prev_chunks.append(hidden_states.detach().to("cpu"))

        # FSQ last token → lm_hidden + precompute stop logits
        last = hidden_states[-1, :].detach()
        lm_hidden = self.fsq_layer(last.unsqueeze(0)).squeeze(0)

        # Precompute stop logits from FSQ'd lm_hidden (native semantics)
        stop_logits = self.stop_head(
            self.stop_actn(self.stop_proj(lm_hidden.unsqueeze(0)))
        )
        self._precomputed_stop_logits = stop_logits.detach()

        return {
            "_prefill_hs_chunks": prev_chunks,
            "lm_hidden": lm_hidden,
            "residual_lm_needs_prefill": True,
        }

    def _postprocess_deferred_prefill(
        self,
        hidden_states: torch.Tensor,
        dev: torch.device,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        """First decode step: run residual_lm prefill on accumulated
        hidden states, then normal decode postprocess."""
        # Concatenate all prefill chunks
        chunks = info.get("_prefill_hs_chunks", [])
        if not chunks:
            # No prefill data — just do normal decode
            return self._postprocess_decode(hidden_states, dev)

        full_hs = torch.cat(
            [c.to(dev, dtype=torch.bfloat16) for c in chunks], dim=0,
        )

        # Setup residual_lm KV cache on the correct device/dtype
        side_dtype = self.fusion_concat_proj.weight.dtype
        if self.residual_lm.kv_cache is None:
            self.residual_lm.setup_cache(
                1, self.config.max_length, dev, side_dtype,
            )

        # Recover masks
        text_mask_cpu = info.get("text_mask_cpu")
        audio_mask_cpu = info.get("audio_mask_cpu")
        combined_embed_cpu = info.get("combined_embed_cpu")

        if text_mask_cpu is None:
            return self._postprocess_decode(hidden_states, dev)

        text_mask = text_mask_cpu.to(dev)
        audio_mask = audio_mask_cpu.to(dev)
        seq_len = full_hs.shape[0]
        mask_len = min(seq_len, text_mask.shape[0])

        tm = text_mask[:mask_len].unsqueeze(-1).float()
        am = audio_mask[:mask_len].unsqueeze(-1).float()

        # Cast to side module dtype (may be float32 if VoxCPM modules
        # weren't cast to bfloat16 by vllm's model loading)
        side_dtype = self.fusion_concat_proj.weight.dtype

        # FSQ selective: audio positions only
        h_slice = full_hs[:mask_len].to(side_dtype)
        fsq_all = self.fsq_layer(h_slice)
        enc_outputs = fsq_all * am.to(side_dtype) + h_slice * tm.to(side_dtype)

        # Reconstruct feat_embed for fusion
        combined_embed = combined_embed_cpu.to(dev, dtype=side_dtype)
        feat_embed_approx = combined_embed[:mask_len] * am.to(side_dtype)

        # Residual LM full forward
        residual_input = self.fusion_concat_proj(
            torch.cat([enc_outputs, feat_embed_approx], dim=-1)
        )
        res_out = self.residual_lm(
            inputs_embeds=residual_input.unsqueeze(0),
            is_causal=True,
        )
        logger.info("DIAG residual_lm output type=%s len=%s", type(res_out).__name__, len(res_out) if isinstance(res_out, (tuple, list)) else "N/A")
        if isinstance(res_out, tuple) and len(res_out) == 2:
            res_outputs, res_kv = res_out
        else:
            res_outputs = res_out if isinstance(res_out, torch.Tensor) else res_out[0]
            res_kv = None
        if res_kv is not None:
            self.residual_lm.kv_cache.fill_caches(res_kv)
        res_hidden = res_outputs[0, -1, :].detach()

        # Use lm_hidden from PREFILL enc_outputs (last token, FSQ'd)
        prefill_lm_hidden = enc_outputs[-1, :].detach()

        # Init prefix_feat_cond (zeros for zero-shot)
        p = self._patch_size
        d = self._feat_dim
        pfc = torch.zeros(p, d, device=dev, dtype=side_dtype)
        self._curr_prefix_feat_cond = pfc

        # Run the FIRST diffusion step here with correct prefill states.
        # talker_mtp ran with zeros for res_hidden, producing garbage.
        # We redo it here with the proper prefill lm_hidden + res_hidden.
        dit_h1 = self.lm_to_dit_proj(prefill_lm_hidden.unsqueeze(0))
        dit_h2 = self.res_to_dit_proj(res_hidden.unsqueeze(0))
        dit_hidden = torch.cat([dit_h1, dit_h2], dim=-1)

        pred_feat = self.feat_decoder(
            mu=dit_hidden.float(),
            patch_size=p,
            cond=pfc.unsqueeze(0).transpose(1, 2).contiguous().float(),
            n_timesteps=self._inference_timesteps,
            cfg_value=self._cfg_value,
        ).to(side_dtype).transpose(1, 2)  # [1, P, D]

        curr_embed_enc = self.feat_encoder(pred_feat.unsqueeze(1))
        curr_embed = self.enc_to_lm_proj(curr_embed_enc).squeeze(1)

        # Update prefix_feat_cond for next step
        self._curr_prefix_feat_cond = pred_feat[0].detach()
        # Store correct curr_embed for this step's residual_lm
        self._curr_embed_for_residual = curr_embed.detach()
        # Store correct first audio patch (replace garbage from talker_mtp)
        self._first_patch_override = pred_feat.reshape(1, -1).detach().cpu().to(torch.float32)

        # FSQ the CURRENT decode step's base_lm output
        last_hidden = hidden_states[-1, :].detach()
        lm_hidden = self.fsq_layer(last_hidden.unsqueeze(0)).squeeze(0)

        # Residual_lm step for current decode token
        # forward_step expects [B, H] input (2D)
        lm_h_2d = lm_hidden.unsqueeze(0)  # [1, H]
        ce_2d = curr_embed if curr_embed.ndim == 2 else curr_embed.unsqueeze(0)
        fusion_input = self.fusion_concat_proj(
            torch.cat([lm_h_2d, ce_2d], dim=-1)
        )  # [1, H]
        step_pos = torch.tensor(
            [self.residual_lm.kv_cache.step()], device=dev,
        )
        res_hidden = self.residual_lm.forward_step(
            fusion_input, step_pos,
        ).clone()  # returns [H]

        # Precompute stop logits for next compute_logits call
        stop_logits = self.stop_head(
            self.stop_actn(self.stop_proj(lm_hidden.unsqueeze(0)))
        )
        self._precomputed_stop_logits = stop_logits.detach()

        return {
            "lm_hidden": lm_hidden,
            "res_hidden": res_hidden,
            "residual_lm_needs_prefill": False,
            "_prefill_hs_chunks": None,
            "text_mask_cpu": None,
            "audio_mask_cpu": None,
            "combined_embed_cpu": None,
        }

    def _postprocess_decode(
        self,
        hidden_states: torch.Tensor,
        dev: torch.device,
    ) -> dict[str, Any]:
        """Normal decode: FSQ + residual_lm step + precompute stop logits."""
        side_dtype = self.fusion_concat_proj.weight.dtype
        last_hidden = hidden_states[-1, :].detach().to(side_dtype)
        lm_hidden = self.fsq_layer(last_hidden.unsqueeze(0)).squeeze(0)

        # Precompute stop logits for next step's compute_logits
        stop_logits = self.stop_head(
            self.stop_actn(self.stop_proj(lm_hidden.unsqueeze(0)))
        )
        self._precomputed_stop_logits = stop_logits.detach()

        curr_embed = getattr(self, "_curr_embed_for_residual", None)
        if curr_embed is None or self.residual_lm.kv_cache is None:
            return {"lm_hidden": lm_hidden}

        curr_embed = curr_embed.to(dev, dtype=side_dtype)

        # forward_step expects [B, H] (2D)
        ce_2d = curr_embed if curr_embed.ndim == 2 else curr_embed.unsqueeze(0)
        fusion_input = self.fusion_concat_proj(
            torch.cat([lm_hidden.unsqueeze(0), ce_2d], dim=-1)
        )  # [1, H] — keep batch dim

        step_pos = torch.tensor(
            [self.residual_lm.kv_cache.step()], device=dev,
        )
        res_hidden = self.residual_lm.forward_step(
            fusion_input, step_pos,
        ).clone()

        return {
            "lm_hidden": lm_hidden,
            "res_hidden": res_hidden,
        }

    # ---- All old talker_mtp / deferred prefill code removed ----
    # Side computation now runs in forward() directly (nanovllm pattern).

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Load weights from VoxCPM2 checkpoint.

        base_lm.* → model.* (via WeightsMapper, handled by MiniCPMModel)
        All other weights → loaded directly to side computation modules.
        """
        # Separate base_lm weights from side computation weights
        side_weights: list[tuple[str, torch.Tensor]] = []

        def _filter_base_lm(
            ws: Iterable[tuple[str, torch.Tensor]],
        ):
            for name, tensor in ws:
                if name.startswith("base_lm."):
                    yield name, tensor
                else:
                    side_weights.append((name, tensor))

        # Load base_lm weights via AutoWeightsLoader + WeightsMapper
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(
            _filter_base_lm(weights), mapper=self.hf_to_vllm_mapper,
        )

        # Load side computation weights directly
        params_dict = dict(self.named_parameters())
        skipped = []
        for name, tensor in side_weights:
            if name not in params_dict:
                skipped.append(name)
                continue
            param = params_dict[name]
            if param.shape != tensor.shape:
                logger.warning(
                    "Shape mismatch for %s: model=%s ckpt=%s, skipping",
                    name, param.shape, tensor.shape,
                )
                continue
            with torch.no_grad():
                param.copy_(tensor)
            loaded.add(name)
        if skipped:
            logger.info(
                "Skipped %d side weights not in model params "
                "(first 5: %s)",
                len(skipped), skipped[:5],
            )

        logger.info(
            "Loaded %d weights for VoxCPM2TalkerForConditionalGeneration",
            len(loaded),
        )
        return loaded
