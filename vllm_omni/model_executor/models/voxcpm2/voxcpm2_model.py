# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VoxCPM2 model for vllm-omni two-stage pipeline.

Stage 0 (latent_generator): Text -> [TSLM -> FSQ -> LocEnc -> RALM -> LocDiT] -> latent
Stage 1 (vae): Latent -> AudioVAE V2 -> 48kHz audio
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .voxcpm2_native_loader import load_voxcpm2_audio_vae, load_voxcpm2_latent_generator

logger = init_logger(__name__)


class VoxCPM2ForConditionalGeneration(nn.Module):
    """VoxCPM2 two-stage TTS model (tokenizer-free diffusion AR, 2B params, 48kHz)."""

    input_modalities = "audio"
    _LATENT_STAGES = {"latent_generator", "latent", "ar_dit"}
    _VAE_STAGES = {"vae", "audio_vae"}
    _FULL_PIPELINE_STAGES = {"full_pipeline"}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.model_stage = getattr(vllm_config.model_config, "model_stage", "latent_generator")
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True
        self.inject_omni_request_id_into_runtime_info = True
        self._pipeline = None
        self._latent_stream_gens: dict[str, Any] = {}
        self._ar_emit_stop_token = True
        self._completed_requests: set[str] = set()
        self._request_outputs: dict[str, dict[str, Any]] = {}

    def _resolve_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        from vllm_omni.platforms import current_omni_platform

        device = current_omni_platform.get_torch_device()
        model_config = getattr(self.vllm_config, "model_config", None)
        dtype = getattr(model_config, "dtype", torch.bfloat16) if model_config else torch.bfloat16
        return device, dtype

    def _ensure_model_loaded(self):
        if self._pipeline is not None:
            return

        device, dtype = self._resolve_device_dtype()
        if self.model_stage in self._FULL_PIPELINE_STAGES:
            # Single-stage: load full VoxCPM model (latent gen + VAE decode)
            self._pipeline = load_voxcpm2_latent_generator(
                self.model_path, device=device, dtype=dtype,
            )
        elif self.model_stage in self._LATENT_STAGES:
            self._pipeline = load_voxcpm2_latent_generator(
                self.model_path, device=device, dtype=dtype,
            )
        elif self.model_stage in self._VAE_STAGES:
            self._pipeline = load_voxcpm2_audio_vae(
                self.model_path, device=device,
            )
        else:
            raise ValueError(f"Unsupported model_stage: {self.model_stage}")

        logger.info("Loaded VoxCPM2 stage '%s' on %s", self.model_stage, device)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        self._ensure_model_loaded()
        return set()

    @staticmethod
    def _extract_val(info: dict[str, Any], key: str, default: Any) -> Any:
        value = info.get(key, default)
        if isinstance(value, list):
            return value[0] if value else default
        return value

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def _get_vocab_size(self) -> int:
        model_config = getattr(self.vllm_config, "model_config", None)
        if model_config is not None:
            hf_config = getattr(model_config, "hf_text_config", None)
            if hf_config is not None and hasattr(hf_config, "vocab_size"):
                return int(hf_config.vocab_size)
        return 32000

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> torch.Tensor:
        del sampling_metadata
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            device, dtype = self._resolve_device_dtype()
            hidden_states = torch.zeros((0, 1), device=device, dtype=dtype)
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(-1)
        elif hidden_states.ndim > 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        vocab_size = self._get_vocab_size()
        num_rows = int(hidden_states.shape[0])
        logits = torch.zeros((num_rows, vocab_size), dtype=torch.float32, device=hidden_states.device)
        eos_id = 2 if vocab_size > 2 else 0
        safe_id = 1 if vocab_size > 1 and 1 != eos_id else 0
        if num_rows > 0:
            if self._ar_emit_stop_token:
                logits[:, eos_id] = 1.0e6
            else:
                logits[:, eos_id] = -1.0e9
                logits[:, safe_id] = 1.0e6
        return logits

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds, kwargs
        self._ensure_model_loaded()
        out_device, out_dtype = self._resolve_device_dtype()

        infos = runtime_additional_information or [{}]
        sample_rate = int(getattr(self._pipeline, "sample_rate", 48000))
        async_chunk = bool(getattr(self.vllm_config.model_config, "async_chunk", False))

        # Determine request key for caching (stable across engine forward calls)
        request_key = "0"
        for info in infos:
            rk = info.get("_omni_req_id")
            if rk is not None:
                request_key = str(rk)
                break


        # --- VAE stage ---
        if self.model_stage in self._VAE_STAGES:
            if all(self._extract_val(info, "latent_audio_feat", None) is None for info in infos):
                self._ar_emit_stop_token = True
                return self._empty_output(out_device, out_dtype, infos, sample_rate, "model_outputs")
            outputs: list[torch.Tensor] = []
            for info in infos:
                latent = self._extract_val(info, "latent_audio_feat", None)
                audio = self._pipeline.decode(latent, trim_streaming_patch=async_chunk)
                outputs.append(audio.float().cpu())
            self._ar_emit_stop_token = True
            return self._build_output(outputs, sample_rate, out_device, out_dtype, "model_outputs")

        # --- Latent generator stage ---
        # Extract text from first info (only present on first engine call)
        text = ""
        for info in infos:
            text = self._extract_val(info, "text", "")
            if text:
                break

        # Dummy/warmup: no text and not a known request — return hidden states
        # sized to match input_ids so the model runner doesn't index OOB.
        if not text and request_key not in self._request_outputs and request_key not in self._latent_stream_gens:
            num_tokens = max(input_ids.shape[0] if input_ids is not None and input_ids.ndim > 0 else 1, 1)
            hidden = torch.zeros((num_tokens, 1), device=out_device, dtype=out_dtype)
            self._ar_emit_stop_token = True
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        # Already completed: keep emitting EOS with cached output
        if request_key in self._completed_requests:
            self._ar_emit_stop_token = True
            cached = self._request_outputs.get(request_key)
            if cached:
                return cached
            return self._empty_output(out_device, out_dtype, infos, sample_rate, "latent_audio_feat")

        # Build gen params from first call's info (text is available)
        info = infos[0]
        gen_params = {
            "text": text,
            "reference_audio": self._extract_val(info, "reference_audio", None),
            "prompt_audio": self._extract_val(info, "prompt_audio", None),
            "prompt_text": self._extract_val(info, "prompt_text", None),
            "control_instruction": self._extract_val(info, "control_instruction", None),
            "cfg_value": float(self._extract_val(info, "cfg_value", 2.0)),
            "inference_timesteps": int(self._extract_val(info, "inference_timesteps", 10)),
            "temperature": float(self._extract_val(info, "temperature", 1.0)),
            "top_p": float(self._extract_val(info, "top_p", 1.0)),
            "min_length": int(self._extract_val(info, "min_len", 2)),
            "max_length": int(self._extract_val(info, "max_len",
                               self._extract_val(info, "max_new_tokens", 4096))),
        }

        # --- Async chunk path: yield one chunk per forward() call ---
        if async_chunk:
            if request_key not in self._latent_stream_gens:
                self._latent_stream_gens[request_key] = self._pipeline.iter_latent_chunks_streaming(
                    **gen_params,
                    streaming_prefix_len=int(self._extract_val(info, "streaming_prefix_len", 3)),
                )
            generator = self._latent_stream_gens[request_key]
            try:
                chunk_latent, is_last = next(generator)
            except StopIteration:
                self._latent_stream_gens.pop(request_key, None)
                self._completed_requests.add(request_key)
                self._ar_emit_stop_token = True
                return self._empty_output(out_device, out_dtype, infos, sample_rate, "latent_audio_feat")
            if is_last:
                self._latent_stream_gens.pop(request_key, None)
                self._completed_requests.add(request_key)
            self._ar_emit_stop_token = is_last
            return self._build_output([chunk_latent.detach().float().cpu()], sample_rate, out_device, out_dtype, "latent_audio_feat")

        # --- Sync path: generate full latents in one call ---
        if self.model_stage in self._FULL_PIPELINE_STAGES:
            import numpy as np
            wav = self._pipeline._model.generate(
                text=gen_params["text"],
                reference_wav_path=gen_params.get("reference_audio"),
                prompt_wav_path=gen_params.get("prompt_audio"),
                prompt_text=gen_params.get("prompt_text"),
                cfg_value=gen_params["cfg_value"],
                inference_timesteps=gen_params["inference_timesteps"],
            )
            result = self._build_output(
                [torch.as_tensor(wav, dtype=torch.float32).reshape(-1)],
                sample_rate, out_device, out_dtype, "model_outputs",
            )
        else:
            latent = self._pipeline.generate_latents(**gen_params)
            result = self._build_output(
                [latent.float().cpu()], sample_rate, out_device, out_dtype, "latent_audio_feat",
            )

        # Cache and mark complete so subsequent engine calls emit EOS
        self._completed_requests.add(request_key)
        self._request_outputs[request_key] = result
        self._ar_emit_stop_token = True
        return result

    def _build_output(
        self,
        outputs: list[torch.Tensor],
        sample_rate: int,
        device: torch.device,
        dtype: torch.dtype,
        key: str,
    ) -> OmniOutput:
        """Build OmniOutput from a list of output tensors."""
        sample_rates = [torch.tensor(sample_rate, dtype=torch.int32) for _ in outputs]
        multimodal_outputs: dict[str, Any] = {key: outputs, "sr": sample_rates}

        if outputs and all(o.numel() > 0 for o in outputs):
            stacked = torch.stack(outputs)
            if stacked.ndim == 1:
                text_hidden_states = stacked.unsqueeze(-1)
            else:
                text_hidden_states = stacked.reshape(-1, max(stacked.shape[-1], 1))
        else:
            # Must have at least 1 row for the model runner's logits indexing
            text_hidden_states = torch.zeros((1, 1), device=device, dtype=dtype)
        text_hidden_states = text_hidden_states.to(device=device, dtype=dtype)
        return OmniOutput(text_hidden_states=text_hidden_states, multimodal_outputs=multimodal_outputs)

    def _empty_output(self, device, dtype, infos, sample_rate, key):
        return OmniOutput(
            text_hidden_states=torch.zeros((max(len(infos), 1), 1), device=device, dtype=dtype),
            multimodal_outputs={
                key: [torch.zeros((0,), dtype=torch.float32) for _ in infos],
                "sr": [torch.tensor(sample_rate, dtype=torch.int32) for _ in infos],
            },
        )

    def make_empty_intermediate_tensors(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        del batch_size, dtype, device
        return {}
