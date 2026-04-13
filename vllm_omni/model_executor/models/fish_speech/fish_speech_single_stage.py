"""Fish Speech S2 Pro -- Single-Stage model (AR + incremental re-vocode).

Folds codec decoding into the same engine process as AR generation.
DAC vocoder uses incremental re-vocode strategy (inspired by sgl-omni):

  Every ``_VOCODE_STRIDE`` new frames, re-decode ALL accumulated codes
  and emit only the new (delta) audio samples.  This approach:
  - Has ZERO audio truncation (final decode covers all frames)
  - Supports streaming (emit audio progressively)
  - Amortizes decode cost (~4 calls per request vs per-step)
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import remove_parametrizations
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .dac_utils import DAC_NUM_CODEBOOKS, DAC_SAMPLE_RATE, build_dac_codec
from .fish_speech_slow_ar import FishSpeechSlowARForConditionalGeneration

logger = init_logger(__name__)

# Re-vocode stride: decode every N new frames.
# Smaller = lower latency, more GPU work.  Larger = higher latency, less work.
_VOCODE_STRIDE = 25
# Initial stride for low-latency first audio chunk.
_INITIAL_VOCODE_STRIDE = 4


class FishSpeechSingleStageForConditionalGeneration(
    FishSpeechSlowARForConditionalGeneration,
):
    """Single-stage Fish Speech: AR + incremental re-vocode."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._dac_codec: nn.Module | None = None
        self._dac_sample_rate: int = DAC_SAMPLE_RATE
        self._dac_num_codebooks: int = DAC_NUM_CODEBOOKS

    # -------------------- DAC codec management --------------------

    @staticmethod
    def _bake_weight_norm(codec: nn.Module) -> None:
        baked = 0
        for module in codec.modules():
            parametrizations = getattr(module, "parametrizations", None)
            if not parametrizations:
                continue
            for name in list(parametrizations.keys()):
                remove_parametrizations(module, name, leave_parametrized=True)
                baked += 1
        if baked > 0:
            logger.info("Baked %d DAC parametrized weights for inference", baked)

    @staticmethod
    def _cache_attention_masks(codec: nn.Module) -> None:
        for module in codec.modules():
            if not hasattr(module, "make_mask") or not hasattr(
                module, "make_window_limited_mask"
            ):
                continue
            base_make_mask = module.make_mask
            base_make_window_mask = module.make_window_limited_mask
            mask_cache: dict[int, torch.Tensor] = {}
            window_mask_cache: dict[int, torch.Tensor] = {}

            def _cached_mask(
                max_length: int,
                x_lens: torch.Tensor | None = None,
                *,
                _orig=base_make_mask,
            ):
                if x_lens is not None:
                    return _orig(max_length, x_lens)
                key = int(max_length)
                if key not in mask_cache:
                    mask_cache[key] = _orig(max_length, x_lens)
                return mask_cache[key]

            def _cached_window_mask(
                max_length: int,
                x_lens: torch.Tensor | None = None,
                *,
                _orig=base_make_window_mask,
            ):
                if x_lens is not None:
                    return _orig(max_length, x_lens)
                key = int(max_length)
                if key not in window_mask_cache:
                    window_mask_cache[key] = _orig(max_length, x_lens)
                return window_mask_cache[key]

            module.make_mask = _cached_mask
            module.make_window_limited_mask = _cached_window_mask

    def _ensure_dac_loaded(self) -> None:
        if self._dac_codec is not None:
            return
        codec_path = os.path.join(self.model_path, "codec.pth")
        if not os.path.exists(codec_path):
            try:
                from transformers.utils.hub import cached_file

                cached = cached_file(self.model_path, "codec.pth")
                if cached is not None:
                    codec_path = cached
            except Exception:
                pass
        if not os.path.exists(codec_path):
            raise FileNotFoundError(f"codec.pth not found at {codec_path}.")

        codec = build_dac_codec()
        state_dict = torch.load(
            codec_path, map_location="cpu", weights_only=True,
        )
        if "generator" in state_dict:
            state_dict = state_dict["generator"]
        codec.load_state_dict(state_dict, strict=False)
        self._bake_weight_norm(codec)
        self._cache_attention_masks(codec)
        codec.encoder = None
        codec.quantizer.pre_module = None
        codec.quantizer.downsample = None

        device = self.vllm_config.device_config.device
        codec = codec.to(device=device, dtype=torch.float32)
        codec.eval()
        self._dac_codec = codec
        logger.info(
            "Single-stage DAC codec loaded from %s (device=%s)",
            codec_path, device,
        )

    # -------------------- DAC decode --------------------

    @torch.no_grad()
    def _decode_all(self, codes_fq: torch.Tensor) -> torch.Tensor:
        """Decode [N, Q] codes → waveform tensor."""
        self._ensure_dac_loaded()
        assert self._dac_codec is not None

        codec_device = next(self._dac_codec.parameters()).device
        codes_qf = codes_fq.to(device=codec_device).transpose(0, 1).long()
        total_frames = codes_qf.shape[1]
        feature_lengths = torch.tensor(
            [total_frames], device=codec_device, dtype=torch.long,
        )

        with torch.amp.autocast("cuda", enabled=False):
            wav_batch, audio_lengths = self._dac_codec.decode(
                codes_qf.unsqueeze(0), feature_lengths,
            )

        audio_len = (
            int(audio_lengths[0].item())
            if audio_lengths.numel() > 0
            else int(wav_batch.shape[-1])
        )
        return wav_batch[0, 0, :audio_len].to(dtype=torch.float32).reshape(-1)

    # -------------------- Override make_omni_output --------------------

    def make_omni_output(
        self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any,
    ) -> OmniOutput:
        """Accumulate codes; re-vocode all codes at stride boundaries.

        Instead of chunked decode (which truncates tail frames), we
        re-decode ALL accumulated codes every _VOCODE_STRIDE frames and
        emit only the delta audio.  The waveform grows monotonically
        and always covers all frames — zero truncation.
        """
        if isinstance(model_outputs, OmniOutput):
            parent_output = model_outputs
        else:
            parent_output = super().make_omni_output(model_outputs, **kwargs)

        mm = parent_output.multimodal_outputs or {}
        latest_codes = mm.get("audio_codes")

        info_dicts = kwargs.get("model_intermediate_buffer") or kwargs.get(
            "runtime_additional_information"
        ) or []

        req_info: dict[str, Any] | None = None
        for info in info_dicts:
            if isinstance(info, dict):
                req_info = info
                break

        if req_info is None:
            return OmniOutput(
                text_hidden_states=parent_output.text_hidden_states,
                multimodal_outputs={},
            )

        # Accumulate latest frame.
        if isinstance(latest_codes, torch.Tensor) and latest_codes.numel() > 0:
            if latest_codes.ndim == 1:
                latest_codes = latest_codes.unsqueeze(0)
            valid = latest_codes.any(dim=1)
            if valid.any():
                codes_list = req_info.get("_all_codes")
                if codes_list is None:
                    codes_list = []
                    req_info["_all_codes"] = codes_list
                codes_list.append(latest_codes[valid].detach().cpu())

        codes_list = req_info.get("_all_codes")
        if not codes_list:
            return OmniOutput(
                text_hidden_states=parent_output.text_hidden_states,
                multimodal_outputs={},
            )

        total_frames = sum(c.shape[0] for c in codes_list)
        last_vocoded_at = req_info.get("_last_vocoded_at", 0)
        new_since_vocode = total_frames - last_vocoded_at

        # Determine if it's time to re-vocode.
        in_initial_phase = last_vocoded_at == 0
        stride = _INITIAL_VOCODE_STRIDE if in_initial_phase else _VOCODE_STRIDE
        should_vocode = new_since_vocode >= stride

        if not should_vocode:
            return OmniOutput(
                text_hidden_states=parent_output.text_hidden_states,
                multimodal_outputs={},
            )

        # Re-vocode all codes; emit only the delta audio (new samples
        # since last emission) so the streaming output is correct.
        all_codes = torch.cat(codes_list, dim=0)
        try:
            full_wav = self._decode_all(all_codes).cpu()
        except Exception as exc:
            logger.error("DAC re-vocode failed: %s", exc)
            return OmniOutput(
                text_hidden_states=parent_output.text_hidden_states,
                multimodal_outputs={},
            )

        emitted_samples = req_info.get("_emitted_samples", 0)
        delta_wav = full_wav[emitted_samples:]
        req_info["_emitted_samples"] = int(full_wav.numel())
        req_info["_last_vocoded_at"] = total_frames

        if delta_wav.numel() == 0:
            return OmniOutput(
                text_hidden_states=parent_output.text_hidden_states,
                multimodal_outputs={},
            )

        sr_tensor = torch.tensor(self._dac_sample_rate, dtype=torch.int32)
        return OmniOutput(
            text_hidden_states=parent_output.text_hidden_states,
            multimodal_outputs={
                "model_outputs": [delta_wav],
                "sr": [sr_tensor],
            },
        )
