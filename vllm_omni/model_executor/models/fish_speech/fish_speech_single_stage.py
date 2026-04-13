"""Fish Speech S2 Pro -- Single-Stage model (AR + DAC decode in one stage).

Folds the DAC codec decoder into the Slow AR model so that AR generation
and audio synthesis run in one vLLM engine process.  This eliminates:
  - Second engine process and ``distributed_executor_backend: "mp"``
  - SharedMemoryConnector serialisation / polling
  - OmniGenerationScheduler overhead

Uses chunked incremental decode: every ``_CHUNK_FRAMES`` new frames,
decode only the new chunk (with left context overlap) instead of the
full history.  This keeps DAC cost O(1) per chunk.
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

from .dac_utils import DAC_HOP_LENGTH, DAC_NUM_CODEBOOKS, DAC_SAMPLE_RATE, build_dac_codec
from .fish_speech_slow_ar import FishSpeechSlowARForConditionalGeneration

logger = init_logger(__name__)

_CHUNK_FRAMES = 25
_LEFT_CONTEXT_FRAMES = 25


class FishSpeechSingleStageForConditionalGeneration(FishSpeechSlowARForConditionalGeneration):
    """Single-stage Fish Speech: Slow AR + Fast AR + inline DAC decode."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._dac_codec: nn.Module | None = None
        self._dac_sample_rate: int = DAC_SAMPLE_RATE
        self._dac_hop_length: int = DAC_HOP_LENGTH
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
            if not hasattr(module, "make_mask") or not hasattr(module, "make_window_limited_mask"):
                continue
            base_make_mask = module.make_mask
            base_make_window_mask = module.make_window_limited_mask
            mask_cache: dict[int, torch.Tensor] = {}
            window_mask_cache: dict[int, torch.Tensor] = {}

            def make_mask_cached(max_length: int, x_lens: torch.Tensor | None = None, *, _orig=base_make_mask):
                if x_lens is not None:
                    return _orig(max_length, x_lens)
                cached = mask_cache.get(int(max_length))
                if cached is None:
                    cached = _orig(max_length, x_lens)
                    mask_cache[int(max_length)] = cached
                return cached

            def make_window_mask_cached(max_length: int, x_lens: torch.Tensor | None = None, *, _orig=base_make_window_mask):
                if x_lens is not None:
                    return _orig(max_length, x_lens)
                cached = window_mask_cache.get(int(max_length))
                if cached is None:
                    cached = _orig(max_length, x_lens)
                    window_mask_cache[int(max_length)] = cached
                return cached

            module.make_mask = make_mask_cached
            module.make_window_limited_mask = make_window_mask_cached

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
        state_dict = torch.load(codec_path, map_location="cpu", weights_only=True)
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
        logger.info("Single-stage DAC codec loaded from %s (device=%s)", codec_path, device)

    # -------------------- DAC decode --------------------

    @torch.no_grad()
    def _decode_chunk(self, codes_fq: torch.Tensor, left_ctx: int = 0) -> torch.Tensor:
        """Decode [F, Q] codes to audio, trimming left_ctx frames."""
        self._ensure_dac_loaded()
        assert self._dac_codec is not None

        codec_device = next(self._dac_codec.parameters()).device
        codes_fq = codes_fq.to(device=codec_device)
        codes_qf = codes_fq.transpose(0, 1).to(dtype=torch.long)
        total_frames = codes_qf.shape[1]
        codes_bqf = codes_qf.unsqueeze(0)
        feature_lengths = torch.tensor([total_frames], device=codec_device, dtype=torch.long)

        with torch.amp.autocast("cuda", enabled=False):
            wav_batch, audio_lengths = self._dac_codec.decode(codes_bqf, feature_lengths)

        audio_len = int(audio_lengths[0].item()) if audio_lengths.numel() > 0 else int(wav_batch.shape[-1])
        wav = wav_batch[0, 0, :audio_len]

        if left_ctx > 0 and total_frames > 0:
            cut = int(left_ctx / total_frames * wav.shape[0])
            cut = max(0, min(cut, wav.shape[0]))
            if cut < wav.shape[0]:
                wav = wav[cut:]
            else:
                return torch.zeros((0,), dtype=torch.float32)

        return wav.to(dtype=torch.float32).reshape(-1)

    # -------------------- Override make_omni_output --------------------

    def make_omni_output(
        self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any
    ) -> OmniOutput:
        """Accumulate audio codes and decode chunks incrementally.

        Every _CHUNK_FRAMES new frames, decode a chunk (with left context)
        and append to the accumulated waveform.  O(1) per chunk, not O(N).
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

        # Accumulate the latest frame.
        if isinstance(latest_codes, torch.Tensor) and latest_codes.numel() > 0:
            if latest_codes.ndim == 1:
                latest_codes = latest_codes.unsqueeze(0)
            valid = latest_codes.any(dim=1)
            if valid.any():
                codes_list = req_info.get("_dac_all_codes")
                if codes_list is None:
                    codes_list = []
                    req_info["_dac_all_codes"] = codes_list
                codes_list.append(latest_codes[valid].detach().cpu())

        codes_list = req_info.get("_dac_all_codes")
        if not codes_list:
            return OmniOutput(
                text_hidden_states=parent_output.text_hidden_states,
                multimodal_outputs={},
            )

        total_frames = sum(c.shape[0] for c in codes_list)
        last_decoded = req_info.get("_dac_decoded_up_to", 0)
        new_frames = total_frames - last_decoded

        if new_frames < _CHUNK_FRAMES:
            # Not enough for a chunk.  Return accumulated audio so far
            # (or empty if no chunks decoded yet).
            wav_chunks = req_info.get("_dac_wav_chunks")
            if wav_chunks:
                full_wav = torch.cat(wav_chunks, dim=0)
                sr_tensor = torch.tensor(self._dac_sample_rate, dtype=torch.int32)
                return OmniOutput(
                    text_hidden_states=parent_output.text_hidden_states,
                    multimodal_outputs={"model_outputs": [full_wav], "sr": [sr_tensor]},
                )
            return OmniOutput(
                text_hidden_states=parent_output.text_hidden_states,
                multimodal_outputs={},
            )

        # Decode new chunk(s).
        all_codes = torch.cat(codes_list, dim=0)  # [total_frames, Q]

        # How many full chunks can we decode?
        chunks_to_decode = new_frames // _CHUNK_FRAMES
        wav_chunks = req_info.get("_dac_wav_chunks")
        if wav_chunks is None:
            wav_chunks = []
            req_info["_dac_wav_chunks"] = wav_chunks

        for i in range(chunks_to_decode):
            chunk_end = last_decoded + (i + 1) * _CHUNK_FRAMES
            ctx_start = max(0, last_decoded + i * _CHUNK_FRAMES - _LEFT_CONTEXT_FRAMES)
            chunk_codes = all_codes[ctx_start:chunk_end]
            left_ctx = last_decoded + i * _CHUNK_FRAMES - ctx_start

            try:
                wav = self._decode_chunk(chunk_codes, left_ctx)
                if wav.numel() > 0:
                    wav_chunks.append(wav.cpu())
            except Exception as exc:
                logger.error("DAC chunk decode failed: %s", exc)

        req_info["_dac_decoded_up_to"] = last_decoded + chunks_to_decode * _CHUNK_FRAMES

        if wav_chunks:
            full_wav = torch.cat(wav_chunks, dim=0)
            sr_tensor = torch.tensor(self._dac_sample_rate, dtype=torch.int32)
            return OmniOutput(
                text_hidden_states=parent_output.text_hidden_states,
                multimodal_outputs={"model_outputs": [full_wav], "sr": [sr_tensor]},
            )

        return OmniOutput(
            text_hidden_states=parent_output.text_hidden_states,
            multimodal_outputs={},
        )
