"""Fish Speech S2 Pro -- Single-Stage model (AR + DAC decode in one stage).

Folds the DAC codec decoder into the Slow AR model so that AR generation
and audio synthesis run in one vLLM engine process.  This eliminates:
  - Second engine process and ``distributed_executor_backend: "mp"``
  - SharedMemoryConnector serialisation / polling
  - OmniGenerationScheduler overhead

Streaming is supported: ``make_omni_output`` tracks per-request decode
state via ``model_intermediate_buffer`` and emits incremental audio
chunks every ``codec_chunk_frames`` frames with left-context overlap
for smooth transitions (same chunking logic as the two-stage
async_chunk connector, but inline).
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

# Chunked streaming defaults (match two-stage connector config).
_DEFAULT_CHUNK_FRAMES = 25
_DEFAULT_LEFT_CONTEXT_FRAMES = 25
_DEFAULT_INITIAL_CHUNK_FRAMES = 4


class FishSpeechSingleStageForConditionalGeneration(FishSpeechSlowARForConditionalGeneration):
    """Single-stage Fish Speech: Slow AR + Fast AR + inline DAC decode.

    Produces audio output directly (engine_output_type: audio) without
    needing a second stage for codec decoding.  Supports per-step
    streaming via chunked DAC decode with configurable chunk size and
    left-context overlap.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._dac_codec: nn.Module | None = None
        self._dac_sample_rate: int = DAC_SAMPLE_RATE
        self._dac_hop_length: int = DAC_HOP_LENGTH
        self._dac_num_codebooks: int = DAC_NUM_CODEBOOKS

        # Chunked streaming config.
        self._chunk_frames: int = _DEFAULT_CHUNK_FRAMES
        self._left_context_frames: int = _DEFAULT_LEFT_CONTEXT_FRAMES
        self._initial_chunk_frames: int = _DEFAULT_INITIAL_CHUNK_FRAMES

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

            def make_mask_cached(
                max_length: int,
                x_lens: torch.Tensor | None = None,
                *,
                _orig=base_make_mask,
            ):
                if x_lens is not None:
                    return _orig(max_length, x_lens)
                key = int(max_length)
                cached = mask_cache.get(key)
                if cached is None:
                    cached = _orig(max_length, x_lens)
                    mask_cache[key] = cached
                return cached

            def make_window_mask_cached(
                max_length: int,
                x_lens: torch.Tensor | None = None,
                *,
                _orig=base_make_window_mask,
            ):
                if x_lens is not None:
                    return _orig(max_length, x_lens)
                key = int(max_length)
                cached = window_mask_cache.get(key)
                if cached is None:
                    cached = _orig(max_length, x_lens)
                    window_mask_cache[key] = cached
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
            raise FileNotFoundError(
                f"codec.pth not found at {codec_path}. "
                "Make sure the Fish Speech S2 Pro model includes codec.pth."
            )

        codec = build_dac_codec()
        state_dict = torch.load(codec_path, map_location="cpu", weights_only=True)
        if "generator" in state_dict:
            state_dict = state_dict["generator"]
        codec.load_state_dict(state_dict, strict=False)
        self._bake_weight_norm(codec)
        self._cache_attention_masks(codec)

        # Prune encoder-only components to save GPU memory.
        codec.encoder = None
        codec.quantizer.pre_module = None
        codec.quantizer.downsample = None

        device = self.vllm_config.device_config.device
        codec = codec.to(device=device, dtype=torch.float32)
        codec.eval()
        self._dac_codec = codec

        logger.info(
            "Single-stage DAC codec loaded from %s (device=%s, sr=%d)",
            codec_path,
            device,
            self._dac_sample_rate,
        )

    # -------------------- DAC decode --------------------

    @torch.no_grad()
    def _decode_codes_to_audio(
        self,
        codes_fq: torch.Tensor,
        left_context_frames: int = 0,
    ) -> torch.Tensor:
        """Decode [num_frames, num_codebooks] codec codes to audio waveform.

        Args:
            codes_fq: Tensor of shape [num_frames, num_codebooks].
            left_context_frames: Number of leading context frames to trim
                from the decoded audio (for overlap-add streaming).

        Returns:
            1-D float32 audio tensor (context-trimmed).
        """
        self._ensure_dac_loaded()
        assert self._dac_codec is not None

        # codes_fq: [F, Q] -> [Q, F]
        codes_qf = codes_fq.transpose(0, 1).to(dtype=torch.long)
        total_frames = codes_qf.shape[1]

        # Batch dim: [1, Q, F]
        codes_bqf = codes_qf.unsqueeze(0)
        feature_lengths = torch.tensor(
            [total_frames], device=codes_bqf.device, dtype=torch.long
        )

        with torch.amp.autocast("cuda", enabled=False):
            wav_batch, audio_lengths = self._dac_codec.decode(
                codes_bqf, feature_lengths
            )

        audio_len = (
            int(audio_lengths[0].item())
            if audio_lengths.numel() > 0
            else int(wav_batch.shape[-1])
        )
        wav = wav_batch[0, 0, :audio_len]

        # Trim left context using proportional cut (same as DAC decoder stage).
        if left_context_frames > 0 and total_frames > 0:
            cut = int(left_context_frames / total_frames * wav.shape[0])
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
        """Decode accumulated audio codes to audio waveform inline.

        Implements chunked streaming: tracks per-request decode state in
        the model_intermediate_buffer and emits incremental audio chunks
        every ``codec_chunk_frames`` frames with left-context overlap.
        """
        # Use parent to extract audio_codes from info_dicts.
        if isinstance(model_outputs, OmniOutput):
            parent_output = model_outputs
        else:
            parent_output = super().make_omni_output(model_outputs, **kwargs)

        mm = parent_output.multimodal_outputs or {}
        audio_codes = mm.get("audio_codes")

        if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
            return parent_output

        # Filter zero-padded frames.
        if audio_codes.ndim == 2:
            valid_mask = audio_codes.any(dim=1)
            audio_codes = audio_codes[valid_mask]

        total_frames = audio_codes.shape[0] if audio_codes.ndim == 2 else 0
        if total_frames == 0:
            return parent_output

        # --- Per-request chunked decode state ---
        # Read/write decode progress from model_intermediate_buffer.
        info_dicts = kwargs.get("model_intermediate_buffer") or kwargs.get(
            "runtime_additional_information"
        ) or []

        last_decoded: int = 0
        req_info: dict[str, Any] | None = None
        for info in info_dicts:
            if isinstance(info, dict):
                last_decoded = info.get("_dac_last_decoded", 0)
                req_info = info
                break

        new_frames = total_frames - last_decoded
        chunk_size = self._chunk_frames
        initial_chunk = self._initial_chunk_frames
        left_ctx = self._left_context_frames

        # Determine if we should decode a chunk now.
        in_initial_phase = initial_chunk > 0 and total_frames <= chunk_size
        should_decode = False
        context_frames = 0

        if in_initial_phase:
            # Small initial chunks for fast TTFA.
            if new_frames >= initial_chunk:
                should_decode = True
                context_frames = max(0, last_decoded)
        else:
            # Regular chunks.
            if new_frames >= chunk_size:
                should_decode = True
                context_frames = min(left_ctx, last_decoded)

        if not should_decode:
            # Not enough frames for a chunk yet.  Return an OmniOutput
            # with empty multimodal_outputs so the output processor
            # does not forward raw codec codes to the serving layer.
            return OmniOutput(
                text_hidden_states=parent_output.text_hidden_states,
                multimodal_outputs={},
            )

        # Select the window to decode: [context + new frames].
        window_start = max(0, total_frames - new_frames - context_frames)
        # Snap to chunk boundary for regular phase.
        if not in_initial_phase:
            decode_end = last_decoded + chunk_size
            decode_end = min(decode_end, total_frames)
            window_start = max(0, decode_end - chunk_size - context_frames)
        else:
            decode_end = total_frames

        window_codes = audio_codes[window_start:decode_end]
        actual_context = max(0, last_decoded - window_start)

        sr_tensor = torch.tensor(self._dac_sample_rate, dtype=torch.int32)

        try:
            wav = self._decode_codes_to_audio(window_codes, actual_context)
        except Exception as exc:
            logger.error("DAC decode failed: %s", exc)
            wav = torch.zeros((0,), dtype=torch.float32)

        # Update per-request decode state.
        if req_info is not None:
            req_info["_dac_last_decoded"] = decode_end

        return OmniOutput(
            text_hidden_states=parent_output.text_hidden_states,
            multimodal_outputs={"model_outputs": [wav], "sr": [sr_tensor]},
        )
