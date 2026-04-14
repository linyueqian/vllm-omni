"""Fish Speech S2 Pro -- Single-Stage model (AR + incremental vocode).

Folds codec decoding into the same engine process as AR generation.
DAC vocoder uses incremental chunked decode:

  Every ``_VOCODE_STRIDE`` new frames, decode only the new chunk
  (with left context overlap for smooth transitions) and emit the
  delta audio.  This approach:
  - Has near-zero audio truncation (final decode covers remaining frames)
  - Supports streaming (emit audio progressively)
  - Scales O(chunk_size) per stride, not O(total_frames)
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

from .dac_utils import DAC_NUM_CODEBOOKS, DAC_SAMPLE_RATE, DAC_HOP_LENGTH, build_dac_codec
from .fish_speech_slow_ar import FishSpeechSlowARForConditionalGeneration

logger = init_logger(__name__)

# Vocode stride: decode every N new frames.
_VOCODE_STRIDE = 10
# Initial stride for low-latency first audio chunk.
_INITIAL_VOCODE_STRIDE = 4
# Left context frames for smooth transitions in incremental decode.
_LEFT_CONTEXT_FRAMES = 4
# Optional secondary device for vocoder (e.g. "cuda:1").
_VOCODER_DEVICE_ENV = "VLLM_OMNI_FISH_VOCODER_DEVICE"


class FishSpeechSingleStageForConditionalGeneration(
    FishSpeechSlowARForConditionalGeneration,
):
    """Single-stage Fish Speech: AR + incremental chunked vocode."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._dac_codec: nn.Module | None = None
        self._dac_sample_rate: int = DAC_SAMPLE_RATE
        self._dac_num_codebooks: int = DAC_NUM_CODEBOOKS
        self._dac_hop_length: int = DAC_HOP_LENGTH
        self._vocoder_device: torch.device | None = None
        self._vocoder_stream: torch.cuda.Stream | None = None

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

        # Allow vocoder on a different device for true GPU parallelism.
        ar_device = self.vllm_config.device_config.device
        vocoder_device_str = os.environ.get(_VOCODER_DEVICE_ENV)
        if vocoder_device_str:
            try:
                vocoder_device = torch.device(vocoder_device_str)
            except Exception:
                logger.warning(
                    "Invalid %s=%s; using AR device",
                    _VOCODER_DEVICE_ENV, vocoder_device_str,
                )
                vocoder_device = ar_device
        else:
            vocoder_device = ar_device
        codec = codec.to(device=vocoder_device, dtype=torch.float32)
        codec.eval()
        self._dac_codec = codec
        self._vocoder_device = vocoder_device
        # Dedicated stream on vocoder device for true async decode.
        if vocoder_device.type == "cuda":
            self._vocoder_stream = torch.cuda.Stream(device=vocoder_device)
        logger.info(
            "Single-stage DAC codec loaded from %s (vocoder_device=%s, ar_device=%s)",
            codec_path, vocoder_device, ar_device,
        )

    # -------------------- DAC decode --------------------

    @torch.no_grad()
    def _decode_chunk(
        self, codes_fq: torch.Tensor,
    ) -> torch.Tensor:
        """Decode [N, Q] codes → waveform tensor (synchronous).

        Args:
            codes_fq: [num_frames, num_codebooks] codes to decode.

        Returns:
            Waveform tensor [samples] (float32, on CPU).
        """
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
        return wav_batch[0, 0, :audio_len].to(dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def _submit_decode_async(
        self, codes_fq: torch.Tensor,
    ) -> tuple[torch.cuda.Event | None, Any]:
        """Submit decode on vocoder stream, return (done_event, result).

        Returns immediately without waiting. When vocoder is on a different
        GPU than AR, this enables true parallel execution.
        """
        self._ensure_dac_loaded()
        assert self._dac_codec is not None
        if self._vocoder_stream is None:
            wav = self._decode_chunk(codes_fq)
            return None, wav

        vocoder_device = self._vocoder_device
        with torch.cuda.stream(self._vocoder_stream):
            codes_qf = codes_fq.to(
                device=vocoder_device, non_blocking=True,
            ).transpose(0, 1).long()
            total_frames = codes_qf.shape[1]
            feature_lengths = torch.tensor(
                [total_frames], device=vocoder_device, dtype=torch.long,
            )
            with torch.amp.autocast("cuda", enabled=False):
                wav_batch, audio_lengths = self._dac_codec.decode(
                    codes_qf.unsqueeze(0), feature_lengths,
                )
            audio_len_t = audio_lengths[0] if audio_lengths.numel() > 0 else None
            done_event = self._vocoder_stream.record_event()

        return done_event, (wav_batch[0, 0], audio_len_t)

    # -------------------- Override make_omni_output --------------------

    def make_omni_output(
        self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any,
    ) -> OmniOutput:
        """Accumulate codes; incremental chunked vocode at stride boundaries.

        Uses left-context overlap for smooth chunk transitions.  Only decodes
        ``left_context + new_frames`` each time (not all accumulated codes).
        """
        if isinstance(model_outputs, OmniOutput):
            parent_output = model_outputs
        else:
            parent_output = super().make_omni_output(model_outputs, **kwargs)

        sr_tensor = torch.tensor(self._dac_sample_rate, dtype=torch.int32)
        empty_wav = torch.zeros((0,), dtype=torch.float32)

        info_dicts = kwargs.get("model_intermediate_buffer") or kwargs.get(
            "runtime_additional_information"
        ) or []
        req_infos: list[dict[str, Any]] = [info for info in info_dicts if isinstance(info, dict)]
        batch_size = max(len(req_infos), 1)

        mm = parent_output.multimodal_outputs or {}
        all_codes_combined = mm.get("audio_codes")

        deltas: list[torch.Tensor] = []
        for i, req_info in enumerate(req_infos):
            # Append this step's codes to per-request accumulator (stay on GPU).
            if isinstance(all_codes_combined, torch.Tensor) and i < all_codes_combined.shape[0]:
                latest_codes = all_codes_combined[i:i + 1]
                # Use the first codebook (semantic) as validity check:
                # a valid frame always has a non-negative semantic code.
                has_valid = latest_codes[:, 0] >= 0
                if has_valid.any():
                    codes_list = req_info.get("_all_codes")
                    if codes_list is None:
                        codes_list = []
                        req_info["_all_codes"] = codes_list
                    # Keep on GPU -- avoid per-step D2H overhead.
                    codes_list.append(latest_codes[has_valid].detach())

            codes_list = req_info.get("_all_codes")
            if not codes_list:
                deltas.append(empty_wav)
                continue

            total_frames = sum(c.shape[0] for c in codes_list)
            last_vocoded_at = req_info.get("_last_vocoded_at", 0)
            new_since_vocode = total_frames - last_vocoded_at

            in_initial_phase = last_vocoded_at == 0
            stride = _INITIAL_VOCODE_STRIDE if in_initial_phase else _VOCODE_STRIDE
            if new_since_vocode < stride:
                deltas.append(empty_wav)
                continue

            # Consolidate list into a single tensor (prevents fragmentation
            # and makes subsequent cat operations O(1) amortized).
            all_codes = torch.cat(codes_list, dim=0)
            codes_list.clear()
            codes_list.append(all_codes)

            # Incremental chunked decode: left_context + new frames.
            ctx_start = max(0, last_vocoded_at - _LEFT_CONTEXT_FRAMES)
            chunk_codes = all_codes[ctx_start:]

            try:
                chunk_wav = self._decode_chunk(chunk_codes)
            except Exception as exc:
                logger.error("DAC vocode failed for req %d: %s", i, exc)
                deltas.append(empty_wav)
                continue

            # Trim left-context audio to get only new samples.
            ctx_frames = last_vocoded_at - ctx_start
            if ctx_frames > 0:
                # Each frame produces hop_length samples in DAC.
                ctx_samples = ctx_frames * self._dac_hop_length
                delta_wav = chunk_wav[ctx_samples:]
            else:
                delta_wav = chunk_wav

            req_info["_last_vocoded_at"] = total_frames
            deltas.append(delta_wav if delta_wav.numel() > 0 else empty_wav)

        while len(deltas) < batch_size:
            deltas.append(empty_wav)

        return OmniOutput(
            text_hidden_states=parent_output.text_hidden_states,
            multimodal_outputs={
                "model_outputs": deltas,
                "sr": [sr_tensor] * batch_size,
            },
        )
