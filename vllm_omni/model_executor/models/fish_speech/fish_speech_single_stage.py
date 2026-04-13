"""Fish Speech S2 Pro -- Single-Stage model (AR + async DAC decode).

Folds DAC codec decoding into the Slow AR model so that AR generation
and audio synthesis run in one vLLM engine process.  This eliminates:
  - Second engine process and ``distributed_executor_backend: "mp"``
  - SharedMemoryConnector serialisation / polling
  - OmniGenerationScheduler overhead

DAC decode runs asynchronously on a CUDA stream separate from the main
AR forward stream.  Each chunk is submitted to the background stream
when enough frames accumulate; results are collected on the next call.
This mirrors two-stage parallelism without inter-process overhead.
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

_CHUNK_FRAMES = 25
_LEFT_CONTEXT_FRAMES = 0


class FishSpeechSingleStageForConditionalGeneration(FishSpeechSlowARForConditionalGeneration):
    """Single-stage Fish Speech: Slow AR + Fast AR + async DAC decode."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._dac_codec: nn.Module | None = None
        self._dac_sample_rate: int = DAC_SAMPLE_RATE
        self._dac_num_codebooks: int = DAC_NUM_CODEBOOKS
        # Async decode state
        self._dac_stream: torch.cuda.Stream | None = None

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
        self._dac_stream = torch.cuda.Stream(device=device)
        logger.info(
            "Single-stage DAC codec loaded from %s (device=%s)", codec_path, device
        )

    @torch.no_grad()
    def _decode_chunk(
        self, codes_fq: torch.Tensor, left_ctx: int = 0
    ) -> torch.Tensor:
        """Decode [F, Q] codes to audio, trimming left_ctx overlap."""
        self._ensure_dac_loaded()
        assert self._dac_codec is not None

        codec_device = next(self._dac_codec.parameters()).device
        codes_qf = codes_fq.to(device=codec_device).transpose(0, 1).long()
        total_frames = codes_qf.shape[1]
        feature_lengths = torch.tensor(
            [total_frames], device=codec_device, dtype=torch.long
        )

        with torch.amp.autocast("cuda", enabled=False):
            wav_batch, audio_lengths = self._dac_codec.decode(
                codes_qf.unsqueeze(0), feature_lengths
            )

        audio_len = (
            int(audio_lengths[0].item())
            if audio_lengths.numel() > 0
            else int(wav_batch.shape[-1])
        )
        wav = wav_batch[0, 0, :audio_len]

        if left_ctx > 0 and total_frames > 0:
            cut = int(left_ctx / total_frames * wav.shape[0])
            cut = max(0, min(cut, wav.shape[0]))
            if cut < wav.shape[0]:
                wav = wav[cut:]
            else:
                return torch.zeros((0,), dtype=torch.float32)

        return wav.to(dtype=torch.float32).reshape(-1)

    # -------------------- Async DAC helpers --------------------

    def _submit_async_decode(
        self, req_info: dict[str, Any], all_codes: torch.Tensor,
        chunk_start: int, chunk_end: int,
    ) -> None:
        """Submit a DAC chunk decode on the background CUDA stream."""
        ctx_start = max(0, chunk_start - _LEFT_CONTEXT_FRAMES)
        left_ctx = chunk_start - ctx_start
        chunk_codes = all_codes[ctx_start:chunk_end].clone()

        self._ensure_dac_loaded()
        assert self._dac_stream is not None

        # Record an event on the current (default) stream so the DAC
        # stream waits for any in-flight AR kernels to finish writing
        # the codes tensor before reading it.
        device = self.vllm_config.device_config.device
        current_stream = torch.cuda.current_stream(device)
        start_event = current_stream.record_event()

        # Launch decode on the DAC stream.
        with torch.cuda.stream(self._dac_stream):
            self._dac_stream.wait_event(start_event)
            try:
                wav = self._decode_chunk(chunk_codes, left_ctx)
            except Exception as exc:
                logger.error("Async DAC chunk decode failed: %s", exc)
                wav = torch.zeros((0,), dtype=torch.float32)
            # Record completion event.
            done_event = self._dac_stream.record_event()

        # Store pending result.
        pending = req_info.get("_pending_decode")
        if pending is None:
            pending = []
            req_info["_pending_decode"] = pending
        pending.append((done_event, wav))

    def _collect_async_results(self, req_info: dict[str, Any]) -> None:
        """Collect completed async DAC decode results into wav_chunks."""
        pending = req_info.get("_pending_decode")
        if not pending:
            return

        wav_chunks = req_info.get("_wav_chunks")
        if wav_chunks is None:
            wav_chunks = []
            req_info["_wav_chunks"] = wav_chunks

        still_pending = []
        for done_event, wav in pending:
            if done_event.query():
                # Already done — collect without blocking.
                if wav.numel() > 0:
                    wav_chunks.append(wav.cpu())
            else:
                # Not done yet — must wait (shouldn't happen often if
                # DAC is faster than _CHUNK_FRAMES AR steps).
                done_event.synchronize()
                if wav.numel() > 0:
                    wav_chunks.append(wav.cpu())
        req_info["_pending_decode"] = still_pending

    # -------------------- Override make_omni_output --------------------

    def make_omni_output(
        self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any
    ) -> OmniOutput:
        """Accumulate audio codes and decode asynchronously.

        Every _CHUNK_FRAMES new frames, submit a chunk decode on a
        background CUDA stream.  Collect finished results on each call.
        This keeps DAC decode off the critical AR forward path.
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

        # Collect any finished async decode results.
        self._collect_async_results(req_info)

        codes_list = req_info.get("_all_codes")
        if not codes_list:
            return OmniOutput(
                text_hidden_states=parent_output.text_hidden_states,
                multimodal_outputs={},
            )

        total_frames = sum(c.shape[0] for c in codes_list)
        decoded_up_to = req_info.get("_decoded_up_to", 0)
        new_frames = total_frames - decoded_up_to

        # Submit complete chunks for async decode.
        if new_frames >= _CHUNK_FRAMES:
            all_codes = torch.cat(codes_list, dim=0)

            n_chunks = new_frames // _CHUNK_FRAMES
            for i in range(n_chunks):
                chunk_start = decoded_up_to + i * _CHUNK_FRAMES
                chunk_end = chunk_start + _CHUNK_FRAMES
                self._submit_async_decode(req_info, all_codes, chunk_start, chunk_end)

            req_info["_decoded_up_to"] = decoded_up_to + n_chunks * _CHUNK_FRAMES

        # Also submit tail decode (< _CHUNK_FRAMES remaining).
        # The tail is re-submitted each step but only the latest result
        # is kept.  Since this runs on the background stream, it does NOT
        # block the AR forward path — the cost is just the GPU overlap.
        decoded_up_to = req_info.get("_decoded_up_to", 0)
        tail_frames = total_frames - decoded_up_to
        if tail_frames > 0:
            all_codes = torch.cat(codes_list, dim=0)
            ctx_start = max(0, decoded_up_to - _LEFT_CONTEXT_FRAMES)
            left_ctx = decoded_up_to - ctx_start
            chunk_codes = all_codes[ctx_start:total_frames].clone()

            self._ensure_dac_loaded()
            assert self._dac_stream is not None

            device = self.vllm_config.device_config.device
            current_stream = torch.cuda.current_stream(device)
            start_event = current_stream.record_event()

            with torch.cuda.stream(self._dac_stream):
                self._dac_stream.wait_event(start_event)
                try:
                    tail_wav = self._decode_chunk(chunk_codes, left_ctx)
                except Exception as exc:
                    logger.error("Async DAC tail decode failed: %s", exc)
                    tail_wav = torch.zeros((0,), dtype=torch.float32)
                done_event = self._dac_stream.record_event()

            req_info["_tail_pending"] = (done_event, tail_wav)
        else:
            req_info["_tail_pending"] = None

        # Return accumulated waveform (collected chunks + tail).
        wav_chunks = req_info.get("_wav_chunks")
        parts: list[torch.Tensor] = list(wav_chunks) if wav_chunks else []

        tail_pending = req_info.get("_tail_pending")
        if tail_pending is not None:
            done_event, tail_wav = tail_pending
            # Wait for tail — this is the only sync point, and it's
            # just the tail (< 25 frames), so very fast (~2ms).
            done_event.synchronize()
            if tail_wav.numel() > 0:
                parts.append(tail_wav.cpu())

        if parts:
            full_wav = torch.cat(parts, dim=0)
            sr_tensor = torch.tensor(self._dac_sample_rate, dtype=torch.int32)
            return OmniOutput(
                text_hidden_states=parent_output.text_hidden_states,
                multimodal_outputs={
                    "model_outputs": [full_wav],
                    "sr": [sr_tensor],
                },
            )

        return OmniOutput(
            text_hidden_states=parent_output.text_hidden_states,
            multimodal_outputs={},
        )
