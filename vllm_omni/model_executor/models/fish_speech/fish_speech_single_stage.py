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
import time
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
_VOCODE_STRIDE = 10
# Initial stride for low-latency first audio chunk.
_INITIAL_VOCODE_STRIDE = 4
# Optional secondary device for vocoder (e.g. "cuda:1") to truly
# parallelize DAC compute with AR generation on different GPUs.
# Set via env var VLLM_OMNI_FISH_VOCODER_DEVICE.  None = same as AR.
_VOCODER_DEVICE_ENV = "VLLM_OMNI_FISH_VOCODER_DEVICE"


class FishSpeechSingleStageForConditionalGeneration(
    FishSpeechSlowARForConditionalGeneration,
):
    """Single-stage Fish Speech: AR + incremental re-vocode."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._dac_codec: nn.Module | None = None
        self._dac_sample_rate: int = DAC_SAMPLE_RATE
        self._dac_num_codebooks: int = DAC_NUM_CODEBOOKS
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
    def _cache_attention_masks(codec: nn.Module, device: torch.device) -> None:
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
                _device=device,
            ):
                if x_lens is not None:
                    return _orig(max_length, x_lens).to(device=_device, non_blocking=True)
                key = int(max_length)
                if key not in mask_cache:
                    mask_cache[key] = _orig(max_length, x_lens).to(device=_device)
                return mask_cache[key]

            def _cached_window_mask(
                max_length: int,
                x_lens: torch.Tensor | None = None,
                *,
                _orig=base_make_window_mask,
                _device=device,
            ):
                if x_lens is not None:
                    return _orig(max_length, x_lens).to(device=_device, non_blocking=True)
                key = int(max_length)
                if key not in window_mask_cache:
                    window_mask_cache[key] = _orig(max_length, x_lens).to(device=_device)
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
        # Convert numpy scalars to Python ints so torch.compile/Dynamo
        # does not synthesize from_numpy() wrappers that fail guard checks.
        if hasattr(codec, "hop_length"):
            codec.hop_length = int(codec.hop_length)
        if hasattr(codec, "frame_length"):
            codec.frame_length = int(codec.frame_length)
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
        # Cache attention masks AFTER .to(device) so cached masks land on
        # the same device as the rest of the codec (Triton-compiled kernels
        # can't accept CPU tensors as pointer arguments).
        self._cache_attention_masks(codec, vocoder_device)
        # Compile decode to fuse kernels and cut Python dispatch overhead.
        # The DAC has ~55 conv layers run eagerly; compile reduces ~570ms
        # CPU/request to a much smaller kernel-launch cost.
        try:
            codec.decode = torch.compile(
                codec.decode, mode="default", dynamic=True, fullgraph=False,
            )
            logger.info("Enabled torch.compile on DAC codec.decode")
        except Exception as exc:
            logger.warning("torch.compile on DAC codec.decode failed: %s", exc)
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
    def _decode_all(self, codes_fq: torch.Tensor) -> torch.Tensor:
        """Decode [N, Q] codes → waveform tensor (synchronous, blocking)."""
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

    @torch.no_grad()
    def _submit_decode_async(
        self, codes_fq: torch.Tensor,
    ) -> tuple[torch.cuda.Event, torch.Tensor] | None:
        """Submit decode on vocoder stream, return (done_event, result_tensor).

        Returns immediately without waiting.  The caller collects the
        result later by querying the event.  When vocoder is on a
        different GPU than AR, this enables true parallel execution.
        """
        self._ensure_dac_loaded()
        assert self._dac_codec is not None
        if self._vocoder_stream is None:
            # No async path available; fall back to sync.
            wav = self._decode_all(codes_fq)
            return None, wav  # signal caller to use directly

        vocoder_device = self._vocoder_device
        # Async cross-device copy onto vocoder stream.
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
            # Stay on GPU; CPU transfer happens when caller collects.
            audio_len_t = audio_lengths[0] if audio_lengths.numel() > 0 else None
            done_event = self._vocoder_stream.record_event()

        return done_event, (wav_batch[0, 0], audio_len_t)

    # -------------------- Override make_omni_output --------------------

    def _collect_pending(
        self, req_info: dict[str, Any]
    ) -> torch.Tensor | None:
        """Collect a previously submitted async decode; return delta wav or None."""
        pending = req_info.get("_pending_decode")
        if pending is None:
            return None
        done_event, wav_info, frames_at_submit = pending
        # Wait for the decode to finish (usually done already since ~10 AR
        # steps have elapsed since submit).
        t_sync = time.perf_counter()
        if done_event is not None:
            done_event.synchronize()
        sync_ms = (time.perf_counter() - t_sync) * 1000
        if sync_ms > 2.0:
            logger.info("collect sync wait %.2fms (frames=%d)", sync_ms, frames_at_submit)
        wav_gpu, audio_len_t = wav_info
        if audio_len_t is not None:
            audio_len = int(audio_len_t.item())
        else:
            audio_len = int(wav_gpu.shape[-1])
        full_wav = wav_gpu[:audio_len].to(dtype=torch.float32, device="cpu")
        emitted_samples = req_info.get("_emitted_samples", 0)
        delta = full_wav[emitted_samples:].contiguous()
        req_info["_emitted_samples"] = int(full_wav.numel())
        req_info["_last_vocoded_at"] = frames_at_submit
        req_info["_pending_decode"] = None
        return delta

    def make_omni_output(
        self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any,
    ) -> OmniOutput:
        """Pipelined async vocode: submit DAC on stride, collect on next.

        On each call:
          1. Accumulate the new frame into the per-request code buffer.
          2. If an async decode is pending, collect it (wait for event) and
             emit the delta audio.  This overlaps DAC compute with AR.
          3. If a new stride boundary is reached, submit an async decode
             on the vocoder stream.  The result will be collected next call.

        The delta for decode submitted at step N is emitted ~stride steps
        later (when decode N+1 is submitted).  Audio is thus delayed by one
        stride on the streaming path, but AR generation never stalls.
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
            # 1) Accumulate this step's codes (keep on GPU).
            if isinstance(all_codes_combined, torch.Tensor) and i < all_codes_combined.shape[0]:
                latest_codes = all_codes_combined[i:i + 1]
                valid = latest_codes.any(dim=1)
                if valid.any():
                    codes_list = req_info.get("_all_codes")
                    if codes_list is None:
                        codes_list = []
                        req_info["_all_codes"] = codes_list
                    codes_list.append(latest_codes[valid].detach())

            # 2) ALWAYS try to collect any pending async decode first.
            #    The DAC has had ~stride AR steps to finish, so sync is cheap.
            #    This ensures deltas are emitted ASAP and no tail is lost.
            delta_from_pending = self._collect_pending(req_info)

            codes_list = req_info.get("_all_codes")
            if not codes_list:
                deltas.append(delta_from_pending if delta_from_pending is not None else empty_wav)
                continue

            # 3) Check stride boundary for submitting a NEW decode.
            total_frames = sum(c.shape[0] for c in codes_list)
            last_vocoded_at = req_info.get("_last_vocoded_at", 0)
            new_since_vocode = total_frames - last_vocoded_at
            in_initial_phase = last_vocoded_at == 0
            stride = _INITIAL_VOCODE_STRIDE if in_initial_phase else _VOCODE_STRIDE

            if new_since_vocode < stride:
                deltas.append(delta_from_pending if delta_from_pending is not None else empty_wav)
                continue

            # 4) Consolidate list → single tensor (prevent fragmentation).
            all_codes = torch.cat(codes_list, dim=0)
            codes_list.clear()
            codes_list.append(all_codes)

            # 5) Submit a new async decode on the vocoder stream.
            try:
                t_voc = time.perf_counter()
                result = self._submit_decode_async(all_codes)
                submit_ms = (time.perf_counter() - t_voc) * 1000
            except Exception as exc:
                logger.error("DAC submit failed for req %d: %s", i, exc)
                deltas.append(delta_from_pending if delta_from_pending is not None else empty_wav)
                continue

            done_event, payload = result
            if done_event is None:
                # Sync fallback (no vocoder stream available).
                full_wav = payload.cpu() if isinstance(payload, torch.Tensor) else None
                if full_wav is None:
                    deltas.append(delta_from_pending if delta_from_pending is not None else empty_wav)
                    continue
                emitted_samples = req_info.get("_emitted_samples", 0)
                delta = full_wav[emitted_samples:].contiguous()
                req_info["_emitted_samples"] = int(full_wav.numel())
                req_info["_last_vocoded_at"] = total_frames
                if delta.numel() > 0:
                    deltas.append(delta)
                elif delta_from_pending is not None:
                    deltas.append(delta_from_pending)
                else:
                    deltas.append(empty_wav)
            else:
                # Async path: store pending, emit the previously pending
                # decode (if any) now.
                req_info["_pending_decode"] = (done_event, payload, total_frames)
                logger.info(
                    "submit_async %d frames in %.2fms (pending delta: %d samples)",
                    total_frames, submit_ms,
                    0 if delta_from_pending is None else delta_from_pending.numel(),
                )
                deltas.append(delta_from_pending if delta_from_pending is not None else empty_wav)

        # Pad deltas list to batch_size (in case there are fewer info_dicts).
        while len(deltas) < batch_size:
            deltas.append(empty_wav)

        return OmniOutput(
            text_hidden_states=parent_output.text_hidden_states,
            multimodal_outputs={
                "model_outputs": deltas,
                "sr": [sr_tensor] * batch_size,
            },
        )
