"""Fish Speech S2 Pro -- Single-Stage model (AR + DAC decode in one stage).

Folds the DAC codec decoder into the Slow AR model so that after AR
generation finishes, codec codes are decoded to audio inline.  This
eliminates the overhead of a second vLLM engine, SharedMemoryConnector
serialisation, and OmniGenerationScheduler.

Inherits from ``FishSpeechSlowARForConditionalGeneration`` and overrides
``make_omni_output`` to run DAC waveform synthesis on the accumulated
audio codes before returning.
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


class FishSpeechSingleStageForConditionalGeneration(FishSpeechSlowARForConditionalGeneration):
    """Single-stage Fish Speech: Slow AR + Fast AR + inline DAC decode.

    Produces audio output directly (engine_output_type: audio) without
    needing a second stage for codec decoding.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._dac_codec: nn.Module | None = None
        self._dac_sample_rate: int = DAC_SAMPLE_RATE
        self._dac_hop_length: int = DAC_HOP_LENGTH
        self._dac_num_codebooks: int = DAC_NUM_CODEBOOKS

    # -------------------- DAC codec management --------------------

    def _bake_weight_norm(self, codec: nn.Module) -> None:
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

    def _cache_attention_masks(self, codec: nn.Module) -> None:
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
    def _decode_codes_to_audio(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Decode [num_frames, num_codebooks] codec codes to audio waveform.

        Args:
            audio_codes: Tensor of shape [num_frames, num_codebooks].

        Returns:
            1-D float32 audio tensor.
        """
        self._ensure_dac_loaded()
        assert self._dac_codec is not None

        q = self._dac_num_codebooks
        # audio_codes: [F, Q] -> codes_qf: [Q, F]
        codes_qf = audio_codes.transpose(0, 1).to(dtype=torch.long)
        frames = codes_qf.shape[1]

        # Batch dim for DAC: [1, Q, F]
        codes_bqf = codes_qf.unsqueeze(0)
        feature_lengths = torch.tensor([frames], device=codes_bqf.device, dtype=torch.long)

        with torch.amp.autocast("cuda", enabled=False):
            wav_batch, audio_lengths = self._dac_codec.decode(codes_bqf, feature_lengths)

        audio_len = int(audio_lengths[0].item()) if audio_lengths.numel() > 0 else int(wav_batch.shape[-1])
        wav = wav_batch[0, 0, :audio_len]
        return wav.to(dtype=torch.float32).reshape(-1)

    # -------------------- Override make_omni_output --------------------

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        """Override: decode accumulated audio codes to audio waveform inline.

        When audio_codes are present (from Fast AR), decode them via the DAC
        codec and return audio directly.  When no codes are available yet
        (e.g. during prefill), fall back to the parent's latent output.
        """
        if isinstance(model_outputs, OmniOutput):
            # Already an OmniOutput -- check if it has audio_codes to decode.
            mm = model_outputs.multimodal_outputs or {}
            audio_codes = mm.get("audio_codes")
            if isinstance(audio_codes, torch.Tensor) and audio_codes.numel() > 0:
                return self._make_audio_omni_output(model_outputs.text_hidden_states, audio_codes)
            return model_outputs

        # Not yet wrapped -- use parent to extract audio_codes from info_dicts.
        parent_output = super().make_omni_output(model_outputs, **kwargs)
        mm = parent_output.multimodal_outputs or {}
        audio_codes = mm.get("audio_codes")

        if isinstance(audio_codes, torch.Tensor) and audio_codes.numel() > 0:
            return self._make_audio_omni_output(parent_output.text_hidden_states, audio_codes)

        return parent_output

    def _make_audio_omni_output(
        self,
        text_hidden_states: torch.Tensor | None,
        audio_codes: torch.Tensor,
    ) -> OmniOutput:
        """Convert audio_codes → waveform and wrap in OmniOutput."""
        sr_tensor = torch.tensor(self._dac_sample_rate, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        # Filter zero-padded frames.
        if audio_codes.ndim == 2:
            valid_mask = audio_codes.any(dim=1)
            audio_codes = audio_codes[valid_mask]

        if audio_codes.numel() == 0:
            return OmniOutput(
                text_hidden_states=text_hidden_states,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        try:
            wav = self._decode_codes_to_audio(audio_codes)
        except Exception as exc:
            logger.error("DAC decode failed: %s", exc)
            wav = empty

        return OmniOutput(
            text_hidden_states=text_hidden_states,
            multimodal_outputs={"model_outputs": [wav], "sr": [sr_tensor]},
        )
