# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MOSS-TTS-Nano single-stage model for vLLM-Omni.

Runs in a single GPUGenerationWorker stage.  The 0.1B AR LM and the
MOSS-Audio-Tokenizer-Nano codec are both loaded here; the full
text → audio-codes → waveform pipeline executes inside forward().

Streaming (realtime) generation is supported via model.inference_stream():
audio chunks are collected as the LM emits tokens and returned as a
concatenated waveform.  Future work can expose per-chunk streaming via
vLLM-Omni's SSE audio endpoint.

Weight loading deliberately happens inside load_weights() -- NOT __init__ --
so that vLLM initialises distributed state before any CUDA allocations occur.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _patch_torchaudio_load() -> None:
    """Patch torchaudio.load to use soundfile if torchcodec is unavailable.

    torchaudio 2.10+ removed set_audio_backend() and defaults to torchcodec,
    which requires libnvrtc (missing on some servers).  soundfile handles all
    common audio formats (WAV, FLAC, OGG) without FFmpeg/NVRTC.
    """
    try:
        import torchaudio
        # Probe if the current torchaudio.load works.
        torchaudio  # noqa -- just importing is enough to check
        import torchcodec  # noqa: F401 -- will raise if broken
        return  # torchcodec is fine, no patch needed
    except Exception:
        pass

    import soundfile as sf
    import numpy as np

    def _soundfile_load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        # data shape: [samples, channels]
        if frame_offset > 0:
            data = data[frame_offset:]
        if num_frames > 0:
            data = data[:num_frames]
        waveform = torch.from_numpy(data)
        if normalize:
            # soundfile already gives float32 in [-1, 1] for integer formats
            pass
        if channels_first:
            waveform = waveform.T  # [channels, samples]
        return waveform, sr

    try:
        import torchaudio
        torchaudio.load = _soundfile_load
        logger.info("Patched torchaudio.load to use soundfile (torchcodec unavailable)")
    except Exception as e:
        logger.warning("Could not patch torchaudio.load: %s", e)


# Default sampling parameters matching the upstream demo defaults.
_DEFAULT_TEXT_TEMPERATURE = 1.0
_DEFAULT_TEXT_TOP_P = 1.0
_DEFAULT_TEXT_TOP_K = 50
_DEFAULT_AUDIO_TEMPERATURE = 0.8
_DEFAULT_AUDIO_TOP_P = 0.95
_DEFAULT_AUDIO_TOP_K = 25
_DEFAULT_AUDIO_REPETITION_PENALTY = 1.2
_DEFAULT_MAX_NEW_FRAMES = 375
_DEFAULT_VOICE = "Junhao"
_DEFAULT_MODE = "voice_clone"


def _pick(info: dict, key: str, default):
    """Extract scalar from additional_information dict (list or plain value)."""
    val = info.get(key, default)
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return val[0]
    return val if val is not None else default


class MossTTSNanoForGeneration(nn.Module):
    """Single-stage MOSS-TTS-Nano generation model.

    Integrates the 0.1B MOSS-TTS-Nano LM and the MOSS-Audio-Tokenizer-Nano
    codec into a single vLLM-Omni generation stage (GPUGenerationWorker).

    The model uses trust_remote_code=True to load the upstream HuggingFace
    model classes, keeping our implementation thin and easy to update as
    the upstream model evolves.

    Supported request fields in additional_information:
      text        (str)  – text to synthesize [required]
      voice       (str)  – built-in voice preset name, default "Junhao"
      mode        (str)  – "voice_clone" (default) or "continuation"
      prompt_audio_path (str) – path to reference WAV/MP3 for custom voice clone
      prompt_text (str)  – reference transcript (for continuation mode)
      max_new_frames    (int)   – max AR frames, default 375 (~14s audio)
      text_temperature  (float) – LM text-layer temperature
      text_top_p        (float) – LM text-layer top-p
      text_top_k        (int)   – LM text-layer top-k
      audio_temperature (float) – LM audio-layer temperature
      audio_top_p       (float) – LM audio-layer top-p
      audio_top_k       (int)   – LM audio-layer top-k
      audio_repetition_penalty (float) – LM audio repetition penalty
      seed        (int)  – optional random seed for reproducibility
    """

    requires_raw_input_tokens = True
    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    enable_update_additional_information = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model_path: str = vllm_config.model_config.model

        # Populated by load_weights(); kept None until then so __init__ is
        # allocation-free (vLLM may call __init__ before CUDA is ready).
        self._lm: nn.Module | None = None
        self._audio_tokenizer: nn.Module | None = None
        self._device: torch.device | None = None
        self._lock = threading.Lock()

        # Dummy parameter so vLLM can resolve the device via next(parameters()).
        self._sentinel = nn.Parameter(torch.zeros(1), requires_grad=False)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the MOSS-TTS-Nano LM and MOSS-Audio-Tokenizer-Nano codec.

        Both models are loaded via trust_remote_code so we stay in sync
        with any upstream architecture changes without reimplementing
        the model code ourselves.
        """
        with self._lock:
            if self._lm is not None:
                return set()

            # torchaudio 2.10+ dropped set_audio_backend() and defaults to
            # torchcodec which requires libnvrtc -- unavailable on some servers.
            # Patch torchaudio.load to fall back to soundfile so reference audio
            # loading works regardless of the FFmpeg/torchcodec installation.
            _patch_torchaudio_load()

            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self._device = device

            # Select compute dtype: bfloat16 on capable CUDA, float32 elsewhere.
            if device.type == "cuda" and torch.cuda.is_bf16_supported():
                tts_dtype = torch.bfloat16
            elif device.type == "cuda":
                tts_dtype = torch.float16
            else:
                tts_dtype = torch.float32

            # --- Load the AR language model ---
            logger.info("Loading MOSS-TTS-Nano LM from %s (dtype=%s)", self.model_path, tts_dtype)
            from transformers import AutoModelForCausalLM

            lm = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=tts_dtype,
            )
            if device.type == "cuda":
                # Use SDPA when flash_attention_2 is unavailable.
                try:
                    lm._set_attention_implementation("flash_attention_2")
                except Exception:
                    lm._set_attention_implementation("sdpa")
            lm.to(device=device)
            lm.eval()
            self._lm = lm
            logger.info("MOSS-TTS-Nano LM loaded on %s", device)

            # --- Load the audio tokenizer (codec) ---
            codec_path: str = getattr(
                self.config,
                "audio_tokenizer_pretrained_name_or_path",
                "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
            )
            logger.info("Loading MOSS-Audio-Tokenizer-Nano from %s", codec_path)
            from transformers import AutoModel

            audio_tokenizer = AutoModel.from_pretrained(
                codec_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # codec runs in fp32 for stability
            )
            audio_tokenizer.to(device=device)
            audio_tokenizer.eval()
            self._audio_tokenizer = audio_tokenizer
            logger.info("MOSS-Audio-Tokenizer-Nano loaded on %s", device)

        # Consume the vLLM weight iterator (we loaded weights ourselves above).
        for _ in weights:
            pass
        return set()

    # ------------------------------------------------------------------
    # Dummy run support (vLLM profiling / KV-cache estimation)
    # ------------------------------------------------------------------

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict]:
        return [{"text": "hello", "voice": _DEFAULT_VOICE, "_is_dummy": True}] * num_reqs

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Run full MOSS-TTS-Nano pipeline: text → audio codes → waveform.

        Each request in the batch is processed independently.  Results are
        returned as lists so the generation runner can demultiplex them.
        """
        sr = getattr(self.config, "audio_tokenizer_sample_rate", 48000)
        sr_tensor = torch.tensor(sr, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if not runtime_additional_information:
            # Profiling / dummy run — return empty audio.
            n = 1 if input_ids is None else max(1, input_ids.shape[0])
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty] * n,
                    "sr": [sr_tensor] * n,
                },
            )

        if self._lm is None or self._audio_tokenizer is None:
            raise RuntimeError("MOSS-TTS-Nano model not loaded.  Was load_weights() called?")

        waveforms: list[torch.Tensor] = []
        srs: list[torch.Tensor] = []

        for info in runtime_additional_information:
            if info.get("_is_dummy"):
                waveforms.append(empty)
                srs.append(sr_tensor)
                continue

            waveform = self._run_single_inference(info, sr)
            waveforms.append(waveform)
            srs.append(sr_tensor)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": waveforms, "sr": srs},
        )

    def _run_single_inference(self, info: dict[str, Any], sr: int) -> torch.Tensor:
        """Run inference for a single request and return the waveform tensor."""
        text: str = str(_pick(info, "text", "") or "")
        if not text.strip():
            logger.warning("MOSS-TTS-Nano received empty text; returning silence.")
            return torch.zeros((sr,), dtype=torch.float32)

        voice: str = str(_pick(info, "voice", _DEFAULT_VOICE))
        mode: str = str(_pick(info, "mode", _DEFAULT_MODE))
        prompt_audio_path: str | None = _pick(info, "prompt_audio_path", None)
        if prompt_audio_path is not None:
            prompt_audio_path = str(prompt_audio_path)
        prompt_text: str | None = _pick(info, "prompt_text", None)
        if prompt_text is not None:
            prompt_text = str(prompt_text)
        max_new_frames: int = int(_pick(info, "max_new_frames", _DEFAULT_MAX_NEW_FRAMES))
        seed: int | None = _pick(info, "seed", None)
        if seed is not None:
            seed = int(seed)

        sampling = {
            "text_temperature": float(_pick(info, "text_temperature", _DEFAULT_TEXT_TEMPERATURE)),
            "text_top_p": float(_pick(info, "text_top_p", _DEFAULT_TEXT_TOP_P)),
            "text_top_k": int(_pick(info, "text_top_k", _DEFAULT_TEXT_TOP_K)),
            "audio_temperature": float(_pick(info, "audio_temperature", _DEFAULT_AUDIO_TEMPERATURE)),
            "audio_top_p": float(_pick(info, "audio_top_p", _DEFAULT_AUDIO_TOP_P)),
            "audio_top_k": int(_pick(info, "audio_top_k", _DEFAULT_AUDIO_TOP_K)),
            "audio_repetition_penalty": float(
                _pick(info, "audio_repetition_penalty", _DEFAULT_AUDIO_REPETITION_PENALTY)
            ),
        }

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        device = self._device or torch.device("cpu")

        # Use a temp file as the output path (the upstream API requires a path).
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        audio_chunks: list[torch.Tensor] = []

        try:
            for event in self._lm.inference_stream(
                text=text,
                output_audio_path=output_path,
                mode=mode,
                prompt_text=prompt_text,
                prompt_audio_path=prompt_audio_path,
                text_tokenizer_path=self.model_path,
                audio_tokenizer=self._audio_tokenizer,
                device=device,
                nq=None,
                max_new_frames=max_new_frames,
                do_sample=True,
                use_kv_cache=True,
                **sampling,
            ):
                event_type = str(event.get("type", ""))
                if event_type == "audio":
                    waveform = event.get("waveform")
                    if waveform is not None:
                        chunk = torch.as_tensor(waveform, dtype=torch.float32).cpu()
                        audio_chunks.append(chunk)
                elif event_type == "result":
                    # Final event -- prefer the full waveform from the result
                    # if no intermediate chunks were collected.
                    if not audio_chunks:
                        waveform = event.get("waveform")
                        if waveform is not None:
                            audio_chunks.append(
                                torch.as_tensor(waveform, dtype=torch.float32).cpu()
                            )
        except Exception:
            logger.exception("MOSS-TTS-Nano inference failed for text=%r", text[:80])
            return torch.zeros((sr,), dtype=torch.float32)
        finally:
            # Clean up temp file.
            try:
                Path(output_path).unlink(missing_ok=True)
            except Exception:
                pass

        if not audio_chunks:
            logger.warning("MOSS-TTS-Nano produced no audio for text=%r", text[:80])
            return torch.zeros((sr,), dtype=torch.float32)

        # Concatenate all streaming chunks into a single waveform.
        # inference_stream yields chunks that are 1D or 2D [samples] / [channels, samples].
        # Flatten to 1D for the vLLM-Omni audio output format.
        processed: list[torch.Tensor] = []
        for chunk in audio_chunks:
            if chunk.ndim == 2:
                # [channels, samples] → interleave to [samples * channels] for stereo
                processed.append(chunk.T.reshape(-1))
            else:
                processed.append(chunk.reshape(-1))

        return torch.cat(processed, dim=0)

    # ------------------------------------------------------------------
    # vLLM boilerplate: embed_input_ids required by generation runner
    # ------------------------------------------------------------------

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        # Generation stage does not use token embeddings -- return a dummy
        # zero tensor of the expected shape so the runner's prefill works.
        hidden = int(getattr(self.config, "hidden_size", 768))
        return torch.zeros(
            (input_ids.shape[0], hidden),
            device=input_ids.device,
            dtype=torch.float32,
        )
