# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MOSS-TTS-Nano single-stage model for vLLM-Omni.

Runs in a single AR worker stage.  The 0.1B AR LM and the
MOSS-Audio-Tokenizer-Nano codec are both loaded here.

Streaming is supported via the VoxCPM-style generator pattern:
  - On first forward() for a request, inference_stream() is started as
    a Python generator and stored in self._stream_gens[request_key].
  - Each subsequent forward() call pops one audio chunk from the generator
    and returns it as multimodal_outputs.
  - compute_logits() emits EOS only when the last chunk has been yielded,
    telling the AR scheduler to finish the request.

Weight loading deliberately happens inside load_weights() -- NOT __init__ --
so that vLLM initialises distributed state before any CUDA allocations occur.
"""

from __future__ import annotations

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
    """Patch torchaudio.load to use soundfile if torchcodec is unavailable."""
    try:
        import torchaudio

        torchaudio  # noqa
        import torchcodec  # noqa: F401

        return
    except Exception:
        pass

    import soundfile as sf

    def _soundfile_load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if frame_offset > 0:
            data = data[frame_offset:]
        if num_frames > 0:
            data = data[:num_frames]
        waveform = torch.from_numpy(data)
        if channels_first:
            waveform = waveform.T
        return waveform, sr

    def _soundfile_save(path, src, sample_rate, channels_first=True, **kwargs):
        wav = src.detach().cpu().float().numpy()
        if channels_first and wav.ndim == 2:
            wav = wav.T
        sf.write(str(path), wav, sample_rate)

    try:
        import torchaudio

        torchaudio.load = _soundfile_load
        torchaudio.save = _soundfile_save
        logger.info("Patched torchaudio.load/save to use soundfile (torchcodec unavailable)")
    except Exception as e:
        logger.warning("Could not patch torchaudio: %s", e)


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
_DEFAULT_MODE = "continuation"


def _pick(info: dict, key: str, default):
    """Extract scalar from additional_information dict (list or plain value)."""
    val = info.get(key, default)
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return val[0]
    return val if val is not None else default


class MossTTSNanoForGeneration(nn.Module):
    """Single-stage MOSS-TTS-Nano model with streaming audio output.

    Uses the VoxCPM-style generator pattern: inference_stream() is stored
    per-request and yields one audio chunk per forward() call.  The AR
    scheduler keeps the request alive until compute_logits() emits EOS.
    """

    requires_raw_input_tokens = True
    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    enable_update_additional_information = True
    inject_omni_request_id_into_runtime_info = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model_path: str = vllm_config.model_config.model

        self._lm: nn.Module | None = None
        self._audio_tokenizer: nn.Module | None = None
        self._device: torch.device | None = None
        self._lock = threading.Lock()

        # Per-request streaming generators (VoxCPM pattern).
        self._stream_gens: dict[str, Any] = {}
        # Controls whether compute_logits() emits EOS.
        self._ar_emit_stop_token = True

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        with self._lock:
            if self._lm is not None:
                return set()
            _patch_torchaudio_load()

            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._device = device

            if device.type == "cuda" and torch.cuda.is_bf16_supported():
                tts_dtype = torch.bfloat16
            elif device.type == "cuda":
                tts_dtype = torch.float16
            else:
                tts_dtype = torch.float32

            logger.info("Loading MOSS-TTS-Nano LM from %s (dtype=%s)", self.model_path, tts_dtype)
            from transformers import AutoModelForCausalLM

            lm = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=tts_dtype,
            )
            if device.type == "cuda":
                try:
                    import flash_attn  # noqa: F401

                    lm._set_attention_implementation("flash_attention_2")
                    logger.info("MOSS-TTS-Nano using flash_attention_2")
                except ImportError:
                    lm._set_attention_implementation("sdpa")
                    logger.info("MOSS-TTS-Nano using sdpa (flash_attn not installed)")
            lm.to(device=device)
            lm.eval()
            self._lm = lm
            logger.info("MOSS-TTS-Nano LM loaded on %s", device)

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
                torch_dtype=torch.float32,
            )
            audio_tokenizer.to(device=device)
            audio_tokenizer.eval()
            self._audio_tokenizer = audio_tokenizer
            logger.info("MOSS-Audio-Tokenizer-Nano loaded on %s", device)

        for _ in weights:
            pass
        return set()

    # ------------------------------------------------------------------
    # Dummy run support
    # ------------------------------------------------------------------

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict]:
        return [{"text": "hello", "voice": _DEFAULT_VOICE, "_is_dummy": True}] * num_reqs

    # ------------------------------------------------------------------
    # Streaming generator management
    # ------------------------------------------------------------------

    def _create_stream_gen(self, info: dict[str, Any]):
        """Create an inference_stream() generator for a request.

        Yields (waveform_tensor, is_last) tuples.
        """
        text: str = str(_pick(info, "text", "") or "")
        if not text.strip():
            logger.warning("MOSS-TTS-Nano received empty text; yielding silence.")
            sr = getattr(self.config, "audio_tokenizer_sample_rate", 48000)
            yield torch.zeros((sr,), dtype=torch.float32), True
            return

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
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

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

        device = self._device or torch.device("cpu")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        # Collect all audio events from inference_stream, then yield them
        # one by one.  We buffer first because inference_stream mixes
        # "audio" events with a final "result" event.
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
                        if chunk.ndim == 2:
                            chunk = chunk.T.reshape(-1)
                        else:
                            chunk = chunk.reshape(-1)
                        audio_chunks.append(chunk)
                        # Yield each chunk as it arrives (is_last=False).
                        yield chunk, False
                elif event_type == "result":
                    if not audio_chunks:
                        waveform = event.get("waveform")
                        if waveform is not None:
                            chunk = torch.as_tensor(waveform, dtype=torch.float32).cpu().reshape(-1)
                            yield chunk, True
                            return
        except Exception:
            logger.exception("MOSS-TTS-Nano inference failed for text=%r", text[:80])
        finally:
            try:
                Path(output_path).unlink(missing_ok=True)
            except Exception:
                pass

        # Signal completion. If we yielded audio chunks above, the last
        # yield was is_last=False, so emit a final empty sentinel.
        yield torch.zeros((0,), dtype=torch.float32), True

    # ------------------------------------------------------------------
    # Core forward pass (streaming, VoxCPM pattern)
    # ------------------------------------------------------------------

    def _make_dummy_hidden(self, input_ids: torch.Tensor | None) -> torch.Tensor:
        """Return a dummy hidden_states tensor for the AR runner."""
        device = self._device or torch.device("cpu")
        hidden = int(getattr(self.config, "hidden_size", 768))
        n = 1 if input_ids is None else max(1, input_ids.shape[0])
        return torch.zeros((n, hidden), device=device, dtype=torch.float32)

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
        sr = getattr(self.config, "audio_tokenizer_sample_rate", 48000)
        sr_tensor = torch.tensor(sr, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)
        hidden = self._make_dummy_hidden(input_ids)

        infos = runtime_additional_information or [{}]

        if not runtime_additional_information or all(info.get("_is_dummy") for info in infos):
            self._ar_emit_stop_token = True
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={
                    "model_outputs": [empty] * len(infos),
                    "sr": [sr_tensor] * len(infos),
                },
            )

        if self._lm is None or self._audio_tokenizer is None:
            raise RuntimeError("MOSS-TTS-Nano model not loaded.  Was load_weights() called?")

        outputs: list[torch.Tensor] = []
        srs: list[torch.Tensor] = []
        last_chunk_flags: list[bool] = []

        for info in infos:
            if info.get("_is_dummy"):
                outputs.append(empty)
                srs.append(sr_tensor)
                last_chunk_flags.append(True)
                continue

            request_key = str(info.get("_omni_req_id", "0"))

            # Create generator on first call for this request.
            if request_key not in self._stream_gens:
                self._stream_gens[request_key] = self._create_stream_gen(info)

            generator = self._stream_gens[request_key]
            try:
                chunk, is_last = next(generator)
            except StopIteration:
                self._stream_gens.pop(request_key, None)
                outputs.append(empty)
                last_chunk_flags.append(True)
            else:
                if is_last:
                    self._stream_gens.pop(request_key, None)
                outputs.append(chunk)
                last_chunk_flags.append(bool(is_last))

            srs.append(sr_tensor)

        # Emit EOS only when ALL requests in this batch have finished.
        self._ar_emit_stop_token = all(last_chunk_flags)

        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"model_outputs": outputs, "sr": srs},
        )

    # ------------------------------------------------------------------
    # AR runner interface
    # ------------------------------------------------------------------

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor:
        """Emit EOS or non-EOS logits to control AR scheduler lifetime.

        When _ar_emit_stop_token is True, the logits strongly favour EOS
        so the scheduler finishes the request.  Otherwise, a non-EOS token
        is favoured to keep the request alive for the next chunk.
        """
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        if hidden_states is None:
            device = self._device or torch.device("cpu")
            hidden_states = torch.zeros((0, 1), device=device, dtype=torch.float32)
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(-1)
        elif hidden_states.ndim > 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        vocab_size = int(getattr(self.config, "vocab_size", 32000))
        num_rows = int(hidden_states.shape[0])
        logits = torch.zeros(
            (num_rows, vocab_size),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        eos_id = 2 if vocab_size > 2 else 0
        safe_id = 1 if vocab_size > 1 and 1 != eos_id else 0
        if num_rows > 0:
            if self._ar_emit_stop_token:
                logits[:, eos_id] = 1.0e6
            else:
                logits[:, eos_id] = -1.0e9
                logits[:, safe_id] = 1.0e6
        return logits

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        hidden = int(getattr(self.config, "hidden_size", 768))
        return torch.zeros(
            (input_ids.shape[0], hidden),
            device=input_ids.device,
            dtype=torch.float32,
        )
