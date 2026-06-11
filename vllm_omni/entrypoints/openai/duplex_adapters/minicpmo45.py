from __future__ import annotations

import base64
import binascii
import os
from typing import Any

import numpy as np
from vllm.multimodal.media import MediaConnector

from vllm_omni.entrypoints.openai.protocol.duplex import DuplexSessionConfig
from vllm_omni.model_executor.models.minicpmo_4_5.duplex_policy import MiniCPMO45DuplexPolicy


class MiniCPMO45PcmAppendBuffer:
    """Accumulates short native-duplex PCM chunks into model-sized appends."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._sample_rate_hz: int | None = None
        self._force_listen = False
        self._emitted_first = False

    def clear(self) -> None:
        self._buffer.clear()
        self._sample_rate_hz = None
        self._force_listen = False
        self._emitted_first = False

    def clear_force_listen(self) -> None:
        self._force_listen = False

    def has_pending(self) -> bool:
        return bool(self._buffer)

    def append(
        self,
        payload: dict[str, object],
        *,
        chunk_period_ms: int,
        flush: bool = False,
        allow_emit: bool = True,
    ) -> dict[str, object] | None:
        fmt = payload.get("format")
        sample_rate_hz = payload.get("sample_rate_hz")
        audio = payload.get("audio")
        if fmt != "pcm_f32le" or not isinstance(sample_rate_hz, int) or not isinstance(audio, str):
            return payload
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError):
            return payload
        if len(raw) % 4 != 0:
            return payload

        if self._sample_rate_hz is not None and self._sample_rate_hz != sample_rate_hz:
            raise ValueError("MiniCPM-o native duplex audio append sample_rate_hz changed within a session")
        self._sample_rate_hz = sample_rate_hz
        self._buffer.extend(raw)
        self._force_listen = self._force_listen or bool(payload.get("force_listen", False))
        if not allow_emit:
            return None

        min_samples = max(1, int(sample_rate_hz * max(1, int(chunk_period_ms)) / 1000))
        buffered_samples = len(self._buffer) // 4
        if not flush and buffered_samples < min_samples:
            return None

        # Emit whole model chunks only: the engine reserves scheduler slots
        # from the payload size and the worker consumes one unit per complete
        # chunk, so partial chunks would turn into pad embeddings inside the
        # model KV. On flush, the tail is zero-padded (silence) up to the
        # chunk boundary instead. The very first emission is capped at one
        # chunk because the worker's first unit consumes the official
        # first-chunk window (1035 ms) rather than a plain chunk period;
        # any remainder stays buffered for the next unit, like the official
        # streaming_prefill buffer.
        if not self._emitted_first:
            emit_samples = min(min_samples, buffered_samples)
            pad_samples = min_samples - emit_samples if flush else 0
            if not flush and emit_samples < min_samples:
                return None
        elif flush:
            emit_samples = buffered_samples
            remainder = emit_samples % min_samples
            pad_samples = (min_samples - remainder) if remainder else 0
        else:
            emit_samples = buffered_samples - (buffered_samples % min_samples)
            pad_samples = 0
        emit_bytes = emit_samples * 4
        emit_raw = bytes(self._buffer[:emit_bytes]) + b"\x00" * (pad_samples * 4)
        del self._buffer[:emit_bytes]
        self._emitted_first = True

        out = dict(payload)
        out["audio"] = base64.b64encode(emit_raw).decode("ascii")
        out["sample_rate_hz"] = sample_rate_hz
        out["force_listen"] = self._force_listen
        self._force_listen = False
        return out

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        if not self._buffer:
            return None
        payload: dict[str, object] = {
            "type": "audio",
            "audio": "",
            "format": "pcm_f32le",
            "sample_rate_hz": self._sample_rate_hz or 16000,
            "force_listen": self._force_listen,
        }
        return self.append(payload, chunk_period_ms=chunk_period_ms, flush=True)


class MiniCPMO45NativeDuplexServingAdapter:
    """Serving-side MiniCPM-o 4.5 native duplex session preparation.

    The generic duplex WebSocket handler should not let client-supplied local
    paths reach workers.  This adapter follows the existing media connector
    boundary: serving resolves client media URIs, then workers receive only
    normalized PCM payloads.  Server-owned model assets remain local paths here.
    """

    @classmethod
    def is_enabled(cls, config: DuplexSessionConfig) -> bool:
        extra_body = config.extra_body
        explicit = extra_body.get("native_duplex") or extra_body.get("minicpmo45_native_duplex")
        if explicit is not None:
            return bool(explicit)
        mode = extra_body.get("duplex_mode") or extra_body.get("runtime")
        return isinstance(mode, str) and mode.lower() in {"native", "model_native", "minicpmo45_native"}

    @classmethod
    async def prepare_session_config(cls, config: DuplexSessionConfig, *, model_config: Any) -> None:
        extra_body = dict(config.extra_body)
        if any(key in extra_body for key in ("ref_audio_path", "tts_ref_audio_path")):
            raise ValueError("ref_audio_path is not accepted by native duplex; use ref_audio URI instead")
        cls._apply_default_scheduler_policy(extra_body, model_config=model_config)

        ref_audio = config.ref_audio
        if ref_audio is None and isinstance(extra_body.get("ref_audio"), str):
            ref_audio = extra_body.pop("ref_audio")
        if ref_audio is None and isinstance(extra_body.get("tts_ref_audio"), str):
            ref_audio = extra_body.pop("tts_ref_audio")

        if ref_audio is None:
            default_ref = cls._default_ref_audio_path(config, model_config=model_config)
            if default_ref is None:
                cls._apply_first_append_context_tokens(
                    extra_body,
                    model_config=model_config,
                    instructions=config.instructions,
                    ref_sample_count=None,
                )
                config.extra_body = extra_body
                return
            wav_np, sr = cls._load_local_ref_audio(default_ref)
        else:
            wav_np, sr = await cls.resolve_ref_audio(ref_audio, model_config=model_config)

        wav_np = cls.normalize_ref_audio(wav_np, int(sr), target_sr=16000)
        # Trim to a whole number of pooled audio embeddings (100 ms frames) so
        # the first-append scheduler reserve can count them exactly.
        usable = (len(wav_np) // MiniCPMO45DuplexPolicy.SAMPLES_PER_AUDIO_TOKEN) * (
            MiniCPMO45DuplexPolicy.SAMPLES_PER_AUDIO_TOKEN
        )
        wav_np = wav_np[:usable]
        ref_audio_bytes = np.ascontiguousarray(wav_np, dtype=np.float32).tobytes()
        extra_body["ref_audio_data"] = base64.b64encode(ref_audio_bytes).decode("ascii")
        extra_body["ref_audio_format"] = "pcm_f32le"
        extra_body["ref_audio_sample_rate_hz"] = 16000
        cls._apply_first_append_context_tokens(
            extra_body,
            model_config=model_config,
            instructions=config.instructions,
            ref_sample_count=len(wav_np),
        )
        config.extra_body = extra_body
        config.ref_audio = None

    @classmethod
    def _apply_default_scheduler_policy(cls, extra_body: dict[str, object], *, model_config: Any) -> None:
        if "duplex_stage_max_tokens" in extra_body:
            raw_stage0 = None
        else:
            raw_stage0 = extra_body.get("duplex_stage0_max_tokens")
            if not isinstance(raw_stage0, int | float) or int(raw_stage0) <= 0:
                raw_stage0 = 20
            extra_body["duplex_stage_max_tokens"] = {"0": int(raw_stage0)}
        if "duplex_stage_sampling_params" not in extra_body:
            stage0_params: dict[str, object] = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 100,
                "repetition_penalty": 1.05,
            }
            stop_token_ids = cls._native_stage0_stop_token_ids(model_config)
            if stop_token_ids:
                stage0_params["stop_token_ids"] = stop_token_ids
            extra_body["duplex_stage_sampling_params"] = {"0": stage0_params}
        if "duplex_scheduler_token_id" not in extra_body:
            scheduler_token_id = cls._native_scheduler_token_id(model_config)
            if scheduler_token_id is not None:
                extra_body["duplex_scheduler_token_id"] = scheduler_token_id

    @classmethod
    def _apply_first_append_context_tokens(
        cls,
        extra_body: dict[str, object],
        *,
        model_config: Any,
        instructions: object,
        ref_sample_count: int | None,
    ) -> None:
        """Precompute the exact session-context token count for the engine.

        The first data-plane append carries the system template and optional
        reference-audio embeddings ahead of the first unit. The engine
        reserves scheduler slots from this count; an inexact count turns into
        pad embeddings inside the model KV (surplus) or truncated context
        (deficit), so it is computed with the same template and pooling math
        the worker uses.
        """
        if "duplex_first_append_context_tokens" in extra_body:
            return
        tokenizer = cls._load_native_tokenizer(model_config)
        if tokenizer is None:
            return
        prefix, suffix = MiniCPMO45DuplexPolicy.session_context_texts(
            instructions,
            ref_sample_count is not None,
        )
        try:
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        except Exception:
            return
        ref_tokens = MiniCPMO45DuplexPolicy.audio_token_count(ref_sample_count or 0)
        extra_body["duplex_first_append_context_tokens"] = len(prefix_ids) + ref_tokens + len(suffix_ids)

    @staticmethod
    def _native_stage0_stop_token_ids(model_config: Any) -> list[int]:
        tokenizer = MiniCPMO45NativeDuplexServingAdapter._load_native_tokenizer(model_config)
        if tokenizer is None:
            return []
        out: list[int] = []
        stop_token_fields = (
            "chunk_eos_token_id",
            "chunk_tts_eos_token_id",
            "listen_token_id",
        )
        for field in stop_token_fields:
            token = MiniCPMO45DuplexPolicy.SPECIAL_TOKEN_FIELDS[field]
            token_id = MiniCPMO45NativeDuplexServingAdapter._convert_token_to_id(tokenizer, token)
            if token_id is not None and token_id not in out:
                out.append(token_id)
        return out

    @staticmethod
    def _native_scheduler_token_id(model_config: Any) -> int | None:
        tokenizer = MiniCPMO45NativeDuplexServingAdapter._load_native_tokenizer(model_config)
        if tokenizer is None:
            return None
        scheduler_tokens = (
            MiniCPMO45DuplexPolicy.SPECIAL_TOKEN_FIELDS["unit_token_id"],
            MiniCPMO45DuplexPolicy.OPTIONAL_TOKEN_FIELDS["audio_placeholder_token_id"],
        )
        for token in scheduler_tokens:
            token_id = MiniCPMO45NativeDuplexServingAdapter._convert_token_to_id(tokenizer, token)
            if token_id is not None:
                return token_id
        eos_id = getattr(tokenizer, "eos_token_id", None)
        try:
            return int(eos_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _load_native_tokenizer(model_config: Any) -> Any | None:
        model_path = getattr(model_config, "model", None)
        if not isinstance(model_path, str) or not model_path:
            return None
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            return None

    @staticmethod
    def _convert_token_to_id(tokenizer: Any, token: str) -> int | None:
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        value = None
        if callable(convert):
            value = convert(token)
            if isinstance(value, list):
                value = value[0] if len(value) == 1 else None
        try:
            token_id = int(value)
        except (TypeError, ValueError):
            token_id = -1
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if token_id >= 0 and token_id != unk_token_id:
            return token_id
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            try:
                ids = list(encode(token, add_special_tokens=False))
            except TypeError:
                ids = list(encode(token))
            if len(ids) == 1:
                try:
                    token_id = int(ids[0])
                except (TypeError, ValueError):
                    token_id = -1
                if token_id >= 0 and token_id != unk_token_id:
                    return token_id
        return None

    @staticmethod
    async def resolve_ref_audio(ref_audio: str, *, model_config: Any) -> tuple[np.ndarray, int]:
        connector = MediaConnector(
            allowed_local_media_path=getattr(model_config, "allowed_local_media_path", None),
            allowed_media_domains=getattr(model_config, "allowed_media_domains", None),
        )
        wav_np, sr = await connector.fetch_audio_async(ref_audio)
        return np.asarray(wav_np, dtype=np.float32), int(sr)

    @staticmethod
    def _default_ref_audio_path(config: DuplexSessionConfig, *, model_config: Any) -> str | None:
        model = config.model or getattr(model_config, "model", None)
        if not isinstance(model, str) or not os.path.isdir(model):
            return None
        ref_path = os.path.join(model, "assets", "HT_ref_audio.wav")
        return ref_path if os.path.exists(ref_path) else None

    @staticmethod
    def _load_local_ref_audio(path: str) -> tuple[np.ndarray, int]:
        import soundfile as sf

        wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
        return wav_np, int(sr)

    @staticmethod
    def normalize_ref_audio(wav_np: np.ndarray, sample_rate: int, *, target_sr: int) -> np.ndarray:
        wav_np = np.asarray(wav_np, dtype=np.float32)
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=-1)
        wav_np = wav_np.reshape(-1)
        if sample_rate <= 0 or sample_rate == target_sr or wav_np.size == 0:
            return wav_np.astype(np.float32, copy=False)
        import torch
        import torchaudio

        audio = torch.from_numpy(wav_np).to(dtype=torch.float32).unsqueeze(0)
        resampled = torchaudio.functional.resample(audio, int(sample_rate), int(target_sr))
        return resampled.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
