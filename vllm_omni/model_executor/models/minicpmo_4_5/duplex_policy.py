from __future__ import annotations

import os
from typing import Any


class MiniCPMO45DuplexPolicy:
    """MiniCPM-o 4.5 native duplex model policy.

    Keep model-specific token names, input modes, and handoff type strings out
    of the generic scheduler/orchestrator path. These names are part of the
    MiniCPM-o 4.5 remote-code contract, not a general vLLM-Omni duplex schema.
    """

    INPUT_AUDIO_MODE = "append_audio_chunk"
    STAGE_HANDOFF_MODE = "append_stage_handoff"
    TTS_HANDOFF_MODE = "append_tts_handoff"
    TTS_HANDOFF_TYPE = "minicpmo45_tts_handoff"

    SPECIAL_TOKEN_FIELDS: dict[str, str] = {
        "unit_token_id": "<unit>",
        "unit_end_token_id": "</unit>",
        "listen_token_id": "<|listen|>",
        "speak_token_id": "<|speak|>",
        "tts_bos_token_id": "<|tts_bos|>",
        "tts_eos_token_id": "<|tts_eos|>",
        "tts_pad_token_id": "<|tts_pad|>",
        "chunk_eos_token_id": "<|chunk_eos|>",
        "chunk_tts_eos_token_id": "<|chunk_tts_eos|>",
        "turn_eos_token_id": "<|turn_eos|>",
    }
    OPTIONAL_TOKEN_FIELDS: dict[str, str] = {
        "audio_placeholder_token_id": "<|audio|>",
    }

    @staticmethod
    def profile_logs_enabled() -> bool:
        return os.environ.get("MINICPMO45_PROFILE_LOGS") == "1"

    @classmethod
    def token_ids_from_tokenizer(cls, tokenizer: Any) -> dict[str, int]:
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)

        def token_id(token: str) -> int:
            if not callable(convert):
                value = None
            else:
                value = convert(token)
                if isinstance(value, list):
                    value = value[0] if len(value) == 1 else None
            unk_token_id = getattr(tokenizer, "unk_token_id", None)
            try:
                candidate = int(value)
            except (TypeError, ValueError):
                candidate = -1
            if candidate >= 0 and candidate != unk_token_id:
                return candidate

            encode = getattr(tokenizer, "encode", None)
            if callable(encode):
                try:
                    ids = list(encode(token, add_special_tokens=False))
                except TypeError:
                    ids = list(encode(token))
                if len(ids) == 1:
                    try:
                        candidate = int(ids[0])
                    except (TypeError, ValueError):
                        candidate = -1
                    if candidate >= 0 and candidate != unk_token_id:
                        return candidate
            return -1

        resolved = {field: token_id(token) for field, token in cls.SPECIAL_TOKEN_FIELDS.items()}
        resolved.update({field: token_id(token) for field, token in cls.OPTIONAL_TOKEN_FIELDS.items()})
        if resolved.get("listen_token_id", -1) < 0:
            eos_id = getattr(tokenizer, "eos_token_id", None)
            try:
                resolved["listen_token_id"] = int(eos_id)
            except (TypeError, ValueError):
                pass
        return resolved

    @staticmethod
    def native_forbidden_token_ids(
        token_ids: dict[str, int],
        *,
        bad_token_ids: list[int] | tuple[int, ...] = (),
    ) -> list[int]:
        return [
            token_ids.get("tts_pad_token_id", -1),
            *list(bad_token_ids),
            token_ids.get("chunk_eos_token_id", -1),
        ]

    @staticmethod
    def native_special_token_ids(
        token_ids: dict[str, int],
        *,
        tokenizer_special_ids: list[int] | tuple[int, ...] = (),
    ) -> set[int]:
        special = set(tokenizer_special_ids or [])
        special.update(value for value in token_ids.values() if isinstance(value, int) and value >= 0)
        return special
