from __future__ import annotations

import asyncio
import base64
import binascii
import json
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import WebSocket

try:
    from audioop import alaw2lin, lin2alaw, lin2ulaw, ulaw2lin
except ImportError:  # pragma: no cover - audioop is removed in newer Python.
    alaw2lin = None
    lin2alaw = None
    lin2ulaw = None
    ulaw2lin = None


def _is_minicpmo45_model(model: str) -> bool:
    normalized = model.lower().replace("_", "-")
    return "minicpm-o-4-5" in normalized or "minicpmo-4-5" in normalized or "minicpmo45" in normalized


REALTIME_INPUT_AUDIO_FORMATS = {
    "pcm16",
    "pcm_s16le",
    "s16le",
    "pcm_f32le",
    "g711_ulaw",
    "g711_alaw",
}
REALTIME_OUTPUT_AUDIO_FORMATS = {
    "pcm16",
    "pcm_s16le",
    "s16le",
    "wav",
    "pcm",
    "g711_ulaw",
    "g711_alaw",
}
REALTIME_ERROR_TYPES_BY_CODE = {
    "bad_event": "invalid_request_error",
    "bad_audio": "invalid_request_error",
    "config_timeout": "invalid_request_error",
    "invalid_json": "invalid_request_error",
    "event_too_large": "invalid_request_error",
    "unknown_event": "invalid_request_error",
    "internal_error": "server_error",
    "runtime_open_failed": "server_error",
    "runtime_open_unsupported": "server_error",
    "runtime_append_failed": "server_error",
    "runtime_append_task_failed": "server_error",
    "runtime_signal_failed": "server_error",
    "runtime_close_failed": "server_error",
    "runtime_abort_failed": "server_error",
    "runtime_data_plane_stream_failed": "server_error",
    "runtime_data_plane_text_without_audio": "server_error",
    "response_error": "server_error",
    "chat_error": "server_error",
    "duplex_session_busy": "rate_limit_error",
    "response_already_active": "invalid_request_error",
    "response_not_active": "invalid_request_error",
    "response_create_without_input": "invalid_request_error",
    "input_audio_buffer_empty": "invalid_request_error",
    "missing_item_id": "invalid_request_error",
    "item_not_found": "invalid_request_error",
    "unsupported_audio_format": "invalid_request_error",
    "unsupported_ref_audio_path": "invalid_request_error",
    "model_update_unsupported": "invalid_request_error",
    "voice_update_after_audio_unsupported": "invalid_request_error",
    "ref_audio_update_unsupported": "invalid_request_error",
    "native_text_append_unsupported": "invalid_request_error",
    "runtime_native_stage_role_required": "server_error",
    "runtime_native_runner_kv_required": "server_error",
}


class NativeRealtimeSessionProtocol:
    """Realtime schema state for the native session actor path.

    The Realtime endpoint uses the same session actor/runtime core as
    ``/v1/duplex``, but WebSocket I/O stays in the Realtime handler. This class
    only owns schema normalization, conversation item bookkeeping, and outbound
    Realtime event construction.
    """

    def __init__(self, query_params: Any) -> None:
        if hasattr(query_params, "query_params"):
            query_params = query_params.query_params
        self._opened = False
        self._autostarted_default_session = False
        self._pending_outbound: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self._held_realtime_payloads: list[dict[str, object]] = []
        self._hold_realtime_output_until_session_created = True
        self._default_model = query_params.get("model") if hasattr(query_params, "get") else None
        self._default_session_id = query_params.get("session_id") if hasattr(query_params, "get") else None
        self._input_audio_format = "pcm16"
        self._input_sample_rate_hz = 16000
        self._output_audio_format = "pcm16"
        self._overlap_silence_rms = 0.003
        self._turn_detection_create_response = False
        audio_alias_events = query_params.get("audio_alias_events") if hasattr(query_params, "get") else None
        if audio_alias_events is None and hasattr(query_params, "get"):
            audio_alias_events = query_params.get("response_audio_events")
        self._emit_response_audio_alias_events = str(
            "true" if audio_alias_events is None else audio_alias_events
        ).lower() not in {"0", "false", "no", "off"}
        self._emit_legacy_audio_events = str(
            (query_params.get("legacy_audio_events") if hasattr(query_params, "get") else None)
            or (query_params.get("vllm_omni_legacy_events") if hasattr(query_params, "get") else None)
            or ""
        ).lower() in {"1", "true", "yes", "on"}
        output_audio_events = query_params.get("output_audio_events") if hasattr(query_params, "get") else None
        if output_audio_events is None and hasattr(query_params, "get"):
            output_audio_events = query_params.get("vllm_omni_output_audio_events")
        self._emit_output_audio_events = self._emit_legacy_audio_events or str(output_audio_events or "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._send_realtime_json = None
        self._initial_session_update = False
        self._input_speech_started = False
        self._done_response_ids: set[str] = set()
        self._audio_done_response_ids: set[str] = set()
        self._audio_delta_response_ids: set[str] = set()
        self._content_done_response_ids: set[str] = set()
        self._output_item_done_response_ids: set[str] = set()
        self._conversation_item_done_response_ids: set[str] = set()
        self._response_items: dict[str, str] = {}
        self._response_transcripts: dict[str, list[str]] = {}
        self._response_audio_formats: dict[str, str] = {}
        self._response_audio_durations_ms: dict[str, int] = {}
        self._response_audio_text_marks: dict[str, list[dict[str, int]]] = {}
        self._response_text_parts: dict[str, list[str]] = {}
        self._item_truncation_cursors: dict[str, tuple[int, int]] = {}
        self._speak_response_ids: set[str] = set()
        self._audio_content_part_added_response_ids: set[str] = set()
        self._text_content_part_added_response_ids: set[str] = set()
        self._text_content_part_done_response_ids: set[str] = set()
        self._output_text_done_response_ids: set[str] = set()
        self._active_response_id: str | None = None
        self._last_response_id: str | None = None
        self._conversation_items: dict[str, dict[str, object]] = {}
        self._pending_commit_item_ids: asyncio.Queue[str] = asyncio.Queue()
        self._last_conversation_item_id: str | None = None
        self._output_sample_rate_hz: int | None = None
        self._active_input_item_id: str | None = None
        self._input_audio_buffer_has_audio = False
        self._input_audio_buffer_had_non_speech = False
        self._input_audio_buffer_transcript_parts: list[str] = []

    def bind_sender(self, send_realtime_json) -> None:
        self._send_realtime_json = send_realtime_json

    async def discard_pending_input_audio(
        self,
        *,
        audio_end_ms: int | None = None,
    ) -> None:
        """Drop Realtime input-buffer state that was consumed as overlap.

        The serving overlap policy may classify a short acknowledgement such
        as "continue" as a side-channel signal instead of user turn input. The
        Realtime protocol has already observed the append and may have emitted
        ``speech_started``; reset that transient buffer so a later commit does
        not accidentally attach the acknowledgement transcript/audio to the
        next real user turn.
        """
        if self._input_speech_started and self._active_input_item_id is not None:
            await self._send_realtime_payload(
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "audio_end_ms": max(0, int(audio_end_ms or 0)),
                    "item_id": self._active_input_item_id,
                }
            )
        self._input_speech_started = False
        self._active_input_item_id = None
        self._input_audio_buffer_has_audio = False
        self._input_audio_buffer_had_non_speech = False
        self._input_audio_buffer_transcript_parts.clear()

    async def receive_internal_event_text(self, websocket: WebSocket) -> str:
        if not self._pending_outbound.empty():
            return json.dumps(await self._pending_outbound.get())
        if not self._opened and not self._autostarted_default_session and self._default_model:
            self._opened = True
            self._autostarted_default_session = True
            return json.dumps(
                self._session_create_from_realtime(
                    {
                        "model": self._default_model,
                        "session_id": self._default_session_id,
                    }
                )
            )
        while True:
            raw = await websocket.receive_text()
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                return raw
            if not isinstance(event, dict):
                return raw
            if not self._opened and event.get("type") != "session.update":
                self._opened = True
                await self._pending_outbound.put(
                    self._session_create_from_realtime(
                        {
                            "model": self._default_model,
                            "session_id": self._default_session_id,
                        }
                    )
                )
                translated = await self._to_duplex_event(event)
                if translated is not None:
                    await self._send_realtime_input_ack(event)
                    await self._pending_outbound.put(translated)
                return json.dumps(await self._pending_outbound.get())
            translated = await self._to_duplex_event(event)
            if translated is None:
                if not self._pending_outbound.empty():
                    return json.dumps(await self._pending_outbound.get())
                continue
            await self._send_realtime_input_ack(event)
            return json.dumps(translated)

    def encode_outbound_event(self, data: dict[str, Any]) -> list[dict[str, object]]:
        payloads = self._from_duplex_event(data)
        for payload in payloads:
            self._attach_event_id(payload)
        return payloads

    @staticmethod
    def _attach_event_id(payload: dict[str, object]) -> None:
        payload.setdefault("event_id", f"event_{uuid4().hex}")

    async def _send_realtime_payload(self, payload: dict[str, object]) -> None:
        self._attach_event_id(payload)
        if (
            self._hold_realtime_output_until_session_created
            and payload.get("type") != "error"
            and payload.get("type") != "session.created"
        ):
            self._held_realtime_payloads.append(payload)
            return
        if self._send_realtime_json is None:
            raise RuntimeError("Native Realtime sender is not bound")
        await self._send_realtime_json(payload)

    @staticmethod
    def _realtime_error_payload(
        code: str,
        message: str,
        *,
        event_id: object | None = None,
        param: object | None = None,
    ) -> dict[str, object]:
        error: dict[str, object] = {
            "type": REALTIME_ERROR_TYPES_BY_CODE.get(code, "invalid_request_error"),
            "code": code,
            "message": message,
        }
        if isinstance(event_id, str) and event_id:
            error["event_id"] = event_id
        if isinstance(param, str) and param:
            error["param"] = param
        return {"type": "error", "error": error}

    async def _send_realtime_input_ack(self, event: dict[str, object]) -> None:
        if event.get("type") != "conversation.item.create":
            return
        item = event.get("item")
        if not isinstance(item, dict):
            return
        item = self._normalize_conversation_item(item)
        previous_item_id = event.get("previous_item_id")
        if isinstance(previous_item_id, str):
            item["_previous_item_id"] = previous_item_id
        if item.get("role") != "user":
            return
        self._conversation_items[str(item["id"])] = item
        for payload in self._conversation_item_added_events(item):
            await self._send_realtime_payload(payload)

    async def _to_duplex_event(self, event: dict[str, object]) -> dict[str, object] | None:
        event_type = event.get("type")
        if event_type == "session.update":
            session_payload = event.get("session") if isinstance(event.get("session"), dict) else event
            format_error = self._validate_realtime_session_audio_formats(session_payload)
            if format_error is not None:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "unsupported_audio_format",
                        format_error,
                        event_id=event.get("event_id"),
                    )
                )
                return None
            self._apply_realtime_session_defaults(session_payload)
            session_payload.update(self._realtime_overlap_fields(session_payload))
            if not self._opened:
                self._opened = True
                self._initial_session_update = True
                return self._session_create_from_realtime(session_payload)
            return {
                "type": "turn.signal",
                "event": "session.update",
                "payload": session_payload,
            }
        if event_type == "conversation.item.create":
            item = event.get("item")
            format_error = self._validate_conversation_item_audio_formats(item)
            if format_error is not None:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "unsupported_audio_format",
                        format_error,
                        event_id=event.get("event_id"),
                    )
                )
                return None
            return self._conversation_item_to_duplex(event)
        if event_type == "conversation.item.delete":
            item_id = event.get("item_id")
            if not isinstance(item_id, str) or not item_id:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "missing_item_id",
                        "conversation.item.delete requires item_id",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            if item_id not in self._conversation_items:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "item_not_found",
                        f"Conversation item not found: {item_id}",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            self._remove_conversation_item(item_id)
            await self._pending_outbound.put(
                {
                    "type": "turn.signal",
                    "event": "conversation.item.delete",
                    "payload": {"item_id": item_id},
                }
            )
            return None
        if event_type == "conversation.item.retrieve":
            item_id = event.get("item_id")
            if not isinstance(item_id, str) or not item_id:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "missing_item_id",
                        "conversation.item.retrieve requires item_id",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            item = self._conversation_items.get(item_id)
            if item is None:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "item_not_found",
                        f"Conversation item not found: {item_id}",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            await self._send_realtime_payload({"type": "conversation.item.retrieved", "item": item})
            return None
        if event_type == "conversation.item.truncate":
            item_id = event.get("item_id")
            audio_end_ms = event.get("audio_end_ms")
            content_index = event.get("content_index", 0)
            if not isinstance(item_id, str) or not item_id:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "missing_item_id",
                        "conversation.item.truncate requires item_id",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            if not isinstance(audio_end_ms, int | float):
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "bad_event",
                        "conversation.item.truncate requires numeric audio_end_ms",
                        event_id=event.get("event_id"),
                        param="audio_end_ms",
                    )
                )
                return None
            item = self._conversation_items.get(item_id)
            if item is None:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "item_not_found",
                        f"Conversation item not found: {item_id}",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            truncate_error = self._validate_realtime_item_truncate(
                item,
                content_index=int(content_index) if isinstance(content_index, int | float) else 0,
                audio_end_ms=int(audio_end_ms),
            )
            if truncate_error is not None:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "bad_event",
                        truncate_error,
                        event_id=event.get("event_id"),
                    )
                )
                return None
            self._item_truncation_cursors[item_id] = (
                int(content_index) if isinstance(content_index, int | float) else 0,
                int(audio_end_ms),
            )
            self._truncate_realtime_item_content(
                item,
                content_index=int(content_index) if isinstance(content_index, int | float) else 0,
                audio_end_ms=int(audio_end_ms),
            )
            await self._pending_outbound.put(
                {
                    "type": "turn.signal",
                    "event": "conversation.item.truncate",
                    "payload": {
                        "item_id": item_id,
                        "content_index": content_index,
                        "audio_end_ms": audio_end_ms,
                    },
                }
            )
            await self._pending_outbound.put(
                {
                    "type": "playback.ack",
                    "item_id": item_id,
                    "committed_ms": int(audio_end_ms),
                    "played_ms": int(audio_end_ms),
                    "truncate": True,
                }
            )
            return None
        if event_type == "input_audio_buffer.append":
            audio = event.get("audio") or event.get("delta")
            fmt, format_rate = self._parse_realtime_audio_format(
                event.get("format") or event.get("input_audio_format") or self._input_audio_format
            )
            sample_rate_hz = (
                event.get("sample_rate_hz") or event.get("sample_rate") or format_rate or self._input_sample_rate_hz
            )
            if not self._is_supported_realtime_input_format(fmt):
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "unsupported_audio_format",
                        f"Unsupported input_audio_format: {fmt}",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            audio, fmt, sample_rate_hz = self._convert_realtime_input_audio_with_rate(
                audio,
                fmt,
                sample_rate_hz=sample_rate_hz if isinstance(sample_rate_hz, int | float) else None,
            )
            looks_like_speech = self._input_looks_like_speech(event, audio=audio, fmt=fmt)
            self._input_audio_buffer_has_audio = self._input_audio_buffer_has_audio or (
                looks_like_speech and isinstance(audio, str) and bool(audio)
            )
            self._input_audio_buffer_had_non_speech = self._input_audio_buffer_had_non_speech or (
                not looks_like_speech and isinstance(audio, str) and bool(audio)
            )
            if looks_like_speech:
                await self._emit_input_speech_started(event)
                self._remember_input_transcript_hint(event)
            payload = {
                "type": "input_audio_buffer.append",
                "audio": audio,
                "format": fmt,
                "sample_rate_hz": sample_rate_hz,
            }
            self._copy_realtime_input_hints(event, payload)
            if not looks_like_speech:
                payload["is_speech"] = False
            return payload
        if event_type == "input_audio_buffer.commit":
            if not self._input_audio_buffer_has_audio:
                if self._input_audio_buffer_had_non_speech:
                    self._input_audio_buffer_had_non_speech = False
                    self._active_input_item_id = None
                    return {
                        "type": "input_audio_buffer.commit",
                        "final": event.get("final", True),
                        "response_create": False,
                        "is_speech": False,
                    }
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "input_audio_buffer_empty",
                        "input_audio_buffer.commit requires a non-empty input audio buffer",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            item_id = self._active_input_item_id or f"item_{uuid4().hex}"
            await self._pending_commit_item_ids.put(item_id)
            await self._emit_input_speech_stopped(event, item_id=item_id)
            transcript = self._consume_input_transcript_hint()
            self._active_input_item_id = None
            self._input_audio_buffer_has_audio = False
            self._input_audio_buffer_had_non_speech = False
            payload = {
                "type": "input_audio_buffer.commit",
                "final": event.get("final", True),
                "realtime_item_id": item_id,
                "response_create": bool(event.get("response_create", self._turn_detection_create_response)),
            }
            if transcript:
                payload["transcript"] = transcript
            return payload
        if event_type == "input_audio_buffer.clear":
            self._input_speech_started = False
            self._active_input_item_id = None
            self._input_audio_buffer_has_audio = False
            self._input_audio_buffer_had_non_speech = False
            self._input_audio_buffer_transcript_parts.clear()
            return {"type": "input_audio_buffer.clear", "reason": event_type}
        if event_type == "output_audio_buffer.clear":
            payload = {"type": "output_audio_buffer.clear", "reason": event_type}
            response_id = event.get("response_id")
            if not isinstance(response_id, str) or not response_id:
                response_id = self._active_response_id or self._last_response_id
            if isinstance(response_id, str) and response_id in self._done_response_ids:
                await self._send_realtime_payload(
                    {
                        "type": "output_audio_buffer.cleared",
                        "response_id": response_id,
                    }
                )
                return None
            if isinstance(response_id, str) and response_id:
                payload["response_id"] = response_id
            return payload
        if event_type == "response.cancel":
            payload = {"type": "response.cancel", "reason": event_type}
            response_id = event.get("response_id")
            if not isinstance(response_id, str) or not response_id:
                response_id = self._active_response_id or self._last_response_id
            if isinstance(response_id, str) and response_id in self._done_response_ids:
                await self._send_realtime_payload(
                    self._realtime_error_payload(
                        "response_not_active",
                        f"Response is already complete: {response_id}",
                        event_id=event.get("event_id"),
                    )
                )
                return None
            if isinstance(response_id, str) and response_id:
                payload["response_id"] = response_id
            return payload
        if event_type == "response.create":
            response_payload = event.get("response")
            if isinstance(response_payload, dict):
                format_error = self._validate_realtime_response_audio_formats(response_payload)
                if format_error is not None:
                    await self._send_realtime_payload(
                        self._realtime_error_payload(
                            "unsupported_audio_format",
                            format_error,
                            event_id=event.get("event_id"),
                        )
                    )
                    return None
            return {
                "type": "response.create",
                "response": response_payload if isinstance(response_payload, dict) else {},
            }
        if event_type in {"session.close", "close"}:
            return {"type": "session.close"}
        return event

    @staticmethod
    def _normalize_conversation_item(item: dict[str, object]) -> dict[str, object]:
        normalized = dict(item)
        normalized.setdefault("id", f"item_{uuid4().hex}")
        normalized.setdefault("object", "realtime.item")
        normalized.setdefault("type", "message")
        normalized.setdefault("status", "completed")
        if "role" not in normalized and normalized.get("type") == "message":
            normalized["role"] = "user"
        if not isinstance(normalized.get("content"), list):
            normalized["content"] = []
        return normalized

    @staticmethod
    def _truncate_realtime_item_content(
        item: dict[str, object],
        *,
        content_index: int,
        audio_end_ms: int,
    ) -> None:
        content = item.get("content")
        if not isinstance(content, list) or not content:
            return
        index = max(0, int(content_index))
        if index >= len(content):
            return
        part = content[index]
        if not isinstance(part, dict):
            return
        transcript = part.get("transcript")
        if not isinstance(transcript, str) or not transcript:
            return
        marks = part.get("audio_text_marks")
        if isinstance(marks, list):
            keep_chars = NativeRealtimeSessionProtocol._text_chars_for_audio_ms_from_marks(
                audio_end_ms,
                len(transcript),
                marks,
                final_ms=part.get("audio_duration_ms") or part.get("duration_ms") or part.get("audio_ms"),
            )
            part["transcript"] = transcript[:keep_chars].rstrip()
            return
        duration_ms = part.get("audio_duration_ms") or part.get("duration_ms") or part.get("audio_ms")
        if isinstance(duration_ms, int | float) and duration_ms > 0:
            keep_chars = int(len(transcript) * max(0.0, min(1.0, int(audio_end_ms) / float(duration_ms))))
            part["transcript"] = transcript[:keep_chars].rstrip()
        elif audio_end_ms <= 0:
            part["transcript"] = ""

    @staticmethod
    def _validate_realtime_item_truncate(
        item: dict[str, object],
        *,
        content_index: int,
        audio_end_ms: int,
    ) -> str | None:
        if item.get("type") != "message" or item.get("role") != "assistant":
            return "conversation.item.truncate only supports assistant message items"
        if audio_end_ms < 0:
            return "conversation.item.truncate requires non-negative audio_end_ms"
        content = item.get("content")
        if not isinstance(content, list) or not content:
            return None
        index = max(0, int(content_index))
        if index >= len(content):
            return f"conversation.item.truncate content_index out of range: {content_index}"
        part = content[index]
        if not isinstance(part, dict):
            return "conversation.item.truncate target content part is invalid"
        if part.get("type") not in {"audio", "output_audio"}:
            return "conversation.item.truncate target content part must be audio"
        duration_ms = part.get("audio_duration_ms") or part.get("duration_ms") or part.get("audio_ms")
        if isinstance(duration_ms, int | float) and int(duration_ms) >= 0 and audio_end_ms > int(duration_ms):
            return "conversation.item.truncate audio_end_ms exceeds item audio duration"
        return None

    @staticmethod
    def _text_chars_for_audio_ms_from_marks(
        audio_end_ms: int,
        text_len: int,
        marks: list[object],
        *,
        final_ms: object | None = None,
    ) -> int:
        if text_len <= 0:
            return 0
        clean_marks: list[tuple[int, int]] = []
        for mark in marks:
            if not isinstance(mark, dict):
                continue
            raw_text_chars = mark.get("text_chars")
            raw_audio_end_ms = mark.get("audio_end_ms", mark.get("audio_ms"))
            if not isinstance(raw_text_chars, int | float) or not isinstance(raw_audio_end_ms, int | float):
                continue
            clean_marks.append((max(0, int(raw_audio_end_ms)), min(text_len, max(0, int(raw_text_chars)))))
        if not clean_marks:
            return 0 if audio_end_ms <= 0 else text_len
        clean_marks.sort(key=lambda item: item[0])
        audio_end_ms = max(0, int(audio_end_ms))
        if audio_end_ms <= 0:
            return 0
        previous_ms = 0
        previous_chars = 0
        for mark_ms, mark_chars in clean_marks:
            mark_ms = max(previous_ms, mark_ms)
            mark_chars = max(previous_chars, min(text_len, mark_chars))
            if audio_end_ms <= mark_ms:
                if mark_ms <= previous_ms:
                    return mark_chars
                ratio = (audio_end_ms - previous_ms) / max(1, mark_ms - previous_ms)
                return int(previous_chars + (mark_chars - previous_chars) * max(0.0, min(1.0, ratio)))
            previous_ms = mark_ms
            previous_chars = mark_chars
        if isinstance(final_ms, int | float) and int(final_ms) > previous_ms:
            if audio_end_ms >= int(final_ms):
                return text_len
            ratio = (audio_end_ms - previous_ms) / max(1, int(final_ms) - previous_ms)
            return int(previous_chars + (text_len - previous_chars) * max(0.0, min(1.0, ratio)))
        return text_len if audio_end_ms >= previous_ms else previous_chars

    def _conversation_item_added_events(self, item: dict[str, object]) -> list[dict[str, object]]:
        item_id = item.get("id")
        explicit_previous_item_id = item.pop("_previous_item_id", None)
        previous_item_id = (
            explicit_previous_item_id if isinstance(explicit_previous_item_id, str) else self._last_conversation_item_id
        )
        if isinstance(item_id, str) and item_id:
            self._last_conversation_item_id = item_id
        return [
            {
                "type": "conversation.item.added",
                "previous_item_id": previous_item_id,
                "item": item,
            },
            {
                "type": "conversation.item.created",
                "previous_item_id": previous_item_id,
                "item": item,
            },
        ]

    def _conversation_item_done_event(self, item: dict[str, object]) -> dict[str, object]:
        item_id = item.get("id")
        previous_item_id = self._previous_item_id(item_id) if isinstance(item_id, str) else None
        if isinstance(item_id, str) and item_id:
            self._conversation_items[item_id] = item
            self._last_conversation_item_id = item_id
        return {
            "type": "conversation.item.done",
            "previous_item_id": previous_item_id,
            "item": item,
        }

    def _remove_conversation_item(self, item_id: str) -> bool:
        removed = self._conversation_items.pop(item_id, None) is not None
        if self._last_conversation_item_id == item_id:
            self._last_conversation_item_id = next(reversed(self._conversation_items), None)
        self._item_truncation_cursors.pop(item_id, None)
        return removed

    def _response_output_item_added_events(
        self,
        *,
        response_id: object,
        item: dict[str, object],
    ) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = [
            {
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": 0,
                "item": item,
            },
        ]
        if self._emit_legacy_audio_events:
            payloads.append(
                {
                    "type": "response.output_item.created",
                    "response_id": response_id,
                    "output_index": 0,
                    "item": item,
                }
            )
        return payloads

    @staticmethod
    def _response_content_part(*, transcript: str = "") -> dict[str, object]:
        return {
            "type": "audio",
            "transcript": transcript,
        }

    @staticmethod
    def _response_text_content_part(*, text: str = "") -> dict[str, object]:
        return {
            "type": "text",
            "text": text,
        }

    @staticmethod
    def _response_item_content_part(
        *,
        transcript: str = "",
        audio_duration_ms: int | None = None,
        audio_text_marks: list[dict[str, int]] | None = None,
    ) -> dict[str, object]:
        part: dict[str, object] = {
            "type": "output_audio",
            "transcript": transcript,
        }
        if audio_duration_ms is not None:
            part["audio_duration_ms"] = int(audio_duration_ms)
        if audio_text_marks:
            part["audio_text_marks"] = [dict(mark) for mark in audio_text_marks]
        return part

    @staticmethod
    def _response_item_text_content_part(*, text: str = "") -> dict[str, object]:
        return {
            "type": "output_text",
            "text": text,
        }

    def _previous_item_id(self, item_id: str) -> str | None:
        previous: str | None = None
        for known_id in self._conversation_items:
            if known_id == item_id:
                return previous
            previous = known_id
        if self._last_conversation_item_id == item_id:
            return previous
        return self._last_conversation_item_id

    def _input_audio_buffer_committed_event(
        self,
        *,
        item_id: str,
        event: dict[str, Any],
    ) -> dict[str, object]:
        return {
            "type": "input_audio_buffer.committed",
            "previous_item_id": self._previous_item_id(item_id),
            "item_id": item_id,
            "event": event,
        }

    def _session_create_from_realtime(self, session_payload: dict[str, object]) -> dict[str, object]:
        self._apply_realtime_session_defaults(session_payload)
        model = session_payload.get("model")
        audio_config = session_payload.get("audio")
        audio_input = audio_config.get("input") if isinstance(audio_config, dict) else None
        audio_output = audio_config.get("output") if isinstance(audio_config, dict) else None
        extra_body = (
            dict(session_payload.get("extra_body")) if isinstance(session_payload.get("extra_body"), dict) else {}
        )
        if isinstance(model, str) and _is_minicpmo45_model(model):
            extra_body.setdefault("minicpmo45_native_duplex", True)
        extra_body["realtime_session_payload"] = self._json_safe_realtime_payload(session_payload)
        if isinstance(session_payload.get("tools"), list):
            extra_body["realtime_tools"] = session_payload["tools"]
        if isinstance(session_payload.get("tool_choice"), str | dict):
            extra_body["realtime_tool_choice"] = session_payload["tool_choice"]
        if isinstance(session_payload.get("metadata"), dict):
            extra_body["realtime_metadata"] = dict(session_payload["metadata"])
        if isinstance(session_payload.get("include"), list):
            extra_body["realtime_include"] = list(session_payload["include"])
        if isinstance(session_payload.get("prompt"), dict):
            extra_body["realtime_prompt"] = dict(session_payload["prompt"])
        turn_detection = self._turn_detection_config(session_payload)
        if isinstance(turn_detection, dict):
            extra_body["realtime_turn_detection"] = dict(turn_detection)
        input_audio_transcription = self._input_audio_transcription_config(session_payload)
        if isinstance(input_audio_transcription, dict):
            extra_body["realtime_input_audio_transcription"] = dict(input_audio_transcription)
        if isinstance(session_payload.get("input_audio_noise_reduction"), dict):
            extra_body["realtime_input_audio_noise_reduction"] = dict(session_payload["input_audio_noise_reduction"])
        if isinstance(audio_input, dict) and isinstance(audio_input.get("noise_reduction"), dict):
            extra_body["realtime_input_audio_noise_reduction"] = dict(audio_input["noise_reduction"])
        if isinstance(session_payload.get("audio"), dict):
            extra_body["realtime_audio"] = dict(session_payload["audio"])
        if isinstance(session_payload.get("tracing"), str | dict):
            extra_body["realtime_tracing"] = session_payload["tracing"]
        response_format = self._duplex_response_format(self._output_audio_format)
        extra_body.setdefault("realtime_output_audio_format", self._output_audio_format)
        overlap_fields = self._realtime_overlap_fields(session_payload)
        voice = session_payload.get("voice")
        if not isinstance(voice, str) and isinstance(audio_output, dict):
            voice = audio_output.get("voice")
        speed = session_payload.get("speed")
        if not isinstance(speed, int | float) and isinstance(audio_output, dict):
            speed = audio_output.get("speed")
        return {
            "type": "session.create",
            "session_id": session_payload.get("session_id") or session_payload.get("id"),
            "session": {
                "model": model,
                "modalities": (
                    session_payload.get("modalities") or session_payload.get("output_modalities") or ["text", "audio"]
                ),
                "instructions": session_payload.get("instructions"),
                "voice": voice,
                "ref_audio": session_payload.get("ref_audio"),
                "response_format": response_format,
                "temperature": session_payload.get("temperature"),
                "max_tokens": self.realtime_max_output_tokens(
                    session_payload.get("max_response_output_tokens")
                    or session_payload.get("max_output_tokens")
                    or session_payload.get("max_tokens")
                ),
                "speed": speed,
                "idle_timeout_s": session_payload.get("idle_timeout_s") or 300.0,
                **overlap_fields,
                "extra_body": extra_body,
            },
        }

    def _apply_realtime_session_defaults(self, session_payload: dict[str, object]) -> None:
        input_format: object = session_payload.get("input_audio_format")
        audio_config = session_payload.get("audio")
        if input_format is None and isinstance(audio_config, dict):
            audio_input = audio_config.get("input")
            if isinstance(audio_input, dict):
                input_format = audio_input.get("format")
        input_format, input_rate = self._parse_realtime_audio_format(input_format)
        if isinstance(input_format, str) and input_format.lower() in REALTIME_INPUT_AUDIO_FORMATS:
            self._input_audio_format = input_format
        output_format: object = session_payload.get("output_audio_format") or session_payload.get("response_format")
        output_rate_raw: object | None = None
        if output_format is None and isinstance(audio_config, dict):
            audio_output = audio_config.get("output")
            if isinstance(audio_output, dict):
                output_format = audio_output.get("format")
                output_rate_raw = audio_output.get("sample_rate_hz") or audio_output.get("sample_rate")
        output_format, output_rate = self._parse_realtime_audio_format(output_format)
        if output_rate is None and isinstance(output_rate_raw, int | float) and output_rate_raw > 0:
            output_rate = int(output_rate_raw)
        if isinstance(output_format, str) and output_format.lower() in REALTIME_OUTPUT_AUDIO_FORMATS:
            self._output_audio_format = self._realtime_output_format(output_format)
        sample_rate = session_payload.get("sample_rate_hz") or session_payload.get("sample_rate")
        if sample_rate is None and isinstance(audio_config, dict):
            audio_input = audio_config.get("input")
            if isinstance(audio_input, dict):
                sample_rate = audio_input.get("sample_rate_hz") or audio_input.get("sample_rate")
        if sample_rate is None:
            sample_rate = input_rate
        if isinstance(sample_rate, int | float) and sample_rate > 0:
            self._input_sample_rate_hz = int(sample_rate)
        if isinstance(output_rate, int | float) and output_rate > 0:
            self._output_sample_rate_hz = int(output_rate)
        turn_detection = self._turn_detection_config(session_payload)
        if isinstance(turn_detection, dict):
            create_response = turn_detection.get("create_response")
            if isinstance(create_response, bool):
                self._turn_detection_create_response = create_response
        elif "turn_detection" in session_payload and session_payload.get("turn_detection") is None:
            self._turn_detection_create_response = False
        audio_alias_events = session_payload.get("audio_alias_events")
        if audio_alias_events is None:
            audio_alias_events = session_payload.get("response_audio_events")
        if isinstance(audio_alias_events, bool):
            self._emit_response_audio_alias_events = audio_alias_events
        output_audio_events = session_payload.get("output_audio_events")
        if output_audio_events is None:
            output_audio_events = session_payload.get("vllm_omni_output_audio_events")
        if isinstance(output_audio_events, bool):
            self._emit_output_audio_events = output_audio_events or self._emit_legacy_audio_events
        overlap_fields = self._realtime_overlap_fields(session_payload)
        overlap_silence_rms = overlap_fields.get("overlap_silence_rms")
        if isinstance(overlap_silence_rms, int | float):
            self._overlap_silence_rms = max(0.0, float(overlap_silence_rms))

    @classmethod
    def _validate_realtime_session_audio_formats(cls, session_payload: dict[str, object]) -> str | None:
        audio_config = session_payload.get("audio")
        input_format: object = session_payload.get("input_audio_format")
        if input_format is None and isinstance(audio_config, dict):
            audio_input = audio_config.get("input")
            if isinstance(audio_input, dict):
                input_format = audio_input.get("format")
        parsed_input, _ = cls._parse_realtime_audio_format(input_format)
        if input_format is not None and not (
            isinstance(parsed_input, str) and parsed_input.lower() in REALTIME_INPUT_AUDIO_FORMATS
        ):
            return f"Unsupported input_audio_format: {input_format}"

        output_format: object = session_payload.get("output_audio_format") or session_payload.get("response_format")
        if output_format is None and isinstance(audio_config, dict):
            audio_output = audio_config.get("output")
            if isinstance(audio_output, dict):
                output_format = audio_output.get("format")
        parsed_output, _ = cls._parse_realtime_audio_format(output_format)
        if output_format is not None and not (
            isinstance(parsed_output, str) and parsed_output.lower() in REALTIME_OUTPUT_AUDIO_FORMATS
        ):
            return f"Unsupported output_audio_format: {output_format}"
        return None

    @classmethod
    def _validate_realtime_response_audio_formats(cls, response_payload: dict[str, object]) -> str | None:
        output_format: object = response_payload.get("output_audio_format") or response_payload.get("response_format")
        audio_config = response_payload.get("audio")
        if output_format is None and isinstance(audio_config, dict):
            audio_output = audio_config.get("output")
            if isinstance(audio_output, dict):
                output_format = audio_output.get("format")
        parsed_output, _ = cls._parse_realtime_audio_format(output_format)
        if output_format is not None and not (
            isinstance(parsed_output, str) and parsed_output.lower() in REALTIME_OUTPUT_AUDIO_FORMATS
        ):
            return f"Unsupported output_audio_format: {output_format}"
        return None

    @classmethod
    def _validate_conversation_item_audio_formats(cls, item: object) -> str | None:
        if not isinstance(item, dict):
            return None
        content = item.get("content")
        if not isinstance(content, list):
            return None
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in {"input_audio", "audio"}:
                continue
            raw_format = part.get("format")
            parsed_format, _ = cls._parse_realtime_audio_format(raw_format)
            if raw_format is not None and not (
                isinstance(parsed_format, str) and parsed_format.lower() in REALTIME_INPUT_AUDIO_FORMATS
            ):
                return f"Unsupported input_audio format in conversation.item.create: {raw_format}"
        return None

    @staticmethod
    def _json_safe_realtime_payload(payload: dict[str, object]) -> dict[str, object]:
        clean: dict[str, object] = {}
        for key, value in payload.items():
            if key == "extra_body":
                continue
            if isinstance(value, str | int | float | bool) or value is None:
                clean[key] = value
            elif isinstance(value, dict):
                clean[key] = NativeRealtimeSessionProtocol._json_safe_realtime_payload(value)
            elif isinstance(value, list):
                clean[key] = [
                    (
                        NativeRealtimeSessionProtocol._json_safe_realtime_payload(item)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value
                    if isinstance(item, str | int | float | bool | dict) or item is None
                ]
        return clean

    @staticmethod
    def _parse_realtime_audio_format(raw_format: object) -> tuple[object, int | None]:
        def normalize_format(fmt: str) -> str:
            normalized = fmt.lower()
            if normalized in {"audio/pcm", "pcm"}:
                return "pcm16"
            if normalized in {"audio/wav", "wav"}:
                return "wav"
            if normalized in {"audio/pcm16", "pcm16", "pcm_s16le", "s16le"}:
                return "pcm16"
            if normalized in {"audio/pcm_f32le", "pcm_f32le", "f32le"}:
                return "pcm_f32le"
            if normalized in {"audio/g711_ulaw", "g711_ulaw", "g711-ulaw", "ulaw", "mulaw"}:
                return "g711_ulaw"
            if normalized in {"audio/g711_alaw", "g711_alaw", "g711-alaw", "alaw"}:
                return "g711_alaw"
            return fmt

        if isinstance(raw_format, str):
            return normalize_format(raw_format), None
        if not isinstance(raw_format, dict):
            return raw_format, None
        rate = raw_format.get("rate") or raw_format.get("sample_rate_hz") or raw_format.get("sample_rate")
        sample_rate_hz = int(rate) if isinstance(rate, int | float) and rate > 0 else None
        fmt = raw_format.get("type") or raw_format.get("format")
        if not isinstance(fmt, str):
            return raw_format, sample_rate_hz
        return normalize_format(fmt), sample_rate_hz

    @staticmethod
    def _duplex_response_format(realtime_format: str) -> str:
        normalized = realtime_format.lower()
        if normalized in {"pcm16", "pcm_s16le", "s16le"}:
            return "pcm"
        if normalized in {"g711_ulaw", "g711_alaw"}:
            return "pcm"
        if normalized in {"wav", "pcm"}:
            return normalized
        return "wav"

    @staticmethod
    def _realtime_output_format(duplex_format: object) -> str:
        if isinstance(duplex_format, str) and duplex_format.lower() in {"g711_ulaw", "g711_alaw"}:
            return duplex_format.lower()
        if isinstance(duplex_format, str) and duplex_format.lower() == "pcm":
            return "pcm16"
        return str(duplex_format or "wav")

    @staticmethod
    def _is_supported_realtime_input_format(fmt: object) -> bool:
        return isinstance(fmt, str) and fmt.lower() in REALTIME_INPUT_AUDIO_FORMATS

    @staticmethod
    def _input_explicitly_non_speech(event: dict[str, object]) -> bool:
        for key in ("is_speech", "speech"):
            value = event.get(key)
            if isinstance(value, bool):
                return not value
        vad = event.get("vad")
        if isinstance(vad, dict):
            value = vad.get("is_speech")
            if isinstance(value, bool):
                return not value
            probability = vad.get("speech_probability", vad.get("probability"))
            if isinstance(probability, int | float):
                return float(probability) < 0.5
        probability = event.get("speech_probability")
        return isinstance(probability, int | float) and float(probability) < 0.5

    def _input_looks_like_speech(self, event: dict[str, object], *, audio: object, fmt: object) -> bool:
        if NativeRealtimeSessionProtocol._input_explicitly_non_speech(event):
            return False
        for key in ("is_speech", "speech"):
            value = event.get(key)
            if isinstance(value, bool):
                return value
        vad = event.get("vad")
        if isinstance(vad, dict):
            probability = vad.get("speech_probability", vad.get("probability"))
            if isinstance(probability, int | float):
                return float(probability) >= 0.5
        probability = event.get("speech_probability")
        if isinstance(probability, int | float):
            return float(probability) >= 0.5
        if fmt != "pcm_f32le" or not isinstance(audio, str):
            return True
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError):
            return True
        if len(raw) < 4 or len(raw) % 4 != 0:
            return True
        samples = np.frombuffer(raw, dtype=np.float32)
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
        threshold = event.get("overlap_silence_rms")
        if not isinstance(threshold, int | float):
            vad = event.get("vad")
            if isinstance(vad, dict):
                threshold = vad.get("silence_rms")
        silence_rms = float(threshold) if isinstance(threshold, int | float) else self._overlap_silence_rms
        return rms >= max(0.0, silence_rms)

    @staticmethod
    def _copy_realtime_input_hints(source: dict[str, object], target: dict[str, object]) -> None:
        for key in (
            "duration_ms",
            "audio_duration_ms",
            "audio_start_ms",
            "audio_end_ms",
            "is_speech",
            "speech",
            "speech_probability",
            "vad",
            "overlap_action",
            "overlap",
            "force_barge_in",
            "force_listen",
            "text",
            "transcript",
        ):
            if key in source:
                target[key] = source[key]

    @staticmethod
    def _convert_realtime_input_audio(audio: object, fmt: object) -> tuple[object, object]:
        audio, fmt, _ = NativeRealtimeSessionProtocol._convert_realtime_input_audio_with_rate(audio, fmt)
        return audio, fmt

    @staticmethod
    def _convert_realtime_input_audio_with_rate(
        audio: object,
        fmt: object,
        *,
        sample_rate_hz: int | float | None = None,
        target_sample_rate_hz: int = 16000,
    ) -> tuple[object, object, int | float | None]:
        if not isinstance(audio, str) or not isinstance(fmt, str):
            return audio, fmt, sample_rate_hz
        normalized = fmt.lower()
        if normalized not in {"pcm16", "pcm_s16le", "s16le", "g711_ulaw", "g711_alaw"}:
            return audio, fmt, sample_rate_hz
        try:
            raw = base64.b64decode(audio.strip(), validate=False)
        except (binascii.Error, ValueError):
            return audio, fmt, sample_rate_hz
        if normalized == "g711_ulaw":
            raw = ulaw2lin(raw, 2) if ulaw2lin is not None else NativeRealtimeSessionProtocol._decode_g711_ulaw(raw)
            if not isinstance(sample_rate_hz, int | float) or sample_rate_hz <= 0:
                sample_rate_hz = 8000
        elif normalized == "g711_alaw":
            raw = alaw2lin(raw, 2) if alaw2lin is not None else NativeRealtimeSessionProtocol._decode_g711_alaw(raw)
            if not isinstance(sample_rate_hz, int | float) or sample_rate_hz <= 0:
                sample_rate_hz = 8000
        elif len(raw) % 2 != 0:
            return audio, fmt, sample_rate_hz
        if (
            isinstance(sample_rate_hz, int | float)
            and sample_rate_hz > 0
            and int(sample_rate_hz) != int(target_sample_rate_hz)
        ):
            raw = NativeRealtimeSessionProtocol._resample_pcm16_mono(
                raw,
                source_rate_hz=int(sample_rate_hz),
                target_rate_hz=int(target_sample_rate_hz),
            )
            sample_rate_hz = int(target_sample_rate_hz)
        pcm16 = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        converted = base64.b64encode(np.ascontiguousarray(pcm16, dtype="<f4").tobytes()).decode("ascii")
        return converted, "pcm_f32le", sample_rate_hz

    @staticmethod
    def _convert_realtime_output_audio(
        audio: str,
        *,
        source_fmt: str,
        target_fmt: str,
        source_sample_rate_hz: int | None = None,
        target_sample_rate_hz: int | None = None,
    ) -> tuple[str, str, int | None]:
        target = target_fmt.lower()
        if target not in {"g711_ulaw", "g711_alaw"}:
            return audio, source_fmt, source_sample_rate_hz
        try:
            raw = base64.b64decode(audio, validate=False)
        except (binascii.Error, ValueError):
            return audio, source_fmt, source_sample_rate_hz
        source = source_fmt.lower()
        if source == "wav":
            pcm_raw, wav_sample_rate_hz = NativeRealtimeSessionProtocol._wav_payload_to_pcm16(raw)
            if pcm_raw is None:
                return audio, source_fmt, source_sample_rate_hz
            raw = pcm_raw
            if source_sample_rate_hz is None:
                source_sample_rate_hz = wav_sample_rate_hz
        elif source not in {"pcm", "pcm16", "pcm_s16le", "s16le"}:
            return audio, source_fmt, source_sample_rate_hz
        if len(raw) % 2 != 0:
            return audio, source_fmt, source_sample_rate_hz
        target_rate = target_sample_rate_hz or 8000
        if source_sample_rate_hz is not None:
            raw = NativeRealtimeSessionProtocol._resample_pcm16_mono(
                raw,
                source_rate_hz=source_sample_rate_hz,
                target_rate_hz=target_rate,
            )
        if target == "g711_ulaw":
            encoded = lin2ulaw(raw, 2) if lin2ulaw is not None else NativeRealtimeSessionProtocol._encode_g711_ulaw(raw)
        else:
            encoded = lin2alaw(raw, 2) if lin2alaw is not None else NativeRealtimeSessionProtocol._encode_g711_alaw(raw)
        return base64.b64encode(encoded).decode("ascii"), target, target_rate

    @staticmethod
    def _wav_payload_to_pcm16(raw: bytes) -> tuple[bytes | None, int | None]:
        try:
            import io
            import wave

            with wave.open(io.BytesIO(raw), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate_hz = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
            if sample_width != 2:
                return None, sample_rate_hz
            if channels <= 1:
                return frames, sample_rate_hz
            pcm = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
            mono = np.mean(pcm.astype(np.float32), axis=1)
            return np.clip(mono, -32768, 32767).astype("<i2").tobytes(), sample_rate_hz
        except Exception:
            return None, None

    @staticmethod
    def _resample_pcm16_mono(raw: bytes, *, source_rate_hz: int, target_rate_hz: int) -> bytes:
        if source_rate_hz <= 0 or target_rate_hz <= 0 or source_rate_hz == target_rate_hz:
            return raw
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        if samples.size <= 1:
            return raw
        target_size = max(1, int(round(samples.size * target_rate_hz / source_rate_hz)))
        source_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=True)
        target_x = np.linspace(0.0, 1.0, num=target_size, endpoint=True)
        resampled = np.interp(target_x, source_x, samples)
        return np.clip(resampled, -32768, 32767).astype("<i2").tobytes()

    @staticmethod
    def _decode_g711_ulaw(raw: bytes) -> bytes:
        data = np.frombuffer(raw, dtype=np.uint8)
        u = np.bitwise_not(data).astype(np.int16)
        sign = u & 0x80
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        sample = ((mantissa << 3) + 0x84) << exponent
        sample = sample - 0x84
        sample = np.where(sign != 0, -sample, sample).astype("<i2")
        return sample.tobytes()

    @staticmethod
    def _decode_g711_alaw(raw: bytes) -> bytes:
        data = np.bitwise_xor(np.frombuffer(raw, dtype=np.uint8), 0x55).astype(np.int16)
        sign = data & 0x80
        exponent = (data >> 4) & 0x07
        mantissa = data & 0x0F
        sample = np.where(
            exponent == 0,
            (mantissa << 4) + 8,
            ((mantissa << 4) + 0x108) << (exponent - 1),
        )
        sample = np.where(sign != 0, sample, -sample).astype("<i2")
        return sample.tobytes()

    @staticmethod
    def _encode_g711_ulaw(raw: bytes) -> bytes:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.int32)
        pcm = np.clip(pcm, -32635, 32635)
        sign = np.where(pcm < 0, 0x80, 0)
        magnitude = np.abs(pcm) + 0x84
        exponent = np.zeros_like(magnitude)
        for exp in range(7):
            exponent = np.where(magnitude > (0xFF << exp), exp + 1, exponent)
        mantissa = (magnitude >> (exponent + 3)) & 0x0F
        encoded = np.bitwise_not(sign | (exponent << 4) | mantissa) & 0xFF
        return encoded.astype(np.uint8).tobytes()

    @staticmethod
    def _encode_g711_alaw(raw: bytes) -> bytes:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.int32)
        sign = np.where(pcm >= 0, 0x80, 0x00)
        magnitude = np.abs(pcm)
        exponent = np.zeros_like(magnitude)
        for exp in range(1, 8):
            exponent = np.where(magnitude >= (1 << (exp + 7)), exp, exponent)
        mantissa = np.where(
            exponent == 0,
            (magnitude >> 4) & 0x0F,
            (magnitude >> (exponent + 3)) & 0x0F,
        )
        encoded = (sign | (exponent << 4) | mantissa) ^ 0x55
        return encoded.astype(np.uint8).tobytes()

    @classmethod
    def _realtime_overlap_fields(cls, session_payload: dict[str, object]) -> dict[str, object]:
        fields: dict[str, object] = {}
        if isinstance(session_payload.get("overlap_policy"), str):
            fields["overlap_policy"] = session_payload["overlap_policy"]
        for key in ("overlap_short_ack_ms", "overlap_barge_in_ms", "overlap_silence_rms"):
            value = session_payload.get(key)
            if isinstance(value, int | float):
                fields[key] = value

        turn_detection = cls._turn_detection_config(session_payload)
        if isinstance(turn_detection, dict):
            interrupt_response = turn_detection.get("interrupt_response")
            if "overlap_policy" not in fields and isinstance(interrupt_response, bool):
                fields["overlap_policy"] = "auto" if interrupt_response else "listen_only"
            silence_duration_ms = turn_detection.get("silence_duration_ms")
            if isinstance(silence_duration_ms, int | float) and "overlap_short_ack_ms" not in fields:
                fields["overlap_short_ack_ms"] = max(0, int(silence_duration_ms))
            threshold = turn_detection.get("threshold")
            if isinstance(threshold, int | float) and "overlap_silence_rms" not in fields:
                fields["overlap_silence_rms"] = max(0.0, min(1.0, float(threshold))) * 0.01

        if isinstance(session_payload.get("playback_commit_policy"), str):
            fields["playback_commit_policy"] = session_payload["playback_commit_policy"]
        return fields

    @staticmethod
    def _turn_detection_config(session_payload: dict[str, object]) -> dict[str, object] | None:
        turn_detection = session_payload.get("turn_detection")
        if isinstance(turn_detection, dict):
            return turn_detection
        audio_config = session_payload.get("audio")
        if not isinstance(audio_config, dict):
            return None
        audio_input = audio_config.get("input")
        if not isinstance(audio_input, dict):
            return None
        turn_detection = audio_input.get("turn_detection")
        return turn_detection if isinstance(turn_detection, dict) else None

    @staticmethod
    def _input_audio_transcription_config(session_payload: dict[str, object]) -> dict[str, object] | None:
        transcription = session_payload.get("input_audio_transcription")
        if isinstance(transcription, dict):
            return transcription
        audio_config = session_payload.get("audio")
        if not isinstance(audio_config, dict):
            return None
        audio_input = audio_config.get("input")
        if not isinstance(audio_input, dict):
            return None
        transcription = audio_input.get("transcription")
        return transcription if isinstance(transcription, dict) else None

    @staticmethod
    def realtime_max_output_tokens(value: object) -> int | None:
        """Normalize Realtime max output tokens.

        OpenAI Realtime clients commonly send ``"inf"`` for unbounded output.
        The duplex core represents that as ``None`` because model-specific
        defaults remain in force.
        """
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"inf", "infinity", "unlimited"}:
            return None
        if isinstance(value, int) and value > 0:
            return int(value)
        return None

    def _conversation_item_to_duplex(self, event: dict[str, object]) -> dict[str, object] | None:
        item = event.get("item")
        if not isinstance(item, dict):
            return None
        item = self._normalize_conversation_item(item)
        previous_item_id = event.get("previous_item_id")
        if isinstance(previous_item_id, str):
            item["_previous_item_id"] = previous_item_id
        item_id = str(item["id"])
        item_type = item.get("type")
        role = item.get("role")
        if item_type != "message" or role in {"assistant", "system"}:
            return {
                "type": "turn.signal",
                "event": "conversation.item.create",
                "payload": {"item": item},
            }
        self._conversation_items[item_id] = item
        content = item.get("content")
        if not isinstance(content, list):
            return None
        text_chunks: list[str] = []
        audio_events: list[dict[str, object]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"input_text", "text"} and isinstance(part.get("text"), str):
                text_chunks.append(str(part["text"]))
            if part.get("type") in {"input_audio", "audio"}:
                audio = part.get("audio") or part.get("data")
                fmt, format_rate = self._parse_realtime_audio_format(part.get("format") or self._input_audio_format)
                sample_rate_hz = (
                    part.get("sample_rate_hz") or part.get("sample_rate") or format_rate or self._input_sample_rate_hz
                )
                if not self._is_supported_realtime_input_format(fmt):
                    continue
                audio, fmt, sample_rate_hz = self._convert_realtime_input_audio_with_rate(
                    audio,
                    fmt,
                    sample_rate_hz=sample_rate_hz if isinstance(sample_rate_hz, int | float) else None,
                )
                if not isinstance(audio, str) or not audio:
                    continue
                speech_hints = dict(event)
                speech_hints.update(part)
                if not self._input_looks_like_speech(speech_hints, audio=audio, fmt=fmt):
                    continue
                self._input_speech_started = True
                payload = {
                    "type": "input_audio_buffer.append",
                    "audio": audio,
                    "format": fmt,
                    "sample_rate_hz": sample_rate_hz,
                }
                self._copy_realtime_input_hints(part, payload)
                self._copy_realtime_input_hints(event, payload)
                audio_events.append(payload)
        if audio_events:
            self._input_audio_buffer_has_audio = True
            transcript = self._input_transcript_from_item(item)
            for extra_event in audio_events[1:]:
                self._pending_outbound.put_nowait(extra_event)
            commit_payload: dict[str, object] = {
                "type": "input_audio_buffer.commit",
                "final": True,
                "realtime_item_id": item_id,
                "response_create": False,
            }
            if transcript:
                commit_payload["transcript"] = transcript
            self._pending_outbound.put_nowait(commit_payload)
            return audio_events[0]
        if not text_chunks:
            return None
        text_item = dict(item)
        text_item["status"] = "completed"
        return {
            "type": "turn.signal",
            "event": "conversation.item.create",
            "payload": {"item": text_item},
        }

    def _from_duplex_event(self, event: dict[str, Any]) -> list[dict[str, object]]:
        event_type = event.get("type")
        if event_type == "session.created":
            session = self._realtime_session_payload(event.get("session"))
            payloads: list[dict[str, object]] = [{"type": "session.created", "session": session}]
            if self._initial_session_update:
                payloads.append({"type": "session.updated", "session": session})
                self._initial_session_update = False
            self._hold_realtime_output_until_session_created = False
            if self._held_realtime_payloads:
                payloads.extend(self._held_realtime_payloads)
                self._held_realtime_payloads = []
            return payloads
        if event_type == "session.updated":
            session = self._realtime_session_payload(event.get("session"))
            return [{"type": "session.updated", "session": session}]
        if event_type == "response.created":
            response_id = event.get("response_id")
            if isinstance(response_id, str) and response_id:
                self._active_response_id = response_id
                self._last_response_id = response_id
            item_id = self._response_item_id(response_id)
            modalities = event.get("modalities")
            has_audio_modality = not isinstance(modalities, list) or "audio" in modalities
            item = {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            }
            self._conversation_items[item_id] = item
            payloads = [
                self._response_created_event(event),
                *self._conversation_item_added_events(item),
                *self._response_output_item_added_events(response_id=response_id, item=item),
            ]
            if has_audio_modality:
                payloads.extend(self._ensure_response_audio_part_added(response_id))
            return payloads
        if event_type == "response.listen":
            return [
                {
                    "type": "response.listen",
                    "session_id": event.get("session_id"),
                    "epoch": event.get("epoch"),
                    "response": {
                        "object": "realtime.response",
                        "status": "listening",
                        "metadata": event,
                    },
                }
            ]
        if event_type == "response.speak":
            response_id = event.get("response_id")
            return [
                {
                    "type": "response.speak",
                    "response_id": response_id,
                    "item_id": self._response_item_id(response_id),
                    "output_index": 0,
                    "content_index": 0,
                    "text": event.get("text", ""),
                    "metadata": event,
                }
            ]
        if event_type == "overlap.decision":
            return [
                {
                    "type": "overlap.decision",
                    "session_id": event.get("session_id"),
                    "epoch": event.get("epoch"),
                    "policy": event.get("policy"),
                    "action": event.get("action"),
                    "reason": event.get("reason"),
                    "metadata": event,
                }
            ]
        if event_type == "response.output_audio.delta":
            response_id = event.get("response_id")
            audio = event.get("audio", "")
            payloads: list[dict[str, object]] = []
            if isinstance(audio, str) and audio:
                payloads.extend(self._ensure_response_audio_part_added(response_id))
                payloads.extend(self._realtime_audio_delta_events(event, response_id, audio))
                self._refresh_in_progress_response_item(response_id)
            text = event.get("text")
            has_text = isinstance(text, str) and bool(text)
            if has_text:
                self._append_response_transcript(response_id, text)
                self._refresh_in_progress_response_item(response_id)
            # Keep the audio.delta + transcript.delta pair invariant even for
            # text-less units (deduplicated continuations, turn-end flush):
            # clients that treat the pair as unit-complete would otherwise
            # wait on a transcript that never comes.
            emit_transcript = has_text or (isinstance(audio, str) and bool(audio))
            if emit_transcript:
                if not has_text:
                    text = ""
                if self._emit_output_audio_events:
                    payloads.append(
                        {
                            "type": "response.output_audio_transcript.delta",
                            "response_id": response_id,
                            "item_id": self._response_item_id(response_id),
                            "output_index": 0,
                            "content_index": 0,
                            "delta": text,
                        }
                    )
                if self._emit_legacy_audio_events:
                    payloads.append(
                        {
                            "type": "response.audio_transcript.delta",
                            "response_id": response_id,
                            "item_id": self._response_item_id(response_id),
                            "output_index": 0,
                            "content_index": 0,
                            "delta": text,
                        }
                    )
                elif self._emit_response_audio_alias_events:
                    payloads.append(
                        {
                            "type": "response.audio_transcript.delta",
                            "response_id": response_id,
                            "item_id": self._response_item_id(response_id),
                            "output_index": 0,
                            "content_index": 0,
                            "delta": text,
                        }
                    )
                if self._emit_legacy_audio_events:
                    payloads.append({"type": "response.text.delta", "response_id": response_id, "delta": text})
            if event.get("end_of_turn") is True:
                payloads.extend(self._realtime_audio_done_events(event, response_id))
                payloads.extend(
                    self._realtime_response_terminal_events(
                        event,
                        response_id,
                        status="completed",
                        status_details={
                            "type": "completed",
                            "reason": event.get("finish_reason") or "stop",
                        },
                    )
                )
            return payloads
        if event_type == "response.text.delta":
            response_id = event.get("response_id")
            text = event.get("delta", "")
            if isinstance(response_id, str) and isinstance(text, str) and text:
                self._response_text_parts.setdefault(response_id, []).append(text)
                self._refresh_in_progress_response_item(response_id)
            payloads = self._ensure_response_text_part_added(response_id)
            payloads.append(
                {
                    "type": "response.output_text.delta",
                    "response_id": response_id,
                    "item_id": self._response_item_id(response_id),
                    "output_index": 0,
                    "content_index": 1
                    if (isinstance(response_id, str) and response_id in self._audio_content_part_added_response_ids)
                    else 0,
                    "delta": text,
                }
            )
            if self._emit_legacy_audio_events:
                payloads.append(
                    {
                        "type": "response.text.delta",
                        "response_id": response_id,
                        "delta": text,
                    }
                )
            return payloads
        if event_type == "response.done":
            response_id = event.get("response_id")
            status = event.get("status") if isinstance(event.get("status"), str) else "completed"
            status_details = event.get("status_details") if isinstance(event.get("status_details"), dict) else None
            return [
                *self._realtime_audio_done_events(event, response_id),
                *self._realtime_response_terminal_events(
                    event,
                    response_id,
                    status=status,
                    status_details=status_details,
                ),
            ]
        if event_type == "input.committed":
            event_item_id = event.get("realtime_item_id")
            item_id = (
                event_item_id
                if isinstance(event_item_id, str) and event_item_id
                else self._pop_pending_commit_item_id()
            )
            item = self._conversation_items.get(item_id)
            created_payload: list[dict[str, object]] = []
            if item is None:
                message = event.get("message")
                no_response = event.get("no_response") is True
                is_speech = event.get("is_speech")
                item = {
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "role": "user",
                    "status": "completed",
                    "content": (
                        [{"type": "input_audio", "transcript": "", "is_speech": False}]
                        if no_response and is_speech is False
                        else self._user_item_content_from_duplex_message(message)
                    ),
                }
                self._conversation_items[item_id] = item
                created_payload.extend(self._conversation_item_added_events(item))
            item["status"] = "completed"
            payloads = created_payload + [
                self._input_audio_buffer_committed_event(item_id=item_id, event=event),
            ]
            transcription_event = self._input_audio_transcription_completed_event(item_id, item)
            if transcription_event is not None:
                payloads.append(transcription_event)
            payloads.append(self._conversation_item_done_event(item))
            return payloads
        if event_type == "input.cancelled":
            return [{"type": "input_audio_buffer.cleared"}]
        if event_type == "input_audio_buffer.cleared":
            return [{"type": "input_audio_buffer.cleared"}]
        if event_type == "audio.cancelled":
            response_id = event.get("response_id")
            if not isinstance(response_id, str) or not response_id:
                response_id = self._active_response_id or self._last_response_id
            payloads: list[dict[str, object]] = []
            if event.get("reason") == "output_audio_buffer_clear":
                payloads.append(
                    {
                        "type": "output_audio_buffer.cleared",
                        "response_id": response_id,
                    }
                )
                if not isinstance(response_id, str) or not response_id:
                    return payloads
            if not isinstance(response_id, str) or not response_id:
                return [
                    {
                        "type": "response.cancelled",
                        "event": event,
                    }
                ]
            committed_ms = event.get("committed_ms")
            if isinstance(committed_ms, int | float):
                item_id = self._response_item_id(response_id)
                committed_audio_ms = max(0, int(committed_ms))
                self._item_truncation_cursors[item_id] = (0, committed_audio_ms)
                item = self._conversation_items.get(item_id)
                if item is not None:
                    self._truncate_realtime_item_content(
                        item,
                        content_index=0,
                        audio_end_ms=committed_audio_ms,
                    )
                payloads.append(
                    {
                        "type": "conversation.item.truncated",
                        "item_id": item_id,
                        "content_index": 0,
                        "audio_end_ms": committed_audio_ms,
                        "event": event,
                    }
                )
            if self._emit_legacy_audio_events:
                payloads.append(
                    {
                        "type": "response.cancelled",
                        "response_id": response_id,
                        "event": event,
                    }
                )
            payloads.extend(self._realtime_audio_done_events(event, response_id))
            payloads.extend(
                self._realtime_response_terminal_events(
                    event,
                    response_id,
                    status="cancelled",
                    status_details={
                        "type": "cancelled",
                        "reason": event.get("reason") or "client_cancelled",
                    },
                )
            )
            if isinstance(response_id, str) and response_id == self._active_response_id:
                self._active_response_id = None
            return payloads
        if event_type == "playback.acknowledged":
            return [{"type": "playback.acknowledged", "event": event}]
        if event_type == "conversation.item.created":
            item = event.get("item")
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                item_id = str(item["id"])
                already_known = item_id in self._conversation_items
                self._conversation_items[item_id] = item
                if already_known:
                    if item.get("status") == "completed":
                        return [self._conversation_item_done_event(item)]
                    return []
                payloads = self._conversation_item_added_events(item)
                if item.get("status") == "completed":
                    payloads.append(self._conversation_item_done_event(item))
                return payloads
            return [{"type": "conversation.item.created", "item": item, "event": event}]
        if event_type == "conversation.item.deleted":
            item_id = event.get("item_id")
            if isinstance(item_id, str):
                self._remove_conversation_item(item_id)
            return [
                {
                    "type": "conversation.item.deleted",
                    "item_id": item_id,
                    "event": event,
                }
            ]
        if event_type == "conversation.item.truncated":
            item_id = event.get("item_id")
            audio_end_ms = event.get("audio_end_ms")
            content_index = event.get("content_index", 0)
            if isinstance(item_id, str):
                item = self._conversation_items.get(item_id)
                if item is not None:
                    self._truncate_realtime_item_content(
                        item,
                        content_index=int(content_index) if isinstance(content_index, int | float) else 0,
                        audio_end_ms=int(audio_end_ms) if isinstance(audio_end_ms, int | float) else 0,
                    )
            return [
                {
                    "type": "conversation.item.truncated",
                    "item_id": item_id,
                    "content_index": content_index,
                    "audio_end_ms": audio_end_ms,
                    "event": event,
                }
            ]
        if event_type == "response.output_item.done":
            response_id = event.get("response_id")
            return self._realtime_response_terminal_events(
                event,
                response_id,
                status="completed",
                status_details={
                    "type": "completed",
                    "reason": event.get("finish_reason") or "stop",
                },
            )
        if event_type == "error":
            raw_error = event.get("error")
            if isinstance(raw_error, dict):
                return [event]
            message = str(raw_error or event.get("message") or "Duplex runtime error")
            code = str(event.get("code") or "duplex_error")
            return [self._realtime_error_payload(code, message)]
        if event_type == "session.closed":
            return [{"type": "session.closed", "event": event}]
        return [{"type": f"duplex.{event_type}", "event": event}]

    @staticmethod
    def _user_item_content_from_duplex_message(message: object) -> list[dict[str, object]]:
        if not isinstance(message, dict):
            return [{"type": "input_audio", "transcript": ""}]
        message_transcript = message.get("transcript")
        content = message.get("content")
        if isinstance(content, str):
            return [{"type": "input_text", "text": content}]
        if not isinstance(content, list):
            return [
                {
                    "type": "input_audio",
                    "transcript": message_transcript if isinstance(message_transcript, str) else "",
                }
            ]
        parts: list[dict[str, object]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in {"text", "input_text"} and isinstance(part.get("text"), str):
                parts.append({"type": "input_text", "text": part["text"]})
                continue
            if part_type == "audio_url":
                transcript = (
                    part.get("transcript")
                    if isinstance(part.get("transcript"), str)
                    else message_transcript
                    if isinstance(message_transcript, str)
                    else ""
                )
                parts.append({"type": "input_audio", "transcript": transcript})
                continue
            if part_type in {"audio", "input_audio"}:
                transcript = part.get("transcript") if isinstance(part.get("transcript"), str) else ""
                parts.append({"type": "input_audio", "transcript": transcript})
        return parts or [{"type": "input_audio", "transcript": ""}]

    def _remember_input_transcript_hint(self, event: dict[str, object]) -> None:
        transcript = event.get("transcript")
        if not isinstance(transcript, str):
            transcript = event.get("text") if isinstance(event.get("text"), str) else None
        if not isinstance(transcript, str):
            hints = event.get("hints")
            if isinstance(hints, dict):
                transcript = hints.get("transcript")
                if not isinstance(transcript, str):
                    transcript = hints.get("text") if isinstance(hints.get("text"), str) else None
        if isinstance(transcript, str) and transcript:
            if (
                self._input_audio_buffer_transcript_parts
                and self._input_audio_buffer_transcript_parts[-1] == transcript
            ):
                return
            self._input_audio_buffer_transcript_parts.append(transcript)

    def _consume_input_transcript_hint(self) -> str:
        transcript = "".join(self._input_audio_buffer_transcript_parts).strip()
        self._input_audio_buffer_transcript_parts.clear()
        return transcript

    @staticmethod
    def _input_transcript_from_item(item: dict[str, object]) -> str:
        content = item.get("content")
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            for key in ("transcript", "text"):
                value = part.get(key)
                if isinstance(value, str) and value:
                    if part.get("type") in {"input_audio", "audio", "audio_transcript", "transcript"}:
                        parts.append(value)
                        break
        return "".join(parts).strip()

    @staticmethod
    def _input_audio_transcription_completed_event(
        item_id: str,
        item: dict[str, object],
    ) -> dict[str, object] | None:
        content = item.get("content")
        if not isinstance(content, list):
            return None
        transcript_parts: list[str] = []
        for index, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            if part.get("type") != "input_audio":
                continue
            transcript = part.get("transcript")
            if isinstance(transcript, str) and transcript:
                transcript_parts.append(transcript)
                return {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item_id,
                    "content_index": index,
                    "transcript": "".join(transcript_parts),
                }
        return None

    def _pop_pending_commit_item_id(self) -> str:
        try:
            return self._pending_commit_item_ids.get_nowait()
        except asyncio.QueueEmpty:
            return f"item_{uuid4().hex}"

    def _realtime_session_payload(self, session: object) -> object:
        if not isinstance(session, dict):
            return session
        payload = dict(session)
        payload.setdefault("object", "realtime.session")
        payload.setdefault("type", "realtime")
        payload.setdefault("id", payload.get("id") or self._default_session_id)
        payload.setdefault("model", payload.get("model") or self._default_model)
        payload.setdefault("input_audio_format", self._input_audio_format)
        payload.setdefault("output_audio_format", self._output_audio_format)
        payload.setdefault("modalities", payload.get("modalities") or ["text", "audio"])
        payload.setdefault("output_modalities", payload.get("output_modalities") or payload.get("modalities"))
        payload.setdefault("object", "realtime.session")
        payload.setdefault(
            "audio",
            {
                "input": {
                    "format": self._realtime_audio_format_object(
                        self._input_audio_format,
                        sample_rate_hz=self._input_sample_rate_hz,
                    ),
                    "sample_rate_hz": self._input_sample_rate_hz,
                },
                "output": {
                    "format": self._realtime_audio_format_object(
                        self._output_audio_format,
                        sample_rate_hz=self._output_sample_rate_hz,
                    ),
                },
            },
        )
        payload.setdefault("turn_detection", payload.get("turn_detection"))
        payload.setdefault("input_audio_transcription", payload.get("input_audio_transcription"))
        payload.setdefault("tracing", payload.get("tracing"))
        return payload

    @staticmethod
    def _realtime_audio_format_object(fmt: object, *, sample_rate_hz: int | None = None) -> dict[str, object]:
        if isinstance(fmt, str) and fmt.lower() in {"pcm16", "pcm_s16le", "s16le", "pcm"}:
            payload: dict[str, object] = {"type": "audio/pcm"}
        elif isinstance(fmt, str) and fmt.lower() == "pcm_f32le":
            payload = {"type": "audio/pcm_f32le"}
        elif isinstance(fmt, str) and fmt.lower() == "g711_ulaw":
            payload = {"type": "audio/g711_ulaw"}
        elif isinstance(fmt, str) and fmt.lower() == "g711_alaw":
            payload = {"type": "audio/g711_alaw"}
        else:
            payload = {"type": "audio/wav"}
        if sample_rate_hz is not None:
            payload["rate"] = int(sample_rate_hz)
        return payload

    def _response_item_id(self, response_id: object) -> str:
        if isinstance(response_id, str) and response_id:
            return self._response_items.setdefault(response_id, f"item_{response_id}")
        return f"item_{uuid4().hex}"

    def _response_done_output_item(
        self,
        response_id: object,
        *,
        status: str,
    ) -> dict[str, object]:
        item_id = self._response_item_id(response_id)
        transcript = ""
        audio_duration_ms: int | None = None
        audio_text_marks: list[dict[str, int]] | None = None
        if isinstance(response_id, str):
            transcript = "".join(self._response_transcripts.get(response_id, []))
            audio_duration_ms = self._response_audio_durations_ms.get(response_id)
            audio_text_marks = self._response_audio_text_marks.get(response_id)
        text = "".join(self._response_text_parts.get(response_id, [])) if isinstance(response_id, str) else ""
        content: list[dict[str, object]] = []
        if (
            (isinstance(response_id, str) and response_id in self._audio_content_part_added_response_ids)
            or transcript
            or audio_duration_ms is not None
        ):
            content.append(
                self._response_item_content_part(
                    transcript=transcript,
                    audio_duration_ms=audio_duration_ms,
                    audio_text_marks=audio_text_marks,
                )
            )
        if text:
            content.append(self._response_item_text_content_part(text=text))
        item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "role": "assistant",
            "status": status,
            "content": content,
        }
        self._apply_pending_item_truncation(item)
        return item

    def _apply_pending_item_truncation(self, item: dict[str, object]) -> None:
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            return
        cursor = self._item_truncation_cursors.get(item_id)
        if cursor is None:
            return
        content_index, audio_end_ms = cursor
        self._truncate_realtime_item_content(
            item,
            content_index=content_index,
            audio_end_ms=audio_end_ms,
        )

    def _refresh_in_progress_response_item(self, response_id: object) -> None:
        if not isinstance(response_id, str) or not response_id:
            return
        item_id = self._response_item_id(response_id)
        item = self._conversation_items.get(item_id)
        if not isinstance(item, dict):
            return
        content = item.get("content")
        if not isinstance(content, list):
            content = []
            item["content"] = content

        transcript = "".join(self._response_transcripts.get(response_id, []))
        audio_duration_ms = self._response_audio_durations_ms.get(response_id)
        audio_text_marks = self._response_audio_text_marks.get(response_id)
        has_audio = (
            response_id in self._audio_content_part_added_response_ids
            or bool(transcript)
            or audio_duration_ms is not None
        )
        if has_audio:
            audio_part = self._response_item_content_part(
                transcript=transcript,
                audio_duration_ms=audio_duration_ms,
                audio_text_marks=audio_text_marks,
            )
            if content and isinstance(content[0], dict) and content[0].get("type") in {"audio", "output_audio"}:
                content[0] = audio_part
            else:
                content.insert(0, audio_part)

        text = "".join(self._response_text_parts.get(response_id, []))
        if text:
            text_index = (
                1
                if content
                and isinstance(content[0], dict)
                and content[0].get("type")
                in {
                    "audio",
                    "output_audio",
                }
                else 0
            )
            text_part = self._response_item_text_content_part(text=text)
            if (
                len(content) > text_index
                and isinstance(content[text_index], dict)
                and content[text_index].get("type") in {"text", "output_text"}
            ):
                content[text_index] = text_part
            else:
                content.insert(text_index, text_part)

        self._apply_pending_item_truncation(item)

    def _append_response_transcript(self, response_id: object, text: str) -> None:
        if not isinstance(response_id, str) or not text:
            return
        self._response_transcripts.setdefault(response_id, []).append(text)

    def _ensure_response_text_part_added(self, response_id: object) -> list[dict[str, object]]:
        if not isinstance(response_id, str) or not response_id:
            return []
        if response_id in self._text_content_part_added_response_ids:
            return []
        self._text_content_part_added_response_ids.add(response_id)
        content_index = 1 if response_id in self._audio_content_part_added_response_ids else 0
        return [
            {
                "type": "response.content_part.added",
                "response_id": response_id,
                "item_id": self._response_item_id(response_id),
                "output_index": 0,
                "content_index": content_index,
                "part": self._response_text_content_part(),
            }
        ]

    def _ensure_response_audio_part_added(self, response_id: object) -> list[dict[str, object]]:
        if not isinstance(response_id, str) or not response_id:
            return []
        if response_id in self._audio_content_part_added_response_ids:
            return []
        self._audio_content_part_added_response_ids.add(response_id)
        return [
            {
                "type": "response.content_part.added",
                "response_id": response_id,
                "item_id": self._response_item_id(response_id),
                "output_index": 0,
                "content_index": 0,
                "part": self._response_content_part(),
            }
        ]

    def _remember_response_audio_metadata(self, response_id: object, event: dict[str, Any]) -> None:
        if not isinstance(response_id, str) or not response_id:
            return
        duration = event.get("audio_duration_ms")
        playback = event.get("playback")
        if not isinstance(duration, int | float) and isinstance(playback, dict):
            duration = playback.get("sent_ms") or playback.get("generated_ms")
        if isinstance(duration, int | float):
            self._response_audio_durations_ms[response_id] = max(
                self._response_audio_durations_ms.get(response_id, 0),
                int(duration),
            )
        marks = event.get("audio_text_marks")
        if not isinstance(marks, list):
            return
        clean_marks: list[dict[str, int]] = []
        for mark in marks:
            if not isinstance(mark, dict):
                continue
            text_chars = mark.get("text_chars")
            audio_end_ms = mark.get("audio_end_ms", mark.get("audio_ms"))
            if not isinstance(text_chars, int | float) or not isinstance(audio_end_ms, int | float):
                continue
            clean_marks.append(
                {
                    "text_chars": max(0, int(text_chars)),
                    "audio_end_ms": max(0, int(audio_end_ms)),
                }
            )
        if clean_marks:
            merged = list(self._response_audio_text_marks.get(response_id, []))
            merged.extend(clean_marks)
            deduped: dict[tuple[int, int], dict[str, int]] = {}
            for mark in merged:
                deduped[(int(mark["audio_end_ms"]), int(mark["text_chars"]))] = mark
            self._response_audio_text_marks[response_id] = sorted(
                deduped.values(),
                key=lambda mark: (mark["audio_end_ms"], mark["text_chars"]),
            )

    def _response_created_event(self, event: dict[str, Any]) -> dict[str, object]:
        response_id = event.get("response_id")
        metadata = event.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = {**metadata, "duplex_event": event}
        return {
            "type": "response.created",
            "response_id": response_id,
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "status": "in_progress",
                "status_details": None,
                "output": [],
                "modalities": event.get("modalities") or ["audio", "text"],
                "metadata": metadata,
            },
        }

    def _realtime_audio_delta_events(
        self,
        event: dict[str, Any],
        response_id: object,
        audio: str,
    ) -> list[dict[str, object]]:
        item_id = self._response_item_id(response_id)
        fmt, format_rate = self._parse_realtime_audio_format(event.get("format", "wav"))
        source_fmt = self._realtime_output_format(fmt)
        source_sample_rate_hz = event.get("sample_rate_hz") or format_rate
        target_sample_rate_hz = (
            self._output_sample_rate_hz if self._output_audio_format in {"g711_ulaw", "g711_alaw"} else None
        )
        audio, fmt, converted_sample_rate_hz = self._convert_realtime_output_audio(
            audio,
            source_fmt=source_fmt,
            target_fmt=self._output_audio_format,
            source_sample_rate_hz=(
                int(source_sample_rate_hz) if isinstance(source_sample_rate_hz, int | float) else None
            ),
            target_sample_rate_hz=target_sample_rate_hz,
        )
        sample_rate_hz = (
            converted_sample_rate_hz
            if fmt in {"g711_ulaw", "g711_alaw"}
            else event.get("sample_rate_hz") or format_rate or self._output_sample_rate_hz
        )
        metadata: dict[str, object] = {}
        for key in ("session_id", "epoch", "model_speak", "end_of_turn", "kv_cache_length", "playback", "vllm_omni"):
            if key in event:
                metadata[key] = event[key]
        duration_ms = event.get("audio_duration_ms")
        if isinstance(duration_ms, int | float):
            metadata["audio_duration_ms"] = int(duration_ms)
        marks = event.get("audio_text_marks")
        if isinstance(marks, list):
            metadata["audio_text_marks"] = marks
        if isinstance(response_id, str):
            self._response_audio_formats[response_id] = fmt
            self._audio_delta_response_ids.add(response_id)
            self._remember_response_audio_metadata(response_id, event)
        payloads: list[dict[str, object]] = []
        if self._emit_output_audio_events:
            payloads.append(
                {
                    "type": "response.output_audio.delta",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": audio,
                    "audio": audio,
                    "format": fmt,
                    **({"sample_rate_hz": int(sample_rate_hz)} if isinstance(sample_rate_hz, int | float) else {}),
                    **({"metadata": metadata} if metadata else {}),
                }
            )
        if (
            isinstance(response_id, str)
            and response_id not in self._speak_response_ids
            and metadata.get("model_speak") is True
        ):
            self._speak_response_ids.add(response_id)
            payloads.insert(
                0,
                {
                    "type": "response.speak",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": event.get("text", ""),
                    "metadata": event,
                },
            )
        if self._emit_response_audio_alias_events or self._emit_legacy_audio_events:
            payloads.append(
                {
                    "type": "response.audio.delta",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": audio,
                    "format": fmt,
                    **({"sample_rate_hz": int(sample_rate_hz)} if isinstance(sample_rate_hz, int | float) else {}),
                    **({"metadata": metadata} if metadata else {}),
                }
            )
        return payloads

    def _realtime_audio_done_events(
        self,
        event: dict[str, Any],
        response_id: object,
    ) -> list[dict[str, object]]:
        item_id = self._response_item_id(response_id)
        transcript = ""
        if isinstance(response_id, str):
            transcript = "".join(self._response_transcripts.get(response_id, []))
        payloads: list[dict[str, object]] = []
        done_key = response_id if isinstance(response_id, str) else str(id(event))
        if isinstance(response_id, str) and response_id not in self._audio_delta_response_ids and not transcript:
            return payloads
        if done_key not in self._audio_done_response_ids:
            self._audio_done_response_ids.add(done_key)
            if self._emit_output_audio_events:
                payloads.append(
                    {
                        "type": "response.output_audio.done",
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                    }
                )
            if self._emit_response_audio_alias_events or self._emit_legacy_audio_events:
                payloads.append(
                    {
                        "type": "response.audio.done",
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                    }
                )
            if transcript:
                if self._emit_output_audio_events:
                    payloads.append(
                        {
                            "type": "response.output_audio_transcript.done",
                            "response_id": response_id,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "transcript": transcript,
                        }
                    )
                if self._emit_legacy_audio_events:
                    payloads.append(
                        {
                            "type": "response.audio_transcript.done",
                            "response_id": response_id,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "transcript": transcript,
                        }
                    )
                elif self._emit_response_audio_alias_events:
                    payloads.append(
                        {
                            "type": "response.audio_transcript.done",
                            "response_id": response_id,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "transcript": transcript,
                        }
                    )
        return payloads

    def _realtime_response_terminal_events(
        self,
        event: dict[str, Any],
        response_id: object,
        *,
        status: str = "completed",
        status_details: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        item_id = self._response_item_id(response_id)
        transcript = ""
        if isinstance(response_id, str):
            transcript = "".join(self._response_transcripts.get(response_id, []))
        done_key = response_id if isinstance(response_id, str) else str(id(event))
        payloads: list[dict[str, object]] = []
        if (
            isinstance(response_id, str)
            and response_id in self._audio_content_part_added_response_ids
            and done_key not in self._content_done_response_ids
        ):
            self._content_done_response_ids.add(done_key)
            payloads.append(
                {
                    "type": "response.content_part.done",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": self._response_content_part(transcript=transcript),
                }
            )
        if (
            isinstance(response_id, str)
            and self._response_text_parts.get(response_id)
            and done_key not in self._output_text_done_response_ids
        ):
            self._output_text_done_response_ids.add(done_key)
            content_index = 1 if response_id in self._audio_content_part_added_response_ids else 0
            payloads.append(
                {
                    "type": "response.output_text.done",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": content_index,
                    "text": "".join(self._response_text_parts.get(response_id, [])),
                }
            )
        if (
            isinstance(response_id, str)
            and self._response_text_parts.get(response_id)
            and done_key not in self._text_content_part_done_response_ids
        ):
            self._text_content_part_done_response_ids.add(done_key)
            content_index = 1 if response_id in self._audio_content_part_added_response_ids else 0
            payloads.append(
                {
                    "type": "response.content_part.done",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": content_index,
                    "part": self._response_text_content_part(
                        text="".join(self._response_text_parts.get(response_id, []))
                    ),
                }
            )
        if done_key not in self._output_item_done_response_ids:
            self._output_item_done_response_ids.add(done_key)
            item = self._response_done_output_item(response_id, status=status)
            payloads.append(
                {
                    "type": "response.output_item.done",
                    "response_id": response_id,
                    "output_index": 0,
                    "item": item,
                }
            )
            if done_key not in self._conversation_item_done_response_ids:
                self._conversation_item_done_response_ids.add(done_key)
                payloads.append(self._conversation_item_done_event(item))
        done_event = self._realtime_response_done_event(
            event,
            status=status,
            status_details=status_details,
        )
        if done_event is not None:
            payloads.append(done_event)
            payloads.append(self._rate_limits_updated_event())
            if isinstance(response_id, str) and response_id == self._active_response_id:
                self._active_response_id = None
        return payloads

    def _realtime_response_done_event(
        self,
        event: dict[str, Any],
        *,
        status: str = "completed",
        status_details: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        response_id = event.get("response_id")
        if isinstance(response_id, str):
            if response_id in self._done_response_ids:
                return None
            self._done_response_ids.add(response_id)
        return {
            "type": "response.done",
            "response_id": response_id,
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "status": status,
                "status_details": status_details,
                "output": [self._response_done_output_item(response_id, status=status)],
                "metadata": event,
            },
        }

    @staticmethod
    def _rate_limits_updated_event() -> dict[str, object]:
        # vLLM-Omni does not yet expose a Realtime-specific quota budget. Emit
        # the terminal event with an empty list so clients that sequence on
        # rate_limits.updated do not need a private vLLM branch.
        return {"type": "rate_limits.updated", "rate_limits": []}

    async def _emit_input_speech_started(self, event: dict[str, object]) -> None:
        if self._input_speech_started:
            return
        self._input_speech_started = True
        if self._active_input_item_id is None:
            self._active_input_item_id = f"item_{uuid4().hex}"
        await self._send_realtime_payload(
            {
                "type": "input_audio_buffer.speech_started",
                "audio_start_ms": int(event.get("audio_start_ms", 0) or 0),
                "item_id": self._active_input_item_id,
            }
        )

    async def _emit_input_speech_stopped(self, event: dict[str, object], *, item_id: str) -> None:
        if not self._input_speech_started:
            return
        self._input_speech_started = False
        audio_end_ms = event.get("audio_end_ms", event.get("audio_ms", 0))
        await self._send_realtime_payload(
            {
                "type": "input_audio_buffer.speech_stopped",
                "audio_end_ms": int(audio_end_ms) if isinstance(audio_end_ms, int | float) else 0,
                "item_id": item_id,
            }
        )
