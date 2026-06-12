from __future__ import annotations

import asyncio
import base64
import json
import struct
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

from vllm_omni.entrypoints.openai.api_server import _should_enable_duplex_endpoint
from vllm_omni.entrypoints.openai.native_realtime_protocol import NativeRealtimeSessionProtocol
from vllm_omni.entrypoints.openai.protocol.duplex import (
    DuplexPlaybackCommitPolicy,
    DuplexSession,
    DuplexSessionConfig,
)
from vllm_omni.entrypoints.openai.serving_duplex import (
    DuplexSessionActor,
    OmniDuplexSessionHandler,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ModelConfig:
    model = "test-model"


class FakeEngineClient:
    output_modalities = ["text", "audio"]

    def __init__(
        self,
        *,
        fail_open: bool = False,
        fail_signal: bool = False,
        fail_signal_events: set[str] | None = None,
        fail_close: bool = False,
        fail_abort: bool = False,
        control_result: dict[str, object] | None = None,
        open_result: dict[str, object] | None = None,
        append_result: dict[str, object] | None = None,
        collect_outputs: list[list[object]] | None = None,
        collect_delay_s: float = 0.0,
        signal_result: dict[str, object] | None = None,
        close_result: dict[str, object] | None = None,
    ) -> None:
        self.fail_open = fail_open
        self.fail_signal = fail_signal
        self.fail_signal_events = set(fail_signal_events or ())
        self.fail_close = fail_close
        self.fail_abort = fail_abort
        self.control_result = control_result
        self.open_result = open_result
        self.append_result = append_result
        self.collect_outputs = list(collect_outputs or [])
        self.collect_delay_s = collect_delay_s
        self.signal_result = signal_result
        self.close_result = close_result
        self.opened: list[str] = []
        self.appended: list[tuple[str, str, object, bool]] = []
        self.opened_configs: list[dict[str, object]] = []
        self.signals: list[tuple[str, str]] = []
        self.closed: list[tuple[str, str]] = []
        self.aborted: list[str] = []
        self.collected: list[tuple[str, int | None]] = []

    async def open_duplex_session_async(
        self,
        session_id: str,
        *,
        session_mode: str = "duplex",
        capabilities: dict[str, object] | None = None,
        session_config: dict[str, object] | None = None,
        timeout: float | None = None,
    ) -> None:
        del timeout
        if self.fail_open:
            raise RuntimeError("open failed")
        self.opened.append(session_id)
        self.opened_configs.append(dict(session_config or {}))
        return self.open_result if self.open_result is not None else self.control_result

    async def append_duplex_input_async(
        self,
        session_id: str,
        *,
        mode: str,
        payload: object,
        final: bool = False,
        timeout: float | None = None,
        collect_outputs: bool = True,
    ) -> None:
        del timeout, collect_outputs
        self.appended.append((session_id, mode, payload, final))
        return self.append_result if self.append_result is not None else self.control_result

    async def collect_duplex_data_plane_outputs_async(
        self,
        request_id: str,
        *,
        response_stage_id: int | None = None,
        timeout: float | None = None,
    ) -> list[object]:
        del timeout
        self.collected.append((request_id, response_stage_id))
        if self.collect_delay_s > 0:
            await asyncio.sleep(self.collect_delay_s)
        if not self.collect_outputs:
            return []
        return self.collect_outputs.pop(0)

    async def signal_duplex_turn_async(
        self,
        session_id: str,
        *,
        event: str,
        payload: dict[str, object] | None = None,
        timeout: float | None = None,
    ) -> None:
        del timeout
        if self.fail_signal or event in self.fail_signal_events:
            raise RuntimeError("signal failed")
        self.signals.append((session_id, event))
        return self.signal_result if self.signal_result is not None else self.control_result

    async def close_duplex_session_async(
        self,
        session_id: str,
        *,
        reason: str = "client_close",
        timeout: float | None = None,
    ) -> None:
        del timeout
        if self.fail_close:
            raise RuntimeError("close failed")
        self.closed.append((session_id, reason))
        return self.close_result if self.close_result is not None else self.control_result

    async def abort(self, request_id: str) -> None:
        if self.fail_abort:
            raise RuntimeError("abort failed")
        self.aborted.append(request_id)


class FakeChatService:
    def __init__(self, engine_client: FakeEngineClient) -> None:
        self.engine_client = engine_client
        self.model_config = _ModelConfig()
        self.seen_request_ids: list[str] = []

    async def create_chat_completion(self, request, raw_request=None):
        self.seen_request_ids.append(request.request_id)

        async def _gen():
            await asyncio.sleep(999)
            yield "data: [DONE]\n\n"

        return _gen()

    def create_audio(self, audio_obj):
        return SimpleNamespace(audio_data=f"wav-{int(audio_obj.audio_tensor.shape[0])}")


class TimedWebSocket:
    def __init__(self, *, on_send=None):
        self._q: asyncio.Queue[str] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.accepted = False
        self._on_send = on_send
        self.query_params: dict[str, str] = {}

    async def accept(self):
        self.accepted = True

    async def receive_text(self) -> str:
        try:
            return await asyncio.wait_for(self._q.get(), timeout=1.0)
        except asyncio.TimeoutError as exc:
            raise WebSocketDisconnect(code=1000) from exc

    async def send_json(self, data: dict[str, Any]):
        self.sent.append(data)
        if self._on_send is not None:
            self._on_send(self, data)

    def put(self, payload: dict[str, Any]) -> None:
        self._q.put_nowait(json.dumps(payload))

    def sent_types(self) -> list[str]:
        return [m.get("type", "") for m in self.sent]


@pytest.mark.asyncio
async def test_native_realtime_protocol_drains_internal_conversation_item_control_event():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    ws.put({"type": "session.update", "model": "test-model", "session_id": "rt-delete"})
    session_create = json.loads(await protocol.receive_internal_event_text(ws))
    assert session_create["type"] == "session.create"

    ws.put({"type": "conversation.item.delete", "item_id": "item-a"})
    translated = json.loads(await protocol.receive_internal_event_text(ws))

    assert translated == {
        "type": "turn.signal",
        "event": "conversation.item.delete",
        "payload": {"item_id": "item-a"},
    }


@pytest.mark.asyncio
async def test_native_realtime_protocol_conversation_item_create_commits_user_text():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    ws.put({"type": "session.update", "model": "test-model", "session_id": "rt-item-create"})
    session_create = json.loads(await protocol.receive_internal_event_text(ws))
    assert session_create["type"] == "session.create"

    ws.put(
        {
            "type": "conversation.item.create",
            "item": {
                "id": "item-user-text",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        }
    )

    append_event = json.loads(await protocol.receive_internal_event_text(ws))
    commit_event = json.loads(await protocol.receive_internal_event_text(ws))

    assert append_event == {"type": "input.text.append", "text": "hello"}
    assert commit_event == {
        "type": "input.commit",
        "final": True,
        "realtime_item_id": "item-user-text",
        "response_create": False,
    }
    assert ws.sent[-1]["type"] == "conversation.item.created"
    assert ws.sent[-1]["item"]["id"] == "item-user-text"


@pytest.mark.asyncio
async def test_native_realtime_protocol_audio_commit_requires_non_empty_buffer():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    ws.put({"type": "session.update", "model": "test-model", "session_id": "rt-audio-commit"})
    session_create = json.loads(await protocol.receive_internal_event_text(ws))
    assert session_create["type"] == "session.create"

    ws.put({"type": "input_audio_buffer.commit", "final": True})
    with pytest.raises(WebSocketDisconnect):
        await protocol.receive_internal_event_text(ws)

    assert ws.sent[-1]["type"] == "error"
    assert ws.sent[-1]["error"]["code"] == "input_audio_buffer_empty"


@pytest.mark.asyncio
async def test_native_realtime_protocol_audio_commit_does_not_auto_create_response():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    ws.put({"type": "session.update", "model": "test-model", "session_id": "rt-audio-commit-full"})
    session_create = json.loads(await protocol.receive_internal_event_text(ws))
    assert session_create["type"] == "session.create"

    pcm = struct.pack("<2h", 1024, -1024)
    ws.put({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii"), "format": "pcm16"})
    append_event = json.loads(await protocol.receive_internal_event_text(ws))
    assert append_event["type"] == "input_audio_buffer.append"

    ws.put({"type": "input_audio_buffer.commit", "final": True})
    commit_event = json.loads(await protocol.receive_internal_event_text(ws))

    assert commit_event["type"] == "input_audio_buffer.commit"
    assert commit_event["response_create"] is False


@pytest.mark.asyncio
async def test_native_realtime_protocol_audio_clear_is_not_barge_in_cancel():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    ws.put({"type": "session.update", "model": "test-model", "session_id": "rt-audio-clear"})
    session_create = json.loads(await protocol.receive_internal_event_text(ws))
    assert session_create["type"] == "session.create"

    ws.put({"type": "input_audio_buffer.clear"})
    clear_event = json.loads(await protocol.receive_internal_event_text(ws))

    assert clear_event == {"type": "input_audio_buffer.clear", "reason": "input_audio_buffer.clear"}


@pytest.mark.asyncio
async def test_native_realtime_protocol_preserves_input_turn_policy_hints():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    translated = await protocol._to_duplex_event(
        {
            "type": "input_audio_buffer.append",
            "audio": "AAAA",
            "format": "pcm_f32le",
            "duration_ms": 240,
            "vad": {"is_speech": True, "speech_probability": 0.9},
            "overlap_action": "listen",
        }
    )

    assert translated is not None
    assert translated["duration_ms"] == 240
    assert translated["vad"] == {"is_speech": True, "speech_probability": 0.9}
    assert translated["overlap_action"] == "listen"


@pytest.mark.asyncio
async def test_native_realtime_protocol_turn_detection_maps_to_overlap_policy():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    ws.put(
        {
            "type": "session.update",
            "model": "test-model",
            "session_id": "rt-turn-detection",
            "turn_detection": {
                "type": "server_vad",
                "interrupt_response": False,
                "silence_duration_ms": 900,
                "threshold": 0.4,
            },
        }
    )
    translated = json.loads(await protocol.receive_internal_event_text(ws))

    assert translated["type"] == "session.create"
    session = translated["session"]
    assert session["overlap_policy"] == "listen_only"
    assert session["overlap_short_ack_ms"] == 900
    assert session["overlap_silence_rms"] == pytest.approx(0.004)


@pytest.mark.asyncio
async def test_realtime_session_update_preserves_tools_metadata_and_turn_detection():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-rt-update-fields"))
    ws.put(
        {
            "type": "turn.signal",
            "event": "session.update",
            "payload": {
                "tools": [{"type": "function", "name": "lookup"}],
                "tool_choice": "auto",
                "metadata": {"demo": "yes"},
                "turn_detection": {"type": "server_vad", "interrupt_response": False},
            },
        }
    )
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    updated = next(m for m in ws.sent if m.get("type") == "session.updated")
    session = updated["session"]
    assert session["tools"] == [{"type": "function", "name": "lookup"}]
    assert session["tool_choice"] == "auto"
    assert session["metadata"] == {"demo": "yes"}
    assert session["turn_detection"] == {"type": "server_vad", "interrupt_response": False}


@pytest.mark.asyncio
async def test_native_realtime_protocol_non_speech_append_does_not_emit_speech_started():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    translated = await protocol._to_duplex_event(
        {
            "type": "input_audio_buffer.append",
            "audio": "AAAA",
            "format": "pcm_f32le",
            "vad": {"is_speech": False},
        }
    )

    assert translated is not None
    assert translated["vad"] == {"is_speech": False}
    assert not any(event.get("type") == "input_audio_buffer.speech_started" for event in ws.sent)


def test_native_realtime_protocol_audio_delta_preserves_sample_rate_hz():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]

    payloads = protocol._from_duplex_event(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-a",
            "audio": "AAAA",
            "format": "pcm",
            "sample_rate_hz": 24000,
        }
    )

    audio_events = [
        payload for payload in payloads if payload["type"] in {"response.output_audio.delta", "response.audio.delta"}
    ]
    assert {payload["type"] for payload in audio_events} == {"response.output_audio.delta"}
    assert {payload["format"] for payload in audio_events} == {"pcm16"}
    assert {payload["sample_rate_hz"] for payload in audio_events} == {24000}


def test_native_realtime_protocol_emits_terminal_audio_transcript_events():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]

    payloads = protocol._from_duplex_event(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-a",
            "audio": "AAAA",
            "text": "hello",
            "format": "pcm",
            "sample_rate_hz": 24000,
            "end_of_turn": True,
        }
    )

    by_type = {payload["type"]: payload for payload in payloads}
    assert "response.output_audio.done" in by_type
    assert "response.output_audio_transcript.delta" in by_type
    assert "response.output_audio_transcript.done" in by_type
    assert by_type["response.content_part.done"]["part"]["transcript"] == "hello"
    assert by_type["response.output_item.done"]["item"]["object"] == "realtime.item"
    assert by_type["response.done"]["response"]["output"][0]["object"] == "realtime.item"


def test_native_realtime_protocol_updates_in_progress_item_for_audio_truncate():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]

    protocol._from_duplex_event({"type": "response.created", "response_id": "resp-truncate"})
    protocol._from_duplex_event(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-truncate",
            "audio": "AAAA",
            "text": "hello",
            "format": "pcm",
            "sample_rate_hz": 24000,
            "audio_duration_ms": 100,
            "audio_text_marks": [{"text_chars": 5, "audio_end_ms": 100}],
        }
    )

    item = protocol._conversation_items["item_resp-truncate"]
    assert item["content"][0]["type"] == "output_audio"
    assert item["content"][0]["transcript"] == "hello"
    assert item["content"][0]["audio_duration_ms"] == 100

    translated = asyncio.run(
        protocol._to_duplex_event(
            {
                "type": "conversation.item.truncate",
                "item_id": "item_resp-truncate",
                "content_index": 0,
                "audio_end_ms": 50,
            }
        )
    )

    assert translated is None
    assert protocol._conversation_items["item_resp-truncate"]["content"][0]["transcript"] == "he"


@pytest.mark.asyncio
async def test_native_realtime_protocol_rejects_truncate_for_user_item():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)
    protocol._conversation_items["item-user"] = {
        "id": "item-user",
        "object": "realtime.item",
        "type": "message",
        "role": "user",
        "status": "completed",
        "content": [{"type": "input_audio", "transcript": "hello"}],
    }

    translated = await protocol._to_duplex_event(
        {
            "type": "conversation.item.truncate",
            "item_id": "item-user",
            "content_index": 0,
            "audio_end_ms": 10,
        }
    )

    assert translated is None
    error = next(event for event in ws.sent if event.get("type") == "error")
    assert error["error"]["code"] == "bad_event"


def test_native_realtime_protocol_response_done_emits_audio_done_before_terminal_events():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol._from_duplex_event({"type": "response.created", "response_id": "resp-done"})

    payloads = protocol._from_duplex_event({"type": "response.done", "response_id": "resp-done"})
    event_types = [payload["type"] for payload in payloads]

    assert event_types.index("response.output_audio.done") < event_types.index("response.content_part.done")
    assert event_types.index("response.content_part.done") < event_types.index("response.output_item.done")
    assert "conversation.item.done" not in event_types
    assert event_types.index("response.output_item.done") < event_types.index("response.done")


def test_native_audio_text_marks_are_normalized_to_session_cumulative_offsets():
    marks = OmniDuplexSessionHandler._normalize_native_audio_text_marks(
        [{"text_chars": 3, "audio_end_ms": 200}],
        audio_offset_ms=800,
        text_offset_chars=5,
    )

    assert marks == [{"text_chars": 8, "audio_end_ms": 1000}]


def test_duplex_session_playback_commit_uses_multi_delta_audio_text_marks():
    session = DuplexSession(
        session_id="sid-marks",
        config=DuplexSessionConfig(playback_commit_policy=DuplexPlaybackCommitPolicy.ACK_ONLY.value),
    )
    session.begin_response()
    session.append_assistant_text("hello ")
    session.mark_audio_sent(1000, text_chars=6)
    session.append_assistant_text("world")
    session.mark_audio_sent(2000, text_chars=11)
    session.acknowledge_playback(played_ms=1500, committed_ms=1500)

    committed = session.end_response(commit_text=True)

    assert committed == {"role": "assistant", "content": "hello wo"}
    assert session.history == [committed]


@pytest.mark.asyncio
async def test_duplex_session_actor_prioritizes_control_over_buffered_input():
    ws = TimedWebSocket()
    actor = DuplexSessionActor(ws)

    await actor.enqueue_event({"type": "input_audio_buffer.append", "audio": "AAAA"})
    await actor.enqueue_event({"type": "response.cancel"})

    first = await actor.next_event()
    second = await actor.next_event()

    assert first["type"] == "response.cancel"
    assert second["type"] == "input_audio_buffer.append"


def _pcm_f32_b64(samples: int, *, value: float = 0.05) -> str:
    return base64.b64encode(struct.pack(f"<{samples}f", *([value] * samples))).decode("ascii")


def _session_create(session_id: str = "duplex-test") -> dict[str, Any]:
    return {
        "type": "session.create",
        "session_id": session_id,
        "session": {
            "model": "test-model",
            "modalities": ["text", "audio"],
            "idle_timeout_s": 1,
        },
    }


def _native_session_create(session_id: str = "duplex-native") -> dict[str, Any]:
    event = _session_create(session_id)
    event["session"]["model"] = "openbmb/MiniCPM-o-4_5"
    event["session"]["instructions"] = "You are a concise assistant."
    event["session"]["extra_body"] = {"minicpmo45_native_duplex": True}
    return event


def test_duplex_endpoint_requires_explicit_session_mode_duplex():
    assert _should_enable_duplex_endpoint(None) is False
    assert _should_enable_duplex_endpoint([]) is False
    assert _should_enable_duplex_endpoint([SimpleNamespace(session_mode="turn")]) is False
    assert _should_enable_duplex_endpoint([{"session_mode": "turn"}]) is False
    assert (
        _should_enable_duplex_endpoint(
            [
                SimpleNamespace(session_mode="turn"),
                SimpleNamespace(session_mode="duplex"),
            ]
        )
        is True
    )
    assert _should_enable_duplex_endpoint([{"session_mode": "duplex"}]) is True


def test_duplex_endpoint_supports_top_level_session_mode(tmp_path):
    config_path = tmp_path / "stage_config.yaml"
    config_path.write_text(
        """
session_mode: duplex
stage_args:
  - stage_id: 0
    engine_args: {}
""",
        encoding="utf-8",
    )

    assert _should_enable_duplex_endpoint([], config_path=str(config_path)) is True


def test_duplex_handler_iter_native_results_does_not_duplicate_single_result():
    result = {
        "stage_results": [
            {
                "stage_id": 1,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "native_result": {
                        "is_listen": False,
                        "text": "hello",
                    },
                },
            }
        ],
    }

    native_results = list(OmniDuplexSessionHandler._iter_native_duplex_results(result))

    assert native_results == [{"is_listen": False, "text": "hello"}]


def test_duplex_handler_splits_data_plane_audio_list_into_deltas():
    import torch

    class _ChatService:
        model_config = _ModelConfig()

        def create_audio(self, audio_obj):
            return SimpleNamespace(audio_data=f"wav-{int(audio_obj.audio_tensor.shape[0])}")

    handler = OmniDuplexSessionHandler(chat_service=_ChatService())
    output = SimpleNamespace(
        finished=True,
        outputs=[
            SimpleNamespace(
                text="hello",
                multimodal_output={
                    "audio": [
                        torch.zeros(10, dtype=torch.float32),
                        torch.zeros(20, dtype=torch.float32),
                    ],
                    "sr": 24000,
                },
            )
        ],
    )

    native_results = list(handler._native_results_from_data_plane_output(output))

    assert [result["audio_data"] for result in native_results] == ["wav-10", "wav-20"]
    assert [result["text"] for result in native_results] == ["hello", ""]
    assert [result["end_of_turn"] for result in native_results] == [False, True]


def test_duplex_listen_latent_does_not_poison_cumulative_audio_offset():
    import torch

    class _ChatService:
        model_config = _ModelConfig()

        def create_audio(self, audio_obj):
            return SimpleNamespace(audio_data=f"wav-{int(audio_obj.audio_tensor.shape[0])}")

    handler = OmniDuplexSessionHandler(chat_service=_ChatService())
    request_id = "duplex-duplex-sess-stage0"

    # A model-listen decision wraps the segment with a latent tensor that is
    # NOT reply audio; it must not advance the cumulative audio offset.
    listen_output = SimpleNamespace(
        request_id=request_id,
        finished=True,
        outputs=[],
        multimodal_output={
            "duplex_native_decision": "listen",
            "model_listen": True,
            "latent": torch.zeros(331776, dtype=torch.float32),
            "meta": {"sr": 24000},
        },
    )
    listen_results = list(handler._native_results_from_data_plane_output(listen_output))
    assert [result.get("is_listen") for result in listen_results] == [True]

    # The first speak unit carries cumulative stage-1 audio far smaller than
    # the listen latent; it must still be delivered from sample 0.
    speak_output = SimpleNamespace(
        request_id=request_id,
        finished=False,
        outputs=[
            SimpleNamespace(
                text=" It was a very",
                multimodal_output={},
            )
        ],
        multimodal_output={
            "audio": torch.zeros(32768, dtype=torch.float32),
            "sr": 24000,
        },
    )
    speak_results = list(handler._native_results_from_data_plane_output(speak_output))
    assert [result.get("audio_data") for result in speak_results] == ["wav-32768"]
    assert speak_results[0]["text"] == " It was a very"


def test_duplex_segment_text_is_attached_once_across_streaming_batches():
    import torch

    class _ChatService:
        model_config = _ModelConfig()

        def create_audio(self, audio_obj):
            return SimpleNamespace(audio_data=f"wav-{int(audio_obj.audio_tensor.shape[0])}")

    handler = OmniDuplexSessionHandler(chat_service=_ChatService())
    request_id = "duplex-duplex-sess-stage0"
    session = DuplexSession(
        session_id="sid-auto-respond",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )

    def _speak(total_samples: int, text: str, *, finished: bool):
        output = SimpleNamespace(
            request_id=request_id,
            finished=finished,
            outputs=[SimpleNamespace(text=text, multimodal_output={})],
            multimodal_output={
                "audio": torch.zeros(total_samples, dtype=torch.float32),
                "sr": 24000,
            },
        )
        return list(handler._data_plane_native_results({"data_plane_outputs": [output]}, session=session))

    # One talker segment streams several cumulative-audio batches, each
    # carrying the SAME segment text; only the first may attach it.
    assert [r["text"] for r in _speak(100, " movie called", finished=False)] == [" movie called"]
    assert [r["text"] for r in _speak(220, " movie called", finished=False)] == [""]
    assert [r["text"] for r in _speak(300, " movie called", finished=True)] == [""]

    # The next segment starts fresh: its text is attached again, even when
    # it repeats the previous segment's text verbatim.
    assert [r["text"] for r in _speak(400, " Titanic", finished=True)] == [" Titanic"]
    assert [r["text"] for r in _speak(500, " Titanic", finished=True)] == [" Titanic"]

    # Text growing within a segment is delivered as suffix deltas.
    assert [r["text"] for r in _speak(600, " by", finished=False)] == [" by"]
    assert [r["text"] for r in _speak(700, " by James", finished=True)] == [" James"]

    # A segment whose finished batch slices to an EMPTY audio delta (all
    # samples already delivered) must still end the segment: the next
    # segment's text may not be suffix-sliced against the previous one.
    assert [r["text"] for r in _speak(800, " James Cameron. The", finished=False)] == [" James Cameron. The"]
    assert _speak(800, " James Cameron. The", finished=True) == []
    assert [r["text"] for r in _speak(900, " 997.", finished=False)] == [" 997."]


@pytest.mark.asyncio
async def test_duplex_chat_audio_stream_uses_output_audio_delta_event():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    session = DuplexSession(session_id="sid-chat-audio", config=DuplexSessionConfig())
    response_id = session.begin_response()
    sent: list[dict[str, Any]] = []

    async def send_json(data: dict[str, Any]) -> None:
        sent.append(data)

    await handler._emit_chat_payload(
        session,
        {
            "modality": "audio",
            "choices": [
                {
                    "delta": {
                        "content": "AAAA",
                    }
                }
            ],
        },
        session.epoch,
        response_id,
        send_json,
    )

    assert sent == [
        {
            "type": "response.output_audio.delta",
            "session_id": "sid-chat-audio",
            "response_id": response_id,
            "epoch": 0,
            "audio": "AAAA",
            "format": "wav",
        }
    ]


@pytest.mark.asyncio
async def test_minicpmo_model_name_does_not_auto_enable_experimental_native_duplex():
    event = _session_create("sid-minicpmo-default-protocol")
    event["session"]["model"] = "openbmb/MiniCPM-o-4_5"
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(event)
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    created = next(m for m in ws.sent if m.get("type") == "session.created")
    capabilities = created["session"]["capabilities"]
    assert capabilities["implementation_level"] == "serving_session_adapter"
    assert capabilities["supports_input_append"] is False


@pytest.mark.asyncio
async def test_duplex_handler_aborts_current_chat_request_id_on_barge_in():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)

    def on_send(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        if data.get("type") == "response.created":
            ws.put({"type": "input.cancel", "reason": "test_barge_in"})

    ws = TimedWebSocket(on_send=on_send)
    ws.put(_session_create("sid-a"))
    ws.put({"type": "input.text.append", "text": "hello"})
    ws.put({"type": "input.commit"})

    await handler.handle_session(ws)

    assert ws.accepted
    assert "response.created" in ws.sent_types()
    assert "audio.cancelled" in ws.sent_types()
    assert chat_service.seen_request_ids == ["duplex-sid-a-0-1"]
    assert engine.aborted == ["chatcmpl-duplex-sid-a-0-1"]


@pytest.mark.asyncio
async def test_duplex_handler_runtime_open_failure_is_reported_to_client():
    engine = FakeEngineClient(fail_open=True)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-open-fail"))
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ws.sent_types() == ["error"]
    assert ws.sent[0]["code"] == "runtime_open_failed"
    assert engine.opened == []


@pytest.mark.asyncio
async def test_duplex_cancel_reports_playback_committed_cursor():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)

    def on_send(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        if data.get("type") == "response.created":
            ws.put({"type": "playback.ack", "played_ms": 1200, "committed_ms": 1000})
            ws.put({"type": "input.cancel", "reason": "test_barge_in"})

    ws = TimedWebSocket(on_send=on_send)
    ws.put(_session_create("sid-playback"))
    ws.put({"type": "input.text.append", "text": "hello"})
    ws.put({"type": "input.commit"})

    await handler.handle_session(ws)

    ack = next(m for m in ws.sent if m.get("type") == "playback.acknowledged")
    cancelled = next(m for m in ws.sent if m.get("type") == "audio.cancelled")
    assert ack["playback"]["played_ms"] == 1200
    assert ack["playback"]["committed_ms"] == 1000
    assert cancelled["committed_ms"] == 1000
    assert cancelled["epoch"] == 1


@pytest.mark.asyncio
async def test_cancel_active_response_commits_only_played_assistant_history():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    session = DuplexSession(
        session_id="sid-playback-partial",
        config=DuplexSessionConfig(playback_commit_policy=DuplexPlaybackCommitPolicy.COMMIT_ALL_ON_DONE.value),
    )
    response_id = session.begin_response()
    session.append_assistant_text("hello world")
    session.mark_audio_sent(1000, text_chars=len("hello world"))
    session.acknowledge_playback(played_ms=500, committed_ms=500)

    cancelled = await handler._cancel_active_response(session, None, ws.send_json, reason="barge_in")

    assert cancelled is True
    assert session.epoch == 1
    assert session.history == [{"role": "assistant", "content": "hello"}]
    assert session.history_item_ids[f"item_{response_id}"] == session.history[0]
    event = next(m for m in ws.sent if m.get("type") == "audio.cancelled")
    assert event["committed_ms"] == 500


@pytest.mark.asyncio
async def test_cancel_active_native_data_plane_request_aborts_stage_request_id():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    session = DuplexSession(session_id="sid-native-abort", config=DuplexSessionConfig())
    session.active_request_id = "duplex-sid-native-abort-e0-stage0-s1"

    cancelled = await handler._cancel_active_response(session, None, ws.send_json, reason="barge_in")

    assert cancelled is True
    assert engine.aborted == ["duplex-sid-native-abort-e0-stage0-s1"]
    assert "audio.cancelled" in ws.sent_types()


@pytest.mark.asyncio
async def test_duplex_handler_explicit_close_closes_runtime_once_with_client_reason():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-close"))
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ws.sent_types() == ["session.created", "session.closed"]
    assert engine.closed == [("sid-close", "session_close")]


@pytest.mark.asyncio
async def test_duplex_handler_idle_timeout_close_does_not_emit_runtime_control():
    control_result = {
        "operation": "close",
        "session_id": "sid-disconnect",
        "ok": True,
        "unsupported_count": 1,
        "error_count": 0,
        "stage_results": [{"stage_id": 0, "replica_id": 0, "result": {"supported": False}}],
    }
    engine = FakeEngineClient(append_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=0.1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-disconnect"))

    await handler.handle_session(ws)

    assert engine.closed == [("sid-disconnect", "timeout")]


@pytest.mark.asyncio
async def test_duplex_handler_runtime_close_failure_is_reported_without_closed_ack():
    engine = FakeEngineClient(fail_close=True)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-close-fail"))
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ws.sent_types() == ["session.created", "error"]
    assert ws.sent[-1]["code"] == "runtime_close_failed"


@pytest.mark.asyncio
async def test_duplex_handler_control_close_failure_is_reported_without_closed_ack():
    control_result = {
        "operation": "close",
        "session_id": "sid-control-close-fail",
        "ok": False,
        "unsupported_count": 0,
        "error_count": 1,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {"supported": False, "error": "stage close failed"},
            }
        ],
    }
    engine = FakeEngineClient(close_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-control-close-fail"))
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ws.sent_types() == ["session.created", "error"]
    assert ws.sent[-1]["code"] == "runtime_close_failed"
    assert ws.sent[-1]["runtime_control"]["error_count"] == 1


@pytest.mark.asyncio
async def test_duplex_cancel_without_active_response_clears_pending_input_and_acks():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-pending-cancel"))
    ws.put({"type": "input.text.append", "text": "hello"})
    ws.put({"type": "input.cancel", "reason": "user_cancel"})
    ws.put({"type": "input.commit"})

    await handler.handle_session(ws)

    cancelled = next(m for m in ws.sent if m.get("type") == "input.cancelled")
    assert cancelled["cancelled"] == {"text_chunks": 1, "audio_chunks": 0}
    assert cancelled["epoch"] == 1
    assert chat_service.seen_request_ids == []
    assert "response.created" not in ws.sent_types()


@pytest.mark.asyncio
async def test_duplex_handler_runtime_signal_failure_is_reported_to_client():
    engine = FakeEngineClient(fail_signal=True)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-signal-fail"))
    ws.put({"type": "turn.signal", "event": "user_started"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    error = next(m for m in ws.sent if m.get("type") == "error")
    assert error["code"] == "runtime_signal_failed"
    assert not engine.signals


@pytest.mark.asyncio
async def test_duplex_barge_in_aborts_active_response_when_runtime_signal_fails():
    engine = FakeEngineClient(fail_signal_events={"barge_in"})
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)

    def on_send(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        if data.get("type") == "response.created":
            ws.put({"type": "input.cancel", "reason": "test_barge_in"})

    ws = TimedWebSocket(on_send=on_send)
    ws.put(_session_create("sid-barge-signal-fail"))
    ws.put({"type": "input.text.append", "text": "hello"})
    ws.put({"type": "input.commit"})

    await handler.handle_session(ws)

    assert "audio.cancelled" in ws.sent_types()
    assert engine.aborted == ["chatcmpl-duplex-sid-barge-signal-fail-0-1"]
    error = next(m for m in ws.sent if m.get("type") == "error")
    assert error["code"] == "runtime_signal_failed"


@pytest.mark.asyncio
async def test_duplex_handler_surfaces_stage_unsupported_result_to_client():
    control_result = {
        "operation": "open",
        "session_id": "sid-unsupported",
        "ok": True,
        "unsupported_count": 1,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {"supported": False, "reason": "not implemented"},
            }
        ],
    }
    engine = FakeEngineClient(open_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-unsupported"))
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    created = next(m for m in ws.sent if m.get("type") == "session.created")
    assert created["runtime_control"]["unsupported_count"] == 1


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_rejects_stage0_result_without_runner_kv():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-listen",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "passive_count": 1,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "native_result": {
                        "is_listen": True,
                        "kv_cache_length": 128,
                        "stage_role": "llm",
                        "runtime_impl": "vllm_omni_minicpmo45_stage0_experimental_worker_runtime",
                        "owned_runtime": False,
                        "uses_model_runner_scheduler": False,
                        "runner_kv_backed": False,
                        "experimental_worker_control_rpc": True,
                        "experimental_eager_decoder": False,
                    },
                },
            }
        ],
    }
    engine = FakeEngineClient(append_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-listen"))
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA", "force_listen": True})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    created = next(m for m in ws.sent if m.get("type") == "session.created")
    assert created["session"]["capabilities"]["implementation_level"] == "model_native_duplex"
    assert created["session"]["capabilities"]["requires_model_runner_kv"] is True
    assert engine.appended == [
        (
            "sid-native-listen",
            "append_audio_chunk",
            {
                "type": "audio",
                "audio": "AAAA",
                "format": "pcm_f32le",
                "sample_rate_hz": 16000,
                "force_listen": True,
            },
            False,
        )
    ]
    assert "response.listen" not in ws.sent_types()
    error = next(m for m in ws.sent if m.get("type") == "error")
    assert error["code"] == "runtime_native_runner_kv_required"
    assert "scheduler/KV-backed" in error["error"]


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_append_control_error_does_not_emit_model_delta():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-append-error",
        "ok": False,
        "unsupported_count": 0,
        "error_count": 1,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "native_result": {
                        "is_listen": False,
                        "text": "stage0 text must not leak as success",
                        "stage_role": "llm",
                        "uses_model_runner_scheduler": True,
                        "runner_kv_backed": True,
                    },
                },
            },
            {
                "stage_id": -1,
                "replica_id": -1,
                "result": {
                    "supported": False,
                    "error": "duplex_stage_handoff_target_missing:tts",
                },
            },
        ],
    }
    engine = FakeEngineClient(append_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-append-error"))
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert "response.output_audio.delta" not in ws.sent_types()
    error = next(m for m in ws.sent if m.get("type") == "error")
    assert error["code"] == "runtime_append_failed"
    assert error["runtime_control"]["error_count"] == 1


@pytest.mark.asyncio
async def test_duplex_handler_control_signal_failure_does_not_emit_turn_event():
    control_result = {
        "operation": "signal",
        "session_id": "sid-control-signal-fail",
        "ok": False,
        "unsupported_count": 0,
        "error_count": 1,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {"supported": False, "error": "stage signal failed"},
            }
        ],
    }
    engine = FakeEngineClient(signal_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-control-signal-fail"))
    ws.put({"type": "turn.signal", "event": "user_started"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert "turn.event" not in ws.sent_types()
    error = next(m for m in ws.sent if m.get("type") == "error")
    assert error["code"] == "runtime_signal_failed"
    assert error["runtime_control"]["error_count"] == 1


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_rejects_model_event_without_stage_role():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-missing-stage-role",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "native_result": {
                        "is_listen": True,
                        "kv_cache_length": 128,
                        "uses_model_runner_scheduler": True,
                        "runner_kv_backed": True,
                    },
                },
            }
        ],
    }
    engine = FakeEngineClient(control_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-missing-stage-role"))
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert "response.listen" not in ws.sent_types()
    error = next(m for m in ws.sent if m.get("type") == "error")
    assert error["code"] == "runtime_native_stage_role_required"
    assert "stage_role" in error["error"]


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_open_unsupported_fails_session_create():
    control_result = {
        "operation": "open",
        "session_id": "sid-native-busy",
        "ok": True,
        "unsupported_count": 1,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {"supported": False, "reason": "native_duplex_session_busy"},
            }
        ],
    }
    engine = FakeEngineClient(control_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-busy"))
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ws.sent_types() == ["error"]
    assert ws.sent[0]["code"] == "runtime_open_unsupported"
    assert ws.sent[0]["runtime_control"]["unsupported_count"] == 1


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_open_error_fails_session_create():
    control_result = {
        "operation": "open",
        "session_id": "sid-native-no-runner-kv",
        "ok": False,
        "unsupported_count": 0,
        "error_count": 1,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": False,
                    "error": ("MiniCPM-o stage0 native duplex requires duplex_forward_with_runner_context"),
                },
            }
        ],
    }
    engine = FakeEngineClient(control_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-no-runner-kv"))
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ws.sent_types() == ["error"]
    assert ws.sent[0]["code"] == "runtime_open_failed"
    assert ws.sent[0]["runtime_control"]["error_count"] == 1
    assert "duplex_forward_with_runner_context" in str(ws.sent[0]["runtime_control"])


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_audio_append_does_not_retain_pending_audio():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-no-pending",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "passive_count": 1,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "native_result": {
                        "is_listen": True,
                        "kv_cache_length": 128,
                        "stage_role": "llm",
                        "uses_model_runner_scheduler": True,
                        "runner_kv_backed": True,
                    },
                },
            }
        ],
    }
    engine = FakeEngineClient(control_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-no-pending"))
    short_chunk = base64.b64encode(b"\x00" * (800 * 4)).decode("ascii")
    ws.put({"type": "input_audio_buffer.append", "audio": short_chunk, "format": "pcm_f32le"})
    ws.put({"type": "input.cancel", "reason": "barge_in"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    cancelled = next(m for m in ws.sent if m.get("type") == "input.cancelled")
    assert cancelled["cancelled"] == {"text_chunks": 0, "audio_chunks": 0}
    assert cancelled["epoch"] == 1


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_rejects_text_append_before_runtime_call():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-text"))
    ws.put({"type": "input.text.append", "text": "hello"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    error = next(m for m in ws.sent if m.get("type") == "error")
    assert error["code"] == "native_text_append_unsupported"
    assert engine.appended == []


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_resolves_ref_audio_before_runtime_open(monkeypatch):
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)

    async def fake_resolve_ref_audio(_, *, model_config):
        del model_config
        return [0.25, -0.25], 16000

    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.duplex_adapters.minicpmo45."
        "MiniCPMO45NativeDuplexServingAdapter.resolve_ref_audio",
        fake_resolve_ref_audio,
    )
    event = _native_session_create("sid-native-ref-audio")
    event["session"]["ref_audio"] = "data:audio/wav;base64,AAAA"
    ws = TimedWebSocket()
    ws.put(event)
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ws.sent_types()[0] == "session.created"
    opened_config = engine.opened_configs[0]
    extra_body = opened_config["extra_body"]
    assert extra_body["ref_audio_format"] == "pcm_f32le"
    assert extra_body["ref_audio_sample_rate_hz"] == 16000
    assert extra_body["duplex_stage_max_tokens"] == {"0": 32}
    assert base64.b64decode(extra_body["ref_audio_data"]) == struct.pack("<ff", 0.25, -0.25)
    assert "ref_audio" not in extra_body
    assert "ref_audio_path" not in extra_body


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_preserves_ref_audio_channels_until_normalize(monkeypatch):
    class FakeMediaConnector:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def fetch_audio_async(self, ref_audio):
            assert ref_audio == "data:audio/wav;base64,AAAA"
            return np.array([[0.25, -0.25], [0.5, -0.5]], dtype=np.float32), 16000

    monkeypatch.setattr("vllm_omni.entrypoints.openai.duplex_adapters.minicpmo45.MediaConnector", FakeMediaConnector)

    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    event = _native_session_create("sid-native-ref-audio-stereo")
    event["session"]["ref_audio"] = "data:audio/wav;base64,AAAA"
    ws = TimedWebSocket()
    ws.put(event)
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    extra_body = engine.opened_configs[0]["extra_body"]
    assert base64.b64decode(extra_body["ref_audio_data"]) == struct.pack("<ff", 0.0, 0.0)


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_rejects_ref_audio_path():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    event = _native_session_create("sid-native-ref-path")
    event["session"]["extra_body"] = {
        "minicpmo45_native_duplex": True,
        "ref_audio_path": "/tmp/ref.wav",
    }
    ws = TimedWebSocket()
    ws.put(event)

    await handler.handle_session(ws)

    assert ws.sent_types() == ["error"]
    assert ws.sent[0]["code"] == "unsupported_ref_audio_path"
    assert engine.opened == []


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_skips_passive_stage_results():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-passive",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "passive_count": 1,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "native_result": {
                        "is_listen": True,
                        "kv_cache_length": 128,
                        "stage_role": "llm",
                        "runtime_impl": "vllm_omni_minicpmo45_stage0_experimental_worker_runtime",
                        "uses_model_runner_scheduler": True,
                        "runner_kv_backed": True,
                    },
                },
            },
            {
                "stage_id": 1,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "native_result": {"passive_stage": True},
                },
            },
        ],
    }
    engine = FakeEngineClient(control_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-passive"))
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ws.sent_types().count("response.listen") == 1
    assert "response.output_audio.delta" not in ws.sent_types()
    runtime_control = next(m for m in ws.sent if m.get("type") == "runtime.control")
    assert runtime_control["result"]["passive_count"] == 1


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_audio_append_emits_output_audio_delta():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-speak",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "native_result": {
                        "is_listen": False,
                        "text": "hello",
                        "audio_data": "BBBB",
                        "end_of_turn": True,
                        "kv_cache_length": 256,
                        "stage_role": "tts",
                        "runtime_impl": "vllm_omni_minicpmo45_stage1_runtime",
                        "owned_runtime": False,
                    },
                },
            }
        ],
    }
    engine = FakeEngineClient(control_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)

    def on_send(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        if data.get("type") == "response.done":
            ws.put({"type": "session.close"})

    ws = TimedWebSocket(on_send=on_send)
    ws.put(_native_session_create("sid-native-speak"))
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA"})

    await handler.handle_session(ws)

    deltas = [m for m in ws.sent if m.get("type") == "response.output_audio.delta"]
    assert len(deltas) == 1
    delta = deltas[0]
    assert delta["text"] == "hello"
    assert delta["audio"] == "BBBB"
    assert delta["end_of_turn"] is True
    assert delta["kv_cache_length"] == 256
    assert delta["vllm_omni"]["runtime_impl"] == "vllm_omni_minicpmo45_stage1_runtime"
    assert delta["vllm_omni"]["owned_runtime"] is False
    created = [m for m in ws.sent if m.get("type") == "response.created"]
    assert len(created) == 1
    assert delta["response_id"] == created[0]["response_id"]
    done = [m for m in ws.sent if m.get("type") == "response.done"]
    assert len(done) == 1
    assert done[0]["response_id"] == delta["response_id"]
    assert done[0]["committed"] is True
    assert "turn.event" not in ws.sent_types()


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_overlap_short_speech_keeps_output_running():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-overlap-listen",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "native_result": {
                        "is_listen": False,
                        "text": "still speaking",
                        "audio_data": "BBBB",
                        "end_of_turn": False,
                        "stage_role": "tts",
                    },
                },
            }
        ],
    }
    engine = FakeEngineClient(control_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)

    overlap_sent = False

    def on_send(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        nonlocal overlap_sent
        if data.get("type") == "response.output_audio.delta" and not overlap_sent:
            overlap_sent = True
            ws.put(
                {
                    "type": "input_audio_buffer.append",
                    "audio": _pcm_f32_b64(16000),
                    "format": "pcm_f32le",
                    "sample_rate_hz": 16000,
                    "duration_ms": 1000,
                    "is_speech": True,
                }
            )
        if data.get("type") == "overlap.decision":
            ws.put({"type": "session.close"})

    event = _native_session_create("sid-native-overlap-listen")
    event["session"]["overlap_short_ack_ms"] = 1000
    ws = TimedWebSocket(on_send=on_send)
    ws.put(event)
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA"})

    await handler.handle_session(ws)

    decision = next(m for m in ws.sent if m.get("type") == "overlap.decision")
    assert decision["action"] == "listen"
    assert decision["reason"] == "short_ack"
    assert ("sid-native-overlap-listen", "barge_in") not in engine.signals
    assert len(engine.appended) >= 2
    second_payload = engine.appended[1][2]
    assert isinstance(second_payload, dict)
    assert second_payload["force_listen"] is True


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_overlap_long_speech_barges_in_and_keeps_buffered_audio():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-overlap-barge",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "native_result": {
                        "is_listen": False,
                        "text": "old output",
                        "audio_data": "BBBB",
                        "end_of_turn": False,
                        "stage_role": "tts",
                    },
                },
            }
        ],
    }
    engine = FakeEngineClient(control_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)

    overlap_sent = False

    def on_send(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        nonlocal overlap_sent
        if data.get("type") == "response.output_audio.delta" and not overlap_sent:
            overlap_sent = True
            ws.put(
                {
                    "type": "input_audio_buffer.append",
                    "audio": _pcm_f32_b64(11200),
                    "format": "pcm_f32le",
                    "sample_rate_hz": 16000,
                    "duration_ms": 700,
                    "is_speech": True,
                }
            )
            ws.put(
                {
                    "type": "input_audio_buffer.append",
                    "audio": _pcm_f32_b64(9600),
                    "format": "pcm_f32le",
                    "sample_rate_hz": 16000,
                    "duration_ms": 600,
                    "is_speech": True,
                }
            )
        if data.get("type") == "audio.cancelled":
            ws.put({"type": "session.close"})

    ws = TimedWebSocket(on_send=on_send)
    ws.put(_native_session_create("sid-native-overlap-barge"))
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA"})

    await handler.handle_session(ws)

    decisions = [m for m in ws.sent if m.get("type") == "overlap.decision"]
    assert [m["action"] for m in decisions[:2]] == ["listen", "barge_in"]
    assert decisions[1]["reason"] == "long_overlap_speech"
    assert ("sid-native-overlap-barge", "barge_in") in engine.signals
    cancelled = next(m for m in ws.sent if m.get("type") == "audio.cancelled")
    assert cancelled["reason"] == "barge_in"
    assert len(engine.appended) >= 2
    overlap_payload = engine.appended[1][2]
    assert isinstance(overlap_payload, dict)
    assert len(base64.b64decode(overlap_payload["audio"])) == 16000 * 4


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_explicit_non_speech_stays_listening_without_append():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-silence",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "native_result": {
                        "is_listen": False,
                        "audio_data": "BBBB",
                        "end_of_turn": True,
                        "stage_role": "tts",
                    },
                },
            }
        ],
    }
    engine = FakeEngineClient(control_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-silence"))
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": "AAAA",
            "is_speech": False,
        }
    )
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert engine.appended == []
    assert "response.listen" in ws.sent_types()
    assert "response.created" not in ws.sent_types()
    assert "response.output_audio.delta" not in ws.sent_types()


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_drains_data_plane_stream_until_done():
    def _stage_output(samples: int, *, finished: bool):
        return SimpleNamespace(
            request_id="duplex-sid-native-stream-e0-stage0-s1",
            finished=finished,
            outputs=[
                SimpleNamespace(
                    text="",
                    multimodal_output={
                        "audio": np.zeros(samples, dtype=np.float32),
                        "sr": 24000,
                    },
                )
            ],
        )

    control_result = {
        "operation": "append",
        "session_id": "sid-native-stream",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "implementation_level": "model_native_duplex",
                    "data_plane_append": True,
                    "request_id": "duplex-sid-native-stream-e0-stage0-s1",
                    "response_stage_id": 1,
                },
            }
        ],
        "data_plane_outputs": [_stage_output(10, finished=False)],
    }
    engine = FakeEngineClient(
        append_result=control_result,
        collect_outputs=[
            [_stage_output(20, finished=False)],
            [_stage_output(30, finished=True)],
        ],
    )
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)

    def close_on_done(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        if data.get("type") == "response.done":
            ws.put({"type": "session.close"})

    ws = TimedWebSocket(on_send=close_on_done)
    ws.put(_native_session_create("sid-native-stream"))
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA"})

    await handler.handle_session(ws)

    deltas = [m for m in ws.sent if m.get("type") == "response.output_audio.delta"]
    assert [m["audio"] for m in deltas] == ["wav-10", "wav-10", "wav-10"]
    assert [m["end_of_turn"] for m in deltas] == [False, False, True]
    assert len([m for m in ws.sent if m.get("type") == "response.done"]) == 1
    assert engine.collected == [
        ("duplex-sid-native-stream-e0-stage0-s1", 1),
        ("duplex-sid-native-stream-e0-stage0-s1", 1),
    ]


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_accepts_next_append_after_response_done():
    def _stage_output(*, finished: bool):
        return SimpleNamespace(
            request_id="duplex-sid-native-next-turn-e0-stage0-s1",
            finished=finished,
            outputs=[
                SimpleNamespace(
                    text="",
                    multimodal_output={
                        "audio": np.zeros(10, dtype=np.float32),
                        "sr": 24000,
                    },
                )
            ],
        )

    control_result = {
        "operation": "append",
        "session_id": "sid-native-next-turn",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "implementation_level": "model_native_duplex",
                    "data_plane_append": True,
                    "request_id": "duplex-sid-native-next-turn-e0-stage0-s1",
                    "response_stage_id": 1,
                },
            }
        ],
        "data_plane_outputs": [_stage_output(finished=True)],
    }
    engine = FakeEngineClient(append_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    done_count = 0

    def send_next_turn_or_close(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        nonlocal done_count
        if data.get("type") != "response.done":
            return
        done_count += 1
        if done_count == 1:
            ws.put({"type": "input_audio_buffer.append", "audio": "BBBB"})
        else:
            ws.put({"type": "session.close"})

    ws = TimedWebSocket(on_send=send_next_turn_or_close)
    ws.put(_native_session_create("sid-native-next-turn"))
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA"})

    await handler.handle_session(ws)

    assert len(engine.appended) == 2
    assert len([m for m in ws.sent if m.get("type") == "response.done"]) == 2


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_cancel_interrupts_background_data_plane_stream():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-cancel-stream",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "implementation_level": "model_native_duplex",
                    "data_plane_append": True,
                    "request_id": "duplex-sid-native-cancel-stream-e0-stage0-s1",
                    "response_stage_id": 1,
                },
            }
        ],
    }
    engine = FakeEngineClient(
        append_result=control_result,
        collect_delay_s=0.2,
        collect_outputs=[
            [
                SimpleNamespace(
                    request_id="duplex-sid-native-cancel-stream-e0-stage0-s1",
                    finished=True,
                    outputs=[
                        SimpleNamespace(
                            text="",
                            multimodal_output={"audio": np.zeros(10, dtype=np.float32), "sr": 24000},
                        )
                    ],
                )
            ]
        ],
    )
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-cancel-stream"))
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA"})
    ws.put({"type": "input.cancel"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert "audio.cancelled" in ws.sent_types()
    assert "response.output_audio.delta" not in ws.sent_types()
    assert engine.signals == [("sid-native-cancel-stream", "barge_in")]


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_buffers_short_pcm_chunks_before_append():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-short-chunks",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "native_result": {
                        "is_listen": False,
                        "audio_data": "BBBB",
                        "end_of_turn": True,
                        "stage_role": "tts",
                    },
                },
            }
        ],
    }
    engine = FakeEngineClient(control_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-short-chunks"))
    chunk = base64.b64encode(b"\x00" * (3200 * 4)).decode("ascii")
    for _ in range(6):
        ws.put(
            {
                "type": "input_audio_buffer.append",
                "audio": chunk,
                "format": "pcm_f32le",
                "sample_rate_hz": 16000,
            }
        )
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert len(engine.appended) == 2
    _, mode, payload, final = engine.appended[0]
    assert mode == "append_audio_chunk"
    assert final is False
    assert isinstance(payload, dict)
    assert len(base64.b64decode(payload["audio"])) == 16000 * 4
    assert "duplex_num_input_tokens" not in payload
    _, second_mode, second_payload, second_final = engine.appended[1]
    assert second_mode == "append_audio_chunk"
    assert second_final is True
    assert isinstance(second_payload, dict)
    assert len(base64.b64decode(second_payload["audio"])) == 3200 * 4
    assert "duplex_num_input_tokens" not in second_payload
    assert len([m for m in ws.sent if m.get("type") == "response.output_audio.delta"]) == 2


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_skips_stage0_stage_handoff_delta():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-stage-handoff",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "native_result": {
                        "is_listen": False,
                        "text": "stage0 text",
                        "stage_role": "llm",
                        "uses_model_runner_scheduler": True,
                        "runner_kv_backed": True,
                        "requires_stage_handoff": True,
                        "stage_handoff": {
                            "target_stage_role": "tts",
                            "mode": "append_stage_handoff",
                            "payload": {"tts_token_ids": [1]},
                        },
                    },
                },
            },
            {
                "stage_id": 1,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "native_result": {
                        "is_listen": False,
                        "text": "stage1 text",
                        "stage_role": "tts",
                        "audio_data": "BBBB",
                    },
                },
            },
        ],
    }
    engine = FakeEngineClient(control_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-stage-handoff"))
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    deltas = [m for m in ws.sent if m.get("type") == "response.output_audio.delta"]
    assert len(deltas) == 1
    assert deltas[0]["text"] == "stage1 text"
    assert deltas[0]["audio"] == "BBBB"


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_runtime_control_redacts_stage_handoff_payload():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-redact",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "passive_count": 1,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "traceback": "server stack must stay private",
                    "native_result": {
                        "is_listen": False,
                        "stage_role": "llm",
                        "uses_model_runner_scheduler": True,
                        "runner_kv_backed": True,
                        "requires_stage_handoff": True,
                        "stage_handoff": {
                            "target_stage_role": "tts",
                            "mode": "append_stage_handoff",
                            "payload": {
                                "tts_token_ids": [1, 2],
                                "tts_hidden_states": ["opaque tensor bytes"],
                            },
                        },
                    },
                },
            },
            {
                "stage_id": 1,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "native_result": {"passive_stage": True},
                },
            },
        ],
    }
    engine = FakeEngineClient(control_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-redact"))
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    runtime_control = next(m for m in ws.sent if m.get("type") == "runtime.control")
    serialized = json.dumps(runtime_control, sort_keys=True)
    assert "stage_handoff" not in serialized
    assert "tts_handoff" not in serialized
    assert "tts_hidden_states" not in serialized
    assert "tts_token_ids" not in serialized
    assert "requires_stage_handoff" not in serialized
    assert "traceback" not in serialized


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_context_full_closes_runtime_session():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-context-full",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "native_result": {
                        "is_listen": True,
                        "kv_cache_length": 8192,
                        "stage_role": "llm",
                        "runtime_impl": "vllm_omni_minicpmo45_stage0_experimental_worker_runtime",
                        "uses_model_runner_scheduler": True,
                        "runner_kv_backed": True,
                    },
                },
            }
        ],
    }
    engine = FakeEngineClient(control_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-context-full"))
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert "response.listen" in ws.sent_types()
    assert ws.sent_types().count("session.closed") == 1
    closed = next(m for m in ws.sent if m.get("type") == "session.closed")
    assert closed["reason"] == "context_full"
    assert engine.closed == [("sid-native-context-full", "context_full")]


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_idle_timeout_closes_runtime_with_timeout_reason():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=0.1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-timeout"))

    await handler.handle_session(ws)

    assert ws.sent_types() == ["session.created", "session.closed"]
    assert ws.sent[-1]["reason"] == "timeout"
    assert engine.closed == [("sid-native-timeout", "timeout")]
