"""End-to-end MiniCPM-o 4.5 Realtime duplex demo client.

This script is intentionally scenario-based instead of a generic chat client.
It validates the full-duplex semantics implemented by vLLM-Omni:

1. normal audio input -> audio output -> response.done,
2. short overlap acknowledgement keeps the current response alive,
3. long overlap speech triggers barge-in/cancel of the previous epoch,
4. silence/noise commits produce listen/no-response behavior,
5. native listen/speak events are surfaced without conflating buffering with
   model-owned listen decisions,
6. playback/truncate events are accepted by the Realtime adapter.

Run only after a MiniCPM-o 4.5 vLLM-Omni server is up:

  python examples/online_serving/minicpmo/realtime_duplex_demo.py \
      --url ws://localhost:8099/v1/realtime?duplex=1 \
      --model openbmb/MiniCPM-o-4_5 \
      --input-wav input_16k_mono_pcm16.wav \
      --output-dir /tmp/minicpmo_duplex_demo
"""

from __future__ import annotations

import argparse
import array
import asyncio
import base64
import json
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError as exc:  # pragma: no cover - demo dependency.
    raise SystemExit("Install websockets first: pip install websockets") from exc


PCM16_SAMPLE_RATE = 16000
PCM16_BYTES_PER_SAMPLE = 2


@dataclass
class DemoState:
    events: list[dict[str, object]] = field(default_factory=list)
    audio_deltas: list[bytes] = field(default_factory=list)
    response_ids: list[str] = field(default_factory=list)
    assistant_item_ids: list[str] = field(default_factory=list)
    done_count: int = 0
    cancelled_count: int = 0
    listen_count: int = 0
    model_listen_count: int = 0
    buffering_listen_count: int = 0
    model_speak_event_count: int = 0
    model_speak_delta_count: int = 0
    playback_ack_count: int = 0
    playback_history_committed_count: int = 0
    truncate_count: int = 0
    input_transcription_count: int = 0
    audio_marks_seen: bool = False
    overlap_decisions: list[dict[str, object]] = field(default_factory=list)
    output_sample_rate_hz: int = 24000

    def add(self, event: dict[str, object]) -> None:
        self.events.append(event)
        event_type = event.get("type")
        if event_type == "response.created":
            response = event.get("response")
            response_id = response.get("id") if isinstance(response, dict) else event.get("response_id")
            if isinstance(response_id, str) and response_id not in self.response_ids:
                self.response_ids.append(response_id)
        elif event_type == "conversation.item.added":
            item = event.get("item")
            if isinstance(item, dict) and item.get("role") == "assistant":
                item_id = item.get("id")
                if isinstance(item_id, str) and item_id not in self.assistant_item_ids:
                    self.assistant_item_ids.append(item_id)
        elif event_type == "response.audio.delta":
            delta = event.get("delta") or event.get("audio")
            if isinstance(delta, str) and delta:
                try:
                    self.audio_deltas.append(base64.b64decode(delta))
                except Exception:
                    pass
            metadata = event.get("metadata")
            if isinstance(metadata, dict):
                if metadata.get("model_speak") is True:
                    self.model_speak_delta_count += 1
                if isinstance(metadata.get("audio_text_marks"), list):
                    self.audio_marks_seen = True
            sample_rate_hz = event.get("sample_rate_hz")
            if isinstance(sample_rate_hz, int) and sample_rate_hz > 0:
                self.output_sample_rate_hz = sample_rate_hz
        elif event_type == "response.done":
            self.done_count += 1
            response = event.get("response")
            if isinstance(response, dict) and response.get("status") == "cancelled":
                self.cancelled_count += 1
        elif event_type == "response.listen":
            self.listen_count += 1
            response = event.get("response")
            metadata = response.get("metadata") if isinstance(response, dict) else None
            if isinstance(metadata, dict) and metadata.get("model_listen") is True:
                self.model_listen_count += 1
            if isinstance(metadata, dict) and metadata.get("buffering") is True:
                self.buffering_listen_count += 1
        elif event_type == "response.speak":
            self.model_speak_event_count += 1
        elif event_type == "overlap.decision":
            self.overlap_decisions.append(event)
        elif event_type == "playback.acknowledged":
            self.playback_ack_count += 1
            payload = event.get("event")
            if isinstance(payload, dict) and payload.get("history_committed") is True:
                self.playback_history_committed_count += 1
        elif event_type == "conversation.item.truncated":
            self.truncate_count += 1
        elif event_type == "conversation.item.input_audio_transcription.completed":
            self.input_transcription_count += 1

    def count(self, event_type: str) -> int:
        return sum(1 for event in self.events if event.get("type") == event_type)

    def first_index(self, event_type: str, predicate=None) -> int | None:
        for index, event in enumerate(self.events):
            if event.get("type") != event_type:
                continue
            if predicate is not None and not predicate(event):
                continue
            return index
        return None

    @staticmethod
    def _event_response_id(event: dict[str, object]) -> str | None:
        response_id = event.get("response_id")
        if isinstance(response_id, str) and response_id:
            return response_id
        response = event.get("response")
        if isinstance(response, dict):
            response_id = response.get("id")
            if isinstance(response_id, str) and response_id:
                return response_id
        return None

    @staticmethod
    def _event_item_id(event: dict[str, object]) -> str | None:
        item_id = event.get("item_id")
        if isinstance(item_id, str) and item_id:
            return item_id
        item = event.get("item")
        if isinstance(item, dict):
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id:
                return item_id
        return None

    def first_response_lifecycle_indices(self) -> dict[str, int]:
        response_created_index = self.first_index("response.created")
        if response_created_index is None:
            return {}
        response_id = self._event_response_id(self.events[response_created_index])
        if not response_id:
            return {}
        item_id = f"item_{response_id}"
        indices: dict[str, int] = {"response.created": response_created_index}
        for event_type in (
            "conversation.item.added",
            "response.output_item.added",
            "response.content_part.added",
            "response.speak",
            "response.audio.delta",
            "response.audio.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.done",
        ):
            index = self.first_index(
                event_type,
                lambda event, event_type=event_type: (
                    self._event_item_id(event) == item_id
                    if event_type == "conversation.item.added"
                    else self._event_response_id(event) == response_id
                ),
            )
            if index is None:
                return {}
            indices[event_type] = index
        return indices

    def event_order_ok(self) -> bool:
        if not self.events or self.events[0].get("type") != "session.created":
            return False
        first_commit_index = self.first_index("input_audio_buffer.committed")
        first_response_index = self.first_index("response.created")
        if first_commit_index is None or first_response_index is None or first_commit_index > first_response_index:
            return False
        indices_by_type = self.first_response_lifecycle_indices()
        if not indices_by_type:
            return False
        ordered_types = [
            "response.created",
            "conversation.item.added",
            "response.output_item.added",
            "response.content_part.added",
            "response.speak",
            "response.audio.delta",
            "response.audio.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.done",
        ]
        indices = [indices_by_type[event_type] for event_type in ordered_types]
        return indices == sorted(indices)

    def model_speak_before_audio_ok(self) -> bool:
        speak_index = self.first_index("response.speak")
        audio_index = self.first_index("response.audio.delta")
        return speak_index is not None and audio_index is not None and speak_index < audio_index

    def response_done(self, response_id: str | None) -> bool:
        if not response_id:
            return False
        return any(
            event.get("type") == "response.done" and self._event_response_id(event) == response_id
            for event in self.events
        )

    def stale_audio_delta_count(self) -> int:
        cancelled_epochs_by_index: list[tuple[int, int]] = []
        for index, event in enumerate(self.events):
            if event.get("type") != "response.done":
                continue
            response = event.get("response")
            if not isinstance(response, dict) or response.get("status") != "cancelled":
                continue
            metadata = response.get("metadata")
            if not isinstance(metadata, dict):
                continue
            cancelled_epoch = metadata.get("cancelled_epoch")
            if isinstance(cancelled_epoch, int):
                cancelled_epochs_by_index.append((index, cancelled_epoch))
        if not cancelled_epochs_by_index:
            return 0
        stale = 0
        for index, event in enumerate(self.events):
            if event.get("type") != "response.audio.delta":
                continue
            metadata = event.get("metadata")
            if not isinstance(metadata, dict):
                continue
            event_epoch = metadata.get("epoch")
            for cancel_index, cancelled_epoch in cancelled_epochs_by_index:
                if index > cancel_index and event_epoch == cancelled_epoch:
                    stale += 1
                    break
        return stale


def _url_with_model(url: str, model: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.setdefault("duplex", "1")
    query.setdefault("model", model)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _read_wav_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError("input WAV must be mono")
        if wf.getsampwidth() != PCM16_BYTES_PER_SAMPLE:
            raise ValueError("input WAV must be 16-bit PCM")
        if wf.getframerate() != PCM16_SAMPLE_RATE:
            raise ValueError("input WAV must be 16 kHz")
        if wf.getcomptype() != "NONE":
            raise ValueError("input WAV must be uncompressed PCM")
        return wf.readframes(wf.getnframes())


def _write_wav(path: Path, pcm_bytes: bytes, *, sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(PCM16_BYTES_PER_SAMPLE)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm_bytes)


def _write_demo_artifacts(state: DemoState, output_dir: Path, *, output_audio_format: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if state.audio_deltas and output_audio_format == "pcm16":
        _write_wav(
            output_dir / "joined_audio_deltas.wav",
            b"".join(state.audio_deltas),
            sample_rate_hz=state.output_sample_rate_hz,
        )
    elif state.audio_deltas:
        (output_dir / "joined_audio_deltas.bin").write_bytes(b"".join(state.audio_deltas))
    (output_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in state.events) + "\n",
        encoding="utf-8",
    )


def _pcm16_silence(duration_ms: int) -> bytes:
    samples = PCM16_SAMPLE_RATE * max(0, duration_ms) // 1000
    return b"\x00\x00" * samples


def _pcm16_slice(pcm16: bytes, duration_ms: int) -> bytes:
    byte_count = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * max(1, duration_ms) // 1000
    return pcm16[: min(len(pcm16), byte_count)]


def _pcm16_active_slice(pcm16: bytes, duration_ms: int) -> bytes:
    byte_count = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * max(1, duration_ms) // 1000
    byte_count = min(len(pcm16), max(PCM16_BYTES_PER_SAMPLE, byte_count))
    byte_count -= byte_count % PCM16_BYTES_PER_SAMPLE
    if byte_count <= 0:
        return _pcm16_slice(pcm16, duration_ms)
    step = max(PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * 20 // 1000, PCM16_BYTES_PER_SAMPLE)
    step -= step % PCM16_BYTES_PER_SAMPLE
    best_offset = 0
    best_energy = -1.0
    for offset in range(0, max(1, len(pcm16) - byte_count + 1), max(PCM16_BYTES_PER_SAMPLE, step)):
        chunk = pcm16[offset : offset + byte_count]
        samples = array.array("h")
        samples.frombytes(chunk)
        if not samples:
            continue
        energy = sum(abs(sample) for sample in samples) / len(samples)
        if energy > best_energy:
            best_energy = energy
            best_offset = offset
    return pcm16[best_offset : best_offset + byte_count]


async def _reader(ws, state: DemoState, stop: asyncio.Event) -> None:
    try:
        while not stop.is_set():
            raw = await ws.recv()
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            if isinstance(event, dict):
                state.add(event)
    except ConnectionClosed:
        return


async def _send_pcm16(
    ws,
    pcm16: bytes,
    *,
    chunk_ms: int,
    realtime_delay: bool,
    hints: dict[str, object] | None = None,
) -> None:
    hints = hints or {}
    chunk_bytes = max(PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * chunk_ms // 1000, PCM16_BYTES_PER_SAMPLE)
    audio_ms = 0
    for offset in range(0, len(pcm16), chunk_bytes):
        chunk = pcm16[offset : offset + chunk_bytes]
        duration_ms = int(len(chunk) / (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE) * 1000)
        audio_ms += duration_ms
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                    "input_audio_format": "pcm16",
                    "sample_rate_hz": PCM16_SAMPLE_RATE,
                    "duration_ms": duration_ms,
                    "audio_end_ms": audio_ms,
                    **hints,
                }
            )
        )
        if realtime_delay:
            await asyncio.sleep(duration_ms / 1000)


async def _wait_for(
    state: DemoState,
    predicate,
    *,
    timeout_s: float,
    label: str,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise TimeoutError(f"Timed out waiting for {label}")


async def run_demo(args: argparse.Namespace) -> dict[str, object]:
    pcm16 = _read_wav_pcm16(Path(args.input_wav))
    if not pcm16:
        raise ValueError("input WAV has no audio")

    url = _url_with_model(args.url, args.model)
    state = DemoState()
    stop = asyncio.Event()
    output_dir = Path(args.output_dir)

    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        reader = asyncio.create_task(_reader(ws, state, stop))
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "model": args.model,
                            "modalities": ["audio", "text"],
                            "input_audio_format": "pcm16",
                            "output_audio_format": args.output_audio_format,
                            "turn_detection": {
                                "type": "server_vad",
                                "interrupt_response": True,
                                "silence_duration_ms": args.short_ack_ms,
                                "threshold": 0.35,
                            },
                            "overlap_policy": "auto",
                            "overlap_short_ack_ms": args.short_ack_ms,
                            "overlap_barge_in_ms": args.barge_in_ms,
                            "playback_commit_policy": "ack_only",
                        },
                    }
                )
            )
            await _wait_for(state, lambda: state.count("session.created") > 0, timeout_s=20, label="session.created")

            await _send_pcm16(
                ws,
                _pcm16_slice(pcm16, args.first_turn_ms),
                chunk_ms=args.chunk_ms,
                realtime_delay=False,
                hints={"transcript": args.first_turn_transcript},
            )
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
            await ws.send(json.dumps({"type": "response.create"}))
            await _wait_for(
                state,
                lambda: state.count("response.created") > 0,
                timeout_s=args.timeout_s,
                label="first response.created",
            )

            # Short acknowledgement while assistant generation is in-flight.
            # It is intentionally sent before first audio because some prompts
            # can finish in one TTS chunk, leaving no post-audio overlap window.
            before_short_ack_cancelled = state.cancelled_count
            await _send_pcm16(
                ws,
                _pcm16_active_slice(pcm16, min(args.short_ack_ms, 300)),
                chunk_ms=min(args.chunk_ms, 100),
                realtime_delay=False,
                hints={"transcript": "继续"},
            )
            await _wait_for(
                state,
                lambda: any(decision.get("action") == "listen" for decision in state.overlap_decisions),
                timeout_s=10,
                label="short-overlap listen decision",
            )
            await asyncio.sleep(0.2)
            short_ack_cancelled = state.cancelled_count > before_short_ack_cancelled
            await _wait_for(state, lambda: len(state.audio_deltas) > 0, timeout_s=args.timeout_s, label="first audio")

            # Playback progress is acknowledged through the Realtime truncate
            # path so the adapter and internal history use the same cursor.
            if state.assistant_item_ids:
                await ws.send(
                    json.dumps(
                        {
                            "type": "conversation.item.truncate",
                            "item_id": state.assistant_item_ids[-1],
                            "content_index": 0,
                            "audio_end_ms": args.playback_ack_ms,
                        }
                    )
                )
                await _wait_for(
                    state,
                    lambda: state.truncate_count > 0 and state.playback_ack_count > 0,
                    timeout_s=10,
                    label="playback truncate acknowledgement",
                )

            # Long overlap speech should barge in and cancel stale output.
            before_barge_audio = len(state.audio_deltas)
            before_barge_done = state.done_count
            before_barge_responses = len(state.response_ids)
            await _send_pcm16(
                ws,
                _pcm16_active_slice(pcm16, args.barge_in_ms + 300),
                chunk_ms=args.chunk_ms,
                realtime_delay=False,
            )
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
            await ws.send(json.dumps({"type": "response.create"}))
            await _wait_for(
                state,
                lambda: (
                    state.cancelled_count > 0 or any(d.get("action") == "barge_in" for d in state.overlap_decisions)
                ),
                timeout_s=args.timeout_s,
                label="barge-in decision",
            )
            await _wait_for(
                state,
                lambda: len(state.audio_deltas) > before_barge_audio,
                timeout_s=args.timeout_s,
                label="post-barge audio",
            )
            await _wait_for(
                state,
                lambda: len(state.response_ids) > before_barge_responses,
                timeout_s=args.timeout_s,
                label="post-barge response.created",
            )
            post_barge_response_id = state.response_ids[-1] if state.response_ids else None
            await _wait_for(
                state,
                lambda: state.done_count > before_barge_done and state.response_done(post_barge_response_id),
                timeout_s=args.timeout_s,
                label="post-barge response.done",
            )
            before_full_ack = state.playback_ack_count
            await ws.send(
                json.dumps(
                    {
                        "type": "playback.ack",
                        "played_ms": 999999,
                        "committed_ms": 999999,
                    }
                )
            )
            await _wait_for(
                state,
                lambda: state.playback_ack_count > before_full_ack,
                timeout_s=10,
                label="post-barge playback ack",
            )

            # Ask the native turn-policy path to keep listening for an
            # intentionally unfinished chunk. This verifies the Stage0
            # scheduler/runner path can surface a native listen decision, not
            # only serving-side silence/noise filtering.
            before_model_listen = state.model_listen_count
            await _send_pcm16(
                ws,
                _pcm16_active_slice(pcm16, min(args.short_ack_ms, 300)),
                chunk_ms=min(args.chunk_ms, 100),
                realtime_delay=False,
            )
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.commit",
                        "final": True,
                        "response_create": True,
                    }
                )
            )
            await _wait_for(
                state,
                lambda: state.model_listen_count > before_model_listen,
                timeout_s=args.timeout_s,
                label="native model listen",
            )

            # Silence/noise should be accepted as a buffer lifecycle event, but
            # not schedule a new response.
            before_done = state.done_count
            await _send_pcm16(
                ws,
                _pcm16_silence(args.silence_ms),
                chunk_ms=args.chunk_ms,
                realtime_delay=False,
            )
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
            await _wait_for(state, lambda: state.listen_count > 0, timeout_s=10, label="silence listen")
            await asyncio.sleep(0.5)
            silence_created_response = state.done_count > before_done

            await ws.send(json.dumps({"type": "session.close"}))
            await _wait_for(state, lambda: state.count("session.closed") > 0, timeout_s=20, label="session.closed")
        finally:
            stop.set()
            reader.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                pass
            _write_demo_artifacts(state, output_dir, output_audio_format=args.output_audio_format)

    overlap_listen = any(decision.get("action") == "listen" for decision in state.overlap_decisions)
    overlap_barge_in = any(decision.get("action") == "barge_in" for decision in state.overlap_decisions)
    event_order_ok = state.event_order_ok()
    playback_commit_ok = (
        state.truncate_count > 0 and state.playback_ack_count > 0 and state.playback_history_committed_count > 0
    )
    input_transcription_ok = state.input_transcription_count > 0
    native_listen_event_ok = state.model_listen_count > 0 or state.buffering_listen_count > 0
    model_listen_policy_observed = state.model_listen_count > 0
    model_speak_event_ok = state.model_speak_before_audio_ok()
    realtime_audio_lifecycle_ok = state.count("response.audio.delta") > 0 and state.count("response.audio.done") > 0
    stale_audio_delta_count = state.stale_audio_delta_count()
    result = {
        "ok": bool(state.audio_deltas)
        and state.count("response.done") > 0
        and state.count("session.closed") > 0
        and overlap_listen
        and overlap_barge_in
        and not short_ack_cancelled
        and state.listen_count > 0
        and native_listen_event_ok
        and model_listen_policy_observed
        and model_speak_event_ok
        and state.model_speak_delta_count > 0
        and event_order_ok
        and playback_commit_ok
        and input_transcription_ok
        and realtime_audio_lifecycle_ok
        and state.audio_marks_seen
        and stale_audio_delta_count == 0
        and not silence_created_response,
        "event_counts": {
            event_type: state.count(event_type)
            for event_type in sorted({str(event.get("type")) for event in state.events})
        },
        "audio_delta_count": len(state.audio_deltas),
        "done_count": state.done_count,
        "cancelled_count": state.cancelled_count,
        "listen_count": state.listen_count,
        "model_listen_count": state.model_listen_count,
        "buffering_listen_count": state.buffering_listen_count,
        "model_speak_event_count": state.model_speak_event_count,
        "model_speak_delta_count": state.model_speak_delta_count,
        "playback_ack_count": state.playback_ack_count,
        "playback_history_committed_count": state.playback_history_committed_count,
        "truncate_count": state.truncate_count,
        "input_transcription_count": state.input_transcription_count,
        "audio_marks_seen": state.audio_marks_seen,
        "overlap_decisions": state.overlap_decisions,
        "overlap_listen": overlap_listen,
        "overlap_barge_in": overlap_barge_in,
        "short_ack_cancelled": short_ack_cancelled,
        "event_order_ok": event_order_ok,
        "playback_commit_ok": playback_commit_ok,
        "input_transcription_ok": input_transcription_ok,
        "native_listen_event_ok": native_listen_event_ok,
        "model_listen_policy_observed": model_listen_policy_observed,
        "model_speak_event_ok": model_speak_event_ok,
        "realtime_audio_lifecycle_ok": realtime_audio_lifecycle_ok,
        "stale_audio_delta_count": stale_audio_delta_count,
        "post_barge_audio_delta_count": max(0, len(state.audio_deltas) - before_barge_audio),
        "post_barge_done_count": max(0, state.done_count - before_barge_done),
        "silence_created_response": silence_created_response,
        "output_dir": str(output_dir),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8099/v1/realtime?duplex=1")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--output-dir", default="/tmp/minicpmo_realtime_duplex_demo")
    parser.add_argument(
        "--output-audio-format",
        default="pcm16",
        choices=["pcm16", "wav", "g711_ulaw", "g711_alaw"],
    )
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--first-turn-ms", type=int, default=1400)
    parser.add_argument("--first-turn-transcript", default="demo input speech")
    parser.add_argument("--short-ack-ms", type=int, default=350)
    parser.add_argument("--barge-in-ms", type=int, default=1200)
    parser.add_argument("--silence-ms", type=int, default=500)
    parser.add_argument("--playback-ack-ms", type=int, default=500)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    result = asyncio.run(run_demo(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
