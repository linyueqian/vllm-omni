"""MetricsAggregator and the immutable snapshot it publishes to the UI."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal

EventKind = Literal["first", "chunk", "done", "error"]
StreamStatus = Literal["pending", "streaming", "done", "error"]


@dataclass(frozen=True)
class StreamEvent:
    """Immutable event emitted by the orchestrator for a single TTS stream."""

    stream_id: int
    kind: EventKind
    ts: float
    byte_count: int = 0
    error_message: str = ""

    @staticmethod
    def first(stream_id: int, ts: float) -> StreamEvent:
        return StreamEvent(stream_id=stream_id, kind="first", ts=ts)

    @staticmethod
    def chunk(stream_id: int, ts: float, byte_count: int) -> StreamEvent:
        return StreamEvent(stream_id=stream_id, kind="chunk", ts=ts, byte_count=byte_count)

    @staticmethod
    def done(stream_id: int, ts: float) -> StreamEvent:
        return StreamEvent(stream_id=stream_id, kind="done", ts=ts)

    @staticmethod
    def error(stream_id: int, ts: float, message: str) -> StreamEvent:
        return StreamEvent(stream_id=stream_id, kind="error", ts=ts, error_message=message)


@dataclass
class StreamState:
    """Mutable per-stream bookkeeping owned by ``MetricsAggregator``."""

    stream_id: int
    status: StreamStatus = "pending"
    request_sent_s: float | None = None
    first_chunk_s: float | None = None
    last_chunk_s: float | None = None
    bytes_received: int = 0
    error_message: str = ""

    @property
    def ttfb_s(self) -> float | None:
        if self.request_sent_s is None or self.first_chunk_s is None:
            return None
        return self.first_chunk_s - self.request_sent_s

    @property
    def audio_seconds(self) -> float:
        # 24 kHz mono int16 = 2 bytes/sample.
        return self.bytes_received / (24_000 * 2)

    @property
    def final_rtf(self) -> float | None:
        if self.status != "done" or self.audio_seconds <= 0:
            return None
        if self.request_sent_s is None or self.last_chunk_s is None:
            return None
        wall = self.last_chunk_s - self.request_sent_s
        return wall / self.audio_seconds


@dataclass(frozen=True)
class MetricsSnapshot:
    """Immutable view of aggregator state published to the UI thread."""

    wall_s: float
    burst_start_s: float | None
    per_stream: tuple[StreamState, ...] = field(default_factory=tuple)
    completed: int = 0
    active: int = 0
    throughput_x: float = 0.0
    ttfb_p99_ms: float | None = None
    rtf_p99: float | None = None
    serial_eta_s: float | None = None
    parallel_eta_s: float | None = None
    speedup_x: float | None = None
    any_failed: bool = False


class MetricsAggregator:
    """Thread-safe aggregator. UI reads immutable snapshots; orchestrator emits events."""

    def __init__(self, n: int) -> None:
        self._lock = threading.Lock()
        self._n = n
        self._states: list[StreamState] = [StreamState(stream_id=i) for i in range(n)]
        self._burst_start_s: float | None = None
        self._ref_t_observed_s: float | None = None

    def reset(self) -> None:
        with self._lock:
            self._states = [StreamState(stream_id=i) for i in range(self._n)]
            self._burst_start_s = None
            self._ref_t_observed_s = None

    def mark_burst_start(self, ts: float) -> None:
        with self._lock:
            self._burst_start_s = ts

    def set_reference(self, t_observed_s: float) -> None:
        with self._lock:
            self._ref_t_observed_s = t_observed_s

    def mark_request_sent(self, stream_id: int, ts: float) -> None:
        with self._lock:
            s = self._states[stream_id]
            s.request_sent_s = ts

    def apply(self, ev: StreamEvent) -> None:
        with self._lock:
            s = self._states[ev.stream_id]
            if ev.kind == "first":
                s.first_chunk_s = ev.ts
                s.last_chunk_s = ev.ts
                if s.status == "pending":
                    s.status = "streaming"
            elif ev.kind == "chunk":
                s.bytes_received += ev.byte_count
                s.last_chunk_s = ev.ts
            elif ev.kind == "done":
                s.last_chunk_s = ev.ts
                s.status = "done"
            elif ev.kind == "error":
                s.status = "error"
                s.error_message = ev.error_message
                s.last_chunk_s = ev.ts
