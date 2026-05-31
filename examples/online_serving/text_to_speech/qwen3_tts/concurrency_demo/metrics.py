"""MetricsAggregator and the immutable snapshot it publishes to the UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EventKind = Literal["first", "chunk", "done", "error"]
StreamStatus = Literal["pending", "streaming", "done", "error"]


@dataclass(frozen=True)
class StreamEvent:
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
        if self.status != "done" or self.audio_seconds <= 0 or self.request_sent_s is None:
            return None
        wall = (self.last_chunk_s or 0.0) - self.request_sent_s
        return wall / self.audio_seconds


@dataclass(frozen=True)
class MetricsSnapshot:
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
