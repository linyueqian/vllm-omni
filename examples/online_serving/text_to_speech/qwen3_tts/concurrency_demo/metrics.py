"""MetricsAggregator and the immutable snapshot it publishes to the UI."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Literal

EventKind = Literal["first", "chunk", "done", "error"]
StreamStatus = Literal["pending", "streaming", "done", "error"]

# Cap on the per-stream (min, max) envelope buffer. The orchestrator emits a
# fixed number of pairs per chunk; once the buffer exceeds this cap we halve
# it via pair-wise extreme reduction so the buffer always covers the FULL
# utterance, just at progressively coarser resolution as audio grows.
MAX_WAVEFORM_SAMPLES = 512


def _percentile(sorted_values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile, p in [0, 1]."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


@dataclass(frozen=True)
class StreamEvent:
    """Immutable event emitted by the orchestrator for a single TTS stream."""

    stream_id: int
    kind: EventKind
    ts: float
    byte_count: int = 0
    error_message: str = ""
    samples: tuple[tuple[float, float], ...] = ()

    @staticmethod
    def first(stream_id: int, ts: float) -> StreamEvent:
        return StreamEvent(stream_id=stream_id, kind="first", ts=ts)

    @staticmethod
    def chunk(
        stream_id: int,
        ts: float,
        byte_count: int,
        samples: tuple[tuple[float, float], ...] = (),
    ) -> StreamEvent:
        return StreamEvent(stream_id=stream_id, kind="chunk", ts=ts, byte_count=byte_count, samples=samples)

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
    waveform_samples: tuple[tuple[float, float], ...] = ()

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
        # Monotonically increases on every state-mutating call so UI timers can
        # cheaply detect whether anything happened since the previous tick.
        self._seq: int = 0

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    def reset(self) -> None:
        with self._lock:
            self._states = [StreamState(stream_id=i) for i in range(self._n)]
            self._burst_start_s = None
            self._ref_t_observed_s = None
            self._seq += 1

    def mark_burst_start(self, ts: float) -> None:
        with self._lock:
            self._burst_start_s = ts
            self._seq += 1

    def set_reference(self, t_observed_s: float) -> None:
        with self._lock:
            self._ref_t_observed_s = t_observed_s
            self._seq += 1

    def mark_request_sent(self, stream_id: int, ts: float) -> None:
        with self._lock:
            s = self._states[stream_id]
            s.request_sent_s = ts
            self._seq += 1

    def apply(self, ev: StreamEvent) -> None:
        with self._lock:
            self._seq += 1
            s = self._states[ev.stream_id]
            if ev.kind == "first":
                s.first_chunk_s = ev.ts
                s.last_chunk_s = ev.ts
                if s.status == "pending":
                    s.status = "streaming"
            elif ev.kind == "chunk":
                s.bytes_received += ev.byte_count
                s.last_chunk_s = ev.ts
                if ev.samples:
                    combined: tuple[tuple[float, float], ...] = s.waveform_samples + ev.samples
                    # Halve via pair-wise extreme reduction while we're over
                    # cap. Preserves the full utterance shape; the envelope
                    # just gets coarser as more audio accumulates.
                    while len(combined) > MAX_WAVEFORM_SAMPLES:
                        reduced: list[tuple[float, float]] = []
                        for i in range(0, len(combined) - 1, 2):
                            lo_a, hi_a = combined[i]
                            lo_b, hi_b = combined[i + 1]
                            reduced.append((min(lo_a, lo_b), max(hi_a, hi_b)))
                        if len(combined) % 2 == 1:
                            reduced.append(combined[-1])
                        combined = tuple(reduced)
                    s.waveform_samples = combined
            elif ev.kind == "done":
                s.last_chunk_s = ev.ts
                s.status = "done"
            elif ev.kind == "error":
                s.status = "error"
                s.error_message = ev.error_message
                s.last_chunk_s = ev.ts

    def snapshot(self, now: float) -> MetricsSnapshot:
        """Return an immutable snapshot of current state + derived metrics."""
        with self._lock:
            per_stream = tuple(
                StreamState(
                    stream_id=s.stream_id,
                    status=s.status,
                    request_sent_s=s.request_sent_s,
                    first_chunk_s=s.first_chunk_s,
                    last_chunk_s=s.last_chunk_s,
                    bytes_received=s.bytes_received,
                    error_message=s.error_message,
                    waveform_samples=s.waveform_samples,
                )
                for s in self._states
            )
            burst_start = self._burst_start_s
            ref_t = self._ref_t_observed_s
            n = self._n

        completed = sum(1 for s in per_stream if s.status == "done")
        active = sum(1 for s in per_stream if s.status == "streaming")
        any_failed = any(s.status == "error" for s in per_stream)

        total_audio_s = sum(s.audio_seconds for s in per_stream)
        elapsed = (now - burst_start) if burst_start is not None else 0.0
        throughput_x = (total_audio_s / elapsed) if elapsed > 0 else 0.0

        ttfb_values = sorted(s.ttfb_s * 1000.0 for s in per_stream if s.ttfb_s is not None)
        rtf_values = sorted(s.final_rtf for s in per_stream if s.final_rtf is not None)
        ttfb_p99_ms = _percentile(ttfb_values, 0.99)
        rtf_p99 = _percentile(rtf_values, 0.99)

        serial_eta_s = (n * ref_t) if ref_t is not None else None

        if completed == n and burst_start is not None and not any_failed:
            last = max((s.last_chunk_s or burst_start) for s in per_stream)
            parallel_eta_s = last - burst_start
        else:
            parallel_eta_s = None

        if serial_eta_s is not None and parallel_eta_s is not None and parallel_eta_s > 0 and not any_failed:
            speedup_x = serial_eta_s / parallel_eta_s
        else:
            speedup_x = None

        return MetricsSnapshot(
            wall_s=elapsed,
            burst_start_s=burst_start,
            per_stream=per_stream,
            completed=completed,
            active=active,
            throughput_x=throughput_x,
            ttfb_p99_ms=ttfb_p99_ms,
            rtf_p99=rtf_p99,
            serial_eta_s=serial_eta_s,
            parallel_eta_s=parallel_eta_s,
            speedup_x=speedup_x,
            any_failed=any_failed,
        )
