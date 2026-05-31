"""Unit tests for the concurrency-demo metrics aggregator."""

from __future__ import annotations

from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.metrics import (
    MetricsSnapshot,
    StreamEvent,
    StreamState,
)


def test_stream_event_factory_defaults() -> None:
    ev = StreamEvent.first(stream_id=3, ts=1.5)
    assert ev.stream_id == 3
    assert ev.kind == "first"
    assert ev.ts == 1.5
    assert ev.byte_count == 0


def test_stream_state_initial_values() -> None:
    s = StreamState(stream_id=7)
    assert s.stream_id == 7
    assert s.ttfb_s is None
    assert s.bytes_received == 0
    assert s.status == "pending"


def test_metrics_snapshot_is_frozen() -> None:
    snap = MetricsSnapshot(
        wall_s=0.0,
        burst_start_s=None,
        per_stream=(),
        completed=0,
        active=0,
        throughput_x=0.0,
        ttfb_p99_ms=None,
        rtf_p99=None,
        serial_eta_s=None,
        parallel_eta_s=None,
        speedup_x=None,
        any_failed=False,
    )
    # Dataclass should be frozen; mutation must raise.
    try:
        snap.completed = 99
    except Exception as e:
        assert "frozen" in str(e).lower() or "cannot assign" in str(e).lower()
    else:
        raise AssertionError("MetricsSnapshot must be frozen")


import pytest

from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.metrics import (
    MetricsAggregator,
)


def test_aggregator_first_chunk_marks_streaming_and_locks_ttfb() -> None:
    agg = MetricsAggregator(n=2)
    agg.mark_request_sent(stream_id=0, ts=10.0)
    agg.mark_request_sent(stream_id=1, ts=10.0)
    agg.apply(StreamEvent.first(stream_id=0, ts=10.4))
    snap = agg.snapshot(now=10.4)
    s0 = snap.per_stream[0]
    s1 = snap.per_stream[1]
    assert s0.status == "streaming"
    assert s0.ttfb_s == pytest.approx(0.4)
    assert s1.status == "pending"


def test_aggregator_chunks_accumulate_bytes() -> None:
    agg = MetricsAggregator(n=1)
    agg.mark_request_sent(stream_id=0, ts=0.0)
    agg.apply(StreamEvent.first(stream_id=0, ts=0.2))
    agg.apply(StreamEvent.chunk(stream_id=0, ts=0.4, byte_count=4800))
    agg.apply(StreamEvent.chunk(stream_id=0, ts=0.6, byte_count=4800))
    snap = agg.snapshot(now=0.6)
    assert snap.per_stream[0].bytes_received == 9600
    # 9600 bytes / (24000 Hz * 2 B/sample) == 0.2 seconds of audio.
    assert snap.per_stream[0].audio_seconds == pytest.approx(0.2)


def test_aggregator_done_event_marks_complete() -> None:
    agg = MetricsAggregator(n=1)
    agg.mark_request_sent(stream_id=0, ts=0.0)
    agg.apply(StreamEvent.first(stream_id=0, ts=0.2))
    agg.apply(StreamEvent.chunk(stream_id=0, ts=1.0, byte_count=48_000))  # 1 s of audio
    agg.apply(StreamEvent.done(stream_id=0, ts=1.05))
    snap = agg.snapshot(now=1.05)
    assert snap.per_stream[0].status == "done"
    assert snap.completed == 1
    assert snap.per_stream[0].final_rtf == pytest.approx(1.05 / 1.0)


def test_aggregator_error_event_marks_failed() -> None:
    agg = MetricsAggregator(n=2)
    agg.mark_request_sent(stream_id=0, ts=0.0)
    agg.apply(StreamEvent.error(stream_id=0, ts=0.3, message="boom"))
    snap = agg.snapshot(now=0.3)
    assert snap.per_stream[0].status == "error"
    assert snap.per_stream[0].error_message == "boom"
    assert snap.any_failed is True
