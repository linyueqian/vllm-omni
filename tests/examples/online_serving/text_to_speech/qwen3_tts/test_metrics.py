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
