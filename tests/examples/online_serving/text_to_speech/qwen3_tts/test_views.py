"""Render-pure HTML view tests — string assertions only, no DOM."""

from __future__ import annotations

from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.metrics import (
    MetricsAggregator,
    MetricsSnapshot,
    StreamEvent,
)
from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.views import (
    render_grid_html,
    render_row_html,
)


def _snapshot_with_one_streaming_one_done() -> MetricsSnapshot:
    agg = MetricsAggregator(n=2)
    agg.mark_burst_start(0.0)
    agg.set_reference(1.0)
    # Stream 0: streaming
    agg.mark_request_sent(stream_id=0, ts=0.0)
    agg.apply(StreamEvent.first(stream_id=0, ts=0.2))
    agg.apply(StreamEvent.chunk(stream_id=0, ts=0.5, byte_count=24_000))
    # Stream 1: done
    agg.mark_request_sent(stream_id=1, ts=0.0)
    agg.apply(StreamEvent.first(stream_id=1, ts=0.2))
    agg.apply(StreamEvent.chunk(stream_id=1, ts=0.4, byte_count=48_000))
    agg.apply(StreamEvent.done(stream_id=1, ts=0.4))
    return agg.snapshot(now=0.5)


def test_render_row_streaming_includes_ttfb_and_progress() -> None:
    snap = _snapshot_with_one_streaming_one_done()
    html = render_row_html(snap, stream_id=0)
    assert "#1" in html
    assert "TTFB" in html
    assert "streaming" in html.lower() or "in-flight" in html.lower()


def test_render_row_done_marks_completion() -> None:
    snap = _snapshot_with_one_streaming_one_done()
    html = render_row_html(snap, stream_id=1)
    assert "#2" in html
    assert "done" in html.lower() or "100" in html


def test_render_grid_returns_64_cells_for_n_64() -> None:
    agg = MetricsAggregator(n=64)
    snap = agg.snapshot(now=0.0)
    html = render_grid_html(snap)
    # 64 dots in the grid.
    assert html.count("data-cell") == 64
