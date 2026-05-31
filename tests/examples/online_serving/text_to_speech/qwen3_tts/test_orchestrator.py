"""Orchestrator tests using an in-process httpx.MockTransport — no real server."""

from __future__ import annotations

import httpx
import pytest

from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.metrics import (
    MetricsAggregator,
)
from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.orchestrator import (
    Orchestrator,
    StreamConfig,
)


def _mock_transport_factory(chunks: list[bytes]) -> httpx.MockTransport:
    """Return a MockTransport that yields the supplied raw byte chunks."""

    def handler(request: httpx.Request) -> httpx.Response:
        async def gen():
            for c in chunks:
                yield c

        return httpx.Response(200, content=gen(), headers={"content-type": "audio/pcm"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_run_one_stream_emits_first_chunks_done_events() -> None:
    aggregator = MetricsAggregator(n=1)
    aggregator.mark_burst_start(0.0)
    transport = _mock_transport_factory([b"\x00" * 4800, b"\x00" * 4800])
    orchestrator = Orchestrator(
        api_base="http://stub",
        ref_audio_b64="UExBQ0VIT0xERVI=",
        transport=transport,
    )
    cfg = StreamConfig(stream_id=0, text="hello", payload_kind="parallel")
    await orchestrator._run_one(cfg, aggregator)
    snap = aggregator.snapshot(now=1.0)
    s0 = snap.per_stream[0]
    assert s0.status == "done"
    assert s0.bytes_received == 9600  # 2 chunks * 4800 bytes


@pytest.mark.asyncio
async def test_run_burst_records_reference_and_runs_n_parallel() -> None:
    aggregator = MetricsAggregator(n=4)
    transport = _mock_transport_factory([b"\x00" * 9600])  # ~0.2 s of audio per request
    orchestrator = Orchestrator(
        api_base="http://stub",
        ref_audio_b64="UExBQ0VIT0xERVI=",
        transport=transport,
    )
    await orchestrator.run_burst(
        n=4,
        prompt="hello world",
        aggregator=aggregator,
    )
    snap = aggregator.snapshot(now=10.0)
    # All 4 streams must be done.
    assert snap.completed == 4
    # serial_eta must be populated (reference ran).
    assert snap.serial_eta_s is not None
    assert snap.serial_eta_s > 0
    # No failures.
    assert snap.any_failed is False
