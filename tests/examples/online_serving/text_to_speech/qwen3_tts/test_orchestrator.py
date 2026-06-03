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
from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.runtime import (
    load_ref_audio_b64,
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
    cfg = StreamConfig(stream_id=0, text="hello")
    await orchestrator._run_one(cfg, aggregator)
    snap = aggregator.snapshot(now=1.0)
    s0 = snap.per_stream[0]
    assert s0.status == "done"
    assert s0.bytes_received == 9600  # 2 chunks * 4800 bytes


@pytest.mark.asyncio
async def test_run_one_handles_odd_byte_chunk_boundaries() -> None:
    """Regression: under high concurrency the server may emit chunks at odd
    byte boundaries (last byte of a sample lands in the next chunk). Without
    a leftover buffer, every odd-sized chunk silently drops its sample data
    while still advancing bytes_received -- leaving streams with a flat-zero
    waveform on top of N seconds of "received" audio. See debug-notes header
    on Orchestrator._run_one for the diagnosis.
    """
    # Build a chunk sequence whose only EVEN-aligned chunk is silent, and
    # whose subsequent chunks (each odd-sized) carry the real loud audio.
    # If the leftover buffer is missing, every loud chunk emits no samples
    # and the only surviving samples are the silent prefix's (min,max)==(0,0).
    silent_first = b"\x00" * 64  # even, all zeros
    # Loud body split into ALL-odd-sized chunks. The peak (~0.3) only shows
    # up if the leftover-byte realignment puts even-sample boundaries back
    # in sync between chunks; otherwise every odd chunk's _compute_samples
    # short-circuits with () and the body never enters the waveform.
    loud_value = 10000  # int16 = ~0.305 normalized
    loud_pcm = b"".join(
        int.to_bytes(loud_value, length=2, byteorder="little", signed=True) for _ in range(2046)
    )
    assert len(loud_pcm) == 4092  # divides cleanly into 4 odd-sized chunks (1023 each)
    chunks = [
        silent_first,
        loud_pcm[:1023],
        loud_pcm[1023:2046],
        loud_pcm[2046:3069],
        loud_pcm[3069:4092],
    ]
    # Sanity: ensure all non-silent chunks are odd.
    for c in chunks[1:]:
        assert len(c) % 2 == 1

    aggregator = MetricsAggregator(n=1)
    aggregator.mark_burst_start(0.0)
    transport = _mock_transport_factory(chunks)
    orchestrator = Orchestrator(
        api_base="http://stub",
        ref_audio_b64="UExBQ0VIT0xERVI=",
        transport=transport,
    )
    cfg = StreamConfig(stream_id=0, text="hello")
    await orchestrator._run_one(cfg, aggregator)
    snap = aggregator.snapshot(now=1.0)
    s0 = snap.per_stream[0]
    assert s0.status == "done"
    assert s0.bytes_received == sum(len(c) for c in chunks)
    # The loud body MUST show up in the envelope; without the leftover
    # buffer this would be flat zeros.
    peak = max(max(abs(lo), abs(hi)) for lo, hi in s0.waveform_samples)
    assert peak > 0.05, f"expected loud envelope, got peak={peak} samples={s0.waveform_samples[:8]}"


@pytest.mark.asyncio
async def test_run_burst_parallel_completes_all_streams() -> None:
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
        concurrency=None,  # full parallel
    )
    snap = aggregator.snapshot(now=10.0)
    assert snap.completed == 4
    assert snap.burst_start_s is not None
    assert snap.any_failed is False


@pytest.mark.asyncio
async def test_run_burst_serial_concurrency_one_completes_all_streams() -> None:
    aggregator = MetricsAggregator(n=4)
    transport = _mock_transport_factory([b"\x00" * 9600])
    orchestrator = Orchestrator(
        api_base="http://stub",
        ref_audio_b64="UExBQ0VIT0xERVI=",
        transport=transport,
    )
    await orchestrator.run_burst(
        n=4,
        prompt="hello world",
        aggregator=aggregator,
        concurrency=1,
    )
    snap = aggregator.snapshot(now=10.0)
    assert snap.completed == 4
    assert snap.any_failed is False


def test_load_ref_audio_b64_returns_nonempty_string() -> None:
    b64 = load_ref_audio_b64()
    assert isinstance(b64, str)
    assert len(b64) > 100  # arbitrary; clone_2.wav is ~757 KB
