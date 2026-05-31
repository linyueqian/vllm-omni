"""Live smoke test: hits a real vllm server. Gated by the full_model marker."""

from __future__ import annotations

import os

import pytest

from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.metrics import (
    MetricsAggregator,
)
from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.orchestrator import (
    Orchestrator,
)
from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.prompts import (
    DEMO_PROMPT,
)
from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.runtime import (
    load_ref_audio_b64,
)

pytestmark = pytest.mark.full_model


@pytest.mark.asyncio
async def test_run_burst_n8_against_live_server() -> None:
    api_base = os.environ.get("QWEN3_TTS_API_BASE", "http://localhost:8000")
    orchestrator = Orchestrator(api_base=api_base, ref_audio_b64=load_ref_audio_b64())
    aggregator = MetricsAggregator(n=8)
    await orchestrator.run_burst(n=8, prompt=DEMO_PROMPT, aggregator=aggregator)
    snap = aggregator.snapshot(now=9999.0)
    assert snap.completed == 8, f"only {snap.completed}/8 completed: {snap.per_stream}"
    assert snap.speedup_x is not None and snap.speedup_x >= 4.0, f"speedup {snap.speedup_x} below 4x target"
