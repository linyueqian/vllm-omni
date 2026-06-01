"""FastAPI app for the Qwen3-TTS concurrency demo.

Replaces the earlier Gradio prototype: serves one static index.html with
inline JS+canvas, exposes ``/api/start/{page}`` and ``/api/reset/{page}``
POST endpoints, and pushes seq-deduped snapshots over an SSE stream at
``/api/stream/{page}``. The browser drives all rendering, so the page
sits visually still between bursts.

Launch::

    python -m examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.app \\
        --api-base http://localhost:8000 --port 7860
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from .metrics import MetricsAggregator, MetricsSnapshot
from .orchestrator import Orchestrator
from .prompts import DEMO_PROMPT
from .runtime import REF_TEXT, WorkerLoop, load_ref_audio_b64

logger = logging.getLogger(__name__)

N_PAGE_A = 8
N_PAGE_B = 64

STATIC_DIR = Path(__file__).parent / "static"


def _snapshot_to_dict(snap: MetricsSnapshot) -> dict[str, Any]:
    return {
        "wall_s": snap.wall_s,
        "completed": snap.completed,
        "active": snap.active,
        "throughput_x": snap.throughput_x,
        "ttfb_p99_ms": snap.ttfb_p99_ms,
        "rtf_p99": snap.rtf_p99,
        "serial_eta_s": snap.serial_eta_s,
        "parallel_eta_s": snap.parallel_eta_s,
        "speedup_x": snap.speedup_x,
        "any_failed": snap.any_failed,
        "streams": [
            {
                "id": s.stream_id,
                "status": s.status,
                "ttfb_s": s.ttfb_s,
                "audio_seconds": s.audio_seconds,
                "final_rtf": s.final_rtf,
                "samples": list(s.waveform_samples),
            }
            for s in snap.per_stream
        ],
    }


def _log_burst_exception(future: Future) -> None:
    exc = future.exception()
    if exc is not None:
        logger.exception("Concurrency-demo burst failed", exc_info=exc)


def build_app(api_base: str) -> FastAPI:
    app = FastAPI(title="Qwen3-TTS Concurrency Demo")
    worker = WorkerLoop()
    ref_b64 = load_ref_audio_b64()
    orchestrator = Orchestrator(api_base=api_base, ref_audio_b64=ref_b64, ref_text=REF_TEXT)
    agg_a = MetricsAggregator(n=N_PAGE_A)
    agg_b = MetricsAggregator(n=N_PAGE_B)
    aggs = {"A": agg_a, "B": agg_b}
    sizes = {"A": N_PAGE_A, "B": N_PAGE_B}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {"api_base": api_base, "n_a": N_PAGE_A, "n_b": N_PAGE_B}

    @app.post("/api/start/{page}")
    def start(page: str) -> dict[str, Any]:
        if page not in aggs:
            return {"error": f"unknown page {page!r}"}
        future = worker.submit(orchestrator.run_burst(n=sizes[page], prompt=DEMO_PROMPT, aggregator=aggs[page]))
        future.add_done_callback(_log_burst_exception)
        return {"started": sizes[page], "page": page}

    @app.post("/api/reset/{page}")
    def reset(page: str) -> dict[str, Any]:
        if page not in aggs:
            return {"error": f"unknown page {page!r}"}
        aggs[page].reset()
        return {"ok": True, "page": page}

    @app.get("/api/stream/{page}")
    async def stream(page: str) -> StreamingResponse:
        if page not in aggs:
            return StreamingResponse(iter(()), media_type="text/event-stream")
        agg = aggs[page]

        async def gen():
            # Send the current state immediately so the client paints on connect.
            snap = agg.snapshot(now=time.perf_counter())
            yield f"data: {json.dumps(_snapshot_to_dict(snap))}\n\n"
            last_seq = agg.seq
            keepalive_at = time.monotonic() + 15.0
            while True:
                current = agg.seq
                if current != last_seq:
                    last_seq = current
                    snap = agg.snapshot(now=time.perf_counter())
                    yield f"data: {json.dumps(_snapshot_to_dict(snap))}\n\n"
                    keepalive_at = time.monotonic() + 15.0
                elif time.monotonic() >= keepalive_at:
                    # SSE keepalive comment so intermediaries don't drop the
                    # idle connection. Browsers ignore comment-only events.
                    yield ": keepalive\n\n"
                    keepalive_at = time.monotonic() + 15.0
                await asyncio.sleep(0.05)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", default="http://localhost:8000")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    app = build_app(api_base=args.api_base)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
