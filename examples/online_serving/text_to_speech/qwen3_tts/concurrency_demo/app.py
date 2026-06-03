"""FastAPI app for the Qwen3-TTS concurrency race.

Runs two side-by-side lanes against two independent vLLM-Omni servers:
- SERIAL lane: ``concurrency=1`` against ``--api-base-serial``
- PARALLEL lane: full ``asyncio.gather`` against ``--api-base-parallel``

A single SSE endpoint ``/api/race/{page}`` pushes one combined snapshot per
seq-bump so the browser can render both lanes (and the headline Speedup x
banner) from a single connection. The page sits visually still between
bursts because the SSE only fires when an aggregator actually changes.

Launch::

    python -m examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.app \\
        --api-base-serial   http://localhost:8000 \\
        --api-base-parallel http://localhost:8001 \\
        --port 7860
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


def _lane_snapshot_to_dict(snap: MetricsSnapshot) -> dict[str, Any]:
    return {
        "wall_s": snap.wall_s,
        "completed": snap.completed,
        "active": snap.active,
        "throughput_x": snap.throughput_x,
        "ttfb_p99_ms": snap.ttfb_p99_ms,
        "rtf_p99": snap.rtf_p99,
        "any_failed": snap.any_failed,
        "streams": [
            {
                "id": s.stream_id,
                "status": s.status,
                "ttfb_s": s.ttfb_s,
                "audio_seconds": s.audio_seconds,
                "final_rtf": s.final_rtf,
                "samples": [[lo, hi] for (lo, hi) in s.waveform_samples],
            }
            for s in snap.per_stream
        ],
    }


def _race_speedup(
    serial: MetricsSnapshot, parallel: MetricsSnapshot, n: int
) -> tuple[float | None, float | None, float | None]:
    """Return (serial_elapsed, parallel_elapsed, speedup_x).

    Each lane's elapsed clock is its observed wall time so far. Speedup is
    computed once the PARALLEL lane has fully drained (all N done). The
    SERIAL lane may still be running — its bar keeps growing on the page.
    """
    serial_done = serial.completed == n and not serial.any_failed
    parallel_done = parallel.completed == n and not parallel.any_failed

    serial_elapsed: float | None = serial.wall_s if serial.burst_start_s is not None else None
    parallel_elapsed: float | None = parallel.wall_s if parallel.burst_start_s is not None else None

    speedup: float | None = None
    if serial_done and parallel_done and parallel_elapsed and parallel_elapsed > 0:
        speedup = serial_elapsed / parallel_elapsed if serial_elapsed else None
    elif parallel_done and parallel_elapsed and parallel_elapsed > 0 and serial.completed > 0:
        # Parallel finished, serial still running — project from observed rate.
        per_stream_s = (serial.wall_s or 0.0) / max(1, serial.completed)
        projected_serial = per_stream_s * n
        speedup = projected_serial / parallel_elapsed
        serial_elapsed = projected_serial
    return serial_elapsed, parallel_elapsed, speedup


def _log_burst_exception(future: Future) -> None:
    exc = future.exception()
    if exc is not None:
        logger.exception("Concurrency-demo burst failed", exc_info=exc)


def build_app(api_base_serial: str, api_base_parallel: str) -> FastAPI:
    app = FastAPI(title="Qwen3-TTS Concurrency Race")
    worker = WorkerLoop()
    ref_b64 = load_ref_audio_b64()
    orch_serial = Orchestrator(api_base=api_base_serial, ref_audio_b64=ref_b64, ref_text=REF_TEXT)
    orch_parallel = Orchestrator(api_base=api_base_parallel, ref_audio_b64=ref_b64, ref_text=REF_TEXT)

    # Per page, one aggregator per lane.
    aggs: dict[str, dict[str, MetricsAggregator]] = {
        "A": {"serial": MetricsAggregator(n=N_PAGE_A), "parallel": MetricsAggregator(n=N_PAGE_A)},
        "B": {"serial": MetricsAggregator(n=N_PAGE_B), "parallel": MetricsAggregator(n=N_PAGE_B)},
    }
    sizes = {"A": N_PAGE_A, "B": N_PAGE_B}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {
            "api_base_serial": api_base_serial,
            "api_base_parallel": api_base_parallel,
            "n_a": N_PAGE_A,
            "n_b": N_PAGE_B,
        }

    @app.post("/api/start/{page}")
    def start(page: str) -> dict[str, Any]:
        if page not in aggs:
            return {"error": f"unknown page {page!r}"}
        n = sizes[page]
        f_serial = worker.submit(
            orch_serial.run_burst(n=n, prompt=DEMO_PROMPT, aggregator=aggs[page]["serial"], concurrency=1)
        )
        f_parallel = worker.submit(
            orch_parallel.run_burst(n=n, prompt=DEMO_PROMPT, aggregator=aggs[page]["parallel"], concurrency=None)
        )
        f_serial.add_done_callback(_log_burst_exception)
        f_parallel.add_done_callback(_log_burst_exception)
        return {"started": n, "page": page}

    @app.post("/api/reset/{page}")
    def reset(page: str) -> dict[str, Any]:
        if page not in aggs:
            return {"error": f"unknown page {page!r}"}
        aggs[page]["serial"].reset()
        aggs[page]["parallel"].reset()
        return {"ok": True, "page": page}

    @app.get("/api/race/{page}")
    async def race(page: str) -> StreamingResponse:
        if page not in aggs:
            return StreamingResponse(iter(()), media_type="text/event-stream")
        agg_s = aggs[page]["serial"]
        agg_p = aggs[page]["parallel"]
        n = sizes[page]

        def _build_payload() -> str:
            now = time.perf_counter()
            snap_s = agg_s.snapshot(now=now)
            snap_p = agg_p.snapshot(now=now)
            serial_eta, parallel_eta, speedup = _race_speedup(snap_s, snap_p, n)
            payload = {
                "n": n,
                "serial": _lane_snapshot_to_dict(snap_s),
                "parallel": _lane_snapshot_to_dict(snap_p),
                "serial_eta_s": serial_eta,
                "parallel_eta_s": parallel_eta,
                "speedup_x": speedup,
                "milestones": [m for m in (8, 16, 32, 64) if m <= n and snap_p.completed >= m],
            }
            return f"data: {json.dumps(payload)}\n\n"

        async def gen():
            yield _build_payload()
            last = (agg_s.seq, agg_p.seq)
            keepalive_at = time.monotonic() + 15.0
            while True:
                cur = (agg_s.seq, agg_p.seq)
                if cur != last:
                    last = cur
                    yield _build_payload()
                    keepalive_at = time.monotonic() + 15.0
                elif time.monotonic() >= keepalive_at:
                    yield ": keepalive\n\n"
                    keepalive_at = time.monotonic() + 15.0
                await asyncio.sleep(0.05)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--api-base-serial",
        default="http://localhost:8000",
        help="vLLM-Omni server URL for the serial (concurrency=1) lane",
    )
    p.add_argument(
        "--api-base-parallel",
        default="http://localhost:8001",
        help="vLLM-Omni server URL for the parallel (concurrency=N) lane",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    app = build_app(
        api_base_serial=args.api_base_serial,
        api_base_parallel=args.api_base_parallel,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
