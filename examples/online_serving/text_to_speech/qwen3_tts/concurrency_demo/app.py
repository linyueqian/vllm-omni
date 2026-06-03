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


def _lane_elapsed(snap: MetricsSnapshot, n: int) -> float | None:
    """Effective lane wall time: frozen at completion, live while running.

    ``snap.wall_s`` is ``now - burst_start_s`` and keeps climbing forever once
    the burst is done. Once all N streams have finished, freeze at the lane's
    actual completion time = ``max(last_chunk) - burst_start`` (already
    computed as ``snap.parallel_eta_s`` by the aggregator). While the lane is
    still running, keep returning the live wall clock.
    """
    if snap.burst_start_s is None:
        return None
    is_done = snap.completed == n and not snap.any_failed
    if is_done and snap.parallel_eta_s is not None:
        return snap.parallel_eta_s
    return snap.wall_s


def _lane_snapshot_to_dict(snap: MetricsSnapshot, n: int) -> dict[str, Any]:
    return {
        "wall_s": _lane_elapsed(snap, n) or 0.0,
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
    serial: MetricsSnapshot,
    parallel: MetricsSnapshot,
    n: int,
    *,
    locked_projection: dict[str, float] | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Return (serial_elapsed, parallel_elapsed, speedup_x).

    Lane elapsed uses ``_lane_elapsed`` so both the live and the frozen
    case stay accurate. Speedup is computed once the PARALLEL lane has
    fully drained.

    When serial finishes too, the speedup is just serial_wall / parallel_wall.
    When serial is still running (the N=64 case), we project its finish time
    from its observed per-stream rate at the moment parallel completed --
    the moment is captured by the caller and passed in as ``locked_projection``
    so the headline value stays stable instead of drifting as serial warms
    up its ref-audio cache and the per-stream rate falls.
    """
    serial_done = serial.completed == n and not serial.any_failed
    parallel_done = parallel.completed == n and not parallel.any_failed

    # The badges show the lane's real clock: live while running, frozen at
    # actual completion time (max(last_chunk) - burst_start) when done.
    serial_elapsed = _lane_elapsed(serial, n)
    parallel_elapsed = _lane_elapsed(parallel, n)

    speedup: float | None = None
    if serial_done and parallel_done and parallel_elapsed and parallel_elapsed > 0 and serial_elapsed:
        speedup = serial_elapsed / parallel_elapsed
    elif parallel_done and parallel_elapsed and parallel_elapsed > 0:
        # Parallel is the headline; speedup uses a projected serial finish
        # time so it can lock in immediately at parallel-done. The badge
        # value (serial_elapsed) stays live so the user still sees serial's
        # real clock tick up while the lane grinds in the background.
        if locked_projection is not None and locked_projection["serial_completed"] > 0:
            wall = locked_projection["serial_wall"]
            comp = locked_projection["serial_completed"]
            projected_serial = (wall / comp) * n
            speedup = projected_serial / parallel_elapsed
        elif serial.completed > 0:
            per_stream_s = (serial.wall_s or 0.0) / max(1, serial.completed)
            projected_serial = per_stream_s * n
            speedup = projected_serial / parallel_elapsed
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
    # Per-page projection lock for the speedup x. Populated the first tick
    # after the parallel lane finishes, cleared on reset/start.
    speedup_locks: dict[str, dict[str, float] | None] = {"A": None, "B": None}

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
        speedup_locks[page] = None
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
        speedup_locks[page] = None
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
            # Latch the projection the first tick parallel completes so the
            # headline speedup stops drifting as serial warms up.
            parallel_done = snap_p.completed == n and not snap_p.any_failed
            if parallel_done and speedup_locks[page] is None:
                speedup_locks[page] = {
                    "serial_wall": float(snap_s.wall_s or 0.0),
                    "serial_completed": float(snap_s.completed),
                }
            serial_eta, parallel_eta, speedup = _race_speedup(
                snap_s,
                snap_p,
                n,
                locked_projection=speedup_locks[page],
            )
            payload = {
                "n": n,
                "serial": _lane_snapshot_to_dict(snap_s, n),
                "parallel": _lane_snapshot_to_dict(snap_p, n),
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
