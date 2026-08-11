"""Async benchmark client for Legato: SSE streams with PCM-exact accounting.

Two load modes:
  * Closed-loop (default): --concurrency workers, each running --n sequential
    requests back to back (sustained load; completions gate arrivals).
  * Open-loop: --arrival-rate R (req/s, Poisson interarrivals from --seed) for
    --duration-s seconds. Arrivals are INDEPENDENT of completions: each request
    task is spawned at its arrival time regardless of how many are in flight
    (fire-and-track). Required for controlled offered-load evaluation, since
    closed-loop clients silently lower offered load when latency grows.

Reuses the exact-accounting summarize/simulate from legato_bench.py (same dir).
Adds an in-process /metrics sampler task so server-side stage state can be
cross-referenced with client-side timelines (absolute epoch timestamps).

Usage:
  python legato_bench_async.py --concurrency 48 --text "..." \
      --out run.ndjson --metrics-out run_metrics.ndjson
  python legato_bench_async.py --arrival-rate 2.4 --duration-s 180 --seed 0 \
      --text "..." --out run.ndjson --metrics-out run_metrics.ndjson
"""

import argparse
import asyncio
import base64
import json
import logging
import random
import re
import time
from typing import Any

import aiohttp

from legato_bench import _extract_audio_b64, summarize

logger = logging.getLogger(__name__)

METRIC_LINE = re.compile(
    r"^([a-zA-Z_:][\w:]*\{[^}]*\}|[a-zA-Z_:][\w:]*)\s+([0-9.eE+-]+)$"
)
METRIC_KEEP = re.compile(r"running|waiting|usage|rtf|stage|queue|preempt|tokens")


async def stream_once(
    session: aiohttp.ClientSession, url: str, text: str, voice: str
) -> dict[str, Any]:
    payload = {
        "input": text,
        "task_type": "VoiceDesign",
        "instructions": voice,
        "response_format": "pcm",
        "stream": True,
    }
    t0_epoch = time.time()
    t0 = time.perf_counter()
    emissions: list[dict[str, float]] = []
    async with session.post(f"{url.rstrip('/')}/v1/audio/speech", json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            body = line[5:].strip()
            if body == b"[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            b64 = _extract_audio_b64(obj)
            if b64 is None:
                continue
            nbytes = len(base64.b64decode(b64))
            emissions.append({"t": time.perf_counter() - t0, "bytes": float(nbytes)})
    return {"status": "ok", "t0_epoch": t0_epoch, "emissions": emissions}


async def metrics_sampler(
    session: aiohttp.ClientSession,
    url: str,
    out_path: str,
    stop: asyncio.Event,
    interval: float,
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        while not stop.is_set():
            try:
                async with session.get(f"{url.rstrip('/')}/metrics") as resp:
                    body = await resp.text()
                sample: dict[str, float] = {}
                for line in body.splitlines():
                    if line.startswith("#") or not METRIC_KEEP.search(line):
                        continue
                    m = METRIC_LINE.match(line)
                    if m:
                        sample[m.group(1)] = float(m.group(2))
                f.write(json.dumps({"ts": time.time(), "m": sample}) + "\n")
                f.flush()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning("metrics sample failed: %s", exc)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass


def base_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Shared summary over ok rows: latency medians and stall fractions."""
    ok = [r for r in rows if r.get("status") == "ok"]
    agg: dict[str, Any] = {"n_ok": len(ok), "n_err": len(rows) - len(ok)}
    if not ok:
        return agg
    agg["ttfa_med"] = round(sorted(r["ttfa_s"] for r in ok)[len(ok) // 2], 4)
    agg["rtf_med"] = round(sorted(r["rtf"] for r in ok)[len(ok) // 2], 4)
    for b in (100, 300, 500):
        key = f"stall_b{b}ms"
        stalled = [r for r in ok if r[key]["stalls"] > 0]
        agg[f"stalled_frac_b{b}"] = round(len(stalled) / len(ok), 3)
        agg[f"stall_time_total_b{b}"] = round(sum(r[key]["stall_time"] for r in ok), 3)
    return agg


def concurrency_stats(
    rows: list[dict[str, Any]], duration_s: float | None
) -> dict[str, Any]:
    """Realized concurrency from per-request [send, last-emission] spans."""
    ok = [r for r in rows if r.get("status") == "ok" and "t0_epoch" in r]
    if not ok:
        return {}
    spans = [(r["t0_epoch"], r["t0_epoch"] + r["wall_s"]) for r in ok]
    t_first = min(s for s, _ in spans)
    t_last = max(e for _, e in spans)
    makespan = max(t_last - t_first, 1e-9)
    busy = sum(e - s for s, e in spans)
    out = {
        "makespan_s": round(makespan, 2),
        "realized_mean_concurrency": round(busy / makespan, 2),
    }
    if duration_s:
        # Concurrency averaged over the arrival window only (drain excluded).
        w_end = t_first + duration_s
        busy_w = sum(max(0.0, min(e, w_end) - min(s, w_end)) for s, e in spans)
        out["mean_concurrency_window"] = round(busy_w / duration_s, 2)
    return out


async def run_closed_loop(
    session: aiohttp.ClientSession, args: argparse.Namespace
) -> list[dict[str, Any]]:
    async def one(i: int) -> list[dict[str, Any]]:
        if args.stagger:
            await asyncio.sleep(i * args.stagger)
        worker_rows = []
        for j in range(args.n):
            res = await stream_once(session, args.url, args.text, args.voice)
            row = summarize(res)
            row["stream"] = i
            row["req"] = j
            row["t0_epoch"] = res["t0_epoch"]
            worker_rows.append(row)
        return worker_rows

    nested = await asyncio.gather(*(one(i) for i in range(args.concurrency)))
    return [r for wr in nested for r in wr]


async def run_open_loop(
    session: aiohttp.ClientSession, args: argparse.Namespace
) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    arrivals: list[float] = []
    t = rng.expovariate(args.arrival_rate)
    while t <= args.duration_s:
        arrivals.append(t)
        t += rng.expovariate(args.arrival_rate)
    logger.info(
        "open-loop: %d arrivals over %.1fs (nominal %.3f req/s, seed %d)",
        len(arrivals), args.duration_s, args.arrival_rate, args.seed,
    )

    async def one(i: int, sched_at: float) -> dict[str, Any]:
        t0_epoch = time.time()
        try:
            res = await stream_once(session, args.url, args.text, args.voice)
            row = summarize(res)
            row["t0_epoch"] = res["t0_epoch"]
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            row = {"status": "error", "error": repr(exc), "t0_epoch": t0_epoch}
        row["req"] = i
        row["sched_arrival_s"] = round(sched_at, 4)
        return row

    t_start = time.perf_counter()
    tasks: list[asyncio.Task] = []
    for i, at in enumerate(arrivals):
        delay = at - (time.perf_counter() - t_start)
        if delay > 0:
            await asyncio.sleep(delay)
        # Fire-and-track: spawn regardless of how many are in flight.
        tasks.append(asyncio.create_task(one(i, at)))
    logger.info("open-loop: arrivals done, draining %d tasks", len(tasks))
    return list(await asyncio.gather(*tasks))


async def run(args: argparse.Namespace) -> None:
    open_loop = args.arrival_rate is not None
    timeout = aiohttp.ClientTimeout(total=args.timeout, sock_read=args.timeout)
    # Open loop must never queue connections client-side (that would re-couple
    # arrivals to completions); closed loop keeps the bounded connector.
    limit = 0 if open_loop else args.concurrency + 8
    conn = aiohttp.TCPConnector(limit=limit)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        warm = await stream_once(session, args.url, "Warm up sentence.", args.voice)
        logger.info("warmup: %d emissions", len(warm["emissions"]))

        stop = asyncio.Event()
        sampler = None
        if args.metrics_out:
            sampler = asyncio.create_task(
                metrics_sampler(
                    session, args.url, args.metrics_out, stop, args.metrics_interval
                )
            )

        if open_loop:
            rows = await run_open_loop(session, args)
        else:
            rows = await run_closed_loop(session, args)
        stop.set()
        if sampler:
            await sampler

    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    agg = base_aggregate(rows)
    if open_loop:
        agg["mode"] = "open-loop"
        agg["arrival_rate"] = args.arrival_rate
        agg["duration_s"] = args.duration_s
        agg["seed"] = args.seed
        agg["n_arrivals"] = len(rows)
        agg["offered_rate"] = round(len(rows) / args.duration_s, 4)
        agg.update(concurrency_stats(rows, args.duration_s))
    else:
        agg["mode"] = "closed-loop"
        agg["concurrency"] = args.concurrency
        agg.update(concurrency_stats(rows, None))
    logger.info("AGGREGATE %s", json.dumps(agg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8091")
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument(
        "--n", type=int, default=1,
        help="sequential requests per worker (closed-loop sustained load)",
    )
    parser.add_argument(
        "--arrival-rate", type=float, default=None,
        help="open-loop offered rate in req/s (Poisson interarrivals); "
        "when set, --concurrency/--n/--stagger are ignored",
    )
    parser.add_argument(
        "--duration-s", type=float, default=180.0,
        help="open-loop arrival window length in seconds",
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for open-loop interarrivals")
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default="A warm female voice with clear pronunciation")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--stagger", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--metrics-interval", type=float, default=1.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
