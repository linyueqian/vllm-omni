"""Async benchmark client for Legato: N concurrent SSE streams, no thread jitter.

Reuses the exact-accounting summarize/simulate from legato_bench.py (same dir).
Adds an in-process /metrics sampler task so server-side stage state can be
cross-referenced with client-side stall windows (absolute epoch timestamps).

Usage:
  python legato_bench_async.py --concurrency 48 --text "..." \
      --out run.ndjson --metrics-out run_metrics.ndjson
"""

import argparse
import asyncio
import base64
import json
import logging
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
    session: aiohttp.ClientSession, url: str, out_path: str, stop: asyncio.Event
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
                await asyncio.wait_for(stop.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass


async def run(args: argparse.Namespace) -> None:
    timeout = aiohttp.ClientTimeout(total=args.timeout, sock_read=args.timeout)
    conn = aiohttp.TCPConnector(limit=args.concurrency + 8)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        warm = await stream_once(session, args.url, "Warm up sentence.", args.voice)
        logger.info("warmup: %d emissions", len(warm["emissions"]))

        stop = asyncio.Event()
        sampler = None
        if args.metrics_out:
            sampler = asyncio.create_task(
                metrics_sampler(session, args.url, args.metrics_out, stop)
            )

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
        rows = [r for wr in nested for r in wr]
        stop.set()
        if sampler:
            await sampler

    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    ok = [r for r in rows if r.get("status") == "ok"]
    agg: dict[str, Any] = {
        "concurrency": args.concurrency,
        "n_ok": len(ok),
        "ttfa_med": round(sorted(r["ttfa_s"] for r in ok)[len(ok) // 2], 4),
        "rtf_med": round(sorted(r["rtf"] for r in ok)[len(ok) // 2], 4),
    }
    for b in (100, 300, 500):
        key = f"stall_b{b}ms"
        stalled = [r for r in ok if r[key]["stalls"] > 0]
        agg[f"stalled_frac_b{b}"] = round(len(stalled) / len(ok), 3)
        agg[f"stall_time_total_b{b}"] = round(sum(r[key]["stall_time"] for r in ok), 3)
    logger.info("AGGREGATE %s", json.dumps(agg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8091")
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument(
        "--n", type=int, default=1,
        help="sequential requests per worker (closed-loop sustained load)",
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default="A warm female voice with clear pronunciation")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--stagger", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics-out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
