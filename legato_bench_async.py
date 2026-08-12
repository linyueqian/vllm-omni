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

PACER_BUFFER_S = 0.3


class Pacer:
    """Live risk-controlled pacing/admission from the exported D* predictor.

    Per arrival: one /metrics scrape, features built with the SAME semantics as
    extract_dstar.py (stage-labeled counters; qtime counter deltas against a
    rolling previous-scrape cache; client_inflight = accepted requests in
    flight, excluding the deciding request). bound = quantile prediction + Qc
    (split-CQR offset). bound > theta => REJECT (request is never sent);
    otherwise the request is sent unmodified and the bound is recorded so the
    paced playback start s = max(ttfa, bound) + 0.3 is computed in the summary.
    """

    def __init__(self, model_path: str, theta: float) -> None:
        import joblib  # deferred: only pacer runs need sklearn in the venv

        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.qc = float(payload["Qc"])
        self.features: list[str] = list(payload["features"])
        self.log_target = bool(payload.get("log_target", True))
        self.theta = theta
        self.prev: dict[str, Any] | None = None
        self.inflight = 0
        logger.info(
            "pacer: model=%s Qc=%.4f theta=%s features=%s",
            model_path, self.qc, theta, self.features,
        )

    @staticmethod
    def _get(m: dict[str, float], name: str, stage: str,
             extra: str | None = None) -> float | None:
        prefix = name + "{"
        for k, v in m.items():
            if k.startswith(prefix) and f'stage="{stage}"' in k:
                if extra is None or extra in k:
                    return v
        return None

    async def _scrape(
        self, session: aiohttp.ClientSession, url: str
    ) -> dict[str, Any] | None:
        try:
            async with session.get(f"{url.rstrip('/')}/metrics") as resp:
                body = await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            logger.warning("pacer scrape failed: %s", exc)
            return None
        m: dict[str, float] = {}
        for line in body.splitlines():
            if line.startswith("#") or not METRIC_KEEP.search(line):
                continue
            mt = METRIC_LINE.match(line)
            if mt:
                m[mt.group(1)] = float(mt.group(2))
        return {
            "ts": time.time(),
            "s0_running": self._get(m, "vllm:num_requests_running", "0"),
            "s1_running": self._get(m, "vllm:num_requests_running", "1"),
            "s0_waiting": self._get(m, "vllm:num_requests_waiting", "0"),
            "s1_waiting": self._get(m, "vllm:num_requests_waiting", "1"),
            "s0_kv_usage": self._get(m, "vllm:kv_cache_usage_perc", "0"),
            "s1_qtime_count": self._get(
                m, "vllm:request_queue_time_seconds_count", "1"),
            "s1_qtime_sum": self._get(m, "vllm:request_queue_time_seconds_sum", "1"),
        }

    async def decide(self, session: aiohttp.ClientSession, url: str) -> dict[str, Any]:
        cur = await self._scrape(session, url)
        scrape_ok = cur is not None
        if cur is None:
            cur = self.prev  # stale fallback; None on the very first arrival
        feats: dict[str, float] = {"client_inflight": float(self.inflight)}
        for k in ("s0_running", "s0_waiting", "s1_running", "s1_waiting",
                  "s0_kv_usage"):
            v = cur.get(k) if cur else None
            feats[k] = float(v) if v is not None else 0.0
        # qtime counter deltas vs rolling previous-scrape cache (0.0 when no
        # previous sample -- same imputation as model training).
        feats["s1_qtime_count_delta"] = 0.0
        feats["s1_qtime_sum_delta"] = 0.0
        dt = None
        if scrape_ok and self.prev is not None:
            dt = cur["ts"] - self.prev["ts"]
            for key, out in (("s1_qtime_count", "s1_qtime_count_delta"),
                             ("s1_qtime_sum", "s1_qtime_sum_delta")):
                if cur.get(key) is not None and self.prev.get(key) is not None:
                    feats[out] = float(cur[key] - self.prev[key])
        vec = [[feats[k] for k in self.features]]
        import math as _math

        pred = float(self.model.predict(vec)[0])
        if self.log_target:
            pred = _math.exp(pred)
        bound = max(0.0, pred + self.qc)
        if scrape_ok:
            self.prev = cur
        return {"bound": bound, "features": feats, "scrape_ok": scrape_ok,
                "scrape_dt_s": round(dt, 3) if dt is not None else None}


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


def paced_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Acceptance and paced-playback stall summary (s = max(ttfa, bound) + 0.3),
    exactly the replay convention: stall time = [dstar - s]+."""
    from extract_dstar import deficits

    rejected = [r for r in rows if r.get("status") == "rejected"]
    acc = [r for r in rows if r.get("status") == "ok" and "pacer_bound" in r]
    out: dict[str, Any] = {
        "n_rejected": len(rejected),
        "n_accepted": len(acc),
        "acceptance": round(len(acc) / max(len(acc) + len(rejected), 1), 3),
    }
    stall_times: list[float] = []
    n_lbl = 0
    for r in acc:
        d = deficits(r)
        if d is None:
            continue
        n_lbl += 1
        s = max(r["ttfa_s"], r["pacer_bound"]) + PACER_BUFFER_S
        st = max(0.0, d[0] - s)
        if st > 1e-9:
            stall_times.append(st)
    out["paced_stalled_frac_b300"] = round(len(stall_times) / max(n_lbl, 1), 3)
    out["paced_stall_total_s"] = round(sum(stall_times), 2)
    return out


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
    session: aiohttp.ClientSession, args: argparse.Namespace,
    pacer: "Pacer | None" = None,
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
        prow: dict[str, Any] = {}
        if pacer is not None:
            dec = await pacer.decide(session, args.url)
            prow = {
                "pacer_bound": round(dec["bound"], 4),
                "pacer_scrape_ok": dec["scrape_ok"],
                "pacer_scrape_dt_s": dec["scrape_dt_s"],
                "pacer_features": dec["features"],
            }
            if dec["bound"] > pacer.theta:
                return {"status": "rejected", "req": i,
                        "sched_arrival_s": round(sched_at, 4),
                        "t0_epoch": time.time(), **prow}
            pacer.inflight += 1
        try:
            res = await stream_once(session, args.url, args.text, args.voice)
            row = summarize(res)
            row["t0_epoch"] = res["t0_epoch"]
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            row = {"status": "error", "error": repr(exc), "t0_epoch": t0_epoch}
        finally:
            if pacer is not None:
                pacer.inflight -= 1
        row.update(prow)
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
    pacer = None
    if args.pacer_model:
        if not open_loop:
            raise SystemExit("--pacer-model requires open-loop mode (--arrival-rate)")
        pacer = Pacer(args.pacer_model, args.pacer_theta)
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
            rows = await run_open_loop(session, args, pacer)
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
        if pacer is not None:
            agg["pacer_theta"] = args.pacer_theta
            agg.update(paced_aggregate(rows))
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
    parser.add_argument(
        "--pacer-model", default=None,
        help="joblib payload from export_pacer.py; enables live pacing/admission",
    )
    parser.add_argument(
        "--pacer-theta", type=float, default=float("inf"),
        help="reject arrivals whose conformalized bound exceeds theta seconds "
        "(default inf = pace-only, never reject)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
