"""Extract D* pacing labels from legato bench ndjson runs.

For each request i with send time a_i (t0_epoch, THE chosen origin), emission
times t_ik = a_i + ttfa_s + cumulative arrival gaps up to chunk k, and exact
chunk audio durations d_ij:

    D*_i = max_k [ t_ik - a_i - sum_{j<k} d_ij ]_+

D* is the maximum cumulative production deficit relative to the request send
time: the minimal playback-start delay (measured from a_i) that eliminates all
stalls at the unchanged service schedule. The k=1 term equals TTFA, so
D* >= TTFA by construction. B* (auxiliary) is the same maximum with the origin
at the first chunk (k >= 2, offsets relative to t_i1): the minimal startup
jitter buffer, used to cross-check against the exact stall simulation
(a request stalls at buffer B iff B* > B).

Each request is joined to the nearest-in-time /metrics sample from the run's
metrics ndjson (server-side stage state at arrival = the predictor features).

Usage:
  python extract_dstar.py --out dstar_dataset.ndjson openloop/*.ndjson
Metrics files (<stem>.metrics.ndjson) are discovered by convention and
filtered from the positional list automatically.
"""

import argparse
import bisect
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

METRICS_SUFFIX = ".metrics.ndjson"


def _get(m: dict[str, float], name: str, stage: str, extra: str | None = None):
    """Look up a labeled prometheus key by metric name + stage label."""
    prefix = name + "{"
    for k, v in m.items():
        if k.startswith(prefix) and f'stage="{stage}"' in k:
            if extra is None or extra in k:
                return v
    return None


def load_metrics(path: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = d.get("m", {})
            samples.append({
                "ts": d["ts"],
                "s0_running": _get(m, "vllm:num_requests_running", "0"),
                "s1_running": _get(m, "vllm:num_requests_running", "1"),
                "s0_waiting": _get(m, "vllm:num_requests_waiting", "0"),
                "s1_waiting": _get(m, "vllm:num_requests_waiting", "1"),
                "s1_waiting_capacity": _get(
                    m, "vllm:num_requests_waiting_by_reason", "1",
                    'reason="capacity"',
                ),
                "s0_kv_usage": _get(m, "vllm:kv_cache_usage_perc", "0"),
                "s1_kv_usage": _get(m, "vllm:kv_cache_usage_perc", "1"),
                "s1_qtime_count": _get(
                    m, "vllm:request_queue_time_seconds_count", "1"
                ),
                "s1_qtime_sum": _get(m, "vllm:request_queue_time_seconds_sum", "1"),
            })
    samples.sort(key=lambda s: s["ts"])
    return samples


def join_metrics(samples: list[dict[str, Any]], arrival: float) -> dict[str, Any]:
    """Nearest-in-time sample features + stage-1 queue-time counter deltas
    over the previous sample interval."""
    if not samples:
        return {}
    ts_list = [s["ts"] for s in samples]
    i = bisect.bisect_left(ts_list, arrival)
    if i >= len(samples):
        i = len(samples) - 1
    elif i > 0 and arrival - ts_list[i - 1] < ts_list[i] - arrival:
        i -= 1
    s = samples[i]
    feat = {
        k: s[k]
        for k in (
            "s0_running", "s0_waiting", "s1_running", "s1_waiting",
            "s1_waiting_capacity", "s0_kv_usage", "s1_kv_usage",
        )
    }
    feat["metrics_gap_s"] = round(s["ts"] - arrival, 3)
    if i > 0:
        prev = samples[i - 1]
        dt = s["ts"] - prev["ts"]
        feat["metrics_dt_s"] = round(dt, 3)
        for key, out in (
            ("s1_qtime_count", "s1_qtime_count_delta"),
            ("s1_qtime_sum", "s1_qtime_sum_delta"),
        ):
            if s[key] is not None and prev[key] is not None:
                feat[out] = round(s[key] - prev[key], 6)
            else:
                feat[out] = None
    else:
        feat["metrics_dt_s"] = None
        feat["s1_qtime_count_delta"] = None
        feat["s1_qtime_sum_delta"] = None
    return feat


def deficits(row: dict[str, Any]) -> tuple[float, float] | None:
    """(dstar, bstar) from a summarize() row; None if not computable."""
    ttfa = row.get("ttfa_s")
    gaps = row.get("arrival_gaps_s")
    durs = row.get("chunk_audio_s")
    if ttfa is None or gaps is None or durs is None:
        return None
    if len(durs) != len(gaps) + 1:
        return None
    t_rel = [ttfa]
    for g in gaps:
        t_rel.append(t_rel[-1] + g)
    cum = 0.0
    dstar = 0.0
    bstar = 0.0
    for k, tk in enumerate(t_rel):
        dstar = max(dstar, tk - cum)  # t_ik - a_i - sum_{j<k} d_ij
        if k > 0:
            bstar = max(bstar, tk - ttfa - cum)
        cum += durs[k]
    return max(dstar, 0.0), max(bstar, 0.0)


def client_inflight(rows: list[dict[str, Any]]) -> dict[int, int]:
    """Requests already in flight (sent, last emission not yet received) at
    each request's send time, from the client's own timeline."""
    spans = [
        (r["t0_epoch"], r["t0_epoch"] + r["wall_s"], id(r))
        for r in rows
        if r.get("status") == "ok" and "t0_epoch" in r and "wall_s" in r
    ]
    out: dict[int, int] = {}
    for a, _, rid in spans:
        out[rid] = sum(1 for s, e, oid in spans if oid != rid and s <= a <= e)
    return out


def quantile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    v = sorted(vals)
    idx = min(int(q * len(v)), len(v) - 1)
    return v[idx]


def process_run(run_path: str, out_f, run_id: str) -> None:
    metrics_path = run_path[: -len(".ndjson")] + METRICS_SUFFIX
    samples = load_metrics(metrics_path) if os.path.exists(metrics_path) else []
    if not samples:
        logger.warning("%s: no metrics samples (%s)", run_id, metrics_path)

    rows = []
    with open(run_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    inflight = client_inflight(rows)
    dstars: list[float] = []
    bstars: list[float] = []
    n_out = 0
    for row in rows:
        if row.get("status") != "ok" or "t0_epoch" not in row:
            continue
        d = deficits(row)
        if d is None:
            continue
        dstar, bstar = d
        rec = {
            "run_id": run_id,
            "req": row.get("req"),
            "arrival_epoch": row["t0_epoch"],
            "client_inflight": inflight.get(id(row)),
            "dstar": round(dstar, 4),
            "bstar": round(bstar, 4),
            "ttfa_s": row["ttfa_s"],
            "audio_s": row["audio_s"],
            "wall_s": row["wall_s"],
            "n_chunks": row["n_emissions"],
        }
        rec.update(join_metrics(samples, row["t0_epoch"]))
        out_f.write(json.dumps(rec) + "\n")
        dstars.append(dstar)
        bstars.append(bstar)
        n_out += 1

    if not dstars:
        logger.warning("%s: no usable rows", run_id)
        return
    summary = {
        "run_id": run_id,
        "n": n_out,
        "dstar_p50": round(quantile(dstars, 0.50), 3),
        "dstar_p90": round(quantile(dstars, 0.90), 3),
        "dstar_p95": round(quantile(dstars, 0.95), 3),
        "dstar_max": round(max(dstars), 3),
    }
    # Cross-check: exact stall sim says stalled(B) iff B* > B.
    ok_rows = [r for r in rows if r.get("status") == "ok" and "stall_b300ms" in r]
    for b_ms in (100, 300, 500):
        key = f"stall_b{b_ms}ms"
        rec_frac = sum(1 for r in ok_rows if r[key]["stalls"] > 0) / max(len(ok_rows), 1)
        cmp_frac = sum(1 for b in bstars if b > b_ms / 1000.0) / len(bstars)
        summary[f"stalled_frac_b{b_ms}_recorded"] = round(rec_frac, 3)
        summary[f"stalled_frac_b{b_ms}_from_bstar"] = round(cmp_frac, 3)
    logger.info("SUMMARY %s", json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="per-request ndjson files")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_paths = [p for p in args.runs if not p.endswith(METRICS_SUFFIX)]
    with open(args.out, "w", encoding="utf-8") as out_f:
        for path in run_paths:
            run_id = os.path.basename(path)[: -len(".ndjson")]
            process_run(path, out_f, run_id)
    logger.info("wrote %s from %d runs", args.out, len(run_paths))


if __name__ == "__main__":
    main()
