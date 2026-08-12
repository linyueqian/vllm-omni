"""Matched-pair analysis of the on-policy pacer A/B (onpolicy/ runs).

Per pair (rate, text): pacer-off vs pacer-on arm, same Poisson seed.
- Off arm: recorded stalls at B=300 (playback starts at ttfa + 0.3).
- On arm: paced playback s = max(ttfa, pacer_bound) + 0.3, stall = [dstar - s]+
  (identical to the replay convention), acceptance from rejected rows.
- Replay prediction for the pair: the SAME frozen model + theta applied
  counterfactually to the OFF-arm trace (features re-joined from the off-arm
  metrics file, exact extract_dstar semantics) -- the pessimistic number the
  on-policy run is supposed to beat.
Run on halo: .venv/bin/python scripts/predictor/analyze_onpolicy.py
"""
import json
import math
import os
import sys
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))  # repo root: extract_dstar

import joblib  # noqa: E402
from extract_dstar import (client_inflight, deficits, join_metrics,  # noqa: E402
                           load_metrics)

RUN_DIR = os.path.join(_HERE, "..", "..", "onpolicy")
MODEL = os.path.join(_HERE, "pacer_model.joblib")
B = 0.3
DUR = 180.0
FEATURES = None  # from payload


def q(vals: List[float], p: float) -> float:
    if not vals:
        return float("nan")
    v = sorted(vals)
    idx = min(int(p * len(v)), len(v) - 1)
    return v[idx]


def load_rows(tag: str) -> List[dict]:
    with open(os.path.join(RUN_DIR, f"{tag}.ndjson")) as f:
        return [json.loads(l) for l in f if l.strip()]


def mean_conc(rows: List[dict]) -> float:
    ok = [r for r in rows if r.get("status") == "ok" and "t0_epoch" in r]
    if not ok:
        return float("nan")
    spans = [(r["t0_epoch"], r["t0_epoch"] + r["wall_s"]) for r in ok]
    t0 = min(s for s, _ in spans)
    w_end = t0 + DUR
    busy = sum(max(0.0, min(e, w_end) - min(s, w_end)) for s, e in spans)
    return busy / DUR


def arm_stats(rows: List[dict], paced: bool) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    rej = [r for r in rows if r.get("status") == "rejected"]
    err = [r for r in rows if r.get("status") == "error"]
    ds, stall_t, tta = [], [], []
    for r in ok:
        d = deficits(r)
        if d is None:
            continue
        dstar = d[0]
        ds.append(dstar)
        s = (max(r["ttfa_s"], r["pacer_bound"]) if paced else r["ttfa_s"]) + B
        stall_t.append(max(0.0, dstar - s))
        tta.append(s)
    stalled = [v for v in stall_t if v > 1e-9]
    n_off = len(ok) + len(rej)
    return {
        "n_arrivals": len(rows), "n_err": len(err),
        "acceptance": len(ok) / max(n_off, 1),
        "mean_conc": mean_conc(rows),
        "stall_inc": len(stalled) / max(len(stall_t), 1),
        "stall_mean": sum(stalled) / len(stalled) if stalled else 0.0,
        "stall_p95": q(stalled, 0.95) if stalled else 0.0,
        "stall_total": sum(stalled),
        "tta_p50": q(tta, 0.50), "tta_p95": q(tta, 0.95),
        "d_p50": q(ds, 0.50), "d_p90": q(ds, 0.90), "d_max": max(ds) if ds else 0.0,
    }


def replay_on_off_trace(rows: List[dict], metrics_path: str, model_payload: dict,
                        theta: float) -> dict:
    """Counterfactual pacer decisions on the unlightened off-arm trace."""
    model = model_payload["model"]
    qc = model_payload["Qc"]
    feats_order = model_payload["features"]
    samples = load_metrics(metrics_path) if os.path.exists(metrics_path) else []
    inflight = client_inflight(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    acc_stall, n_acc, n_tot, bounds = [], 0, 0, []
    for r in ok:
        d = deficits(r)
        if d is None:
            continue
        n_tot += 1
        feat = join_metrics(samples, r["t0_epoch"])
        fv = {
            "client_inflight": float(inflight.get(id(r)) or 0),
            **{k: float(feat.get(k) or 0.0)
               for k in ("s0_running", "s0_waiting", "s1_running", "s1_waiting",
                         "s0_kv_usage", "s1_qtime_count_delta",
                         "s1_qtime_sum_delta")},
        }
        pred = float(model.predict([[fv[k] for k in feats_order]])[0])
        if model_payload.get("log_target", True):
            pred = math.exp(pred)
        bound = max(0.0, pred + qc)
        bounds.append(bound)
        if bound > theta:
            continue
        n_acc += 1
        s = max(r["ttfa_s"], bound) + B
        st = max(0.0, d[0] - s)
        if st > 1e-9:
            acc_stall.append(st)
    return {
        "acceptance": n_acc / max(n_tot, 1),
        "stall_inc": len(acc_stall) / max(n_acc, 1),
        "stall_total": sum(acc_stall),
        "bound_p50": q(bounds, 0.5),
    }


def main() -> None:
    payload = joblib.load(MODEL)
    pairs = []
    for rate in ["2.4", "3.0"]:
        for ti in ["0", "3"]:
            arms = [("off", False, None), ("th2", True, 2.0)]
            if rate == "2.4":
                arms.append(("th8", True, 8.0))
            pairs.append((rate, ti, arms))

    print("== Matched-pair table (B=300ms; on-arm stalls under paced playback) ==")
    hdr = (f"{'run':<18} {'arr':>4} {'acc':>6} {'conc':>6} {'stallInc':>8} "
           f"{'stMean':>7} {'stP95':>6} {'stTot':>7} {'ttaP50':>7} {'ttaP95':>7} "
           f"{'D*p50':>7} {'D*p90':>7} {'D*max':>7} {'err':>4}")
    print(hdr)
    all_stats: Dict[str, dict] = {}
    for rate, ti, arms in pairs:
        for arm, paced, _ in arms:
            tag = f"r{rate}_t{ti}_s7_{arm}"
            path = os.path.join(RUN_DIR, f"{tag}.ndjson")
            if not os.path.exists(path):
                print(f"{tag:<18} MISSING")
                continue
            st = arm_stats(load_rows(tag), paced)
            all_stats[tag] = st
            print(f"{tag:<18} {st['n_arrivals']:>4} {st['acceptance']:>6.3f} "
                  f"{st['mean_conc']:>6.1f} {st['stall_inc']:>8.3f} "
                  f"{st['stall_mean']:>7.2f} {st['stall_p95']:>6.2f} "
                  f"{st['stall_total']:>7.1f} {st['tta_p50']:>7.2f} "
                  f"{st['tta_p95']:>7.2f} {st['d_p50']:>7.2f} {st['d_p90']:>7.2f} "
                  f"{st['d_max']:>7.2f} {st['n_err']:>4}")

    print("\n== Replay-prediction (pessimistic, off-trace) vs on-policy realized ==")
    print(f"{'pair':<14} {'theta':>5} {'acc rep':>8} {'acc onp':>8} "
          f"{'stallInc rep':>12} {'stallInc onp':>12} {'D*p50 off':>9} {'D*p50 onp':>9}")
    for rate, ti, arms in pairs:
        off_tag = f"r{rate}_t{ti}_s7_off"
        off_path = os.path.join(RUN_DIR, f"{off_tag}.ndjson")
        if not os.path.exists(off_path):
            continue
        off_rows = load_rows(off_tag)
        mpath = os.path.join(RUN_DIR, f"{off_tag}.metrics.ndjson")
        for arm, _, theta in arms:
            if theta is None:
                continue
            on_tag = f"r{rate}_t{ti}_s7_{arm}"
            if on_tag not in all_stats:
                continue
            rep = replay_on_off_trace(off_rows, mpath, payload, theta)
            onp = all_stats[on_tag]
            print(f"r{rate}_t{ti:<8} {theta:>5.0f} {rep['acceptance']:>8.3f} "
                  f"{onp['acceptance']:>8.3f} {rep['stall_inc']:>12.3f} "
                  f"{onp['stall_inc']:>12.3f} {all_stats[off_tag]['d_p50']:>9.2f} "
                  f"{onp['d_p50']:>9.2f}")


if __name__ == "__main__":
    main()
