"""Closed-form replay of the two-regime pacer on the recorded 26 runs.

Exact replay identity (validated: dstar == ttfa + bstar to 1e-14 on all rows;
frac(bstar>0.3) reproduces the recorded B=300ms stall fractions): with
pause-resume playback and start delay s after arrival, a request stalls iff
dstar > s and its total stall time is [dstar - s]+.

Conventions (stated, not tuned):
- The pacer never reduces the client's default 300 ms prebuffer: paced start
  delay s = max(ttfa, bound) + 0.3.  Baseline (no pacing) s = ttfa + 0.3.
  Oracle s = dstar + 0.3 (dstar >= ttfa always) -> zero stalls by construction.
- Time-to-audible (TTA) = s.  Oracle-gap = TTA - TTA_oracle among accepted.
- Two-regime policy: REJECT iff conformalized bound > theta, theta in {1,2,4,8}s;
  accepted requests are paced.  Replay keeps the recorded trace, so rejection
  does NOT lighten the queue: stall numbers among accepted are PESSIMISTIC.
- Bounds are always out-of-sample: blocked-split model scored on its test runs;
  lolo-knife model on the 4 knife runs; lolo-cc model on the 22 clean+collapse
  runs.  Model/calibration frozen from dstar_predictor.py (log-target gbm-q0.9,
  one-sided split CQR).
"""
import json
from typing import Dict, List

import numpy as np

from dstar_predictor import (DATA, alloc_blocked, alloc_lolo, conformal_offset,
                             fit_gbm, load, masks, regime_of, run_table)

B = 0.3
THETAS = [1.0, 2.0, 4.0, 8.0, float("inf")]
LOG_TARGET = True  # frozen choice from dstar_predictor.py


def q(a: np.ndarray, p: float) -> float:
    return float(np.quantile(a, p)) if len(a) else float("nan")


def policy_row(label: str, dstar, ttfa, s, accept) -> dict:
    """Metrics for one policy on one row subset. s = start delay per request."""
    acc = accept.astype(bool)
    stall_t = np.maximum(0.0, dstar[acc] - s[acc])
    stalled = stall_t > 1e-9
    tta = s[acc]
    gap = s[acc] - (dstar[acc] + B)  # excess vs oracle TTA (can be negative)
    n = len(dstar)
    return {
        "label": label,
        "acc_rate": float(acc.mean()),
        "stall_inc": float(stalled.mean()) if acc.any() else float("nan"),
        "stall_mean": float(stall_t[stalled].mean()) if stalled.any() else 0.0,
        "stall_p95": q(stall_t[stalled], 0.95) if stalled.any() else 0.0,
        "tta_p50": q(tta, 0.5), "tta_p95": q(tta, 0.95),
        "gap_mean": float(gap.mean()) if acc.any() else float("nan"),
        "gap_p95": q(gap, 0.95),
        # certificate triple over ALL offered requests in the subset
        "frac_ok": float((acc & (np.maximum(0.0, dstar - s) <= 1e-9)).sum() / n),
        "frac_rej": float((~acc).mean()),
        "frac_stall": float((acc & (np.maximum(0.0, dstar - s) > 1e-9)).sum() / n),
    }


def print_rows(rows: List[dict]) -> None:
    hdr = (f"  {'policy':<22} {'accept':>6} {'stallInc':>8} {'stallMean':>9} "
           f"{'stallP95':>8} {'ttaP50':>7} {'ttaP95':>7} {'gapMean':>8} {'gapP95':>7}")
    print(hdr)
    for r in rows:
        print(f"  {r['label']:<22} {r['acc_rate']:>6.3f} {r['stall_inc']:>8.3f} "
              f"{r['stall_mean']:>9.2f} {r['stall_p95']:>8.2f} {r['tta_p50']:>7.2f} "
              f"{r['tta_p95']:>7.2f} {r['gap_mean']:>8.2f} {r['gap_p95']:>7.2f}")


def print_cert(rows: List[dict]) -> None:
    print(f"  {'policy':<22} {'ok(accept,nostall)':>18} {'rejected':>9} {'stalled':>8}")
    for r in rows:
        print(f"  {r['label']:<22} {r['frac_ok']:>18.3f} {r['frac_rej']:>9.3f} "
              f"{r['frac_stall']:>8.3f}")


def replay(name: str, alloc: Dict[str, str], X, y, ttfa, run_ids, regimes) -> None:
    m = masks(alloc, run_ids)
    f, _ = fit_gbm(X[m["train"]], y[m["train"]], "quantile", LOG_TARGET)
    Qc = conformal_offset(y[m["cal"]], f(X[m["cal"]]))
    te = m["test"]
    bound = np.maximum(0.0, f(X[te]) + Qc)
    d_te, t_te, reg_te = y[te], ttfa[te], regimes[te]
    test_runs = sorted({r for r, a in alloc.items() if a == "test"})
    print(f"\n#### Replay: {name}  (Qc={Qc:.2f}s, {len(test_runs)} test runs, "
          f"n={te.sum()}) ####")

    groups = [("pooled", np.ones(len(d_te), bool))]
    for reg in ["clean", "knife", "collapse"]:
        gm = reg_te == reg
        if gm.any():
            groups.append((reg, gm))

    for gname, gm in groups:
        d, t, bd = d_te[gm], t_te[gm], bound[gm]
        all_acc = np.ones(len(d), bool)
        rows = [
            policy_row("no-pacing", d, t, t + B, all_acc),
            policy_row("oracle", d, t, d + B, all_acc),
        ]
        for theta in THETAS:
            acc = bd <= theta
            s = np.maximum(t, bd) + B
            lab = "paced (no reject)" if np.isinf(theta) else f"two-regime th={theta:g}"
            rows.append(policy_row(lab, d, t, s, acc))
        print(f"\n-- {name} :: {gname} (n={gm.sum()}) --")
        print_rows(rows)
        print(" certificate (fractions of all offered):")
        print_cert(rows)


def main() -> None:
    X, y, rates, run_ids, t_arr = load()
    rows = [json.loads(l) for l in open(DATA)]
    ttfa = np.array([r["ttfa_s"] for r in rows])
    regimes = np.array([regime_of(r) for r in rates])
    runs = run_table(run_ids, rates, t_arr)

    replay("blocked-test", alloc_blocked(runs), X, y, ttfa, run_ids, regimes)
    replay("lolo-knife", alloc_lolo(runs, ["knife"]), X, y, ttfa, run_ids, regimes)
    replay("lolo-cc", alloc_lolo(runs, ["clean", "collapse"]), X, y, ttfa,
           run_ids, regimes)


if __name__ == "__main__":
    main()
