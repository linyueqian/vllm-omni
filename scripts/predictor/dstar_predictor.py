"""D* per-request deficit predictor: baselines, split CQR, load-regime transfer.

Design (frozen before looking at any leave-one-regime-out result):
- Features at arrival: client_inflight, s0_running, s0_waiting, s1_running,
  s1_waiting, s0_kv_usage, s1_qtime_count_delta, s1_qtime_sum_delta.
  Dropped: s1_waiting_capacity (== s1_waiting everywhere), s1_kv_usage (const 0).
  Null qtime deltas (first request of seed-1 runs, no prior scrape) -> 0.
- Regimes by offered-rate band: clean <= 2.2 req/s, knife = 2.4, collapse >= 2.8.
- Splits group BY RUN, time-ordered (earliest runs -> train), never split a run.
- One-sided split CQR: residual E = y - f(x) on cal block, Q = ceil((n+1)*0.9)-th
  order statistic, bound = f(x) + Q. Coverage = P(y <= bound) on test.
- gbm target space (raw vs log) is chosen once on the blocked split's CAL block.
"""
import json
import math
import os
import re
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

DATA = os.environ.get(
    "DSTAR_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "dstar_dataset.ndjson"))
FEATURES = ["client_inflight", "s0_running", "s0_waiting", "s1_running",
            "s1_waiting", "s0_kv_usage", "s1_qtime_count_delta", "s1_qtime_sum_delta"]
Q_TARGET = 0.9
SEED = 0


def load() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    rows = [json.loads(l) for l in open(DATA)]
    X = np.array([[float(r[k]) if r.get(k) is not None else 0.0 for k in FEATURES]
                  for r in rows])
    y = np.array([r["dstar"] for r in rows])
    run_ids = [r["run_id"] for r in rows]
    rates = np.array([float(re.match(r"r([\d.]+)_", rid).group(1)) for rid in run_ids])
    t = np.array([r["arrival_epoch"] for r in rows])
    return X, y, rates, run_ids, t


def regime_of(rate: float) -> str:
    if rate <= 2.2:
        return "clean"
    if rate <= 2.4:
        return "knife"
    return "collapse"


def run_table(run_ids: List[str], rates: np.ndarray, t: np.ndarray) -> List[dict]:
    seen: Dict[str, dict] = {}
    for i, rid in enumerate(run_ids):
        e = seen.setdefault(rid, {"run": rid, "rate": rates[i], "start": t[i]})
        e["start"] = min(e["start"], t[i])
    runs = sorted(seen.values(), key=lambda e: e["start"])
    for e in runs:
        e["regime"] = regime_of(e["rate"])
    return runs


def alloc_blocked(runs: List[dict]) -> Dict[str, str]:
    """Stratified blocked-time split: within each regime, time-ordered runs go
    first ~50% -> train, next ~25% -> cal, last ~25% -> test."""
    out: Dict[str, str] = {}
    for reg in ["clean", "knife", "collapse"]:
        rr = [e for e in runs if e["regime"] == reg]
        n = len(rr)
        n_tr = round(n * 0.5)
        n_cal = round(n * 0.25)
        for i, e in enumerate(rr):
            out[e["run"]] = "train" if i < n_tr else ("cal" if i < n_tr + n_cal else "test")
    return out


def alloc_lolo(runs: List[dict], held_out: List[str]) -> Dict[str, str]:
    """Train+cal on the non-held-out regimes (time-ordered ~2/3 train, 1/3 cal),
    test = ALL runs of the held-out regime(s)."""
    out: Dict[str, str] = {}
    for reg in ["clean", "knife", "collapse"]:
        rr = [e for e in runs if e["regime"] == reg]
        if reg in held_out:
            for e in rr:
                out[e["run"]] = "test"
        else:
            n_tr = round(len(rr) * 2 / 3)
            for i, e in enumerate(rr):
                out[e["run"]] = "train" if i < n_tr else "cal"
    return out


def pinball(y: np.ndarray, f: np.ndarray, q: float = Q_TARGET) -> float:
    d = y - f
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def conformal_offset(y_cal: np.ndarray, f_cal: np.ndarray, q: float = Q_TARGET) -> float:
    e = np.sort(y_cal - f_cal)
    n = len(e)
    k = math.ceil((n + 1) * q)
    return float(e[min(k, n) - 1])


# ---------------- methods: fit(train) -> predict fn ----------------

def fit_global_q(Xtr, ytr) -> Callable[[np.ndarray], np.ndarray]:
    c = float(np.quantile(ytr, Q_TARGET))
    return lambda X: np.full(len(X), c)


def fit_bucket_q(Xtr, ytr, n_bins: int = 12) -> Callable[[np.ndarray], np.ndarray]:
    """Moon-style percentile baseline keyed on s1_running+s1_waiting at arrival."""
    i_r, i_w = FEATURES.index("s1_running"), FEATURES.index("s1_waiting")
    load_tr = Xtr[:, i_r] + Xtr[:, i_w]
    edges = np.unique(np.quantile(load_tr, np.linspace(0, 1, n_bins + 1)[1:-1]))
    idx_tr = np.digitize(load_tr, edges)
    glob = float(np.quantile(ytr, Q_TARGET))
    table = {b: float(np.quantile(ytr[idx_tr == b], Q_TARGET))
             for b in range(len(edges) + 1) if (idx_tr == b).sum() >= 20}

    def predict(X: np.ndarray) -> np.ndarray:
        load = X[:, i_r] + X[:, i_w]
        idx = np.digitize(load, edges)
        return np.array([table.get(b, glob) for b in idx])
    return predict


def fit_gbm(Xtr, ytr, loss: str, log_target: bool) -> Callable[[np.ndarray], np.ndarray]:
    kw = dict(learning_rate=0.06, max_iter=400, min_samples_leaf=30,
              l2_regularization=1.0, early_stopping=False, random_state=SEED)
    if loss == "quantile":
        model = HistGradientBoostingRegressor(loss="quantile", quantile=Q_TARGET, **kw)
    else:
        model = HistGradientBoostingRegressor(loss="squared_error", **kw)
    yt = np.log(ytr) if log_target else ytr
    model.fit(Xtr, yt)
    if log_target:
        return lambda X: np.exp(model.predict(X)), model
    return (lambda X: model.predict(X)), model


# ---------------- evaluation ----------------

def masks(alloc: Dict[str, str], run_ids: List[str]) -> Dict[str, np.ndarray]:
    a = np.array([alloc[r] for r in run_ids])
    return {s: a == s for s in ["train", "cal", "test"]}


def evaluate_split(name: str, alloc: Dict[str, str], X, y, run_ids, regimes,
                   log_target: bool) -> List[dict]:
    m = masks(alloc, run_ids)
    Xtr, ytr = X[m["train"]], y[m["train"]]
    Xca, yca = X[m["cal"]], y[m["cal"]]
    Xte, yte = X[m["test"]], y[m["test"]]
    reg_te = regimes[m["test"]]
    reg_ca = regimes[m["cal"]]

    methods = {
        "global-q0.9": fit_global_q(Xtr, ytr),
        "load-bucket-q0.9": fit_bucket_q(Xtr, ytr),
        "gbm-q0.9": fit_gbm(Xtr, ytr, "quantile", log_target)[0],
        "gbm-mean": fit_gbm(Xtr, ytr, "squared_error", log_target)[0],
    }
    out = []
    for mname, f in methods.items():
        f_te, f_ca = f(Xte), f(Xca)
        Qc = conformal_offset(yca, f_ca)
        bound = f_te + Qc
        row = {"split": name, "method": mname,
               "pinball": pinball(yte, f_te),
               "cov": float(np.mean(yte <= bound)),
               "raw_cov": float(np.mean(yte <= f_te)),
               "Qc": Qc, "n_test": len(yte)}
        # clean-run tax: bound on clean TEST requests; if none, use clean CAL (*)
        clean_te = reg_te == "clean"
        if clean_te.any():
            tax = bound[clean_te]
            row["tax_src"] = "test"
        else:
            tax = f_ca[reg_ca == "clean"] + Qc
            row["tax_src"] = "cal*"
        row["tax_mean"] = float(np.mean(tax)) if len(tax) else float("nan")
        row["tax_p95"] = float(np.quantile(tax, 0.95)) if len(tax) else float("nan")
        # per-regime test breakdown
        row["per_regime"] = {}
        for reg in ["clean", "knife", "collapse"]:
            s = reg_te == reg
            if s.any():
                row["per_regime"][reg] = {
                    "n": int(s.sum()),
                    "pinball": pinball(yte[s], f_te[s]),
                    "cov": float(np.mean(yte[s] <= bound[s])),
                    "bound_mean": float(np.mean(bound[s])),
                    "bound_p95": float(np.quantile(bound[s], 0.95)),
                }
        out.append(row)
    return out


def main() -> None:
    X, y, rates, run_ids, t = load()
    regimes = np.array([regime_of(r) for r in rates])
    runs = run_table(run_ids, rates, t)

    print("== Run allocation (blocked split) ==")
    alloc_b = alloc_blocked(runs)
    for e in runs:
        print(f"  {e['run']:<14} {e['regime']:<9} -> {alloc_b[e['run']]}")

    # choose gbm target space on blocked CAL block only (frozen thereafter)
    m = masks(alloc_b, run_ids)
    cal_scores = {}
    for lt in [False, True]:
        f, _ = fit_gbm(X[m["train"]], y[m["train"]], "quantile", lt)
        cal_scores[lt] = pinball(y[m["cal"]], f(X[m["cal"]]))
    log_target = min(cal_scores, key=cal_scores.get)
    print(f"\ngbm target space chosen on blocked CAL: raw={cal_scores[False]:.4f} "
          f"log={cal_scores[True]:.4f} -> log_target={log_target}\n")

    splits = {
        "blocked": alloc_b,
        "lolo-knife (train clean+collapse)": alloc_lolo(runs, ["knife"]),
        "lolo-cc (train knife only)": alloc_lolo(runs, ["clean", "collapse"]),
    }
    all_rows = []
    for name, alloc in splits.items():
        all_rows += evaluate_split(name, alloc, X, y, run_ids, regimes, log_target)

    hdr = (f"{'split':<34} {'method':<17} {'pinball':>8} {'cov@90':>7} {'rawcov':>7} "
           f"{'Qc':>7} {'taxMean':>8} {'taxP95':>7} {'taxSrc':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in all_rows:
        print(f"{r['split']:<34} {r['method']:<17} {r['pinball']:>8.4f} "
              f"{r['cov']:>7.3f} {r['raw_cov']:>7.3f} {r['Qc']:>7.2f} "
              f"{r['tax_mean']:>8.2f} {r['tax_p95']:>7.2f} {r['tax_src']:>6}")

    print("\n== Per-regime test breakdown (gbm-q0.9 rows plus baselines) ==")
    for r in all_rows:
        for reg, d in r["per_regime"].items():
            print(f"  {r['split']:<34} {r['method']:<17} {reg:<9} n={d['n']:>5} "
                  f"pinball={d['pinball']:>8.4f} cov={d['cov']:.3f} "
                  f"bound_mean={d['bound_mean']:>7.2f} bound_p95={d['bound_p95']:>7.2f}")

    # permutation importance: gbm-q0.9 on blocked test block, pinball scorer
    print("\n== Permutation importance (gbm-q0.9, blocked split, test block, pinball) ==")
    fpred, model = fit_gbm(X[m["train"]], y[m["train"]], "quantile", log_target)

    def scorer(est, Xs, ys):
        p = np.exp(est.predict(Xs)) if log_target else est.predict(Xs)
        return -pinball(ys, p)
    r = permutation_importance(model, X[m["test"]], y[m["test"]], scoring=scorer,
                               n_repeats=20, random_state=SEED)
    order = np.argsort(-r.importances_mean)
    for i in order:
        print(f"  {FEATURES[i]:<22} dPinball {r.importances_mean[i]:>8.4f} "
              f"+- {r.importances_std[i]:.4f}")

    # per-test-run coverage for the gbm on the blocked split (pathwise view)
    print("\n== Per-test-run coverage, gbm-q0.9 conformalized (blocked split) ==")
    f_ca = fpred(X[m["cal"]])
    Qc = conformal_offset(y[m["cal"]], f_ca)
    bound_all = fpred(X) + Qc
    for e in runs:
        if alloc_b[e["run"]] == "test":
            s = np.array([rid == e["run"] for rid in run_ids])
            print(f"  {e['run']:<14} {e['regime']:<9} n={s.sum():>4} "
                  f"cov={np.mean(y[s] <= bound_all[s]):.3f} "
                  f"bound_mean={bound_all[s].mean():>6.2f}")


if __name__ == "__main__":
    main()
