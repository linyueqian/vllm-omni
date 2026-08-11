"""Post-hoc diagnostics for the D* predictor report (no method changes)."""
import numpy as np

from dstar_predictor import (DATA, FEATURES, Q_TARGET, alloc_blocked, alloc_lolo,
                             conformal_offset, fit_gbm, load, masks, regime_of,
                             run_table)

X, y, rates, run_ids, t = load()
regimes = np.array([regime_of(r) for r in rates])
runs = run_table(run_ids, rates, t)
LOG_TARGET = True  # frozen choice from main script


def run_mask(rid):
    return np.array([r == rid for r in run_ids])


def block(alloc, X, y):
    m = masks(alloc, run_ids)
    return m


print("== Extrapolation ceiling ==")
alloc_b = alloc_blocked(runs)
m = masks(alloc_b, run_ids)
print(f"  blocked train max D*: {y[m['train']].max():.1f}s  "
      f"cal max: {y[m['cal']].max():.1f}s  test max: {y[m['test']].max():.1f}s")
f, _ = fit_gbm(X[m["train"]], y[m["train"]], "quantile", LOG_TARGET)
pred_all = f(X)
print(f"  gbm-q0.9 max prediction anywhere: {pred_all.max():.1f}s")

print("\n== Knife test run r2.4_t3_s1: raw prediction vs truth (blocked model) ==")
s = run_mask("r2.4_t3_s1")
print(f"  actual  p50 {np.quantile(y[s],0.5):.2f} p90 {np.quantile(y[s],0.9):.2f} "
      f"max {y[s].max():.2f}")
print(f"  rawpred p50 {np.quantile(pred_all[s],0.5):.2f} p90 {np.quantile(pred_all[s],0.9):.2f} "
      f"max {pred_all[s].max():.2f}")

print("\n== Clean test runs, actual D* q0.9 (tax context, blocked) ==")
for rid in ["r2.0_t3_s0", "r2.2_t0_s0", "r2.2_t3_s0"]:
    s = run_mask(rid)
    print(f"  {rid}: actual q0.9 = {np.quantile(y[s],0.9):.3f}s  "
          f"bound_mean would-be tax vs actual need")

print("\n== Deep-collapse test run r3.3_t3_s1: where coverage dies (blocked) ==")
s = run_mask("r3.3_t3_s1")
f_ca = f(X[m["cal"]])
Qc = conformal_offset(y[m["cal"]], f_ca)
bound = pred_all[s] + Qc
idx = np.argsort(t[np.where(s)[0]])
ys, bs = y[s][idx], bound[idx]
n = len(ys)
for frac in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
    a, b = int(frac[0] * n), int(frac[1] * n)
    print(f"  time-quarter {frac}: actual p50 {np.quantile(ys[a:b],0.5):>6.1f}s  "
          f"bound p50 {np.quantile(bs[a:b],0.5):>6.1f}s  "
          f"cov {np.mean(ys[a:b] <= bs[a:b]):.3f}")

print("\n== LOLO-knife: per-run coverage on the 4 knife test runs ==")
alloc_k = alloc_lolo(runs, ["knife"])
mk = masks(alloc_k, run_ids)
fk, _ = fit_gbm(X[mk["train"]], y[mk["train"]], "quantile", LOG_TARGET)
Qk = conformal_offset(y[mk["cal"]], fk(X[mk["cal"]]))
print(f"  Qc = {Qk:.2f}s; cal runs: "
      f"{sorted(set(r for r, a in alloc_k.items() if a == 'cal'))}")
for rid in ["r2.4_t0_s0", "r2.4_t3_s0", "r2.4_t0_s1", "r2.4_t3_s1"]:
    s = run_mask(rid)
    b = fk(X[s]) + Qk
    raw = fk(X[s])
    print(f"  {rid}: actual p50/p90 {np.quantile(y[s],0.5):>5.2f}/{np.quantile(y[s],0.9):>5.2f}  "
          f"rawpred p50 {np.quantile(raw,0.5):>5.2f}  rawcov {np.mean(y[s]<=raw):.3f}  "
          f"cov {np.mean(y[s]<=b):.3f}  bound_mean {b.mean():>5.2f}")

print("\n== LOLO-cc: per-clean-run coverage (train knife only) ==")
alloc_c = alloc_lolo(runs, ["clean", "collapse"])
mc = masks(alloc_c, run_ids)
fc, _ = fit_gbm(X[mc["train"]], y[mc["train"]], "quantile", LOG_TARGET)
Qcc = conformal_offset(y[mc["cal"]], fc(X[mc["cal"]]))
print(f"  Qc = {Qcc:.2f}s")
for e in runs:
    if e["regime"] == "clean":
        s = run_mask(e["run"])
        b = fc(X[s]) + Qcc
        print(f"  {e['run']:<12}: actual q0.9 {np.quantile(y[s],0.9):>5.2f}  "
              f"cov {np.mean(y[s]<=b):.3f}  bound_mean {b.mean():>5.2f}")

print("\n== Effective run-level heterogeneity in cal (blocked): residual q0.9 per cal run ==")
for e in runs:
    if alloc_b[e["run"]] == "cal":
        s = run_mask(e["run"])
        res = y[s] - pred_all[s]
        print(f"  {e['run']:<12} {e['regime']:<9} residual q0.9 = {np.quantile(res, 0.9):>7.2f}s")
