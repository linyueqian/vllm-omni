"""EDA for the D* dataset: distributions, per-run stats, missingness, correlations."""
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

DATA = os.environ.get(
    "DSTAR_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "dstar_dataset.ndjson"))

FEATURES = [
    "client_inflight", "s0_running", "s0_waiting", "s1_running", "s1_waiting",
    "s1_waiting_capacity", "s0_kv_usage", "s1_kv_usage",
    "s1_qtime_count_delta", "s1_qtime_sum_delta",
]
AUX = ["metrics_gap_s", "metrics_dt_s", "ttfa_s", "audio_s", "wall_s", "n_chunks", "bstar"]

rows = []
with open(DATA) as f:
    for line in f:
        rows.append(json.loads(line))

print(f"rows: {len(rows)}")
all_keys = set()
for r in rows:
    all_keys.update(r.keys())
print(f"columns: {sorted(all_keys)}")

# Missingness
print("\n== Missingness (missing key or null/NaN) ==")
for k in sorted(all_keys):
    n_missing = sum(1 for r in rows if k not in r or r[k] is None
                    or (isinstance(r[k], float) and math.isnan(r[k])))
    if n_missing:
        print(f"  {k}: {n_missing}")
print("  (columns not listed have zero missing)")

# Per-run stats
runs = defaultdict(list)
for r in rows:
    runs[r["run_id"]].append(r)

def q(a, p):
    return float(np.quantile(np.asarray(a), p))

print("\n== Per-run D* stats (sorted by run start time) ==")
print(f"{'run_id':<16} {'n':>5} {'start_epoch':>14} {'rate':>5} "
      f"{'p50':>8} {'p90':>8} {'p99':>8} {'max':>8} {'frac>1s':>8}")
run_order = sorted(runs, key=lambda k: min(r["arrival_epoch"] for r in runs[k]))
for rid in run_order:
    ds = [r["dstar"] for r in runs[rid]]
    m = re.match(r"r([\d.]+)_", rid)
    rate = m.group(1) if m else "?"
    start = min(r["arrival_epoch"] for r in runs[rid])
    frac1 = sum(1 for d in ds if d > 1.0) / len(ds)
    print(f"{rid:<16} {len(ds):>5} {start:>14.1f} {rate:>5} "
          f"{q(ds,0.5):>8.3f} {q(ds,0.9):>8.3f} {q(ds,0.99):>8.2f} {max(ds):>8.2f} {frac1:>8.3f}")

# Overall D* distribution
ds = np.array([r["dstar"] for r in rows])
print("\n== Overall D* quantiles (s) ==")
for p in [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]:
    print(f"  q{p}: {q(ds, p):.3f}")
print(f"  min {ds.min():.4f}  max {ds.max():.2f}  mean {ds.mean():.3f}")
print(f"  frac dstar==0: {(ds == 0).mean():.4f}")
print(f"  frac <0.3s: {(ds < 0.3).mean():.3f}  frac <1s: {(ds < 1).mean():.3f}  "
      f"frac >5s: {(ds > 5).mean():.3f}  frac >10s: {(ds > 10).mean():.3f}")

# Log-space histogram (text)
print("\n== D* histogram (log-spaced bins, all rows) ==")
eps = 1e-3
bins = [0, 0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10, 20, 40, 80, 1e9]
hist, _ = np.histogram(ds, bins=bins)
for lo, hi, c in zip(bins[:-1], bins[1:], hist):
    bar = "#" * int(60 * c / hist.max())
    hi_s = f"{hi:g}" if hi < 1e8 else "inf"
    print(f"  [{lo:g},{hi_s}): {c:>5} {bar}")

# metrics_gap_s sanity
gap = np.array([r["metrics_gap_s"] for r in rows])
print("\n== metrics_gap_s (join gap) ==")
print(f"  min {gap.min():.3f} p5 {q(gap,0.05):.3f} p50 {q(gap,0.5):.3f} "
      f"p95 {q(gap,0.95):.3f} max {gap.max():.3f}")
print(f"  frac |gap|>2s: {(np.abs(gap) > 2).mean():.4f}  frac |gap|>5s: {(np.abs(gap) > 5).mean():.4f}")
dt = np.array([r["metrics_dt_s"] for r in rows if r.get("metrics_dt_s") is not None])
print(f"  metrics_dt_s (n={len(dt)}): p50 {q(dt,0.5):.3f} p95 {q(dt,0.95):.3f} max {dt.max():.3f}")
null_rows = [(r["run_id"], r["req"]) for r in rows if r.get("metrics_dt_s") is None]
print(f"  null metrics_dt_s rows: {null_rows}")

# Correlations with dstar
def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    # handle ties crudely via average is skipped; fine for screening
    return float(np.corrcoef(rx, ry)[0, 1])

print("\n== Feature correlation with dstar (Pearson / Spearman) ==")
for k in FEATURES + ["metrics_gap_s"]:
    x = np.array([float(r[k]) if r.get(k) is not None else 0.0 for r in rows])
    pear = float(np.corrcoef(x, ds)[0, 1]) if x.std() > 0 else float("nan")
    spr = spearman(x, ds) if x.std() > 0 else float("nan")
    print(f"  {k:<22} pearson {pear:>7.3f}   spearman {spr:>7.3f}")

# Feature ranges
print("\n== Feature ranges ==")
for k in FEATURES:
    x = np.array([float(r[k]) if r.get(k) is not None else 0.0 for r in rows])
    print(f"  {k:<22} min {x.min():>10.4f}  p50 {q(x,0.5):>10.4f}  p95 {q(x,0.95):>10.4f}  max {x.max():>10.4f}")

# load key used by baseline (b)
load = np.array([float(r["s1_running"] + r["s1_waiting"]) for r in rows])
print(f"\n== s1_running+s1_waiting (load-bucket key): min {load.min():.0f} p50 {q(load,0.5):.0f} "
      f"p90 {q(load,0.9):.0f} max {load.max():.0f}, unique {len(np.unique(load))}")
