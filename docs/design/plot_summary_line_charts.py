# -*- coding: utf-8 -*-
"""Generate summary line charts: E2EL / TTFP / RTF vs stacked features, with concurrency 1/4/10."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import FuncFormatter

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 横轴：Baseline → Batch → CUDA Graph → Async Chunk + Streaming
X_LABELS = ["Baseline", "Batch", "CUDA Graph", "Async Chunk\n+ Streaming Output"]
x = np.arange(len(X_LABELS))

# 与 plot_comparison 一致：蓝、绿，再加一色区分三条线
COLORS = ["#1E90FF", "#32CD32", "#E67E22"]  # blue, green, orange
MARKERS = ["o", "s", "^"]
CONCURRENCIES = [1, 4, 10]

# 按 (stage_order, concurrency) 从 plot_comparison 数据整理
# stage 顺序: Baseline, Batch, Batch+CG, Async Chunk
DATA = {
    "E2EL_ms": {
        1: [325865.36, 258556.1, 67381.19, 60435.5],
        4: [988038.79, 259586.2, 98981.04, 90969.2],
        10: [1523135.4, 262400.45, 153351.92, 130681.82],
    },
    "TTFP_ms": {
        1: [325517.15, 258256.7, 67120.89, 1263.0],
        4: [987707.63, 259342.8, 98678.91, 3174.99],
        10: [1522803.53, 262157.5, 152792.3, 12262.39],
    },
    "RTF": {
        1: [1.52, 1.48, 0.43, 0.33],
        4: [6.66, 1.62, 0.46, 0.43],
        10: [6.94, 2.18, 0.88, 0.74],
    },
}


def y_formatter_ms(x_val, pos):
    if x_val >= 1000:
        return f"{x_val:,.0f}"
    return f"{x_val:.2f}"


def make_line_chart(ylabel, data_by_conc, unit_label, outpath, y_formatter=None, use_log_scale=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    all_vals = [v for vals in data_by_conc.values() for v in vals]
    if use_log_scale:
        ax.set_yscale("log")
        lo, hi = max(1, min(all_vals) * 0.5), max(all_vals) * 2
        ax.set_ylim(lo, hi)
    else:
        y_max = max(all_vals) * 1.12
        ax.set_ylim(0, y_max)

    for idx, conc in enumerate(CONCURRENCIES):
        vals = data_by_conc[conc]
        ax.plot(
            x,
            vals,
            color=COLORS[idx],
            marker=MARKERS[idx],
            markersize=7,
            linewidth=2,
            label=f"Concurrency {conc}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Stacked optimization", fontsize=12)
    if y_formatter and not use_log_scale:
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {outpath}")


def main():
    make_line_chart(
        "End-to-End Latency (ms)",
        DATA["E2EL_ms"],
        "ms",
        FIG_DIR / "Summary_E2EL_ms_vs_features.png",
        y_formatter=y_formatter_ms,
        use_log_scale=True,
    )
    make_line_chart(
        "Time to First Audio (ms)",
        DATA["TTFP_ms"],
        "ms",
        FIG_DIR / "Summary_TTFP_ms_vs_features.png",
        y_formatter=y_formatter_ms,
        use_log_scale=True,
    )
    make_line_chart(
        "Real-Time Factor",
        DATA["RTF"],
        "",
        FIG_DIR / "Summary_RTF_vs_features.png",
    )
    print("All summary line charts saved to docs/design/figures/")


if __name__ == "__main__":
    main()
