# -*- coding: utf-8 -*-
"""Generate vLLM-Omni vs Hugging Face Transformers comparison bar charts for the blog."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

# vLLM-Omni blue, HF transformers yellow for clear contrast
COLORS = ["#1E90FF", "#E6A800"]  # blue, yellow/amber
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def make_chart(title, ylabel, labels, values, outpath, use_log_scale=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=COLORS, width=0.6, edgecolor="gray", linewidth=0.5)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("System", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if use_log_scale:
        ax.set_yscale("log")
        lo, hi = min(values), max(values)
        ax.set_ylim(lo * 0.5, hi * 2)
    else:
        ax.set_ylim(0, max(values) * 1.12)

    for bar, v in zip(bars, values):
        label = f"{v:.2f}" if v < 100 else f"{v:,.0f}"
        y_pos = bar.get_height()
        if use_log_scale and y_pos < ax.get_ylim()[1] / 20:
            y_pos = y_pos * 1.4  # place label above bar so it's visible
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {outpath}")


def main():
    labels = ["vLLM-Omni (streaming)", "HF transformers (offline)"]

    make_chart(
        "E2E Latency (Single Request)",
        "End-to-End Latency (s)",
        labels,
        [23.78, 336.10],
        FIG_DIR / "E2EL_s_vllm_omni_vs_transformers.png",
        use_log_scale=True,
    )
    make_chart(
        "Streaming Latency (TTFP)",
        "Time to First Audio (s)",
        labels,
        [0.934, 336.10],
        FIG_DIR / "TTFP_s_vllm_omni_vs_transformers.png",
        use_log_scale=True,
    )
    make_chart(
        "Real-Time Factor (RTF) (Single Request)",
        "Real-Time Factor",
        labels,
        [0.32, 3.776],
        FIG_DIR / "RTF_vllm_omni_vs_transformers.png",
    )
    print("All three charts saved to docs/design/figures/")


if __name__ == "__main__":
    main()
