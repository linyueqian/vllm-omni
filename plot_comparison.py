# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.ticker import FuncFormatter

# Save to blog figures dir with names expected by qwen3_omni_blog.md
FIG_DIR = Path(__file__).resolve().parent / "docs" / "design" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =======================
# 数据
# =======================
data = [
    ["Batch + CUDA Graph + Async Chunk",1,60435.5,1263,0.33],
    ["Batch + CUDA Graph + Async Chunk",4,90969.2,3174.99,0.43],
    ["Batch + CUDA Graph + Async Chunk",10,130681.82,12262.39,0.74],

    ["Batch",1,258556.1,258256.7,1.48],
    ["Batch",4,259586.2,259342.8,1.62],
    ["Batch",10,262400.45,262157.5,2.18],

    ["Batch + CUDA Graph",1,67381.19,67120.89,0.43],
    ["Batch + CUDA Graph",4,98981.04,98678.91,0.46],
    ["Batch + CUDA Graph",10,153351.92,152792.3,0.88],

    ["Baseline",1,325865.36,325517.15,1.52],
    ["Baseline",4,988038.79,987707.63,6.66],
    ["Baseline",10,1523135.4,1522803.53,6.94],
]

df = pd.DataFrame(
    data,
    columns=[
        "stage",
        "concurrency",
        "Mean E2EL (ms)",
        "Mean AUDIO_TTFP (ms)",
        "Mean AUDIO_RTF"
    ]
)

# =======================
# 分组
# =======================
groups = [
    ("Baseline","Batch"),
    ("Batch","Batch + CUDA Graph"),
    ("Batch + CUDA Graph","Batch + CUDA Graph + Async Chunk")
]

metrics = [
    "Mean E2EL (ms)",
    "Mean AUDIO_TTFP (ms)",
    "Mean AUDIO_RTF"
]

colors = ["#1E90FF", "#32CD32"]  # 蓝色 & 绿色

# =======================
# Y轴格式化
# =======================
def y_formatter(x, pos):
    if x >= 1000:
        return f"{x:,.0f}"
    else:
        return f"{x:.2f}"

# =======================
# 绘图
# =======================
for g1, g2 in groups:
    for metric in metrics:

        plt.figure(figsize=(6,4))

        # 横坐标等间距
        x = np.arange(3)

        # 柱子宽度 & 间隙
        width = 0.32
        gap = 0.08              # 👈 控制蓝绿柱子之间间距
        offset = width/2 + gap/2

        vals1 = df[df["stage"]==g1][metric].values
        vals2 = df[df["stage"]==g2][metric].values

        bars1 = plt.bar(x - offset, vals1, width=width, color=colors[0])
        bars2 = plt.bar(x + offset, vals2, width=width, color=colors[1])

        # 自动调整Y轴上限
        max_val = max(max(vals1), max(vals2))
        plt.ylim(0, max_val * 1.12)

        # Y轴加逗号
        plt.gca().yaxis.set_major_formatter(FuncFormatter(y_formatter))

        # =======================
        # 添加柱顶数字
        # =======================
        for bar in bars1:
            height = bar.get_height()
            label = f"{height:,.2f}" if height >= 1000 else f"{height:.2f}"
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height,
                label,
                ha="center",
                va="bottom",
                fontsize=8
            )

        for bar in bars2:
            height = bar.get_height()
            label = f"{height:,.2f}" if height >= 1000 else f"{height:.2f}"
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height,
                label,
                ha="center",
                va="bottom",
                fontsize=8
            )

        # 坐标轴设置
        plt.xlabel("Concurrency")
        plt.ylabel(metric)
        plt.title(f"{metric} comparison: {g1} vs {g2}")

        plt.xticks(x, [1,4,10])
        plt.xlim(-0.6, 2.6)

        # 图例在下方；Async Chunk 组显示为 + Streaming Output 与 blog 一致
        legend_g1 = g1
        legend_g2 = "Batch + CUDA Graph + Async Chunk + Streaming Output" if g2 == "Batch + CUDA Graph + Async Chunk" else g2
        plt.legend(
            [legend_g1, legend_g2],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False
        )

        plt.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()

        # Blog expects exact names: Mean_E2EL_ms_Baseline_vs_Batch.png, etc.
        metric_key = {"Mean E2EL (ms)": "Mean_E2EL_ms", "Mean AUDIO_TTFP (ms)": "Mean_AUDIO_TTFP_ms", "Mean AUDIO_RTF": "Mean_AUDIO_RTF"}[metric]
        group_key = {"Baseline": "Baseline_vs_Batch", "Batch": "Batch_vs_Batch_CUDA_Graph", "Batch + CUDA Graph": "Batch_CUDA_Graph_vs_Async_Chunk"}[g1]
        filename = FIG_DIR / f"{metric_key}_{group_key}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved {filename}")

print("All comparison plots saved.")
