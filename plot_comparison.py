# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter

# =======================
# 数据
# =======================
data = [
    ["Batch + CUDA Graph + Async Chunk",1,8666.5,795.22,0.33],
    ["Batch + CUDA Graph + Async Chunk",4,11463.45,1184.42,0.43],
    ["Batch + CUDA Graph + Async Chunk",10,17407.82,2156.9,0.64],

    ["Batch",1,36746.06,36617.84,1.4],
    ["Batch",4,39559.16,39434.84,1.52],
    ["Batch",10,45076.47,44946.48,1.65],

    ["Batch + CUDA Graph",1,9817.34,9692.73,0.36],
    ["Batch + CUDA Graph",4,11090.13,10961.62,0.41],
    ["Batch + CUDA Graph",10,14576.12,14438.77,0.53],

    ["Baseline",1,37300.42,37173.82,1.42],
    ["Baseline",4,120200.23,120074.43,4.68],
    ["Baseline",10,302722.3,302595.15,11.72],
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

        # 图例在下方
        plt.legend(
            [g1, g2],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False
        )

        plt.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()

        filename = f"{metric}_{g1.replace(' ','_')}_vs_{g2.replace(' ','_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved {filename}")

print("All comparison plots saved.")
