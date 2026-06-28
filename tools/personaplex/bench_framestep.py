# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure per-frame step latency vs the real-time budget (80ms @ 12.5Hz).

If a single step() exceeds frame_size/sample_rate seconds, the duplex stream
cannot keep up and the agent audio comes out choppy (时断时续).

    CUDA_VISIBLE_DEVICES=2 python tools/personaplex/bench_framestep.py --native 1 --frames 200
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", type=int, default=1)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    import torch

    from vllm_omni.experimental.fullduplex.personaplex.config import PersonaPlexConfig
    from vllm_omni.experimental.fullduplex.personaplex.engine import PersonaPlexEngine

    cfg = PersonaPlexConfig(use_native_components=bool(args.native), greedy=True, seed=42)
    eng = PersonaPlexEngine(cfg).load()
    eng.open_session()
    fs, sr = eng.frame_size, eng.sample_rate
    budget_ms = 1000.0 * fs / sr
    silence = np.zeros(fs, dtype=np.float32)

    times: list[float] = []
    for i in range(args.frames + args.warmup):
        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        eng.step(silence)
        torch.accelerator.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= args.warmup:
            times.append(dt)

    times.sort()
    p50 = statistics.median(times)
    p95 = times[int(0.95 * len(times)) - 1]
    mean = statistics.mean(times)
    over = sum(1 for t in times if t > budget_ms)
    rtf = mean / budget_ms
    print(f"native={bool(args.native)} frames={len(times)} budget={budget_ms:.1f}ms")
    print(f"  mean={mean:.1f}ms  p50={p50:.1f}ms  p95={p95:.1f}ms  max={times[-1]:.1f}ms")
    print(f"  RTF={rtf:.2f}  ({'REAL-TIME OK' if rtf < 1 else 'TOO SLOW'})  over-budget frames={over}/{len(times)}")


if __name__ == "__main__":
    main()
