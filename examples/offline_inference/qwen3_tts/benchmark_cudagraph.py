"""Benchmark CUDA graph vs eager mode for Qwen3 TTS.

Runs N prompts from benchmark_prompts.txt (or inline defaults) with and without
CUDA graphs on the Talker stage, then prints a latency summary.

Usage:
    python benchmark_cudagraph.py [--num-prompts N] [--warmup W] [--mode cuda|eager|both]
"""

import os
import time

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import statistics
import tempfile
from pathlib import Path

import yaml

from vllm_omni import Omni

SCRIPT_DIR = Path(__file__).parent
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
BASE_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "vllm_omni/model_executor/stage_configs/qwen3_tts.yaml"
PROMPTS_FILE = SCRIPT_DIR / "benchmark_prompts.txt"

DEFAULT_PROMPTS = [
    "Hello, welcome to the voice synthesis benchmark test.",
    "She said she would be here by noon, but nobody showed up.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "I can't believe how beautiful the sunset looks from up here on the mountain.",
    "Please remember to bring your identification documents to the appointment tomorrow morning.",
]


def load_prompts(n: int) -> list[str]:
    if PROMPTS_FILE.exists():
        lines = [line.strip() for line in PROMPTS_FILE.read_text().splitlines() if line.strip()]
    else:
        lines = DEFAULT_PROMPTS
    # Cycle if n > available
    prompts = []
    for i in range(n):
        prompts.append(lines[i % len(lines)])
    return prompts


def build_input(text: str, model_name: str) -> dict:
    from transformers import AutoTokenizer

    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
        Qwen3TTSTalkerForConditionalGeneration,
    )

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    cfg = Qwen3TTSConfig.from_pretrained(model_name, trust_remote_code=True)
    tcfg = getattr(cfg, "talker_config", None)

    additional_information = {
        "task_type": ["CustomVoice"],
        "text": [text],
        "language": ["English"],
        "speaker": ["Ryan"],
        "instruct": [""],
        "max_new_tokens": [2048],
    }
    prompt_len = Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
        additional_information=additional_information,
        task_type="CustomVoice",
        tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
        codec_language_id=getattr(tcfg, "codec_language_id", None),
        spk_is_dialect=getattr(tcfg, "spk_is_dialect", None),
    )
    return {
        "prompt_token_ids": [0] * prompt_len,
        "additional_information": additional_information,
    }


def make_config(enforce_eager_stage0: bool) -> str:
    """Write a temp yaml with enforce_eager set for stage 0 and return its path."""
    with open(BASE_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    cfg["stage_args"][0]["engine_args"]["enforce_eager"] = enforce_eager_stage0

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix=f"qwen3_tts_{'eager' if enforce_eager_stage0 else 'cudagraph'}_"
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def run_benchmark(label: str, enforce_eager: bool, prompts: list[str], warmup: int) -> list[float]:
    config_path = make_config(enforce_eager_stage0=enforce_eager)
    print(f"\n{'=' * 60}")
    print(f"  Config: {label}  (enforce_eager stage-0={enforce_eager})")
    print(f"  Prompts: {len(prompts)}  Warmup: {warmup}")
    print(f"{'=' * 60}")

    # Pre-build all inputs before starting Omni (avoids counting model-download time)
    print("  Building inputs...")
    inputs = [build_input(p, MODEL_NAME) for p in prompts]

    print("  Initializing Omni (stage init may take ~30-60s for CUDA graph capture)...")
    t0 = time.perf_counter()
    omni = Omni(model=MODEL_NAME, stage_configs_path=config_path, log_stats=False, stage_init_timeout=600)
    init_s = time.perf_counter() - t0
    print(f"  Init time: {init_s:.1f}s")

    latencies = []
    for i, inp in enumerate(inputs):
        tag = "WARMUP" if i < warmup else f"run {i - warmup + 1}/{len(prompts) - warmup}"
        t_start = time.perf_counter()
        for stage_outputs in omni.generate([inp], sampling_params_list=None):
            pass  # consume generator fully
        elapsed = time.perf_counter() - t_start
        print(f"  [{tag}] {elapsed:.3f}s  — '{prompts[i][:50]}...'")
        if i >= warmup:
            latencies.append(elapsed)

    try:
        os.unlink(config_path)
    except OSError:
        pass

    return latencies


def print_stats(label: str, latencies: list[float]):
    if not latencies:
        print(f"  {label}: no data")
        return
    s = sorted(latencies)
    print(f"\n  {label}")
    print(
        f"    n={len(s)}  mean={statistics.mean(s):.3f}s  median={statistics.median(s):.3f}s"
        f"  stdev={statistics.stdev(s) if len(s) > 1 else 0:.3f}s"
    )
    print(f"    min={s[0]:.3f}s  p90={s[int(len(s) * 0.9)]:.3f}s  max={s[-1]:.3f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA graph vs eager for Qwen3 TTS")
    parser.add_argument(
        "--num-prompts", type=int, default=6, help="Total prompts to run per config (includes warmup, default: 6)"
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests to discard (default: 1)")
    parser.add_argument(
        "--mode", choices=["cuda", "eager", "both"], default="both", help="Which config(s) to run (default: both)"
    )
    args = parser.parse_args()

    prompts = load_prompts(args.num_prompts)

    results = {}
    if args.mode in ("cuda", "both"):
        results["CUDA graphs (enforce_eager=false)"] = run_benchmark(
            "CUDA graphs (enforce_eager=false)", enforce_eager=False, prompts=prompts, warmup=args.warmup
        )
    if args.mode in ("eager", "both"):
        results["Eager (enforce_eager=true)"] = run_benchmark(
            "Eager (enforce_eager=true)", enforce_eager=True, prompts=prompts, warmup=args.warmup
        )

    print(f"\n{'=' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for label, lats in results.items():
        print_stats(label, lats)

    if len(results) == 2:
        vals = list(results.values())
        a, b = statistics.mean(vals[0]), statistics.mean(vals[1])
        speedup = b / a if a > 0 else float("nan")
        print(f"\n  Speedup (CUDA graphs vs eager): {speedup:.2f}x  ({(speedup - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
