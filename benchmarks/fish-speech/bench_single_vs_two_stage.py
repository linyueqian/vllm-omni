"""Benchmark: single-stage vs two-stage Fish Speech S2 Pro.

Sends N requests at concurrency 1, measures per-request metrics,
and writes structured JSON results for comparison.

Usage:
    .venv/bin/python benchmarks/fish-speech/bench_single_vs_two_stage.py \
        --port 8091 --tag two-stage --num-prompts 10 --concurrency 1

    .venv/bin/python benchmarks/fish-speech/bench_single_vs_two_stage.py \
        --port 8092 --tag single-stage --num-prompts 10 --concurrency 1
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import requests

SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2  # 16-bit PCM

PROMPTS = [
    "Hello, welcome to the voice synthesis benchmark test.",
    "She said she would be here by noon, but nobody showed up.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "I can't believe how beautiful the sunset looks from up here.",
    "Please remember to bring your identification documents tomorrow morning.",
    "Have you ever wondered what it would be like to travel through time?",
    "The restaurant on the corner serves the best pasta I have ever tasted.",
    "After the meeting, we should discuss the quarterly results.",
    "Learning a new language takes patience and genuine curiosity.",
    "The train leaves at half past seven, so we need to arrive early.",
]


def send_tts_request(
    host: str, port: int, text: str, timeout: float = 120.0
) -> dict:
    """Send a single TTS request and measure timing."""
    url = f"http://{host}:{port}/v1/audio/speech"
    payload = {
        "model": "tts",
        "input": text,
        "voice": "alloy",
        "response_format": "pcm",
        "stream": True,
    }

    t_start = time.perf_counter()
    t_first_byte = None
    audio_bytes = bytearray()

    with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                if t_first_byte is None:
                    t_first_byte = time.perf_counter()
                audio_bytes.extend(chunk)

    t_end = time.perf_counter()

    audio_samples = len(audio_bytes) // SAMPLE_WIDTH
    audio_duration_s = audio_samples / SAMPLE_RATE if SAMPLE_RATE > 0 else 0.0

    ttfb_ms = (t_first_byte - t_start) * 1000 if t_first_byte else 0.0
    e2e_ms = (t_end - t_start) * 1000
    rtf = e2e_ms / 1000.0 / audio_duration_s if audio_duration_s > 0 else float("inf")

    return {
        "text": text,
        "text_len": len(text),
        "ttfb_ms": round(ttfb_ms, 2),
        "e2e_ms": round(e2e_ms, 2),
        "audio_duration_s": round(audio_duration_s, 3),
        "audio_bytes": len(audio_bytes),
        "rtf": round(rtf, 4),
    }


def run_benchmark(
    host: str,
    port: int,
    tag: str,
    num_prompts: int,
    concurrency: int,
) -> dict:
    """Run benchmark and return structured results."""
    prompts = (PROMPTS * ((num_prompts // len(PROMPTS)) + 1))[:num_prompts]

    print(f"\n{'='*60}")
    print(f"  Benchmark: {tag}")
    print(f"  Server: {host}:{port}")
    print(f"  Prompts: {num_prompts}, Concurrency: {concurrency}")
    print(f"{'='*60}\n")

    # Warmup: 2 requests
    print("Warmup (2 requests)...")
    for i in range(2):
        try:
            send_tts_request(host, port, "Warmup request number one.")
        except Exception as e:
            print(f"  Warmup {i} failed: {e}")

    # Actual benchmark
    results = []
    for i, prompt in enumerate(prompts):
        try:
            r = send_tts_request(host, port, prompt)
            results.append(r)
            print(
                f"  [{i+1}/{num_prompts}] "
                f"TTFB={r['ttfb_ms']:.0f}ms  "
                f"E2E={r['e2e_ms']:.0f}ms  "
                f"Audio={r['audio_duration_s']:.2f}s  "
                f"RTF={r['rtf']:.3f}"
            )
        except Exception as e:
            print(f"  [{i+1}/{num_prompts}] FAILED: {e}")
            results.append({"text": prompt, "error": str(e)})

    # Compute aggregates
    valid = [r for r in results if "error" not in r]
    if valid:
        agg = {
            "count": len(valid),
            "mean_ttfb_ms": round(sum(r["ttfb_ms"] for r in valid) / len(valid), 2),
            "mean_e2e_ms": round(sum(r["e2e_ms"] for r in valid) / len(valid), 2),
            "mean_rtf": round(sum(r["rtf"] for r in valid) / len(valid), 4),
            "mean_audio_s": round(
                sum(r["audio_duration_s"] for r in valid) / len(valid), 3
            ),
            "total_audio_s": round(
                sum(r["audio_duration_s"] for r in valid), 3
            ),
            "total_wall_s": round(sum(r["e2e_ms"] for r in valid) / 1000, 3),
            "throughput_audio_s_per_wall_s": round(
                sum(r["audio_duration_s"] for r in valid)
                / (sum(r["e2e_ms"] for r in valid) / 1000),
                4,
            ),
            "p50_e2e_ms": round(
                sorted(r["e2e_ms"] for r in valid)[len(valid) // 2], 2
            ),
            "p90_e2e_ms": round(
                sorted(r["e2e_ms"] for r in valid)[int(len(valid) * 0.9)], 2
            ),
            "p50_ttfb_ms": round(
                sorted(r["ttfb_ms"] for r in valid)[len(valid) // 2], 2
            ),
        }
    else:
        agg = {"count": 0, "error": "all requests failed"}

    output = {
        "tag": tag,
        "config": {
            "host": host,
            "port": port,
            "num_prompts": num_prompts,
            "concurrency": concurrency,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "aggregate": agg,
        "requests": results,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Results: {tag}")
    print(f"{'='*60}")
    if "error" not in agg:
        print(f"  Mean TTFB:     {agg['mean_ttfb_ms']:.0f} ms")
        print(f"  Mean E2E:      {agg['mean_e2e_ms']:.0f} ms")
        print(f"  P50 E2E:       {agg['p50_e2e_ms']:.0f} ms")
        print(f"  P90 E2E:       {agg['p90_e2e_ms']:.0f} ms")
        print(f"  Mean RTF:      {agg['mean_rtf']:.3f}")
        print(f"  Mean Audio:    {agg['mean_audio_s']:.2f} s")
        print(f"  Throughput:    {agg['throughput_audio_s_per_wall_s']:.3f} audio-s/wall-s")
        print(f"  P50 TTFB:      {agg['p50_ttfb_ms']:.0f} ms")
    print()

    return output


def main():
    parser = argparse.ArgumentParser(description="Fish Speech single vs two stage benchmark")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--tag", required=True, help="Label for this run (e.g. two-stage, single-stage)")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output-dir", default="/tmp/fish_bench_results")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output = run_benchmark(args.host, args.port, args.tag, args.num_prompts, args.concurrency)

    out_path = Path(args.output_dir) / f"{args.tag}_c{args.concurrency}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
