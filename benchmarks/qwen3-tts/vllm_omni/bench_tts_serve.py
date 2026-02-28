"""Benchmark client for Qwen3-TTS via /v1/audio/speech endpoint.

Measures TTFP (Time-to-First-Packet), E2E latency, and RTF (Real-Time Factor)
across configurable concurrency levels. Saves results as JSON for plotting.

Usage:
    python bench_tts_serve.py \
        --host 127.0.0.1 --port 8000 \
        --num-prompts 50 \
        --max-concurrency 1 4 10 \
        --result-dir results/
"""

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

PROMPTS = [
    "Hello, welcome to the voice synthesis benchmark test.",
    "She said she would be here by noon, but nobody showed up.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "I can't believe how beautiful the sunset looks from up here on the mountain.",
    "Please remember to bring your identification documents to the appointment tomorrow morning.",
    "Have you ever wondered what it would be like to travel through time and visit ancient civilizations?",
    "The restaurant on the corner serves the best pasta I have ever tasted in my entire life.",
    "After the meeting, we should discuss the quarterly results and plan for the next phase.",
    "Learning a new language takes patience, practice, and a genuine curiosity about other cultures.",
    "The train leaves at half past seven, so we need to arrive at the station before then.",
    "Could you please turn down the music a little bit, I'm trying to concentrate on my work.",
    "It was a dark and stormy night when the old lighthouse keeper heard a knock at the door.",
]


@dataclass
class RequestResult:
    success: bool = False
    ttfp: float = 0.0  # Time to first audio packet (seconds)
    e2e: float = 0.0  # End-to-end latency (seconds)
    audio_bytes: int = 0  # Total audio bytes received
    audio_duration: float = 0.0  # Audio duration in seconds (estimated from PCM)
    rtf: float = 0.0  # Real-time factor = e2e / audio_duration
    prompt: str = ""
    error: str = ""


@dataclass
class BenchmarkResult:
    config_name: str = ""
    concurrency: int = 0
    num_prompts: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    # TTFP stats (ms)
    mean_ttfp_ms: float = 0.0
    median_ttfp_ms: float = 0.0
    std_ttfp_ms: float = 0.0
    p90_ttfp_ms: float = 0.0
    p95_ttfp_ms: float = 0.0
    p99_ttfp_ms: float = 0.0
    # E2E stats (ms)
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    std_e2e_ms: float = 0.0
    p90_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    # RTF stats
    mean_rtf: float = 0.0
    median_rtf: float = 0.0
    std_rtf: float = 0.0
    # Audio stats
    mean_audio_duration_s: float = 0.0
    total_audio_duration_s: float = 0.0
    audio_throughput: float = 0.0  # audio_duration / wall_time
    request_throughput: float = 0.0  # requests / second
    # Per-request details
    per_request: list = field(default_factory=list)


def pcm_bytes_to_duration(num_bytes: int, sample_rate: int = 24000, sample_width: int = 2) -> float:
    """Convert raw PCM byte count to duration in seconds."""
    num_samples = num_bytes / sample_width
    return num_samples / sample_rate


async def send_tts_request(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt: str,
    voice: str = "vivian",
    language: str = "English",
    pbar: tqdm | None = None,
) -> RequestResult:
    """Send a streaming TTS request and measure latency metrics."""
    payload = {
        "input": prompt,
        "voice": voice,
        "language": language,
        "stream": True,
        "response_format": "pcm",
    }

    result = RequestResult(prompt=prompt)
    st = time.perf_counter()

    try:
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
                result.success = False
                return result

            first_chunk = True
            total_bytes = 0

            async for chunk in response.content.iter_any():
                if first_chunk and len(chunk) > 0:
                    result.ttfp = time.perf_counter() - st
                    first_chunk = False
                total_bytes += len(chunk)

            result.e2e = time.perf_counter() - st
            result.audio_bytes = total_bytes
            result.audio_duration = pcm_bytes_to_duration(total_bytes)

            if result.audio_duration > 0:
                result.rtf = result.e2e / result.audio_duration
            result.success = True

    except Exception as e:
        result.error = str(e)
        result.success = False
        result.e2e = time.perf_counter() - st

    if pbar:
        pbar.update(1)
    return result


async def run_benchmark(
    host: str,
    port: int,
    num_prompts: int,
    max_concurrency: int,
    num_warmups: int = 3,
    voice: str = "vivian",
    language: str = "English",
) -> BenchmarkResult:
    """Run benchmark at a given concurrency level."""
    api_url = f"http://{host}:{port}/v1/audio/speech"

    connector = aiohttp.TCPConnector(
        limit=max_concurrency,
        limit_per_host=max_concurrency,
        keepalive_timeout=60,
    )
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=600),
    )

    # Warmup
    if num_warmups > 0:
        print(f"  Warming up with {num_warmups} requests...")
        warmup_tasks = []
        for i in range(num_warmups):
            prompt = PROMPTS[i % len(PROMPTS)]
            warmup_tasks.append(send_tts_request(session, api_url, prompt, voice, language))
        await asyncio.gather(*warmup_tasks)
        print("  Warmup done.")

    # Build request list
    request_prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_prompts)]

    # Run benchmark
    print(f"  Running {num_prompts} requests with concurrency={max_concurrency}...")
    semaphore = asyncio.Semaphore(max_concurrency)
    pbar = tqdm(total=num_prompts, desc=f"  concurrency={max_concurrency}")

    async def limited_request(prompt):
        async with semaphore:
            return await send_tts_request(session, api_url, prompt, voice, language, pbar)

    start_time = time.perf_counter()
    tasks = [asyncio.create_task(limited_request(p)) for p in request_prompts]
    results: list[RequestResult] = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time
    pbar.close()

    await session.close()

    # Compute stats
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    bench = BenchmarkResult(
        concurrency=max_concurrency,
        num_prompts=num_prompts,
        completed=len(successful),
        failed=len(failed),
        duration_s=duration,
    )

    if successful:
        ttfps = [r.ttfp * 1000 for r in successful]  # convert to ms
        e2es = [r.e2e * 1000 for r in successful]
        rtfs = [r.rtf for r in successful]
        audio_durs = [r.audio_duration for r in successful]

        bench.mean_ttfp_ms = float(np.mean(ttfps))
        bench.median_ttfp_ms = float(np.median(ttfps))
        bench.std_ttfp_ms = float(np.std(ttfps))
        bench.p90_ttfp_ms = float(np.percentile(ttfps, 90))
        bench.p95_ttfp_ms = float(np.percentile(ttfps, 95))
        bench.p99_ttfp_ms = float(np.percentile(ttfps, 99))

        bench.mean_e2e_ms = float(np.mean(e2es))
        bench.median_e2e_ms = float(np.median(e2es))
        bench.std_e2e_ms = float(np.std(e2es))
        bench.p90_e2e_ms = float(np.percentile(e2es, 90))
        bench.p95_e2e_ms = float(np.percentile(e2es, 95))
        bench.p99_e2e_ms = float(np.percentile(e2es, 99))

        bench.mean_rtf = float(np.mean(rtfs))
        bench.median_rtf = float(np.median(rtfs))
        bench.std_rtf = float(np.std(rtfs))

        bench.mean_audio_duration_s = float(np.mean(audio_durs))
        bench.total_audio_duration_s = float(np.sum(audio_durs))
        bench.audio_throughput = bench.total_audio_duration_s / duration
        bench.request_throughput = len(successful) / duration

        bench.per_request = [
            {
                "ttfp_ms": r.ttfp * 1000,
                "e2e_ms": r.e2e * 1000,
                "rtf": r.rtf,
                "audio_duration_s": r.audio_duration,
                "prompt": r.prompt,
            }
            for r in successful
        ]

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  Concurrency: {max_concurrency}  |  Completed: {bench.completed}  |  Failed: {bench.failed}")
    print(f"  Duration: {duration:.2f}s  |  Throughput: {bench.request_throughput:.2f} req/s")
    print(
        f"  {'TTFP (ms):':<25} mean={bench.mean_ttfp_ms:.1f}  median={bench.median_ttfp_ms:.1f}"
        f"  p90={bench.p90_ttfp_ms:.1f}  p99={bench.p99_ttfp_ms:.1f}"
    )
    print(
        f"  {'E2E (ms):':<25} mean={bench.mean_e2e_ms:.1f}  median={bench.median_e2e_ms:.1f}"
        f"  p90={bench.p90_e2e_ms:.1f}  p99={bench.p99_e2e_ms:.1f}"
    )
    print(f"  {'RTF:':<25} mean={bench.mean_rtf:.3f}  median={bench.median_rtf:.3f}")
    print(f"  {'Audio throughput:':<25} {bench.audio_throughput:.2f} audio-sec/wall-sec")
    print(f"{'=' * 60}\n")

    if failed:
        for r in failed[:3]:
            print(f"  [ERROR] {r.error[:200]}")

    return bench


async def main(args):
    all_results = []

    for concurrency in args.max_concurrency:
        result = await run_benchmark(
            host=args.host,
            port=args.port,
            num_prompts=args.num_prompts,
            max_concurrency=concurrency,
            num_warmups=args.num_warmups,
            voice=args.voice,
            language=args.language,
        )
        result.config_name = args.config_name
        all_results.append(asdict(result))

    # Save results
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"bench_{args.config_name}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {result_file}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Benchmark Client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-prompts", type=int, default=50, help="Number of prompts per concurrency level")
    parser.add_argument(  # noqa: E501
        "--max-concurrency", type=int, nargs="+", default=[1, 4, 10], help="Concurrency levels to test"
    )
    parser.add_argument("--num-warmups", type=int, default=3)
    parser.add_argument("--voice", type=str, default="vivian")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument(
        "--config-name", type=str, default="async_chunk", help="Label for this config (used in filenames)"
    )
    parser.add_argument("--result-dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
