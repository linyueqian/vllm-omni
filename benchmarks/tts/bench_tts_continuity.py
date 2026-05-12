"""Self-contained TTS benchmark client.

Posts text prompts to an OpenAI-compatible /v1/audio/speech endpoint with controlled
concurrency, measures per-request latency / streaming metrics, and emits aggregate
JSON in the same shape regardless of server version.

Metrics per request (all wall-clock):
  ttft_ms    : request_start → first response byte
  ttfp_ms    : request_start → first non-empty audio chunk (post HTTP header)
  e2el_ms    : request_start → last byte
  tpot_ms    : (e2el - ttft) / max(1, num_chunks)
  audio_duration_s : total_audio_bytes / (sample_rate * 2)
  audio_rtf  : e2el_s / audio_duration_s

Aggregates: mean, median, p99 via numpy. Output JSON stable across vllm-omni versions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

import aiohttp


# Default text prompts — embed locally to avoid v0.18/v0.20 dataset divergence.
DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog near the riverbank at sunset.",
    "Welcome to the future of text-to-speech synthesis in production systems.",
    "Yesterday the team finished rolling out the new authentication flow for real.",
    "The sunset painted the sky in brilliant shades of orange and pink today.",
    "Please remain seated until the captain has turned off the seatbelt sign.",
    "She quickly adjusted the radar and called the control tower for a status update.",
    "Researchers found that the new model handled long contexts better than expected.",
    "I would like a tall iced latte with oat milk and one extra shot of espresso, please.",
    "The autumn leaves drifted slowly past the open kitchen window this morning.",
    "After the meeting concluded, we walked back to the office through the park.",
]


async def one_request(session: aiohttp.ClientSession, url: str, payload: dict, sample_rate: int) -> dict:
    start = time.perf_counter()
    ttft_ms = ttfp_ms = e2el_ms = None
    audio_bytes = 0
    num_chunks = 0
    saw_first_byte = False
    saw_first_audio = False
    chunk_timeline = []  # list of (t_ms_since_start, bytes_in_chunk)

    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {body[:300]}")
        async for chunk in resp.content.iter_any():
            now = time.perf_counter()
            t_ms = (now - start) * 1000.0
            if not saw_first_byte:
                ttft_ms = t_ms
                saw_first_byte = True
            if not saw_first_audio and chunk:
                ttfp_ms = t_ms
                saw_first_audio = True
            if chunk:
                chunk_timeline.append((t_ms, len(chunk)))
            audio_bytes += len(chunk)
            num_chunks += 1
        e2el_ms = (time.perf_counter() - start) * 1000.0

    if not saw_first_audio:
        raise RuntimeError("no audio bytes received")
    audio_duration_s = audio_bytes / (sample_rate * 2.0)  # PCM16 mono
    rtf = (e2el_ms / 1000.0) / max(audio_duration_s, 1e-6)
    tpot_ms = (e2el_ms - (ttft_ms or 0.0)) / max(1, num_chunks)

    # Streaming continuity metrics
    # Underrun: at time t, cumulative bytes received < cumulative bytes the
    # player has consumed since first audio chunk. Player consumes at
    # sample_rate*2 bytes/sec from ttfp_ms onward.
    bytes_per_ms = sample_rate * 2.0 / 1000.0
    cum_bytes = 0
    max_underrun_ms = 0.0
    num_underrun_chunks = 0
    max_inter_chunk_gap_ms = 0.0
    prev_t = None
    for t_ms, b in chunk_timeline:
        cum_bytes += b
        # Bytes the player has already consumed at this t_ms
        played_bytes = max(0.0, (t_ms - ttfp_ms)) * bytes_per_ms
        # Negative buffer = underrun. Convert deficit (bytes) to ms of silence.
        deficit_bytes = played_bytes - cum_bytes
        if deficit_bytes > 0:
            deficit_ms = deficit_bytes / bytes_per_ms
            if deficit_ms > max_underrun_ms:
                max_underrun_ms = deficit_ms
            num_underrun_chunks += 1
        if prev_t is not None:
            gap = t_ms - prev_t
            if gap > max_inter_chunk_gap_ms:
                max_inter_chunk_gap_ms = gap
        prev_t = t_ms

    return {
        "ttft_ms": ttft_ms,
        "ttfp_ms": ttfp_ms,
        "e2el_ms": e2el_ms,
        "tpot_ms": tpot_ms,
        "audio_duration_s": audio_duration_s,
        "audio_rtf": rtf,
        "audio_bytes": audio_bytes,
        "num_chunks": num_chunks,
        "max_underrun_ms": max_underrun_ms,
        "num_underrun_chunks": num_underrun_chunks,
        "max_inter_chunk_gap_ms": max_inter_chunk_gap_ms,
        "continuity_ok": max_underrun_ms < 100.0,  # 100ms slack
    }


async def run_concurrent(url: str, payloads: Sequence[dict], sample_rate: int, concurrency: int, label: str) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    failures = 0
    failure_msgs: list[str] = []

    async with aiohttp.ClientSession() as session:
        async def _wrap(idx: int, p: dict):
            nonlocal failures
            async with semaphore:
                try:
                    r = await one_request(session, url, p, sample_rate)
                    r["request_index"] = idx
                    results.append(r)
                except Exception as e:
                    failures += 1
                    failure_msgs.append(str(e))

        tasks = [asyncio.create_task(_wrap(i, p)) for i, p in enumerate(payloads)]
        wall_start = time.perf_counter()
        await asyncio.gather(*tasks)
        wall_s = time.perf_counter() - wall_start

    if failures:
        print(f"  [{label}] {failures}/{len(payloads)} failures; first: {failure_msgs[:2]}")

    print(f"  [{label}] {len(results)} OK in {wall_s:.1f}s ({len(results)/max(wall_s,1e-6):.2f} req/s)")
    return results


def _agg(values: list[float], key: str) -> dict:
    if not values:
        return {f"mean_{key}": None, f"median_{key}": None, f"p99_{key}": None}
    sv = sorted(values)
    return {
        f"mean_{key}": statistics.fmean(values),
        f"median_{key}": statistics.median(values),
        f"p99_{key}": sv[max(0, int(0.99 * len(sv)) - 1)],
        f"min_{key}": sv[0],
        f"max_{key}": sv[-1],
    }


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {"num_requests_ok": 0}
    out: dict = {"num_requests_ok": len(results)}
    for k in ("ttft_ms", "ttfp_ms", "e2el_ms", "tpot_ms", "audio_duration_s", "audio_rtf", "max_underrun_ms", "max_inter_chunk_gap_ms"):
        vals = [r.get(k) for r in results if r.get(k) is not None]
        if vals:
            out.update(_agg(vals, k))
    # Per-cell continuity verdict
    underruns = [r.get("max_underrun_ms", 0) for r in results]
    out["underrun_rate"] = sum(1 for u in underruns if u >= 100.0) / len(underruns)
    out["num_continuity_ok"] = sum(1 for r in results if r.get("continuity_ok", False))
    return out


async def main_async(args):
    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [
            line.split("|||")[-1].strip()
            for line in Path(args.prompts_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    if args.extra_body_file is not None:
        args.extra_body = args.extra_body_file.read_text()
    base_payload = {
        "model": args.model,
        "input": "",
        "response_format": args.response_format,
        "stream": True,
        "stream_options": {"include_usage": False},
    }
    if args.extra_body:
        base_payload.update(json.loads(args.extra_body))

    payloads = []
    n = max(args.num_prompts, 1)
    for i in range(n):
        p = dict(base_payload)
        p["input"] = prompts[i % len(prompts)]
        payloads.append(p)

    url = f"{args.base_url}/v1/audio/speech"

    # Warmup
    if args.num_warmups > 0:
        warmup_payloads = payloads[: args.num_warmups]
        print(f"[warmup x{args.num_warmups}]")
        await run_concurrent(url, warmup_payloads, args.sample_rate, 1, "warmup")

    print(f"[bench c={args.concurrency} n={n}]")
    results = await run_concurrent(url, payloads, args.sample_rate, args.concurrency, "bench")

    agg = aggregate(results)
    agg.update({
        "model": args.model,
        "concurrency": args.concurrency,
        "num_prompts": n,
        "num_warmups": args.num_warmups,
        "extra_body": json.loads(args.extra_body) if args.extra_body else {},
        "base_url": args.base_url,
        "sample_rate": args.sample_rate,
        "response_format": args.response_format,
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(agg, indent=2))
    print(f"\nwrote {args.output}\n  TTFT median={agg.get('median_ttft_ms', 0):.0f}ms  "
          f"TTFP median={agg.get('median_ttfp_ms', 0):.0f}ms  "
          f"E2EL median={agg.get('median_e2el_ms', 0):.0f}ms  "
          f"RTF median={agg.get('median_audio_rtf', 0):.3f}")
    return 0 if agg["num_requests_ok"] >= max(1, args.num_prompts // 2) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--num-warmups", type=int, default=2)
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument("--extra-body", default="")
    parser.add_argument("--extra-body-file", type=Path, default=None)
    parser.add_argument("--response-format", default="pcm")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rc = asyncio.run(main_async(args))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
