"""Legato week-0 benchmark client: PCM-exact audio accounting for streaming TTS.

Measures, per request, with no assumed chunk durations:
  - arrival wall time of every audio emission
  - exact PCM bytes per emission -> audio seconds (16-bit mono @ sample rate)
  - TTFA, generation-vs-realtime ratio
  - exact playback stall simulation for several jitter buffers

Usage:
  python legato_bench.py --url http://localhost:8091 --n 4 --out bench.ndjson
"""

import argparse
import base64
import json
import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

BYTES_PER_SECOND = 24000 * 2  # 24 kHz, 16-bit mono PCM

AUDIO_KEYS = ("audio", "audio_b64", "data", "delta")


def _extract_audio_b64(obj: dict[str, Any]) -> str | None:
    """Find a base64 audio payload under common key shapes."""
    for key in AUDIO_KEYS:
        val = obj.get(key)
        if isinstance(val, str) and len(val) > 16:
            return val
        if isinstance(val, dict):
            inner = _extract_audio_b64(val)
            if inner:
                return inner
    return None


def stream_once(url: str, text: str, voice: str, timeout: float) -> dict[str, Any]:
    """Run one streaming speech request, return exact emission timeline."""
    payload = {
        "input": text,
        "task_type": "VoiceDesign",
        "instructions": voice,
        "response_format": "pcm",
        "stream": True,
    }
    t0 = time.perf_counter()
    emissions: list[dict[str, float]] = []
    with requests.post(
        f"{url.rstrip('/')}/v1/audio/speech",
        json=payload,
        stream=True,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        if "event-stream" in ctype:
            for line in resp.iter_lines():
                if not line or not line.startswith(b"data:"):
                    continue
                body = line[5:].strip()
                if body == b"[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue
                b64 = _extract_audio_b64(obj)
                if b64 is None:
                    continue
                nbytes = len(base64.b64decode(b64))
                emissions.append(
                    {"t": time.perf_counter() - t0, "bytes": float(nbytes)}
                )
        else:
            for chunk in resp.iter_content(chunk_size=None):
                if chunk:
                    emissions.append(
                        {"t": time.perf_counter() - t0, "bytes": float(len(chunk))}
                    )
    return {"status": "ok", "content_type": ctype, "emissions": emissions}


def simulate_playback(emissions: list[dict[str, float]], buffer_s: float) -> dict[str, float]:
    """Exact stall simulation: playback starts at first-audio + buffer.

    The player consumes audio continuously; a stall occurs whenever the play
    cursor reaches the end of delivered audio before the next emission arrives.
    """
    if not emissions:
        return {"stalls": 0.0, "stall_time": 0.0}
    start = emissions[0]["t"] + buffer_s
    cursor = start  # wall time at which the player is about to need new audio
    delivered_until = start  # wall time up to which playback is covered
    stalls = 0
    stall_time = 0.0
    for e in emissions:
        dur = e["bytes"] / BYTES_PER_SECOND
        avail = max(e["t"], cursor)
        if e["t"] > delivered_until and delivered_until > start:
            stalls += 1
            stall_time += e["t"] - delivered_until
            cursor = e["t"]
        delivered_until = max(delivered_until, avail) + dur
        cursor = min(cursor, delivered_until)
    return {"stalls": float(stalls), "stall_time": round(stall_time, 4)}


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    ems = result["emissions"]
    if not ems:
        return {"status": "empty"}
    total_audio = sum(e["bytes"] for e in ems) / BYTES_PER_SECOND
    ttfa = ems[0]["t"]
    wall = ems[-1]["t"]
    gaps = [round(b["t"] - a["t"], 4) for a, b in zip(ems, ems[1:])]
    durs = [round(e["bytes"] / BYTES_PER_SECOND, 4) for e in ems]
    out = {
        "status": "ok",
        "n_emissions": len(ems),
        "ttfa_s": round(ttfa, 4),
        "wall_s": round(wall, 4),
        "audio_s": round(total_audio, 4),
        "rtf": round((wall - ttfa) / total_audio, 4) if total_audio else None,
        "chunk_audio_s": durs,
        "arrival_gaps_s": gaps,
    }
    for b in (0.1, 0.3, 0.5):
        out[f"stall_b{int(b * 1000)}ms"] = simulate_playback(ems, b)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8091")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument(
        "--text",
        default=(
            "The quick brown fox jumps over the lazy dog while the river "
            "keeps moving under the old stone bridge. Every evening the town "
            "grows quiet, and the lamps come on one by one along the water."
        ),
    )
    parser.add_argument(
        "--voice", default="A warm female voice with clear pronunciation"
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", default="bench.ndjson")
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="number of simultaneous streams (each runs --n requests back to back)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("warmup request...")
    warm = stream_once(args.url, "Warm up sentence for the compiler.", args.voice, args.timeout)
    logger.info("warmup: %d emissions, ct=%s", len(warm["emissions"]), warm["content_type"])

    def worker(stream_id: int) -> list[dict[str, Any]]:
        rows = []
        for i in range(args.n):
            res = stream_once(args.url, args.text, args.voice, args.timeout)
            summary = summarize(res)
            summary["stream"] = stream_id
            summary["req"] = i
            rows.append(summary)
        return rows

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        all_rows = [r for rows in pool.map(worker, range(args.concurrency)) for r in rows]

    with open(args.out, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    ok = [r for r in all_rows if r.get("status") == "ok"]
    if ok:
        agg = {
            "concurrency": args.concurrency,
            "n_ok": len(ok),
            "ttfa_med": round(sorted(r["ttfa_s"] for r in ok)[len(ok) // 2], 4),
            "rtf_med": round(sorted(r["rtf"] for r in ok)[len(ok) // 2], 4),
        }
        for b in (100, 300, 500):
            key = f"stall_b{b}ms"
            stalled = [r for r in ok if r[key]["stalls"] > 0]
            agg[f"stalled_frac_b{b}"] = round(len(stalled) / len(ok), 3)
            agg[f"stall_time_total_b{b}"] = round(
                sum(r[key]["stall_time"] for r in ok), 3
            )
        logger.info("AGGREGATE %s", json.dumps(agg))


if __name__ == "__main__":
    main()
