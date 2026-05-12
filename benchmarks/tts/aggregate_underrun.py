"""Aggregate per-chunk underrun stats from bench_tts_continuity output cells.

Usage:

    python aggregate_underrun.py <results_root> [<results_root> ...]
        [--prefix <prefix>=<label> ...]
        [--output <path>]

Each ``<results_root>`` is a directory of subdirectories where each subdirectory
is one experiment cell (e.g. ``<root>/V0_default__base_voice_clone/c128.json``).
Subdirectory names are split as ``<prefix><task>`` so a per-experiment label
is recovered for the report.

Example:

    python aggregate_underrun.py /path/to/bench/continuity \
        --prefix V0_default_="default" \
        --prefix V_two_gpu_="2-GPU split"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def collect_rows(roots: list[str], prefixes: list[tuple[str, str]]) -> list[dict]:
    rows: list[dict] = []
    for root in roots:
        if not os.path.isdir(root):
            print(f"warn: {root} is not a directory; skipping", file=sys.stderr)
            continue
        for d in sorted(os.listdir(root)):
            full = os.path.join(root, d)
            if not os.path.isdir(full):
                continue
            label = None
            task = None
            for pre, name in prefixes:
                if d.startswith(pre):
                    label = name
                    task = d[len(pre) :]
                    break
            if label is None:
                # Default behavior: use the directory name as label, no task split.
                label = d
                task = ""
            for f in sorted(glob.glob(os.path.join(full, "c*.json"))):
                try:
                    c = int(os.path.basename(f).split(".")[0][1:])
                except ValueError:
                    continue
                data = json.load(open(f))
                n = data.get("num_requests_ok", 0)
                if not n:
                    continue
                undr_rate = data.get("underrun_rate")
                rows.append(
                    {
                        "config": label,
                        "task": task,
                        "c": c,
                        "n": n,
                        "ttft_p50": data.get("median_ttft_ms"),
                        "ttft_p99": data.get("p99_ttft_ms"),
                        "ttft_tail": (data.get("p99_ttft_ms") / data.get("median_ttft_ms"))
                        if data.get("median_ttft_ms")
                        else None,
                        "rtf_p50": data.get("median_audio_rtf"),
                        "rtf_p99": data.get("p99_audio_rtf"),
                        "max_underrun_p50": data.get("median_max_underrun_ms"),
                        "max_underrun_p99": data.get("p99_max_underrun_ms"),
                        "max_underrun_worst": data.get("max_max_underrun_ms"),
                        "underrun_rate": undr_rate,
                        "max_gap_p99": data.get("p99_max_inter_chunk_gap_ms"),
                        "continuity_ok_n": data.get("num_continuity_ok"),
                        "continuity_ok_rate": (data.get("num_continuity_ok") or 0) / n,
                    }
                )
    return rows


def _fmt(v, f: str) -> str:
    if v is None:
        return "-"
    try:
        return f % v
    except (TypeError, ValueError):
        return str(v)


def print_table(rows: list[dict]) -> None:
    print()
    print(
        f"{'config':36s} {'task':24s} {'c':>4s} {'n':>4s} "
        f"{'ttft_p50':>8s} {'ttft_p99':>8s} {'tail':>5s} "
        f"{'rtf_p50':>7s} {'rtf_p99':>7s} "
        f"{'undr_p50':>8s} {'undr_p99':>8s} {'undr_max':>9s} "
        f"{'undr_rate':>9s} {'gap_p99':>8s} {'cont_OK':>8s}"
    )
    print("-" * 180)
    for r in sorted(rows, key=lambda x: (x["task"], x["c"], x["config"])):
        undr_rate_pct = r["underrun_rate"] * 100 if r.get("underrun_rate") is not None else None
        print(
            f"{r['config']:36s} {r['task']:24s} {r['c']:>4d} {r['n']:>4d} "
            f"{_fmt(r['ttft_p50'], '%8.0f'):>8s} {_fmt(r['ttft_p99'], '%8.0f'):>8s} "
            f"{_fmt(r['ttft_tail'], '%5.2f'):>5s} "
            f"{_fmt(r['rtf_p50'], '%7.2f'):>7s} {_fmt(r['rtf_p99'], '%7.2f'):>7s} "
            f"{_fmt(r['max_underrun_p50'], '%8.0f'):>8s} {_fmt(r['max_underrun_p99'], '%8.0f'):>8s} "
            f"{_fmt(r['max_underrun_worst'], '%9.0f'):>9s} "
            f"{_fmt(undr_rate_pct, '%8.1f%%'):>9s} "
            f"{_fmt(r['max_gap_p99'], '%8.0f'):>8s} "
            f"{_fmt(r['continuity_ok_rate'] * 100, '%7.1f%%'):>8s}"
        )


def _parse_prefix(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--prefix expects <prefix>=<label>, got {s!r}")
    prefix, label = s.split("=", 1)
    return prefix, label


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="One or more results-root directories")
    ap.add_argument(
        "--prefix",
        type=_parse_prefix,
        action="append",
        default=[],
        help="Prefix=Label mapping (repeatable). Without prefixes the directory name is used as label.",
    )
    ap.add_argument("--output", default=None, help="Optional JSON output path for the raw rows")
    args = ap.parse_args()

    rows = collect_rows(args.roots, args.prefix)
    print_table(rows)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
