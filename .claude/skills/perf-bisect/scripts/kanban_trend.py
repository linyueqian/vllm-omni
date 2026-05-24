#!/usr/bin/env python3
"""Extract a metric time series from the vllm-omni-kanban repo.

Works for any cell the kanban tracks (TTS TTFP / RTF / throughput,
diffusion image latency, omni-audio throughput) — pass the cell's filename
prefix and the script prints a per-build table with rolling-delta percent
and `←REG` / `←IMP` markers at the 10% threshold.

Usage:
    # Clone or update the kanban first:
    #   git clone --depth 1 https://github.com/hsliuustc0106/vllm-omni-kanban /tmp/kanban
    python kanban_trend.py /tmp/kanban [cell-prefix]

Examples (TTS):
    # Base voice_clone c=64 N=128
    python kanban_trend.py /tmp/kanban result_test_qwen3_tts_base_seed-tts_64_128

    # CustomVoice default_voice c=64 N=128
    python kanban_trend.py /tmp/kanban result_test_qwen3_tts_customvoice_seed-tts-text_64_128

    # Every TTS cell across every build (overview)
    python kanban_trend.py /tmp/kanban result_test_qwen3_tts

Filename convention in the kanban:
    result_test_<family>_<variant>_<dataset>_<concurrency>_<num_prompts>_<…>_YYYYMMDD-HHMMSS.json

Limitation: build dirs in the kanban are NOT strictly 1:1 with main-branch
nightly commits — some are PR-CI runs. Use the date column (parsed from the
JSON filename) to resolve to an actual main commit via
`git log --first-parent` on vllm-omni.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    kanban_root = pathlib.Path(sys.argv[1]) / "data" / "buildkite_nightly_raw"
    if not kanban_root.exists():
        print(f"ERROR: {kanban_root} not found", file=sys.stderr)
        sys.exit(1)
    prefix = sys.argv[2] if len(sys.argv) > 2 else "result_test_qwen3_tts"

    # build -> date (from any file's timestamp suffix)
    series = defaultdict(list)
    for bd in sorted(kanban_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        if not bd.is_dir():
            continue
        results = bd / "tests/dfx/perf/results"
        if not results.exists():
            continue
        # Pick a representative date from any result file
        date = None
        for any_json in results.glob("*.json"):
            m = re.search(r"_(\d{8})-\d{6}\.json$", any_json.name)
            if m:
                d = m.group(1)
                date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                break
        if date is None:
            continue
        # Now extract metrics for the matching cells
        for f in sorted(results.glob(f"{prefix}*.json")):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            ttfp = d.get("median_audio_ttfp_ms")
            rtf = d.get("median_audio_rtf")
            tput = d.get("audio_throughput")
            c = d.get("max_concurrency")
            n = d.get("num_prompts")
            # Build a friendly cell label from the filename
            stem = f.stem.replace("result_test_qwen3_tts_", "")
            stem = re.sub(r"_\d{8}-\d{6}$", "", stem)
            series[stem].append((date, bd.name, c, n, ttfp, rtf, tput))

    if not series:
        print(f"No cells matched prefix {prefix!r}", file=sys.stderr)
        sys.exit(1)

    for cell_name, entries in series.items():
        if not entries:
            continue
        c0 = entries[0][2]
        n0 = entries[0][3]
        print(f"\n## {cell_name} (c={c0}, N={n0})\n")
        print(f"{'date':12s} {'build':>6s} {'TTFP_ms':>10s} {'Δ%':>7s} {'RTF':>7s} {'Δ%':>7s} {'tput':>6s} {'Δ%':>7s}")
        prev = None
        for date, build, c, n, ttfp, rtf, tput in entries:
            if prev:

                def pct(cur, pv, inv=False):
                    if cur is None or pv is None or pv == 0:
                        return ""
                    d = (cur - pv) / pv * 100
                    if inv:
                        d = -d
                    return f"{d:+5.1f}%"

                d1 = pct(ttfp, prev[0])
                d2 = pct(rtf, prev[1])
                d3 = pct(tput, prev[2], inv=True)
            else:
                d1 = d2 = d3 = ""
            mark = ""
            if d1 and abs(float(d1.rstrip("%"))) >= 10:
                mark = " ←REG" if d1.startswith("+") else " ←IMP"
            t1 = f"{ttfp:10.1f}" if ttfp is not None else "       n/a"
            t2 = f"{rtf:7.3f}" if rtf is not None else "    n/a"
            t3 = f"{tput:6.1f}" if tput is not None else "   n/a"
            print(f"{date:12s} {build:>6s} {t1} {d1:>7s} {t2} {d2:>7s} {t3} {d3:>7s}{mark}")
            prev = (ttfp, rtf, tput)


if __name__ == "__main__":
    main()
