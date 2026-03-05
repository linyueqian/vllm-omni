#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Parse async chunk event logs and compute overlap ratio grouped by Stage.

Example log:

[Stage-2] INFO ... [ASYNC_CHUNK_EVENT] gpu_runner_model_forward start ts_ns=... tid=... | stage=code2wav num_tokens=5312
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


EVENT_RE = re.compile(
    r"\[Stage-(?P<stage>\d+)\].*?"
    r"\[ASYNC_CHUNK_EVENT\]\s+"
    r"(?P<tag>\S+)\s+"
    r"(?P<phase>start|end)\s+"
    r"ts_ns=(?P<ts_ns>\d+)\s+"
    r"tid=(?P<tid>\d+)"
    r"(?:\s+\|\s*(?P<extra>.*))?"
)

KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


@dataclass(frozen=True)
class Interval:
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)


def _percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * p

    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)

    frac = pos - lo

    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _merge_intervals(intervals: list[Interval]) -> list[Interval]:

    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x.start_ns)

    merged: list[Interval] = []

    cur_s = intervals[0].start_ns
    cur_e = intervals[0].end_ns

    for iv in intervals[1:]:

        if iv.start_ns <= cur_e:
            cur_e = max(cur_e, iv.end_ns)

        else:
            merged.append(Interval(cur_s, cur_e))
            cur_s = iv.start_ns
            cur_e = iv.end_ns

    merged.append(Interval(cur_s, cur_e))

    return merged


def _intersect_duration_ns(interval: Interval, merged: list[Interval]) -> int:

    total = 0

    for m in merged:

        if m.end_ns <= interval.start_ns:
            continue

        if m.start_ns >= interval.end_ns:
            break

        s = max(interval.start_ns, m.start_ns)
        e = min(interval.end_ns, m.end_ns)

        if e > s:
            total += e - s

    return total


def _ns_to_ms(x: float) -> float:
    return x / 1_000_000.0


def _parse_extra(extra: str | None) -> dict[str, str]:

    if not extra:
        return {}

    return {m.group(1): m.group(2) for m in KV_RE.finditer(extra)}


def parse_events(log_file: Path):

    stacks = defaultdict(list)

    intervals = defaultdict(list)

    unmatched = 0

    with log_file.open("r", encoding="utf-8", errors="replace") as f:

        for line in f:

            m = EVENT_RE.search(line)

            if not m:
                continue

            stage = m.group("stage")
            tag = m.group("tag")
            phase = m.group("phase")

            ts_ns = int(m.group("ts_ns"))
            tid = m.group("tid")

            extra_kv = _parse_extra(m.group("extra"))

            req_id = extra_kv.get("req_id", "-")
            chunk = extra_kv.get("chunk", "-")

            key = (stage, tag, tid, req_id, chunk)

            if phase == "start":

                stacks[key].append(ts_ns)

            else:

                if not stacks[key]:
                    unmatched += 1
                    continue

                start_ns = stacks[key].pop()

                if ts_ns <= start_ns:
                    unmatched += 1
                    continue

                intervals[(stage, tag)].append(
                    Interval(start_ns, ts_ns)
                )

    for pending in stacks.values():
        unmatched += len(pending)

    return intervals, unmatched


def _stats_line(name: str, durations_ns: list[int]) -> str:

    if not durations_ns:
        return f"{name}: count=0"

    count = len(durations_ns)

    total = sum(durations_ns)

    avg = total / count

    p50 = _percentile(durations_ns, 0.50)
    p95 = _percentile(durations_ns, 0.95)
    p99 = _percentile(durations_ns, 0.99)

    return (
        f"{name}: count={count} "
        f"total={_ns_to_ms(total):.3f}ms "
        f"avg={_ns_to_ms(avg):.3f}ms "
        f"p50={_ns_to_ms(p50):.3f}ms "
        f"p95={_ns_to_ms(p95):.3f}ms "
        f"p99={_ns_to_ms(p99):.3f}ms"
    )


def compute_overlap(async_intervals, model_intervals):

    total_async = sum(i.duration_ns for i in async_intervals)

    if total_async <= 0:
        return 0, 0, 0.0

    merged_model = _merge_intervals(model_intervals)

    overlap = sum(
        _intersect_duration_ns(i, merged_model)
        for i in async_intervals
    )

    ratio = overlap / total_async

    return total_async, overlap, ratio


def main():

    parser = argparse.ArgumentParser(
        description="Compute async overlap grouped by Stage"
    )

    parser.add_argument(
        "--log-file",
        required=True,
        help="Log file path",
    )

    parser.add_argument(
        "--model-tag",
        default="gpu_runner_model_forward",
    )

    parser.add_argument(
        "--async-tags",
        default="chunk_adapter_save_work,chunk_adapter_load_work",
    )

    args = parser.parse_args()

    log_file = Path(args.log_file)

    if not log_file.exists():
        raise FileNotFoundError(log_file)

    async_tags = [
        t.strip() for t in args.async_tags.split(",") if t.strip()
    ]

    intervals, unmatched = parse_events(log_file)

    stages = sorted({stage for stage, _ in intervals.keys()})

    for stage in stages:

        print(f"\n===== Stage-{stage} =====")

        model_intervals = intervals.get((stage, args.model_tag), [])

        print(
            _stats_line(
                args.model_tag,
                [i.duration_ns for i in model_intervals],
            )
        )

        for tag in async_tags:

            ivs = intervals.get((stage, tag), [])

            print(
                _stats_line(
                    tag,
                    [i.duration_ns for i in ivs],
                )
            )

        print("\nOverlap ratios:")

        all_async = []

        for tag in async_tags:

            async_intervals = intervals.get((stage, tag), [])

            all_async.extend(async_intervals)

            total_async, overlap, ratio = compute_overlap(
                async_intervals,
                model_intervals,
            )

            print(
                f"- {tag}: "
                f"{_ns_to_ms(overlap):.3f}ms / {_ns_to_ms(total_async):.3f}ms "
                f"=> overlap_ratio={ratio:.4f}"
            )

        total_async, overlap, ratio = compute_overlap(
            all_async,
            model_intervals,
        )

        print(
            f"- ALL_ASYNC: "
            f"{_ns_to_ms(overlap):.3f}ms / {_ns_to_ms(total_async):.3f}ms "
            f"=> overlap_ratio={ratio:.4f}"
        )

    print(f"\nUnmatched events: {unmatched}")

    if unmatched > 0:
        print(
            "Note: unmatched events usually mean missing start/end logs."
        )


if __name__ == "__main__":
    main()