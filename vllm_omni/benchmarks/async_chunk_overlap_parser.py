#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


EVENT_RE = re.compile(
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


def _parse_extra(extra: str | None) -> dict[str, str]:
    if not extra:
        return {}
    return {m.group(1): m.group(2) for m in KV_RE.finditer(extra)}


def _merge_intervals(intervals):
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x.start_ns)
    merged = []

    cur_s, cur_e = intervals[0].start_ns, intervals[0].end_ns

    for iv in intervals[1:]:
        if iv.start_ns <= cur_e:
            cur_e = max(cur_e, iv.end_ns)
        else:
            merged.append(Interval(cur_s, cur_e))
            cur_s, cur_e = iv.start_ns, iv.end_ns

    merged.append(Interval(cur_s, cur_e))
    return merged


def _intersect_duration_ns(interval, merged):
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


def parse_events(log_file: Path):

    stacks = defaultdict(list)

    # (stage, tag) -> intervals
    intervals = defaultdict(list)

    unmatched = 0

    with log_file.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:

            m = EVENT_RE.search(line)

            if not m:
                continue

            tag = m.group("tag")
            phase = m.group("phase")
            ts_ns = int(m.group("ts_ns"))
            tid = m.group("tid")

            extra_kv = _parse_extra(m.group("extra"))
            stage = extra_kv.get("stage", "unknown")

            key = (stage, tag, tid)

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

                intervals[(stage, tag)].append(Interval(start_ns, ts_ns))

    return intervals, unmatched


def compute_overlap(async_intervals, model_intervals):

    total_async = sum(i.duration_ns for i in async_intervals)

    if total_async <= 0:
        return 0, 0, 0

    merged_model = _merge_intervals(model_intervals)

    overlap = sum(
        _intersect_duration_ns(i, merged_model)
        for i in async_intervals
    )

    return total_async, overlap, overlap / total_async


def ns_to_ms(x):
    return x / 1e6


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--log-file", required=True)

    parser.add_argument(
        "--model-tag",
        default="gpu_runner_model_forward"
    )

    parser.add_argument(
        "--async-tags",
        default="chunk_adapter_save_work,chunk_adapter_load_work"
    )

    args = parser.parse_args()

    async_tags = [t.strip() for t in args.async_tags.split(",")]

    intervals, unmatched = parse_events(Path(args.log_file))

    stages = sorted({stage for stage, _ in intervals.keys()})

    for stage in stages:

        print(f"\n===== Stage: {stage} =====")

        model_intervals = intervals.get((stage, args.model_tag), [])

        for tag in [args.model_tag] + async_tags:

            ivs = intervals.get((stage, tag), [])

            total = sum(i.duration_ns for i in ivs)

            count = len(ivs)

            if count == 0:
                print(f"{tag}: count=0")
                continue

            avg = total / count

            print(
                f"{tag}: count={count} "
                f"total={ns_to_ms(total):.3f}ms "
                f"avg={ns_to_ms(avg):.3f}ms"
            )

        print("Overlap ratios:")

        for tag in async_tags:

            async_intervals = intervals.get((stage, tag), [])

            total_async, overlap, ratio = compute_overlap(
                async_intervals,
                model_intervals,
            )

            print(
                f"- {tag}: "
                f"{ns_to_ms(overlap):.3f}ms / {ns_to_ms(total_async):.3f}ms "
                f"=> {ratio:.4f}"
            )

    print(f"\nUnmatched events: {unmatched}")


if __name__ == "__main__":
    main()