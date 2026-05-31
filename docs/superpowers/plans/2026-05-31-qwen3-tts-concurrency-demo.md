# Qwen3-TTS Concurrency Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a video-recording-first Gradio demo (two tabs, N=8 + N=64) that
visualises parallel-vs-serial speedup of streaming Qwen3-TTS-Base under high
concurrency, against a stock `vllm serve` instance.

**Architecture:** Async orchestrator fires one c=1 reference stream and N
parallel `/v1/audio/speech` streams via `httpx.AsyncClient.stream(...).aiter_bytes()`
on a background worker loop. A thread-safe `MetricsAggregator` publishes
immutable snapshots; Gradio's `gr.Timer` ticks pull the latest snapshot and
re-render row/grid HTML. AudioWorklet preview reused from the existing
`qwen3_tts/gradio_demo.py` for one selectable stream on Page A.

**Tech Stack:** Python 3.12, Gradio, httpx (async), pytest + pytest-asyncio,
`httpx.MockTransport` for orchestrator tests, the vendored fixture
`tests/assets/qwen3_tts/clone_2.wav`.

**Spec:** [docs/superpowers/specs/2026-05-31-qwen3-tts-concurrency-demo-design.md](../specs/2026-05-31-qwen3-tts-concurrency-demo-design.md)

---

## File Structure

All new code lives under
`examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/`:

| File | Responsibility | Approx LOC |
|------|----------------|-----------|
| `__init__.py` | Package marker; re-export public types | 10 |
| `prompts.py` | `DEMO_PROMPT` constant | 15 |
| `metrics.py` | `StreamEvent`, `StreamState`, `MetricsSnapshot`, `MetricsAggregator` | 200 |
| `orchestrator.py` | `Orchestrator` class; `_run_one`, `run_burst` | 180 |
| `views.py` | `render_row_html(snapshot, i)`, `render_grid_html(snapshot)`, `render_counters_html(snapshot)` | 180 |
| `app.py` | Gradio app: two `gr.Tabs`, Start/Reset, `gr.Timer`, AudioWorklet preview | 250 |
| `run.sh` | Convenience launcher (server check + demo) | 30 |
| `README.md` | Server bring-up, recording flow, troubleshooting | 100 |

Tests live in `tests/examples/online_serving/text_to_speech/qwen3_tts/`:

| File | Responsibility |
|------|----------------|
| `__init__.py` | Package marker |
| `test_metrics.py` | `MetricsAggregator` unit tests (no network) |
| `test_orchestrator.py` | Orchestrator with `httpx.MockTransport` |
| `test_views.py` | HTML golden-file renders |
| `test_concurrency_demo_smoke.py` | CLI smoke test gated by `--full-model` (optional) |

Linting rules (already in `pyproject.toml`):
- `line-length = 120`
- `[tool.ruff.lint.per-file-ignores]` already exempts `examples/**` from `E501`.
- Tests are exempt from `E501` too.

Pre-commit must pass before push: `pre-commit run --files <paths>` or
`uvx ruff@0.14.10 check --fix <paths> && uvx ruff@0.14.10 format <paths>`.

---

### Task 1: Scaffold the package and the locked prompt

**Files:**
- Create: `examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/__init__.py`
- Create: `examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/prompts.py`

- [ ] **Step 1: Create the package init**

```python
# examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/__init__.py
"""Qwen3-TTS streaming concurrency demo.

See ../README.md and run.sh for usage.
"""
```

- [ ] **Step 2: Create prompts.py with the locked DEMO_PROMPT**

```python
# examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/prompts.py
"""Locked prompt used by every stream in the concurrency demo.

A single fixed English string is used for both the c=1 serial reference and
all N parallel streams. This makes serial_eta = N * t_observed an exact
identity rather than an extrapolation across length-varying prompts.
"""

DEMO_PROMPT: str = (
    "Modern text-to-speech models can stream audio in real time. "
    "When many users speak to the same model at once, batching makes the "
    "throughput multiply, not divide. Watch the streams finish together."
)
```

- [ ] **Step 3: Sanity import**

Run: `python -c "from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.prompts import DEMO_PROMPT; print(len(DEMO_PROMPT))"`
Expected: a positive integer between 150 and 300.

- [ ] **Step 4: Commit**

```bash
git add examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/__init__.py \
        examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/prompts.py
git commit -s -m "feat(qwen3-tts-demo): scaffold concurrency-demo package with DEMO_PROMPT"
```

---

### Task 2: StreamEvent + StreamState + MetricsSnapshot dataclasses

**Files:**
- Create: `examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/metrics.py` (partial)
- Create: `tests/examples/online_serving/text_to_speech/__init__.py`
- Create: `tests/examples/online_serving/text_to_speech/qwen3_tts/__init__.py`
- Create: `tests/examples/online_serving/text_to_speech/qwen3_tts/test_metrics.py` (partial)

- [ ] **Step 1: Write the failing test**

```python
# tests/examples/online_serving/text_to_speech/qwen3_tts/test_metrics.py
"""Unit tests for the concurrency-demo metrics aggregator."""

from __future__ import annotations

from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.metrics import (
    MetricsSnapshot,
    StreamEvent,
    StreamState,
)


def test_stream_event_factory_defaults() -> None:
    ev = StreamEvent.first(stream_id=3, ts=1.5)
    assert ev.stream_id == 3
    assert ev.kind == "first"
    assert ev.ts == 1.5
    assert ev.byte_count == 0


def test_stream_state_initial_values() -> None:
    s = StreamState(stream_id=7)
    assert s.stream_id == 7
    assert s.ttfb_s is None
    assert s.bytes_received == 0
    assert s.status == "pending"


def test_metrics_snapshot_is_frozen() -> None:
    snap = MetricsSnapshot(
        wall_s=0.0,
        burst_start_s=None,
        per_stream=(),
        completed=0,
        active=0,
        throughput_x=0.0,
        ttfb_p99_ms=None,
        rtf_p99=None,
        serial_eta_s=None,
        parallel_eta_s=None,
        speedup_x=None,
        any_failed=False,
    )
    # Dataclass should be frozen; mutation must raise.
    try:
        snap.completed = 99
    except Exception as e:
        assert "frozen" in str(e).lower() or "cannot assign" in str(e).lower()
    else:
        raise AssertionError("MetricsSnapshot must be frozen")
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `pytest tests/examples/online_serving/text_to_speech/qwen3_tts/test_metrics.py -v`
Expected: ImportError (`metrics` module does not exist yet).

- [ ] **Step 3: Implement the dataclasses**

```python
# examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/metrics.py
"""MetricsAggregator and the immutable snapshot it publishes to the UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EventKind = Literal["first", "chunk", "done", "error"]
StreamStatus = Literal["pending", "streaming", "done", "error"]


@dataclass(frozen=True)
class StreamEvent:
    stream_id: int
    kind: EventKind
    ts: float
    byte_count: int = 0
    error_message: str = ""

    @staticmethod
    def first(stream_id: int, ts: float) -> "StreamEvent":
        return StreamEvent(stream_id=stream_id, kind="first", ts=ts)

    @staticmethod
    def chunk(stream_id: int, ts: float, byte_count: int) -> "StreamEvent":
        return StreamEvent(stream_id=stream_id, kind="chunk", ts=ts, byte_count=byte_count)

    @staticmethod
    def done(stream_id: int, ts: float) -> "StreamEvent":
        return StreamEvent(stream_id=stream_id, kind="done", ts=ts)

    @staticmethod
    def error(stream_id: int, ts: float, message: str) -> "StreamEvent":
        return StreamEvent(stream_id=stream_id, kind="error", ts=ts, error_message=message)


@dataclass
class StreamState:
    stream_id: int
    status: StreamStatus = "pending"
    request_sent_s: float | None = None
    first_chunk_s: float | None = None
    last_chunk_s: float | None = None
    bytes_received: int = 0
    error_message: str = ""

    @property
    def ttfb_s(self) -> float | None:
        if self.request_sent_s is None or self.first_chunk_s is None:
            return None
        return self.first_chunk_s - self.request_sent_s

    @property
    def audio_seconds(self) -> float:
        # 24 kHz mono int16 = 2 bytes/sample.
        return self.bytes_received / (24_000 * 2)

    @property
    def final_rtf(self) -> float | None:
        if self.status != "done" or self.audio_seconds <= 0 or self.request_sent_s is None:
            return None
        wall = (self.last_chunk_s or 0.0) - self.request_sent_s
        return wall / self.audio_seconds


@dataclass(frozen=True)
class MetricsSnapshot:
    wall_s: float
    burst_start_s: float | None
    per_stream: tuple[StreamState, ...] = field(default_factory=tuple)
    completed: int = 0
    active: int = 0
    throughput_x: float = 0.0
    ttfb_p99_ms: float | None = None
    rtf_p99: float | None = None
    serial_eta_s: float | None = None
    parallel_eta_s: float | None = None
    speedup_x: float | None = None
    any_failed: bool = False
```

- [ ] **Step 4: Run the tests, confirm they pass**

Run: `pytest tests/examples/online_serving/text_to_speech/qwen3_tts/test_metrics.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/metrics.py \
        tests/examples/online_serving/text_to_speech/__init__.py \
        tests/examples/online_serving/text_to_speech/qwen3_tts/__init__.py \
        tests/examples/online_serving/text_to_speech/qwen3_tts/test_metrics.py
git commit -s -m "feat(qwen3-tts-demo): add StreamEvent/StreamState/MetricsSnapshot dataclasses"
```

---

### Task 3: MetricsAggregator core (event apply + per-stream state)

**Files:**
- Modify: `examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/metrics.py`
- Modify: `tests/examples/online_serving/text_to_speech/qwen3_tts/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `test_metrics.py`:

```python
import pytest

from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.metrics import (
    MetricsAggregator,
)


def test_aggregator_first_chunk_marks_streaming_and_locks_ttfb() -> None:
    agg = MetricsAggregator(n=2)
    agg.mark_request_sent(stream_id=0, ts=10.0)
    agg.mark_request_sent(stream_id=1, ts=10.0)
    agg.apply(StreamEvent.first(stream_id=0, ts=10.4))
    snap = agg.snapshot(now=10.4)
    s0 = snap.per_stream[0]
    s1 = snap.per_stream[1]
    assert s0.status == "streaming"
    assert s0.ttfb_s == pytest.approx(0.4)
    assert s1.status == "pending"


def test_aggregator_chunks_accumulate_bytes() -> None:
    agg = MetricsAggregator(n=1)
    agg.mark_request_sent(stream_id=0, ts=0.0)
    agg.apply(StreamEvent.first(stream_id=0, ts=0.2))
    agg.apply(StreamEvent.chunk(stream_id=0, ts=0.4, byte_count=4800))
    agg.apply(StreamEvent.chunk(stream_id=0, ts=0.6, byte_count=4800))
    snap = agg.snapshot(now=0.6)
    assert snap.per_stream[0].bytes_received == 9600
    # 9600 bytes / (24000 Hz * 2 B/sample) == 0.2 seconds of audio.
    assert snap.per_stream[0].audio_seconds == pytest.approx(0.2)


def test_aggregator_done_event_marks_complete() -> None:
    agg = MetricsAggregator(n=1)
    agg.mark_request_sent(stream_id=0, ts=0.0)
    agg.apply(StreamEvent.first(stream_id=0, ts=0.2))
    agg.apply(StreamEvent.chunk(stream_id=0, ts=1.0, byte_count=48_000))  # 1 s of audio
    agg.apply(StreamEvent.done(stream_id=0, ts=1.05))
    snap = agg.snapshot(now=1.05)
    assert snap.per_stream[0].status == "done"
    assert snap.completed == 1
    assert snap.per_stream[0].final_rtf == pytest.approx(1.05 / 1.0)


def test_aggregator_error_event_marks_failed() -> None:
    agg = MetricsAggregator(n=2)
    agg.mark_request_sent(stream_id=0, ts=0.0)
    agg.apply(StreamEvent.error(stream_id=0, ts=0.3, message="boom"))
    snap = agg.snapshot(now=0.3)
    assert snap.per_stream[0].status == "error"
    assert snap.per_stream[0].error_message == "boom"
    assert snap.any_failed is True
```

- [ ] **Step 2: Run the tests, confirm they fail**

Run: `pytest tests/examples/online_serving/text_to_speech/qwen3_tts/test_metrics.py -v`
Expected: 4 new tests fail with `ImportError` on `MetricsAggregator`.

- [ ] **Step 3: Implement MetricsAggregator (core)**

Append to `metrics.py`:

```python
import threading


class MetricsAggregator:
    """Thread-safe aggregator. UI reads immutable snapshots; orchestrator emits events."""

    def __init__(self, n: int) -> None:
        self._lock = threading.Lock()
        self._n = n
        self._states: list[StreamState] = [StreamState(stream_id=i) for i in range(n)]
        self._burst_start_s: float | None = None
        self._ref_t_observed_s: float | None = None

    def reset(self) -> None:
        with self._lock:
            self._states = [StreamState(stream_id=i) for i in range(self._n)]
            self._burst_start_s = None
            self._ref_t_observed_s = None

    def mark_burst_start(self, ts: float) -> None:
        with self._lock:
            self._burst_start_s = ts

    def set_reference(self, t_observed_s: float) -> None:
        with self._lock:
            self._ref_t_observed_s = t_observed_s

    def mark_request_sent(self, stream_id: int, ts: float) -> None:
        with self._lock:
            s = self._states[stream_id]
            s.request_sent_s = ts

    def apply(self, ev: StreamEvent) -> None:
        with self._lock:
            s = self._states[ev.stream_id]
            if ev.kind == "first":
                s.first_chunk_s = ev.ts
                s.last_chunk_s = ev.ts
                if s.status == "pending":
                    s.status = "streaming"
            elif ev.kind == "chunk":
                s.bytes_received += ev.byte_count
                s.last_chunk_s = ev.ts
            elif ev.kind == "done":
                s.last_chunk_s = ev.ts
                s.status = "done"
            elif ev.kind == "error":
                s.status = "error"
                s.error_message = ev.error_message
                s.last_chunk_s = ev.ts
```

- [ ] **Step 4: Run the tests; the snapshot-dependent ones will still fail**

Run: `pytest tests/examples/online_serving/text_to_speech/qwen3_tts/test_metrics.py -v`
Expected: 3 of 4 new tests still fail because `snapshot()` is not implemented yet. The first/chunk/done state transitions on `StreamState` should compute correctly through the dataclass properties; only the `snapshot()` calls fail. Task 4 implements `snapshot()`.

- [ ] **Step 5: Commit**

```bash
git add examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/metrics.py \
        tests/examples/online_serving/text_to_speech/qwen3_tts/test_metrics.py
git commit -s -m "feat(qwen3-tts-demo): MetricsAggregator core event apply + per-stream state"
```

---

### Task 4: MetricsAggregator.snapshot() and derived metrics

**Files:**
- Modify: `examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/metrics.py`
- Modify: `tests/examples/online_serving/text_to_speech/qwen3_tts/test_metrics.py`

- [ ] **Step 1: Add the failing tests for derived metrics**

Append to `test_metrics.py`:

```python
def test_aggregator_snapshot_throughput_after_complete_run() -> None:
    agg = MetricsAggregator(n=2)
    agg.mark_burst_start(ts=0.0)
    for i in (0, 1):
        agg.mark_request_sent(stream_id=i, ts=0.0)
        agg.apply(StreamEvent.first(stream_id=i, ts=0.2))
        # Each stream produces 48_000 bytes => 1.0 s of audio.
        agg.apply(StreamEvent.chunk(stream_id=i, ts=0.5, byte_count=48_000))
        agg.apply(StreamEvent.done(stream_id=i, ts=0.5))
    snap = agg.snapshot(now=0.5)
    # Sum audio = 2.0 s; wall = 0.5 s => 4x real-time throughput.
    assert snap.throughput_x == pytest.approx(4.0)
    assert snap.completed == 2
    assert snap.active == 0


def test_aggregator_snapshot_serial_and_speedup() -> None:
    agg = MetricsAggregator(n=4)
    agg.set_reference(t_observed_s=2.0)  # c=1 reference: 2 s wall time.
    agg.mark_burst_start(ts=10.0)
    for i in range(4):
        agg.mark_request_sent(stream_id=i, ts=10.0)
        agg.apply(StreamEvent.first(stream_id=i, ts=10.2))
        agg.apply(StreamEvent.chunk(stream_id=i, ts=12.0, byte_count=48_000 * 2))
        agg.apply(StreamEvent.done(stream_id=i, ts=12.0))
    snap = agg.snapshot(now=12.0)
    # serial_eta = 4 * 2.0 = 8.0 s; parallel_eta = 12.0 - 10.0 = 2.0 s.
    assert snap.serial_eta_s == pytest.approx(8.0)
    assert snap.parallel_eta_s == pytest.approx(2.0)
    assert snap.speedup_x == pytest.approx(4.0)


def test_aggregator_speedup_suppressed_if_any_failed() -> None:
    agg = MetricsAggregator(n=2)
    agg.set_reference(t_observed_s=1.0)
    agg.mark_burst_start(ts=0.0)
    agg.mark_request_sent(stream_id=0, ts=0.0)
    agg.apply(StreamEvent.first(stream_id=0, ts=0.2))
    agg.apply(StreamEvent.chunk(stream_id=0, ts=0.5, byte_count=48_000))
    agg.apply(StreamEvent.done(stream_id=0, ts=0.5))
    agg.mark_request_sent(stream_id=1, ts=0.0)
    agg.apply(StreamEvent.error(stream_id=1, ts=0.3, message="boom"))
    snap = agg.snapshot(now=0.5)
    assert snap.any_failed is True
    assert snap.speedup_x is None
```

- [ ] **Step 2: Run the tests; confirm they fail**

Run: `pytest tests/examples/online_serving/text_to_speech/qwen3_tts/test_metrics.py -v`
Expected: snapshot-based tests fail because `snapshot()` is not implemented.

- [ ] **Step 3: Implement snapshot()**

Append to `metrics.py`:

```python
import math


def _percentile(sorted_values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile, p in [0, 1]."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


class MetricsAggregator(MetricsAggregator):  # noqa: F811 — extend in same module
    pass


def _snapshot_impl(self: MetricsAggregator, now: float) -> MetricsSnapshot:
    with self._lock:
        per_stream = tuple(
            StreamState(
                stream_id=s.stream_id,
                status=s.status,
                request_sent_s=s.request_sent_s,
                first_chunk_s=s.first_chunk_s,
                last_chunk_s=s.last_chunk_s,
                bytes_received=s.bytes_received,
                error_message=s.error_message,
            )
            for s in self._states
        )
        burst_start = self._burst_start_s
        ref_t = self._ref_t_observed_s
        n = self._n

    completed = sum(1 for s in per_stream if s.status == "done")
    active = sum(1 for s in per_stream if s.status == "streaming")
    any_failed = any(s.status == "error" for s in per_stream)

    total_audio_s = sum(s.audio_seconds for s in per_stream)
    elapsed = (now - burst_start) if burst_start is not None else 0.0
    throughput_x = (total_audio_s / elapsed) if elapsed > 0 else 0.0

    ttfb_values = sorted(s.ttfb_s * 1000.0 for s in per_stream if s.ttfb_s is not None)
    rtf_values = sorted(s.final_rtf for s in per_stream if s.final_rtf is not None)
    ttfb_p99_ms = _percentile(ttfb_values, 0.99)
    rtf_p99 = _percentile(rtf_values, 0.99)

    serial_eta_s = (n * ref_t) if ref_t is not None else None

    if completed == n and burst_start is not None and not any_failed:
        last = max((s.last_chunk_s or burst_start) for s in per_stream)
        parallel_eta_s = last - burst_start
    else:
        parallel_eta_s = None

    if serial_eta_s is not None and parallel_eta_s is not None and parallel_eta_s > 0 and not any_failed:
        speedup_x = serial_eta_s / parallel_eta_s
    else:
        speedup_x = None

    return MetricsSnapshot(
        wall_s=elapsed,
        burst_start_s=burst_start,
        per_stream=per_stream,
        completed=completed,
        active=active,
        throughput_x=throughput_x,
        ttfb_p99_ms=ttfb_p99_ms,
        rtf_p99=rtf_p99,
        serial_eta_s=serial_eta_s,
        parallel_eta_s=parallel_eta_s,
        speedup_x=speedup_x,
        any_failed=any_failed,
    )


MetricsAggregator.snapshot = _snapshot_impl  # type: ignore[assignment]
```

Note: this avoids re-opening the class definition inline; the
`_snapshot_impl` function is attached as a method at module import time.
A cleaner equivalent is to put `snapshot()` inside the original
`MetricsAggregator` class body when you write Task 3 if you prefer — just
keep the test API identical.

- [ ] **Step 4: Run all metrics tests; confirm they pass**

Run: `pytest tests/examples/online_serving/text_to_speech/qwen3_tts/test_metrics.py -v`
Expected: all tests in the file pass (3 from Task 2 + 4 from Task 3 + 3 from Task 4 = 10).

- [ ] **Step 5: Commit**

```bash
git add examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/metrics.py \
        tests/examples/online_serving/text_to_speech/qwen3_tts/test_metrics.py
git commit -s -m "feat(qwen3-tts-demo): MetricsAggregator.snapshot() with throughput/p99/speedup"
```

---

### Task 5: Orchestrator: single-stream coroutine

**Files:**
- Create: `examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/orchestrator.py`
- Create: `tests/examples/online_serving/text_to_speech/qwen3_tts/test_orchestrator.py`

- [ ] **Step 1: Write the failing test using httpx.MockTransport**

```python
# tests/examples/online_serving/text_to_speech/qwen3_tts/test_orchestrator.py
"""Orchestrator tests using an in-process httpx.MockTransport — no real server."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.metrics import (
    MetricsAggregator,
    StreamEvent,
)
from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.orchestrator import (
    Orchestrator,
    StreamConfig,
)


def _mock_transport_factory(chunks: list[bytes]):
    """Return a MockTransport that yields the supplied raw byte chunks."""

    def handler(request: httpx.Request) -> httpx.Response:
        async def gen():
            for c in chunks:
                yield c

        return httpx.Response(200, content=gen(), headers={"content-type": "audio/pcm"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_run_one_stream_emits_first_chunks_done_events() -> None:
    aggregator = MetricsAggregator(n=1)
    aggregator.mark_burst_start(0.0)
    transport = _mock_transport_factory([b"\x00" * 4800, b"\x00" * 4800])
    orchestrator = Orchestrator(
        api_base="http://stub",
        ref_audio_b64="UExBQ0VIT0xERVI=",
        transport=transport,
    )
    cfg = StreamConfig(stream_id=0, text="hello", payload_kind="parallel")
    await orchestrator._run_one(cfg, aggregator)
    snap = aggregator.snapshot(now=1.0)
    s0 = snap.per_stream[0]
    assert s0.status == "done"
    assert s0.bytes_received == 9600  # 2 chunks * 4800 bytes
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `pytest tests/examples/online_serving/text_to_speech/qwen3_tts/test_orchestrator.py -v`
Expected: ImportError for `orchestrator` module.

- [ ] **Step 3: Implement the orchestrator core**

```python
# examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/orchestrator.py
"""Async orchestrator that fires N concurrent /v1/audio/speech streams.

Reads PCM bytes from each response, emits StreamEvents into the aggregator,
and computes a c=1 reference latency for the serial-ETA badge.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Literal

import httpx

from .metrics import MetricsAggregator, StreamEvent

PayloadKind = Literal["reference", "parallel"]


@dataclass(frozen=True)
class StreamConfig:
    stream_id: int
    text: str
    payload_kind: PayloadKind


class Orchestrator:
    """Owns an httpx.AsyncClient and a worker asyncio loop."""

    def __init__(
        self,
        api_base: str,
        ref_audio_b64: str,
        *,
        timeout_s: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._ref_audio_b64 = ref_audio_b64
        self._timeout_s = timeout_s
        self._transport = transport

    def _build_payload(self, text: str) -> dict:
        return {
            "input": text,
            "response_format": "pcm",
            "stream": True,
            "task_type": "Base",
            "ref_audio": f"data:audio/wav;base64,{self._ref_audio_b64}",
        }

    async def _run_one(self, cfg: StreamConfig, aggregator: MetricsAggregator) -> float:
        """Runs a single streaming request. Returns wall-clock duration.

        Emits events into the aggregator; on error, emits a single error event
        and re-raises only if it is not a transport-level failure.
        """
        sent_at = time.perf_counter()
        aggregator.mark_request_sent(stream_id=cfg.stream_id, ts=sent_at)
        payload = self._build_payload(cfg.text)
        first_seen = False
        bytes_total = 0
        async with httpx.AsyncClient(
            timeout=self._timeout_s,
            transport=self._transport,
        ) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self._api_base}/v1/audio/speech",
                    json=payload,
                    headers={"Authorization": "Bearer EMPTY"},
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        msg = f"HTTP {response.status_code}: {body[:200]!r}"
                        aggregator.apply(StreamEvent.error(cfg.stream_id, time.perf_counter(), msg))
                        return time.perf_counter() - sent_at
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        now = time.perf_counter()
                        if not first_seen:
                            aggregator.apply(StreamEvent.first(cfg.stream_id, now))
                            first_seen = True
                        aggregator.apply(
                            StreamEvent.chunk(cfg.stream_id, now, byte_count=len(chunk))
                        )
                        bytes_total += len(chunk)
            except (httpx.HTTPError, asyncio.CancelledError) as exc:
                aggregator.apply(
                    StreamEvent.error(cfg.stream_id, time.perf_counter(), str(exc))
                )
                return time.perf_counter() - sent_at
        aggregator.apply(StreamEvent.done(cfg.stream_id, time.perf_counter()))
        return time.perf_counter() - sent_at
```

- [ ] **Step 4: Install pytest-asyncio if missing**

Run: `python -c "import pytest_asyncio" 2>&1` — if it errors, install it
with `uv pip install pytest-asyncio` (the demo's dev requirement). Verify
the project's existing dev deps; pytest-asyncio is already used by the repo
(`pytest -m asyncio` flags exist), so no install should be needed.

- [ ] **Step 5: Run the test, confirm it passes**

Run: `pytest tests/examples/online_serving/text_to_speech/qwen3_tts/test_orchestrator.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/orchestrator.py \
        tests/examples/online_serving/text_to_speech/qwen3_tts/test_orchestrator.py
git commit -s -m "feat(qwen3-tts-demo): orchestrator single-stream runner with MockTransport test"
```

---

### Task 6: Orchestrator: run_burst (c=1 reference + N parallel)

**Files:**
- Modify: `examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/orchestrator.py`
- Modify: `tests/examples/online_serving/text_to_speech/qwen3_tts/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Append to `test_orchestrator.py`:

```python
@pytest.mark.asyncio
async def test_run_burst_records_reference_and_runs_n_parallel() -> None:
    aggregator = MetricsAggregator(n=4)
    transport = _mock_transport_factory([b"\x00" * 9600])  # ~0.2 s of audio
    orchestrator = Orchestrator(
        api_base="http://stub",
        ref_audio_b64="UExBQ0VIT0xERVI=",
        transport=transport,
    )
    await orchestrator.run_burst(
        n=4,
        prompt="hello world",
        aggregator=aggregator,
    )
    snap = aggregator.snapshot(now=10.0)  # late enough that wall_s is reasonable
    # All 4 streams must be done.
    assert snap.completed == 4
    # serial_eta must be populated (reference ran).
    assert snap.serial_eta_s is not None
    assert snap.serial_eta_s > 0
    # No failures.
    assert snap.any_failed is False
```

- [ ] **Step 2: Run the test, confirm it fails (missing `run_burst`)**

Run: `pytest tests/examples/online_serving/text_to_speech/qwen3_tts/test_orchestrator.py -v`

- [ ] **Step 3: Implement run_burst**

Append to `orchestrator.py`:

```python
    async def _measure_reference(self, prompt: str, aggregator: MetricsAggregator) -> float:
        """Run one c=1 stream against the same server BEFORE the parallel burst."""
        # Use a synthetic stream_id of -1 so it does not appear in the per-stream UI.
        # Aggregator only tracks N parallel streams; ref timing is captured separately.
        sent_at = time.perf_counter()
        payload = self._build_payload(prompt)
        async with httpx.AsyncClient(
            timeout=self._timeout_s,
            transport=self._transport,
        ) as client:
            async with client.stream(
                "POST",
                f"{self._api_base}/v1/audio/speech",
                json=payload,
                headers={"Authorization": "Bearer EMPTY"},
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise RuntimeError(
                        f"Reference c=1 stream failed: HTTP {response.status_code}: {body[:200]!r}"
                    )
                async for _chunk in response.aiter_bytes():
                    pass
        return time.perf_counter() - sent_at

    async def run_burst(
        self,
        n: int,
        prompt: str,
        aggregator: MetricsAggregator,
    ) -> None:
        """Run one c=1 reference, then N concurrent streams of the same prompt."""
        aggregator.reset()
        t_ref = await self._measure_reference(prompt, aggregator)
        aggregator.set_reference(t_ref)
        aggregator.mark_burst_start(time.perf_counter())
        configs = [
            StreamConfig(stream_id=i, text=prompt, payload_kind="parallel")
            for i in range(n)
        ]
        await asyncio.gather(*(self._run_one(cfg, aggregator) for cfg in configs))
```

- [ ] **Step 4: Run the tests, confirm both pass**

Run: `pytest tests/examples/online_serving/text_to_speech/qwen3_tts/test_orchestrator.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/orchestrator.py \
        tests/examples/online_serving/text_to_speech/qwen3_tts/test_orchestrator.py
git commit -s -m "feat(qwen3-tts-demo): run_burst with c=1 reference + N-parallel gather"
```

---

### Task 7: Views: render_row_html for the N=8 page

**Files:**
- Create: `examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/views.py`
- Create: `tests/examples/online_serving/text_to_speech/qwen3_tts/test_views.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/examples/online_serving/text_to_speech/qwen3_tts/test_views.py
"""Render-pure HTML view tests — string assertions only, no DOM."""

from __future__ import annotations

from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.metrics import (
    MetricsAggregator,
    MetricsSnapshot,
    StreamEvent,
    StreamState,
)
from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.views import (
    render_grid_html,
    render_row_html,
)


def _snapshot_with_one_streaming_one_done() -> MetricsSnapshot:
    agg = MetricsAggregator(n=2)
    agg.mark_burst_start(0.0)
    agg.set_reference(1.0)
    # Stream 0: streaming
    agg.mark_request_sent(stream_id=0, ts=0.0)
    agg.apply(StreamEvent.first(stream_id=0, ts=0.2))
    agg.apply(StreamEvent.chunk(stream_id=0, ts=0.5, byte_count=24_000))
    # Stream 1: done
    agg.mark_request_sent(stream_id=1, ts=0.0)
    agg.apply(StreamEvent.first(stream_id=1, ts=0.2))
    agg.apply(StreamEvent.chunk(stream_id=1, ts=0.4, byte_count=48_000))
    agg.apply(StreamEvent.done(stream_id=1, ts=0.4))
    return agg.snapshot(now=0.5)


def test_render_row_streaming_includes_ttfb_and_progress() -> None:
    snap = _snapshot_with_one_streaming_one_done()
    html = render_row_html(snap, stream_id=0)
    assert "#1" in html
    assert "TTFB" in html
    assert "streaming" in html.lower() or "in-flight" in html.lower()


def test_render_row_done_marks_completion() -> None:
    snap = _snapshot_with_one_streaming_one_done()
    html = render_row_html(snap, stream_id=1)
    assert "#2" in html
    assert "done" in html.lower() or "100" in html


def test_render_grid_returns_64_cells_for_n_64() -> None:
    agg = MetricsAggregator(n=64)
    snap = agg.snapshot(now=0.0)
    html = render_grid_html(snap)
    # 64 dots in the grid.
    assert html.count("data-cell") == 64
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `pytest tests/examples/online_serving/text_to_speech/qwen3_tts/test_views.py -v`
Expected: ImportError on `views`.

- [ ] **Step 3: Implement views.py**

```python
# examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/views.py
"""HTML rendering helpers for the concurrency demo.

All renderers are pure functions of a MetricsSnapshot — no Gradio state.
"""

from __future__ import annotations

from .metrics import MetricsSnapshot, StreamState


def _row_progress_pct(s: StreamState, est_total_audio_s: float = 7.0) -> float:
    if s.status == "done":
        return 100.0
    pct = (s.audio_seconds / est_total_audio_s) * 100.0
    return min(99.0, max(0.0, pct))


def render_row_html(snap: MetricsSnapshot, stream_id: int) -> str:
    s: StreamState = snap.per_stream[stream_id]
    label = f"#{stream_id + 1}"
    status = s.status
    pct = _row_progress_pct(s)
    ttfb_str = f"{int(s.ttfb_s * 1000)} ms" if s.ttfb_s else "—"
    rtf = s.final_rtf
    rtf_str = f"{rtf:.2f}" if rtf is not None else "—"
    color = {
        "pending": "#888",
        "streaming": "#1f77b4",
        "done": "#2ca02c",
        "error": "#d62728",
    }[status]
    return (
        f'<div class="ccd-row" data-stream="{stream_id}" '
        f'data-status="{status}" style="display:flex;align-items:center;gap:8px;font-family:monospace">'
        f'<span style="width:32px">{label}</span>'
        f'<div style="flex:1;height:14px;background:#eee;border-radius:7px;overflow:hidden">'
        f'<div style="width:{pct:.1f}%;height:100%;background:{color}"></div></div>'
        f'<span style="width:80px">TTFB {ttfb_str}</span>'
        f'<span style="width:60px">RTF {rtf_str}</span>'
        f'<span style="width:64px;color:{color}">{status}</span>'
        f"</div>"
    )


def render_grid_html(snap: MetricsSnapshot) -> str:
    cells = []
    for s in snap.per_stream:
        color = {
            "pending": "#cccccc",
            "streaming": "#1f77b4",
            "done": "#2ca02c",
            "error": "#d62728",
        }[s.status]
        cells.append(
            f'<span data-cell="{s.stream_id}" '
            f'style="display:inline-block;width:18px;height:18px;margin:2px;'
            f"border-radius:9px;background:{color}\"></span>"
        )
    # Group every 8 cells per row so the 8x8 grid is visually obvious.
    rows = []
    for i in range(0, len(cells), 8):
        rows.append(f'<div style="line-height:0">{"".join(cells[i : i + 8])}</div>')
    return f'<div class="ccd-grid">{"".join(rows)}</div>'


def render_counters_html(snap: MetricsSnapshot) -> str:
    return (
        f'<div class="ccd-counters" style="display:flex;gap:24px;font-family:monospace">'
        f'<div><b>Active</b><br/>{snap.active}</div>'
        f'<div><b>Done</b><br/>{snap.completed}</div>'
        f'<div><b>Throughput</b><br/>{snap.throughput_x:.1f}×</div>'
        f'<div><b>TTFB p99</b><br/>{int(snap.ttfb_p99_ms) if snap.ttfb_p99_ms else "—"} ms</div>'
        f'<div><b>RTF p99</b><br/>{snap.rtf_p99:.2f if snap.rtf_p99 else "—"}</div>'
        f"</div>"
    )
```

- [ ] **Step 4: Run the tests, confirm they pass**

Run: `pytest tests/examples/online_serving/text_to_speech/qwen3_tts/test_views.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/views.py \
        tests/examples/online_serving/text_to_speech/qwen3_tts/test_views.py
git commit -s -m "feat(qwen3-tts-demo): HTML renderers for rows, grid, and counters"
```

---

### Task 8: Worker-loop runner + ref-audio loader

**Files:**
- Create: `examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/runtime.py`

- [ ] **Step 1: Write the worker-loop runner**

```python
# examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/runtime.py
"""Background asyncio worker loop and ref-audio loading.

Gradio click handlers call submit() from the UI thread; the burst executes on
this background loop. Snapshots remain thread-safe via MetricsAggregator's lock.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from concurrent.futures import Future
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
REF_AUDIO_PATH = REPO_ROOT / "tests" / "assets" / "qwen3_tts" / "clone_2.wav"


def load_ref_audio_b64() -> str:
    """Read the in-repo ref-audio fixture and return base64-encoded WAV bytes."""
    if not REF_AUDIO_PATH.is_file():
        raise FileNotFoundError(
            f"Ref audio not found at {REF_AUDIO_PATH}. The demo expects the vendored "
            "tests/assets/qwen3_tts/clone_2.wav. Re-clone or pull the repo to restore it."
        )
    return base64.b64encode(REF_AUDIO_PATH.read_bytes()).decode("ascii")


class WorkerLoop:
    """Owns a thread + asyncio event loop. submit() schedules a coroutine."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ccd-worker")
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def submit(self, coro) -> Future:
        if self._loop is None:
            raise RuntimeError("Worker loop not initialised")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
```

- [ ] **Step 2: Add a small smoke test for ref-audio loading**

Append to `tests/examples/online_serving/text_to_speech/qwen3_tts/test_orchestrator.py`:

```python
def test_load_ref_audio_b64_returns_nonempty_string() -> None:
    from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.runtime import (
        load_ref_audio_b64,
    )

    b64 = load_ref_audio_b64()
    assert isinstance(b64, str)
    assert len(b64) > 100  # arbitrary; clone_2.wav is ~757 KB
```

- [ ] **Step 3: Run the test; confirm it passes**

Run: `pytest tests/examples/online_serving/text_to_speech/qwen3_tts/test_orchestrator.py::test_load_ref_audio_b64_returns_nonempty_string -v`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/runtime.py \
        tests/examples/online_serving/text_to_speech/qwen3_tts/test_orchestrator.py
git commit -s -m "feat(qwen3-tts-demo): worker loop + in-repo ref-audio loader"
```

---

### Task 9: Gradio app — shell, two Tabs, Start/Reset wiring

**Files:**
- Create: `examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/app.py`

- [ ] **Step 1: Create the app skeleton**

```python
# examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/app.py
"""Gradio app: two tabs (N=8 per-stream, N=64 aggregate dashboard).

Launch:
    python -m examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.app \
        --api-base http://localhost:8000 --port 7860
"""

from __future__ import annotations

import argparse
import logging
import time

try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "gradio is required to run this demo. Install with: pip install 'vllm-omni[demo]'"
    ) from None

from .metrics import MetricsAggregator
from .orchestrator import Orchestrator
from .prompts import DEMO_PROMPT
from .runtime import WorkerLoop, load_ref_audio_b64
from .views import render_counters_html, render_grid_html, render_row_html

logger = logging.getLogger(__name__)

N_PAGE_A = 8
N_PAGE_B = 64


def build_ui(api_base: str) -> gr.Blocks:
    worker = WorkerLoop()
    ref_b64 = load_ref_audio_b64()
    orchestrator = Orchestrator(api_base=api_base, ref_audio_b64=ref_b64)
    agg_a = MetricsAggregator(n=N_PAGE_A)
    agg_b = MetricsAggregator(n=N_PAGE_B)

    def _eta_label(snap):
        ser = snap.serial_eta_s
        par = snap.parallel_eta_s
        sx = snap.speedup_x
        ser_s = f"{ser:.1f} s" if ser is not None else "—"
        par_s = f"{par:.1f} s" if par is not None else "—"
        sx_s = f"{sx:.1f}×" if sx is not None else "—"
        return f"<b>Serial ETA</b>&nbsp;{ser_s}&nbsp;&nbsp;<b>Parallel ETA</b>&nbsp;{par_s}&nbsp;&nbsp;<b>Speedup</b>&nbsp;{sx_s}"

    def _on_start(which: str):
        agg = agg_a if which == "A" else agg_b
        n = N_PAGE_A if which == "A" else N_PAGE_B
        worker.submit(orchestrator.run_burst(n=n, prompt=DEMO_PROMPT, aggregator=agg))
        return gr.update(value=f"Started N={n}…")

    def _on_reset(which: str):
        agg = agg_a if which == "A" else agg_b
        agg.reset()
        return gr.update(value=f"Reset N={N_PAGE_A if which == 'A' else N_PAGE_B}")

    def _tick_a():
        now = time.perf_counter()
        snap = agg_a.snapshot(now=now)
        rows_html = "".join(render_row_html(snap, i) for i in range(N_PAGE_A))
        return rows_html, _eta_label(snap)

    def _tick_b():
        now = time.perf_counter()
        snap = agg_b.snapshot(now=now)
        return render_counters_html(snap), render_grid_html(snap), _eta_label(snap)

    with gr.Blocks(title="Qwen3-TTS Concurrency Demo") as ui:
        gr.Markdown("# Qwen3-TTS Concurrency Demo")
        with gr.Tabs():
            with gr.Tab("Page A — N=8"):
                status_a = gr.Markdown("Idle.")
                eta_a = gr.HTML()
                rows_a = gr.HTML()
                start_a = gr.Button("▶ Start 8-stream race", variant="primary")
                reset_a = gr.Button("Reset")
                timer_a = gr.Timer(0.1)
                start_a.click(fn=lambda: _on_start("A"), outputs=status_a)
                reset_a.click(fn=lambda: _on_reset("A"), outputs=status_a)
                timer_a.tick(fn=_tick_a, outputs=[rows_a, eta_a])
            with gr.Tab("Page B — N=64"):
                status_b = gr.Markdown("Idle.")
                eta_b = gr.HTML()
                counters_b = gr.HTML()
                grid_b = gr.HTML()
                start_b = gr.Button("▶ Start 64-stream race", variant="primary")
                reset_b = gr.Button("Reset")
                timer_b = gr.Timer(0.2)
                start_b.click(fn=lambda: _on_start("B"), outputs=status_b)
                reset_b.click(fn=lambda: _on_reset("B"), outputs=status_b)
                timer_b.tick(fn=_tick_b, outputs=[counters_b, grid_b, eta_b])

    return ui


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", default="http://localhost:8000")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    ui = build_ui(api_base=args.api_base)
    ui.queue().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity import (no server needed)**

Run: `python -c "from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.app import build_ui; print('OK')"`
Expected: prints `OK` (provided `gradio` is installed in the env). If gradio is missing, the import will raise the friendly `ImportError`.

- [ ] **Step 3: Commit**

```bash
git add examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/app.py
git commit -s -m "feat(qwen3-tts-demo): Gradio app with two tabs (N=8 / N=64) + worker-loop wiring"
```

---

### Task 10: Convenience launcher script + README

**Files:**
- Create: `examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/run.sh`
- Create: `examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/README.md`

- [ ] **Step 1: Create run.sh**

```bash
#!/usr/bin/env bash
# Launch the concurrency demo against a running Qwen3-TTS-Base server.
#
# Prerequisite: start the server in another terminal:
#   vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni --port 8000
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8000}"
PORT="${PORT:-7860}"

if ! curl -sf "${API_BASE}/v1/models" > /dev/null; then
    echo "Server not reachable at ${API_BASE}. Start it with:"
    echo "  vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni --port 8000"
    exit 1
fi

exec python -m examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.app \
    --api-base "${API_BASE}" --port "${PORT}"
```

Then make it executable:

```bash
chmod +x examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/run.sh
```

- [ ] **Step 2: Create README.md**

```markdown
# Qwen3-TTS Concurrency Demo

A video-recording-first Gradio demo showing parallel-vs-serial throughput for
streaming Qwen3-TTS-Base under high concurrency.

Two tabs:

- **Page A (N=8)** — per-stream rows: progress bar, TTFB, RTF.
- **Page B (N=64)** — aggregate dashboard: counters, 8×8 stream-state grid.

Both pages share the headline `Serial ETA → Parallel ETA → Speedup ×` row.
The same fixed `DEMO_PROMPT` is sent for the c=1 reference and all N parallel
streams, so `serial_eta = N × t_observed` is an exact identity.

## Prerequisites

- A vllm-omni install with `gradio` available (`pip install 'vllm-omni[demo]'`).
- A running vllm server:
  ```bash
  vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni --port 8000
  ```

## Run

```bash
bash examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/run.sh
# or:
python -m examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.app \
    --api-base http://localhost:8000 --port 7860
```

Open http://localhost:7860 in your browser.

## Reference audio

The demo loads `tests/assets/qwen3_tts/clone_2.wav` from the repo. Nothing to
configure; nothing to download.

## Recording flow (~60 s clip)

| Time | Action |
|------|--------|
| 00:00 – 00:05 | Open Page A (N=8). Click Start. |
| 00:05 – 00:20 | Page A finishes. Speedup × locks in. |
| 00:20 – 00:25 | Switch to Page B tab. |
| 00:25 – 00:35 | Click Start. Brief c=1 pre-roll. |
| 00:35 – 00:55 | Stream grid lights up, counters climb, Speedup × spins to final. |
| 00:55 – 01:00 | Final frame: throughput dial peaked, 64/64 done, Speedup × locked. |

## Troubleshooting

- "Server not reachable" — make sure `vllm serve … --omni` is up.
- File-descriptor limit at N=64 — `ulimit -n 4096` before launching the demo
  on the client machine.
- Speedup × stuck at "—" — at least one stream failed; check the server log.

## Acceptance thresholds (h200-hsliu)

- Page A: Speedup × ≥ 4× over the c=1 reference.
- Page B: Speedup × ≥ 6× over the c=1 reference (calibrated against
  `tests/dfx/perf/tests/test_tts.json` Base c=64 audio throughput ≈ 14).
```

- [ ] **Step 3: Commit**

```bash
git add examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/run.sh \
        examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/README.md
git commit -s -m "docs(qwen3-tts-demo): run.sh launcher + README with recording flow"
```

---

### Task 11: Pre-commit + final lint pass

**Files:** all under the new package.

- [ ] **Step 1: Run ruff check + format on the new files**

```bash
TARGETS=(
  examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo
  tests/examples/online_serving/text_to_speech/qwen3_tts
)
uvx ruff@0.14.10 check --fix "${TARGETS[@]}"
uvx ruff@0.14.10 format "${TARGETS[@]}"
```

Expected: no remaining errors. If ruff introduces changes, stage them.

- [ ] **Step 2: Run the unit test suite for the new package**

Run:

```bash
pytest tests/examples/online_serving/text_to_speech/qwen3_tts -v
```

Expected: all tests pass (10 metrics + 3 orchestrator + 3 views + 1 ref-audio = 17 passes).

- [ ] **Step 3: Run pre-commit on staged files**

Run:

```bash
pre-commit run --files \
  examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/*.py \
  examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/run.sh \
  examples/online_serving/text_to_speech/qwen3_tts/concurrency_demo/README.md \
  tests/examples/online_serving/text_to_speech/__init__.py \
  tests/examples/online_serving/text_to_speech/qwen3_tts/*.py
```

Expected: all hooks pass.

- [ ] **Step 4: If pre-commit modified anything, stage and commit**

```bash
git add -p   # review carefully
git commit -s -m "chore(qwen3-tts-demo): satisfy pre-commit / ruff format"
```

(Skip this step if pre-commit had nothing to fix.)

---

### Task 12: Live smoke test against a real server (optional, gated)

**Files:**
- Create: `tests/examples/online_serving/text_to_speech/qwen3_tts/test_concurrency_demo_smoke.py`

This test is gated by `pytest.mark.full_model` and only runs on a box with
GPU + a launched vllm server. It is not part of the unit-test pass; CI skips
it unless the marker is selected.

- [ ] **Step 1: Write the live smoke test**

```python
# tests/examples/online_serving/text_to_speech/qwen3_tts/test_concurrency_demo_smoke.py
"""Live smoke test: hits a real vllm server. Gated by the full_model marker."""

from __future__ import annotations

import asyncio
import os

import pytest

from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.metrics import (
    MetricsAggregator,
)
from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.orchestrator import (
    Orchestrator,
)
from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.prompts import (
    DEMO_PROMPT,
)
from examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.runtime import (
    load_ref_audio_b64,
)

pytestmark = pytest.mark.full_model


@pytest.mark.asyncio
async def test_run_burst_n8_against_live_server() -> None:
    api_base = os.environ.get("QWEN3_TTS_API_BASE", "http://localhost:8000")
    orchestrator = Orchestrator(api_base=api_base, ref_audio_b64=load_ref_audio_b64())
    aggregator = MetricsAggregator(n=8)
    await orchestrator.run_burst(n=8, prompt=DEMO_PROMPT, aggregator=aggregator)
    snap = aggregator.snapshot(now=9999.0)
    assert snap.completed == 8, f"only {snap.completed}/8 completed: {snap.per_stream}"
    assert snap.speedup_x is not None and snap.speedup_x >= 4.0, (
        f"speedup {snap.speedup_x} below 4x target"
    )
```

- [ ] **Step 2: Run it locally against a real server**

On a machine with the server running:

```bash
QWEN3_TTS_API_BASE=http://localhost:8000 \
  pytest tests/examples/online_serving/text_to_speech/qwen3_tts/test_concurrency_demo_smoke.py -v -m full_model
```

Expected: 1 passed with `speedup_x >= 4.0`. If the speedup is below 4×, check
that the server is warm and that no other workload is contending for the GPU.

- [ ] **Step 3: Commit**

```bash
git add tests/examples/online_serving/text_to_speech/qwen3_tts/test_concurrency_demo_smoke.py
git commit -s -m "test(qwen3-tts-demo): live full-model smoke test for N=8 burst"
```

---

### Task 13: Push and open the PR

- [ ] **Step 1: Push the feature branch**

```bash
git push origin feat/qwen3-tts-concurrency-demo
```

- [ ] **Step 2: Open a PR against `upstream/main` via the GitHub UI**

Title:

```
[Demo][TTS] Qwen3-TTS streaming concurrency demo (parallel-vs-serial visualisation)
```

Body should reference:
- The spec: `docs/superpowers/specs/2026-05-31-qwen3-tts-concurrency-demo-design.md`
- Page A target ≥4×, Page B target ≥6×, with a screenshot of a recorded run.
- The recording flow from `README.md` §Recording flow.

- [ ] **Step 3: Record the video**

Follow the recording flow in the README; embed the video link in the PR body.

---

## Self-Review

This block records the post-write check against the spec.

**Spec coverage:**

| Spec section | Implementing task(s) |
|---|---|
| §3.1 Two pages as Tabs | Task 9 |
| §3.2 Page A: per-stream rows | Task 7 (views) + Task 9 (wiring) |
| §3.3 Page B: aggregate dashboard | Task 7 (grid + counters) + Task 9 (wiring) |
| §4 File layout | Tasks 1, 2, 5, 7, 8, 9, 10 |
| §5 Data flow + event-loop safety | Task 8 (WorkerLoop) + Task 6 (run_burst) + Task 3 (threading.Lock) |
| §6 Metric formulas | Task 2 (StreamState.audio_seconds) + Task 4 (snapshot()) |
| §7 DEMO_PROMPT | Task 1 |
| §8 Reference audio from tests/assets/qwen3_tts/clone_2.wav | Task 8 |
| §9 Error handling, speedup suppression on failure | Task 3 (apply error) + Task 4 (speedup_x is None if any_failed) |
| §10 Recording flow | Task 10 (README) |
| §11.5 Deployment notes (httpx pool, fd limit) | Task 10 (README troubleshooting) |
| §13 Acceptance criteria 4× / 6× | Task 12 (gated live smoke test for 4×) |

**Placeholder scan:** no "TBD", no "implement later", no "similar to Task N";
every step contains the code or command an engineer needs.

**Type consistency:** `MetricsSnapshot`, `MetricsAggregator`, `StreamEvent`,
`StreamState`, `Orchestrator`, `StreamConfig` are referenced with the same
names across all tasks. The `apply` / `mark_request_sent` / `mark_burst_start`
/ `set_reference` / `reset` / `snapshot` method signatures match between the
producer (Tasks 3–4) and the consumers (Tasks 5–7, 9). Page A target 4× and
Page B target 6× match the spec §13.

**Known limitations of this plan:**

- AudioWorklet preview for stream #1 on Page A is **not** implemented in this
  plan. It is listed as a stretch follow-up in the README; the core "Speedup ×"
  story does not depend on audible playback.
- The live smoke test for Page B (N=64, ≥6×) is left as a follow-on; only the
  N=8 ≥4× smoke test is wired in Task 12.
- No CI integration. Pre-commit gates style; the unit tests do not require a
  GPU and could be opted into CI in a follow-up PR.
