# Qwen3-TTS Concurrency Demo — Design Spec

**Date:** 2026-05-31
**Owner:** Yueqian Lin
**Status:** Draft, pending implementation
**Target branch:** new feature branch off `main`

## 1. Goals and non-goals

### 1.1 Goal

A video-recordable Gradio demo that visualises the throughput advantage of
vllm-omni's streaming Qwen3-TTS server under high concurrency. The headline
story is parallel-vs-serial speedup expressed as a single bold number, plus a
live concurrency visualisation that reads clearly on a screen recording.

### 1.2 Non-goals

* Multi-model comparison (Qwen3-TTS vs VoxCPM2 / Fish Speech / MOSS-TTS-Nano).
* Multi-framework comparison (vllm-omni vs sglang / vanilla vLLM / HF
  Transformers).
* Prefix-cache ON vs OFF A/B (PR #3665 story).
* Interactive playground UX. Demo is video-first; controls are minimal.

These are documented as future work in §11.

## 2. Audience and constraints

* **Primary use:** screen-recorded 60–90 s video clip suitable for embedding in
  release notes, conference talks, and social posts.
* **Secondary use:** local interactive demonstration when running the repo
  locally.
* **Hardware target:** h200-hsliu (single GPU). Other H100/H200/H20 boxes work
  but baseline numbers in the README are pinned to h200-hsliu.
* **Server:** stock `vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni`. No
  prefix-cache flag manipulation, no custom deploy YAML overrides.

## 3. User-visible behaviour

### 3.1 Two pages, served as Gradio Tabs

| Page | Concurrency | Visualisation |
|------|-------------|---------------|
| A | N = 8 | Per-stream rows: progress bar + TTFB + RTF, plus one selectable live audio preview |
| B | N = 64 | Aggregate dashboard: top counters, Speedup × badge, 8×8 stream-state grid |

Both pages share the same top-level stopwatch, the same Start/Reset controls,
and the same "Serial ETA — Parallel ETA — Speedup ×" row.

### 3.2 Page A: N = 8

ASCII reference layout:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Qwen3-TTS — 8 concurrent streams                          ⏱  00:00.000     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Serial reference  ████░░░░░░░░░░░░░░░  ETA  00:42.3   Speedup  █×          │
├─────────────────────────────────────────────────────────────────────────────┤
│  #1       ▌▌▌▌▌▌▌▌░░░░  TTFB 312ms  RTF 0.18  ▶ live audio                  │
│  …  (8 rows total, all running the same fixed prompt) …                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  [ ▶  Start 8-stream race ]   [ Reset ]   ☐ play stream #1                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

Behaviour on Start:

1. Fire one live c=1 reference stream against the same server. Record
   `t_observed`.
2. Fire 8 concurrent `/v1/audio/speech` requests via `asyncio.gather`.
3. Per stream, update its row at 10 Hz: progress bar fills proportional to
   received audio bytes vs expected duration (text length × pace estimate);
   TTFB locks at first chunk; RTF computes live.
4. Serial reference bar fills to `8 × t_observed`. Speedup × locks in once all
   8 streams finish.
5. If the **play stream #1** checkbox is on, route stream #1's PCM through the
   AudioWorklet from `qwen3_tts/gradio_demo.py`. Streams #2 – #8 are visualised
   silently.

### 3.3 Page B: N = 64

ASCII reference layout:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Qwen3-TTS — 64 concurrent streams                        ⏱  00:00.000      │
├─────────────────────────────────────────────────────────────────────────────┤
│   [Active]    [Done]    [Throughput]    [TTFB p99]    [RTF p99]             │
│      0          0          0.0×           — ms          —                   │
├─────────────────────────────────────────────────────────────────────────────┤
│   Serial ETA   00:05:42   ↔   Parallel ETA   00:00:08   →   Speedup  42.6× │
├─────────────────────────────────────────────────────────────────────────────┤
│   Stream grid (8 × 8, lights up live; pending → streaming → done)           │
├─────────────────────────────────────────────────────────────────────────────┤
│  [ ▶  Start 64-stream race ]   [ Reset ]   [ ▶ play sample (post-run) ]     │
└─────────────────────────────────────────────────────────────────────────────┘
```

Behaviour on Start:

1. Fire one live c=1 reference stream as a pre-roll. Record `t_observed`.
2. Fire 64 concurrent `/v1/audio/speech` requests.
3. Five top counters update at 5 Hz from the metrics aggregator.
4. Stream grid: each of 64 dots transitions from `pending` to `streaming` (on
   first chunk) to `done` (on stream close). Re-rendered as a single `gr.HTML`
   block.
5. No live audio during the burst. After the burst, a **play sample** button
   plays stream #0's already-buffered PCM.

Because all 64 streams share the same prompt and the same ref-audio, the
8×8 grid lights up in roughly synchronised waves, which reads cleanly on
video. Length variation that would have spread out finish times is
deliberately avoided — see §7.

## 4. Architecture and file layout

```
examples/online_serving/text_to_speech/qwen3_tts/
├── gradio_demo.py                  # existing single-stream demo (unchanged)
├── concurrency_demo/               # NEW
│   ├── __init__.py
│   ├── app.py                      # Gradio entrypoint; two pages (gr.Tabs)
│   ├── orchestrator.py             # asyncio gather over N streams + c=1 ref
│   ├── metrics.py                  # MetricsAggregator + snapshot dataclass
│   ├── views.py                    # row renderer (N=8) + grid renderer (N=64)
│   ├── prompts.py                  # DEMO_PROMPT (single fixed string)
│   ├── run.sh                      # convenience launcher
│   └── README.md                   # how to record the video
```

### 4.1 Reuse

* Payload shape and HTTP client pattern: `qwen3_tts/gradio_demo.py` and
  `qwen3_tts/tts_common.py`. The demo calls `POST /v1/audio/speech` and consumes
  the raw streaming PCM body via `httpx.AsyncClient.stream(...).aiter_bytes()` —
  **not** the WebSocket path at `/v1/audio/speech/stream` exposed by
  `qwen3_tts/streaming_speech_client.py`.
* AudioWorklet JS for gap-free playback: `qwen3_tts/gradio_demo.py` (used only
  on Page A for the single selectable preview stream).
* Payload builder: `qwen3_tts.tts_common.build_payload`.

### 4.2 New code

* Orchestrator: thin asyncio harness, ~150 LOC.
* Metrics aggregator: dataclass + reducer over `StreamEvent`s, ~100 LOC.
* Views: row and grid HTML renderers, ~150 LOC.
* Gradio app: tabs, timers, wiring, ~200 LOC.
* Prompts: literal strings, ~100 LOC.
* README + run.sh: ~100 LOC.

Total estimate: 600 – 800 LOC.

## 5. Data flow

```
Gradio app.py (Page A or B)
       │ on Start click
       ▼
orchestrator.run_burst(N, prompts, ref_audio_path)
       │
       ├──► (1) c=1 reference: single async POST /v1/audio/speech, stream=true
       │         collect t_first_chunk, t_last_chunk → t_observed
       │
       └──► (2) parallel burst: asyncio.gather([stream(i) for i in range(N)])
                 each stream(i):
                   - opens httpx.AsyncClient.stream("POST", "/v1/audio/speech", ...)
                   - iterates response.aiter_bytes()
                   - on first byte chunk: emit StreamEvent(id=i, kind="first", ts)
                   - on each chunk:       emit StreamEvent(id=i, kind="chunk", bytes, ts)
                   - on close:            emit StreamEvent(id=i, kind="done", ts)

asyncio.Queue[StreamEvent]
       │ consumed by
       ▼
metrics.MetricsAggregator
       │ maintains per-stream: ttfb, last_chunk_ts, bytes_received, status
       │ derives: throughput_dial, p99 ttfb, p99 rtf, completed_count,
       │          serial_eta, parallel_eta, speedup_x
       ▼
views.snapshot()  →  dict for gr.update
       │ polled by gr.Timer at 10 Hz (Page A) / 5 Hz (Page B)
       ▼
Gradio components re-render
```

**Event-loop safety.** The Start handler schedules the burst on a background
asyncio task (started via `asyncio.run_coroutine_threadsafe` against a worker
loop owned by the orchestrator) and returns immediately so Gradio can keep
ticking. `MetricsAggregator` publishes immutable snapshots under a short
`threading.Lock`; `gr.Timer` callbacks acquire the lock only to read the
latest snapshot and never block on I/O. Timer callbacks are registered with
`queue=False` so stale ticks cannot accumulate in Gradio's event queue under
load.

## 6. Metrics (exact definitions)

Sample rate is 24 kHz mono int16 (2 bytes/sample) for Qwen3-TTS streamed PCM
(confirmed by `tts_common.py:30` — `PCM_SAMPLE_RATE = 24000`).

| Name | Formula |
|------|---------|
| `ttfb_i` | `t_first_chunk_i − t_request_sent_i`, ms |
| `audio_seconds_i` | `bytes_received_i / (24_000 × 2)` |
| `rtf_i` (live) | `(now − t_request_sent_i) / audio_seconds_i` |
| `rtf_i` (final) | `(t_last_chunk_i − t_request_sent_i) / total_audio_seconds_i` |
| `throughput_dial` | `Σ audio_seconds_i / (now − t_burst_start)` |
| `serial_eta` | `N × (t_last_chunk_ref − t_first_chunk_ref + ttfb_ref)` |
| `parallel_eta` | `max_i(t_last_chunk_i) − t_burst_start` |
| `speedup_x` | `serial_eta / parallel_eta`, computed only after all N done |

* `t_request_sent_i` is logged immediately before `httpx.AsyncClient.stream(...)`
  to avoid client-side scheduling bias.
* p50 and p99 are reported over `ttfb_i` and final `rtf_i`.

## 7. Prompts

`prompts.py` exposes a single module-level constant:

* `DEMO_PROMPT` — one fixed English string expected to produce ~6 – 8 s of
  audio at standard pace.

Both pages use this same string for **all** streams: the c=1 serial reference
and every parallel stream send identical text. This makes `serial_eta` an
exact identity, not an extrapolation, modulo the small server-side cache
warm-up effect documented in §8. Visual variety is sacrificed in exchange for
defensible math — the speedup number on screen is the only thing the audience
is asked to trust.

Prompts are hard-coded literals. No dataset dependency at runtime.

## 8. Server and reference audio

* Server command: `vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni --port 8000`.
  No deploy-YAML override, no flag flips. Whatever defaults the repo ships
  with at the time of recording are what gets demoed.
* Reference audio: `tests/assets/qwen3_tts/clone_2.wav` (already committed in
  the repo, 757 KB; same fixture used by
  `tests/e2e/online_serving/test_qwen3_tts_base.py:24-27`). The demo loads it
  from the repo at startup and base64-encodes it into the request payload.
  No symlink, no external dataset checkout, no README repointing step.
* Server-side ref-audio caching: orchestrator issues one warmup request at
  startup before opening the Gradio UI, ensuring the ref-audio artifact
  cache is hot and per-request overhead is minimised. The c=1 reference
  stream therefore reflects post-warmup steady-state latency, and
  `serial_eta = N × t_observed` is exact up to per-call jitter. This
  warmup is not visible in the recorded video.

## 9. Error handling

* Per-stream failures (HTTP error, timeout, transport drop): mark that row
  (Page A) or dot (Page B) red and continue. The final **Speedup ×** metric
  is suppressed if any stream failed; instead the UI shows an `n/N completed`
  annotation. Partial-completion runs do not satisfy the acceptance criteria
  in §13.
* Burst-level timeout: a run is considered complete when all N streams
  return **or** 60 s have elapsed, whichever first. The metrics aggregator
  freezes its final snapshot at that point.
* No retries. This is a demo, not a load test; failures should be visible,
  not papered over.

## 10. Recording flow

Target clip length: ~60 s.

| Time | Action |
|------|--------|
| 00:00 – 00:05 | Open Page A (N=8). Click Start. |
| 00:05 – 00:20 | Page A finishes. Speedup × locks in. |
| 00:20 – 00:25 | Tab-switch to Page B (N=64). |
| 00:25 – 00:35 | Click Start. Brief c=1 pre-roll. |
| 00:35 – 00:55 | Stream grid lights up; counters climb; Speedup × spins to its final value. |
| 00:55 – 01:00 | Final frame: throughput dial peaked, 64/64 done, Speedup × locked. |

Pages are `gr.Tabs`. Switching is one click, no URL change, no reload.

## 11. Future work (out of scope for this spec)

* **Multi-model zoo extension.** Add VoxCPM2, Fish Speech, MOSS-TTS-Nano as
  selectable backends; same UI, different server.
* **Prefix-cache ON vs OFF mode.** Add a dropdown that switches between two
  pre-launched servers (one with `--no-enable-prefix-caching`). Tells the
  PR #3665 story directly.
* **Framework baseline.** Stand up sglang or vanilla vLLM serving the same
  Qwen3-TTS checkpoint as a side-by-side baseline.
* **Time-series throughput chart.** Live line chart of throughput vs wall
  time during the burst.
* **Persist a run-summary JSON.** After each burst, dump prompt text,
  per-stream latencies, throughput trace, and final speedup × to an
  output directory next to the recorded video. Reuse the output-dir
  pattern from `benchmarks/tts/bench_tts.py:139-145, 298-302`.
* **Varied-prompt mode.** Add an option to draw prompts from a fixed
  multilingual mix; expose a UI disclaimer that the speedup number then
  becomes a coarser projection. Keeps the math defensible while making
  the visualisation more interesting if the audience asks.

## 11.5 Deployment notes

* The client machine opens 64 concurrent HTTP streaming requests at peak.
  Set `httpx.AsyncClient(limits=httpx.Limits(max_connections=128, max_keepalive_connections=128))`
  and ensure the process file-descriptor ceiling is at least 256
  (`ulimit -n 4096` is safe).
* `gr.HTML` re-render cost at 5 Hz with 64 grid cells is negligible
  (<1 ms per tick); the bottleneck is the metrics aggregator's snapshot
  copy, which is O(N).

## 12. Open questions

None at spec-approval time. All decisions captured above.

## 13. Acceptance criteria

* `python -m examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.app`
  launches a Gradio UI with two working tabs against a running
  `vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni` server.
* On Page A: Start runs N=8 successfully on h200-hsliu, all 8 rows complete
  within the 60 s budget, Speedup × is at least **4×** over the c=1
  reference.
* On Page B: Start runs N=64 successfully on h200-hsliu, all 64 dots
  transition to `done` within the 60 s budget, Speedup × is at least
  **6×** over the c=1 reference. This 6× target is calibrated from
  `tests/dfx/perf/tests/test_tts.json` (Base c=1 median RTF ~0.6,
  c=64 audio throughput ~14 → ~6–7× real-time ratio improvement). A
  higher target is treated as a stretch goal and will only replace the
  6× number after a checked-in h200-hsliu run confirms it.
* README documents the exact server command, the in-repo ref-audio path
  (`tests/assets/qwen3_tts/clone_2.wav`), and the recording flow.
