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
