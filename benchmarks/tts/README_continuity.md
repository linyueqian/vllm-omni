# TTS streaming continuity benchmark

Measures the **audio gap** problem visible in streaming TTS at high concurrency:
how long does a listener wait between consecutive PCM chunks once playback has
started, and how does that p99 inter-chunk gap evolve as concurrency grows.

Outputs are stable JSON that `aggregate_underrun.py` summarizes across runs.

## What it measures

For every streamed request the client records the wall-clock arrival of each
non-empty audio chunk. From that timeline it derives:

| metric | definition |
|---|---|
| `ttft_ms` | request_start → first response byte |
| `ttfp_ms` | request_start → first non-empty audio chunk (first PCM the listener hears) |
| `e2el_ms` | request_start → last byte |
| `audio_duration_s` | total PCM bytes / (sample_rate × 2) |
| `audio_rtf` | `e2el_s / audio_duration_s` (1.0 = realtime; >1 = slower than realtime) |
| `max_inter_chunk_gap_ms` | max wall-clock gap between consecutive chunks (post-first-chunk) |
| `max_underrun_ms` | max amount by which the next chunk landed **after** the listener would have run out of audio |
| `underrun_rate` | fraction of requests that had ≥ 1 underrun > 100 ms |
| `num_continuity_ok` | requests with zero underruns |

`max_underrun_ms` is the metric that maps directly to "would the user hear a
gap" — it accounts for both the inter-chunk gap and the length of audio already
buffered.

## Quick start (single config)

```bash
# 1. Launch a server with the deploy yaml you want to test.
.venv/bin/vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni \
    --deploy-config configs/my_deploy.yaml --port 8091

# 2. Bench it at concurrency 128 (2 prompts per worker).
python benchmarks/tts/bench_tts_continuity.py \
    --base-url http://127.0.0.1:8091 \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --num-prompts 256 --concurrency 128 --num-warmups 2 \
    --extra-body-file benchmarks/tts/extra_bodies/qwen3_default_voice.json \
    --sample-rate 24000 \
    --output /tmp/c128.json
```

## Sweep one (variant, model) across concurrencies

`run_continuity_variant.sh` launches the server, waits ready, runs the bench at
each concurrency, then tears down. Path-agnostic via env vars; see the script
header for the full list.

```bash
REPO=$(pwd) \
OUT_ROOT=/tmp/tts_bench/results \
LOG_ROOT=/tmp/tts_bench/logs \
bash benchmarks/tts/run_continuity_variant.sh \
    my_variant configs/my_deploy.yaml custom 0 8091 "64 128 256"
```

`base` runs `Qwen3-TTS-Base` + voice_clone (requires
`benchmarks/tts/extra_bodies/qwen3_voice_clone.json` — see that directory's
README for how to generate one). `custom` runs `Qwen3-TTS-CustomVoice` + the
default-voice extra body which ships in the repo.

## Aggregate across runs

```bash
python benchmarks/tts/aggregate_underrun.py \
    /tmp/tts_bench/results \
    --prefix my_variant_=my_variant \
    --output /tmp/tts_bench/summary.json
```

Prints one row per `(label, task, concurrency)` cell with the underrun /
TTFT / RTF percentiles. Without `--prefix` the directory name is used as the
label and `task` is left blank.

## Layout the scripts assume

```
$OUT_ROOT/
  <variant>_base_voice_clone/c{64,128,256}.json
  <variant>_custom_default_voice/c{64,128,256}.json
$LOG_ROOT/
  <variant>_<task>/server.log
```

Each `c<N>.json` carries both per-request fields (under `per_request_metrics`)
and the aggregated `*_p50` / `*_p99` / `*_max` columns the aggregator reads.
