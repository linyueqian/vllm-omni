#!/bin/bash
cd ~/proj/vllm-omni-legato
pkill -f "[v]llm serve" ; sleep 8
export PATH=$HOME/proj/vllm-omni-legato/.venv/bin:$PATH
CUDA_VISIBLE_DEVICES=1 nohup vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni --port 8091 --deploy-config legato_configs/qwen3_tts_ramp_dense.yaml > serve_dense.log 2>&1 &
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 2 http://localhost:8091/health 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 5
done
[ "$code" != "200" ] && { echo SERVER_FAILED > density.marker; exit 1; }
T0="The quick brown fox jumps over the lazy dog while the river keeps moving under the old stone bridge. Every evening the town grows quiet, and the lamps come on one by one along the water."
T3="Seventy three percent of the survey respondents reported that their commute changed substantially after the new line opened in twenty twenty four, with average travel times falling from forty one minutes to twenty six."
mkdir -p density
for c in 64 96 128; do
  .venv/bin/python legato_bench_async.py --concurrency $c --text "$T0 $T0 $T0 $T0" --out density/c${c}_t0.ndjson --metrics-out density/c${c}_t0.metrics.ndjson > density/c${c}_t0.log 2>&1
  sleep 10
  .venv/bin/python legato_bench_async.py --concurrency $c --text "$T3 $T3 $T3 $T3" --out density/c${c}_t3.ndjson --metrics-out density/c${c}_t3.metrics.ndjson > density/c${c}_t3.log 2>&1
  sleep 10
done
echo DONE > density.marker
