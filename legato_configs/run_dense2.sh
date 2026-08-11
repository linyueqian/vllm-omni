#!/bin/bash
cd ~/proj/vllm-omni-legato
pkill -f "[v]llm serve" ; sleep 8
export PATH=$HOME/proj/vllm-omni-legato/.venv/bin:$PATH
CUDA_VISIBLE_DEVICES=1 nohup vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni --port 8091 --deploy-config legato_configs/qwen3_tts_ramp_dense2.yaml > serve_dense2.log 2>&1 &
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 2 http://localhost:8091/health 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 5
done
[ "$code" != "200" ] && { echo SERVER_FAILED > dense2.marker; exit 1; }
T0="The quick brown fox jumps over the lazy dog while the river keeps moving under the old stone bridge. Every evening the town grows quiet, and the lamps come on one by one along the water."
mkdir -p dense2
.venv/bin/python legato_bench_async.py --concurrency 96 --n 4 --stagger 0.15 --text "$T0 $T0 $T0 $T0" --out dense2/c96_n4.ndjson --metrics-out dense2/c96_n4.metrics.ndjson > dense2/c96_n4.log 2>&1
sleep 10
.venv/bin/python legato_bench_async.py --concurrency 120 --n 4 --stagger 0.15 --text "$T0 $T0 $T0 $T0" --out dense2/c120_n4.ndjson --metrics-out dense2/c120_n4.metrics.ndjson > dense2/c120_n4.log 2>&1
echo DONE > dense2.marker
