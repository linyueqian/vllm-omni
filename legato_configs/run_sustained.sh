#!/bin/bash
cd ~/proj/vllm-omni-legato
T0="The quick brown fox jumps over the lazy dog while the river keeps moving under the old stone bridge. Every evening the town grows quiet, and the lamps come on one by one along the water."
mkdir -p sustained
.venv/bin/python legato_bench_async.py --concurrency 120 --n 4 --stagger 0.15 --text "$T0 $T0 $T0 $T0" --out sustained/c120_n4.ndjson --metrics-out sustained/c120_n4.metrics.ndjson > sustained/c120_n4.log 2>&1
sleep 10
.venv/bin/python legato_bench_async.py --concurrency 96 --n 4 --stagger 0.15 --text "$T0 $T0 $T0 $T0" --out sustained/c96_n4.ndjson > sustained/c96_n4.log 2>&1
echo DONE > sustained.marker
