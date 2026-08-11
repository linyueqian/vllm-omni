#!/bin/bash
cd ~/proj/vllm-omni-legato
code=$(curl -s -o /dev/null -w "%{http_code}" -m 2 http://localhost:8091/health 2>/dev/null)
[ "$code" != "200" ] && { echo SERVER_DOWN > openloop_cal.marker; exit 1; }
T0="The quick brown fox jumps over the lazy dog while the river keeps moving under the old stone bridge. Every evening the town grows quiet, and the lamps come on one by one along the water."
mkdir -p openloop
.venv/bin/python legato_bench_async.py --arrival-rate 1.6 --duration-s 180 --seed 0 --text "$T0 $T0 $T0 $T0" --out openloop/r1.6_t0_s0.ndjson --metrics-out openloop/r1.6_t0_s0.metrics.ndjson > openloop/r1.6_t0_s0.log 2>&1
echo DONE > openloop_cal.marker
