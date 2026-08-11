#!/bin/bash
# Open-loop mixed-load dataset collection for the D* label extractor.
# Server must already be running the static config (qwen3_tts_ramp_dense.yaml)
# on port 8091. Rates calibrated from r1.6_t0_s0 (realized N=25.5, T~17s)
# plus closed-loop T anchors: 2.4->~64, 2.8->~100, 3.0->~112, 3.3->~128.
cd ~/proj/vllm-omni-legato
code=$(curl -s -o /dev/null -w "%{http_code}" -m 2 http://localhost:8091/health 2>/dev/null)
[ "$code" != "200" ] && { echo SERVER_DOWN > openloop.marker; exit 1; }
T0="The quick brown fox jumps over the lazy dog while the river keeps moving under the old stone bridge. Every evening the town grows quiet, and the lamps come on one by one along the water."
T3="Seventy three percent of the survey respondents reported that their commute changed substantially after the new line opened in twenty twenty four, with average travel times falling from forty one minutes to twenty six."
mkdir -p openloop
for seed in 0 1; do
  for rate in 1.6 2.4 2.8 3.0 3.3; do
    for ti in 0 3; do
      tag="r${rate}_t${ti}_s${seed}"
      [ -f "openloop/${tag}.ndjson" ] && continue
      if [ "$ti" = "0" ]; then TXT="$T0 $T0 $T0 $T0"; else TXT="$T3 $T3 $T3 $T3"; fi
      .venv/bin/python legato_bench_async.py --arrival-rate "$rate" --duration-s 180 \
        --seed "$seed" --text "$TXT" \
        --out "openloop/${tag}.ndjson" --metrics-out "openloop/${tag}.metrics.ndjson" \
        > "openloop/${tag}.log" 2>&1
      echo "${tag} $(date +%H:%M:%S) $(grep -o "AGGREGATE.*" openloop/${tag}.log | head -c 400)" >> openloop/progress.txt
      sleep 20
    done
  done
done
echo DONE > openloop.marker
