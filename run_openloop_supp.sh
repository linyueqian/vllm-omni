#!/bin/bash
cd ~/proj/vllm-omni-legato
code=$(curl -s -o /dev/null -w "%{http_code}" -m 2 http://localhost:8091/health 2>/dev/null)
[ "$code" != "200" ] && { echo SERVER_DOWN > openloop_supp.marker; exit 1; }
T0="The quick brown fox jumps over the lazy dog while the river keeps moving under the old stone bridge. Every evening the town grows quiet, and the lamps come on one by one along the water."
T3="Seventy three percent of the survey respondents reported that their commute changed substantially after the new line opened in twenty twenty four, with average travel times falling from forty one minutes to twenty six."
for rate in 1.8 2.0 2.2; do
  for ti in 0 3; do
    tag="r${rate}_t${ti}_s0"
    [ -f "openloop/${tag}.ndjson" ] && continue
    if [ "$ti" = "0" ]; then TXT="$T0 $T0 $T0 $T0"; else TXT="$T3 $T3 $T3 $T3"; fi
    .venv/bin/python legato_bench_async.py --arrival-rate "$rate" --duration-s 180 \
      --seed 0 --text "$TXT" \
      --out "openloop/${tag}.ndjson" --metrics-out "openloop/${tag}.metrics.ndjson" \
      > "openloop/${tag}.log" 2>&1
    echo "${tag} $(date +%H:%M:%S)" >> openloop/progress.txt
    sleep 20
  done
done
echo DONE > openloop_supp.marker
