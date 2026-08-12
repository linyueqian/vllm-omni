#!/bin/bash
# On-policy A/B for the two-regime pacer: rates {2.4,3.0} x texts {t0,t3} x
# {pacer off, pacer theta=2}, 180 s each, seed 7 (unseen), matched arrivals.
# Optional phase 2: theta=8 arms at 2.4 (off arms reused), only if phase 1 is
# fully healthy. Server (port 8091, qwen3_tts_ramp_dense.yaml) used as-is.
cd ~/proj/vllm-omni-legato
code=$(curl -s -o /dev/null -w "%{http_code}" -m 2 http://localhost:8091/health 2>/dev/null)
[ "$code" != "200" ] && { echo SERVER_DOWN > onpolicy.marker; exit 1; }
T0="The quick brown fox jumps over the lazy dog while the river keeps moving under the old stone bridge. Every evening the town grows quiet, and the lamps come on one by one along the water."
T3="Seventy three percent of the survey respondents reported that their commute changed substantially after the new line opened in twenty twenty four, with average travel times falling from forty one minutes to twenty six."
TXT0="$T0 $T0 $T0 $T0"
TXT3="$T3 $T3 $T3 $T3"
SEED=7
mkdir -p onpolicy

run_one() {
  tag="$1"; rate="$2"; txt="$3"; shift 3
  [ -f "onpolicy/${tag}.ndjson" ] && return 0
  .venv/bin/python legato_bench_async.py --arrival-rate "$rate" --duration-s 180 \
    --seed "$SEED" --text "$txt" \
    --out "onpolicy/${tag}.ndjson" --metrics-out "onpolicy/${tag}.metrics.ndjson" \
    "$@" > "onpolicy/${tag}.log" 2>&1
  echo "${tag} $(date +%H:%M:%S) $(grep -o 'AGGREGATE.*' onpolicy/${tag}.log | head -c 600)" >> onpolicy/progress.txt
  sleep 20
}

PM=scripts/predictor/pacer_model.joblib
for rate in 2.4 3.0; do
  for ti in 0 3; do
    if [ "$ti" = "0" ]; then TXT="$TXT0"; else TXT="$TXT3"; fi
    run_one "r${rate}_t${ti}_s7_off" "$rate" "$TXT"
    run_one "r${rate}_t${ti}_s7_th2" "$rate" "$TXT" --pacer-model "$PM" --pacer-theta 2
  done
done

# Phase 2 (optional theta=8 at 2.4): only if all 8 phase-1 runs produced output
# and none errored at the client level.
n_ok=0
for f in onpolicy/r2.4_t0_s7_off onpolicy/r2.4_t0_s7_th2 onpolicy/r2.4_t3_s7_off \
         onpolicy/r2.4_t3_s7_th2 onpolicy/r3.0_t0_s7_off onpolicy/r3.0_t0_s7_th2 \
         onpolicy/r3.0_t3_s7_off onpolicy/r3.0_t3_s7_th2; do
  [ -s "${f}.ndjson" ] && n_ok=$((n_ok+1))
done
if [ "$n_ok" = "8" ]; then
  run_one "r2.4_t0_s7_th8" 2.4 "$TXT0" --pacer-model "$PM" --pacer-theta 8
  run_one "r2.4_t3_s7_th8" 2.4 "$TXT3" --pacer-model "$PM" --pacer-theta 8
  echo DONE > onpolicy.marker
else
  echo "PARTIAL n_ok=${n_ok}" > onpolicy.marker
fi
