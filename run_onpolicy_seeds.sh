#!/bin/bash
# Multi-seed replication of the on-policy pacer A/B: heavy pairs
# {2.4/t3, 3.0/t0, 3.0/t3} x {off, theta=2} at seeds 8 and 9 (12 runs),
# then optional {2.4/t0} pairs if all 12 are healthy. Same protocol as
# run_onpolicy.sh (180 s, metrics-out, frozen model, server untouched).
cd ~/proj/vllm-omni-legato
code=$(curl -s -o /dev/null -w "%{http_code}" -m 2 http://localhost:8091/health 2>/dev/null)
[ "$code" != "200" ] && { echo SERVER_DOWN > onpolicy_seeds.marker; exit 1; }
T0="The quick brown fox jumps over the lazy dog while the river keeps moving under the old stone bridge. Every evening the town grows quiet, and the lamps come on one by one along the water."
T3="Seventy three percent of the survey respondents reported that their commute changed substantially after the new line opened in twenty twenty four, with average travel times falling from forty one minutes to twenty six."
TXT0="$T0 $T0 $T0 $T0"
TXT3="$T3 $T3 $T3 $T3"
mkdir -p onpolicy

run_one() {
  tag="$1"; rate="$2"; seed="$3"; txt="$4"; shift 4
  [ -f "onpolicy/${tag}.ndjson" ] && return 0
  .venv/bin/python legato_bench_async.py --arrival-rate "$rate" --duration-s 180 \
    --seed "$seed" --text "$txt" \
    --out "onpolicy/${tag}.ndjson" --metrics-out "onpolicy/${tag}.metrics.ndjson" \
    "$@" > "onpolicy/${tag}.log" 2>&1
  echo "${tag} $(date +%H:%M:%S) $(grep -o 'AGGREGATE.*' onpolicy/${tag}.log | head -c 600)" >> onpolicy/progress_seeds.txt
  sleep 20
}

PM=scripts/predictor/pacer_model.joblib
for seed in 8 9; do
  for pair in "2.4 3" "3.0 0" "3.0 3"; do
    rate=${pair% *}; ti=${pair#* }
    if [ "$ti" = "0" ]; then TXT="$TXT0"; else TXT="$TXT3"; fi
    run_one "r${rate}_t${ti}_s${seed}_off" "$rate" "$seed" "$TXT"
    run_one "r${rate}_t${ti}_s${seed}_th2" "$rate" "$seed" "$TXT" \
      --pacer-model "$PM" --pacer-theta 2
  done
done

n_ok=0
for seed in 8 9; do
  for tag in "r2.4_t3_s${seed}_off" "r2.4_t3_s${seed}_th2" \
             "r3.0_t0_s${seed}_off" "r3.0_t0_s${seed}_th2" \
             "r3.0_t3_s${seed}_off" "r3.0_t3_s${seed}_th2"; do
    [ -s "onpolicy/${tag}.ndjson" ] && n_ok=$((n_ok+1))
  done
done
if [ "$n_ok" = "12" ]; then
  for seed in 8 9; do
    run_one "r2.4_t0_s${seed}_off" 2.4 "$seed" "$TXT0"
    run_one "r2.4_t0_s${seed}_th2" 2.4 "$seed" "$TXT0" \
      --pacer-model "$PM" --pacer-theta 2
  done
  echo DONE > onpolicy_seeds.marker
else
  echo "PARTIAL n_ok=${n_ok}" > onpolicy_seeds.marker
fi
