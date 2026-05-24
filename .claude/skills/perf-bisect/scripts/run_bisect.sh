#!/usr/bin/env bash
# perf-bisect bench-loop template.
#
# Iterates a list of commits, for each one starts `vllm serve`, runs
# `vllm bench serve`, parses the result JSON, kills the server, and moves on.
# Produces one result.json per (commit, cell) under $RESULTS/<label>/result.json.
#
# Usage: copy to /tmp/, edit the EDIT-HERE block, run on the remote host:
#   bash run_bisect.sh
#
# The defaults below describe a TTS cell. Swap MODEL / DEPLOY_YAML /
# DATASET_NAME / EXTRA_BODY for diffusion or omni-audio. See `cells/` for
# canned cell definitions and `../SKILL.md` for the cell-definition discipline.

set -u
set -o pipefail

# ─── EDIT HERE ────────────────────────────────────────────────────────────────
# Commits to test. Order matters only for log readability. Replace the
# placeholder SHAs and labels for each new bisect.
COMMITS=("<good-sha>" "<midpoint-sha>" "<bad-sha>")
LABELS=("good_baseline" "midpoint" "bad_main")

# Cell definition (the 7-tuple from SKILL.md "cell-definition discipline").
MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEPLOY_YAML="vllm_omni/deploy/qwen3_tts_high_concurrency.yaml"
DATASET_NAME="seed-tts-text"
DATASET_PATH="benchmarks/build_dataset/seed_tts_smoke"
EXTRA_BODY='{"voice":"Vivian","language":"English","task_type":"CustomVoice"}'
NUM_PROMPTS=512
MAX_CONCURRENCY=64
NUM_WARMUPS=8

# Remote-host paths. Adapt to your host class per the remote-gpu skill;
# on H20 hosts these must live under /mnt/<volume>/<your-dir>/ (NOT /home).
VENV=/path/to/vllm-omni/.venv
WT=/path/to/vllm-omni-bisect              # dedicated git worktree
RESULTS=/path/to/bisect-results           # output dir
HF_CACHE=/path/to/hf_cache

# GPU pinning. nvidia-smi first; pick free GPUs.
export CUDA_VISIBLE_DEVICES=4,5

PORT=8092

# ─── END EDIT ─────────────────────────────────────────────────────────────────

mkdir -p "$RESULTS"
source "$VENV/bin/activate"
export PATH="$VENV/bin:$PATH"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$HF_CACHE"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

log() { echo "[$(date +%H:%M:%S)] $*"; }

wait_for_ready() {
    local timeout=${1:-1800}
    local start=$(date +%s)
    while true; do
        if curl -s --max-time 5 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$MODEL"; then
            sleep 30   # settling time: model in /v1/models, but graphs may still be compiling
            log "Server ready (settled)."
            return 0
        fi
        local now=$(date +%s)
        if [ $((now - start)) -gt $timeout ]; then
            log "Server-ready timeout after ${timeout}s."
            return 1
        fi
        sleep 10
    done
}

kill_server() {
    pkill -9 -f "vllm.*serve\|OmniServer\|StageEngineCore\|stage_diffusion" 2>/dev/null || true
    sleep 8
}

run_one() {
    local sha="$1" label="$2"
    log "=== Testing $label ($sha) ==="
    cd "$WT"
    kill_server
    git checkout "$sha" 2>&1 | tail -3
    git log -1 --format="HEAD: %H %s"

    local outdir="$RESULTS/$label"; mkdir -p "$outdir"

    log "Starting vllm serve (deploy=$DEPLOY_YAML)..."
    nohup vllm serve "$MODEL" --omni \
        --deploy-config "$DEPLOY_YAML" \
        --port "$PORT" \
        > "$outdir/server.log" 2>&1 &
    local server_pid=$!
    log "Server PID=$server_pid; waiting for ready..."

    if ! wait_for_ready 1800; then
        log "$label SERVER FAILED — tail of server.log:"
        tail -50 "$outdir/server.log"
        kill_server
        return 1
    fi

    log "Running vllm bench serve (N=$NUM_PROMPTS, c=$MAX_CONCURRENCY, $NUM_WARMUPS warmups)..."
    vllm bench serve --omni \
        --base-url "http://localhost:$PORT" \
        --backend openai-audio-speech \
        --endpoint /v1/audio/speech \
        --model "$MODEL" \
        --dataset-name "$DATASET_NAME" \
        --dataset-path "$DATASET_PATH" \
        --extra-body "$EXTRA_BODY" \
        --num-prompts "$NUM_PROMPTS" \
        --num-warmups "$NUM_WARMUPS" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --request-rate inf \
        --percentile-metrics 'ttft,e2el,audio_rtf,audio_ttfp,audio_duration' \
        --save-result \
        --result-dir "$outdir" \
        --result-filename "result.json" \
        2>&1 | tee "$outdir/bench.log" | tail -60 || true

    python - <<PY
import json, glob
files = sorted(glob.glob('$outdir/result*.json'))
if not files:
    print("  NO RESULT JSON FOUND")
else:
    d = json.load(open(files[0]))
    print(f"  median_TTFP={d.get('median_audio_ttfp_ms', 0):8.1f}ms  "
          f"p99={d.get('p99_audio_ttfp_ms', 0):8.1f}ms  "
          f"RTF={d.get('median_audio_rtf', 0):.3f}  "
          f"tput={d.get('audio_throughput', 0):6.1f}  "
          f"completed={d.get('completed')}/{d.get('num_prompts')}")
PY
    kill_server
}

for i in "${!COMMITS[@]}"; do
    run_one "${COMMITS[$i]}" "${LABELS[$i]}"
done

log "ALL DONE"
log "=== FINAL SUMMARY ==="
for label in "${LABELS[@]}"; do
    echo "--- $label ---"
    python - <<PY
import json, glob
files = sorted(glob.glob('$RESULTS/$label/result*.json'))
if not files:
    print("  NO RESULT")
else:
    d = json.load(open(files[0]))
    print(f"  median_TTFP={d.get('median_audio_ttfp_ms', 0):8.1f}ms  "
          f"p99={d.get('p99_audio_ttfp_ms', 0):8.1f}ms  "
          f"RTF={d.get('median_audio_rtf', 0):.3f}  "
          f"tput={d.get('audio_throughput', 0):6.1f}")
PY
done
