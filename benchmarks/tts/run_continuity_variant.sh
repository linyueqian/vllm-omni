#!/bin/bash
# Run the continuity-instrumented TTS benchmark against one served variant.
#
# Required positional args:
#   $1  variant_tag          — short label that becomes part of the output dir
#   $2  yaml                  — deploy yaml path
#   $3  model_short           — "base" (voice_clone) or "custom" (default_voice)
#   $4  devices               — CUDA_VISIBLE_DEVICES value e.g. "0" or "0,1"
#   $5  port                  — listening port
#   $6  concurrencies         — space-separated list e.g. "64 128 256"
#
# Required env vars (defaults shown):
#   REPO        — path to vllm-omni checkout (must contain a usable .venv)
#                 default: $(pwd)
#   EXTRA_DIR   — directory containing extra-body JSON files
#                 default: $REPO/benchmarks/tts/extra_bodies
#   OUT_ROOT    — where per-cell JSON results go
#                 default: $REPO/benchmarks/tts/_results
#   LOG_ROOT    — where vllm-serve logs go
#                 default: $REPO/benchmarks/tts/_logs
#
# Optional env vars:
#   MODEL_BASE       — Qwen3-TTS-Base model id (default Qwen/Qwen3-TTS-12Hz-1.7B-Base)
#   MODEL_CUSTOM     — CustomVoice model id (default Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
#   EXTRA_BASE       — extra-body json for base voice_clone
#                      (default: $EXTRA_DIR/qwen3_voice_clone.json)
#   EXTRA_CUSTOM     — extra-body json for custom default voice
#                      (default: $EXTRA_DIR/qwen3_default_voice.json)
#   SAMPLE_RATE      — sample rate for audio_rtf (default 24000)
#
# Notes:
#   - The voice_clone extra-body json carries a base64-encoded ref-audio and is
#     therefore NOT shipped in the repo. Generate one locally before running.
#     See benchmarks/tts/extra_bodies/README.md.
#   - The script kills any process holding $port before launching the server.
set -eu

VARIANT=$1
YAML=$2
MODEL_SHORT=$3
DEVICES=$4
PORT=$5
CONCS=$6

REPO=${REPO:-$(pwd)}
EXTRA_DIR=${EXTRA_DIR:-$REPO/benchmarks/tts/extra_bodies}
OUT_ROOT=${OUT_ROOT:-$REPO/benchmarks/tts/_results}
LOG_ROOT=${LOG_ROOT:-$REPO/benchmarks/tts/_logs}

MODEL_BASE=${MODEL_BASE:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}
MODEL_CUSTOM=${MODEL_CUSTOM:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}
EXTRA_BASE=${EXTRA_BASE:-$EXTRA_DIR/qwen3_voice_clone.json}
EXTRA_CUSTOM=${EXTRA_CUSTOM:-$EXTRA_DIR/qwen3_default_voice.json}
SAMPLE_RATE=${SAMPLE_RATE:-24000}

BENCH_CLIENT=$REPO/benchmarks/tts/bench_tts_continuity.py
VLLM_BIN=$REPO/.venv/bin/vllm
PYTHON_BIN=$REPO/.venv/bin/python
[ -x "$VLLM_BIN" ] || { echo "error: $VLLM_BIN not executable; set REPO to a checkout with .venv" >&2; exit 2; }

if [ "$MODEL_SHORT" = "base" ]; then
  MODEL=$MODEL_BASE
  EXTRA=$EXTRA_BASE
  TASK="base_voice_clone"
elif [ "$MODEL_SHORT" = "custom" ]; then
  MODEL=$MODEL_CUSTOM
  EXTRA=$EXTRA_CUSTOM
  TASK="custom_default_voice"
else
  echo "error: model_short must be 'base' or 'custom', got '$MODEL_SHORT'" >&2
  exit 2
fi

[ -f "$EXTRA" ] || { echo "error: extra-body file $EXTRA not found" >&2; exit 2; }
[ -f "$YAML" ] || { echo "error: yaml file $YAML not found" >&2; exit 2; }
[ -f "$BENCH_CLIENT" ] || { echo "error: bench client $BENCH_CLIENT not found" >&2; exit 2; }

OUT_DIR=$OUT_ROOT/${VARIANT}_${TASK}
LOG_DIR=$LOG_ROOT/${VARIANT}_${TASK}
mkdir -p "$OUT_DIR" "$LOG_DIR"

fuser -k "$PORT/tcp" 2>/dev/null || true
sleep 2

cd "$REPO"
CUDA_VISIBLE_DEVICES=$DEVICES setsid \
  "$VLLM_BIN" serve "$MODEL" --omni \
    --deploy-config "$YAML" \
    --host 127.0.0.1 --port "$PORT" \
    > "$LOG_DIR/server.log" 2>&1 < /dev/null &
SERVER_PID=$!

for i in $(seq 1 240); do
  curl -fs http://127.0.0.1:"$PORT"/v1/models > /dev/null 2>&1 && { echo "[$VARIANT/$MODEL_SHORT] ready ${i}s"; break; }
  grep -qE "Traceback|RuntimeError|out of memory" "$LOG_DIR/server.log" 2>/dev/null && {
    tail -30 "$LOG_DIR/server.log" >&2
    kill -9 "$SERVER_PID" 2>/dev/null || true
    exit 1
  }
  sleep 1
done

for c in $CONCS; do
  np=$((c * 2))
  echo "[$VARIANT/$MODEL_SHORT] c=$c"
  "$PYTHON_BIN" "$BENCH_CLIENT" \
    --base-url http://127.0.0.1:"$PORT" --model "$MODEL" \
    --num-prompts $np --concurrency $c --num-warmups 2 \
    --extra-body-file "$EXTRA" --sample-rate "$SAMPLE_RATE" \
    --output "$OUT_DIR/c${c}.json" 2>&1 | tail -3
done

kill -INT "$SERVER_PID" 2>/dev/null || true
for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
kill -9 "$SERVER_PID" 2>/dev/null || true
fuser -k "$PORT/tcp" 2>/dev/null || true
echo "[$VARIANT/$MODEL_SHORT] DONE"
