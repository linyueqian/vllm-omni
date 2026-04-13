#!/bin/bash
# Launch vLLM-Omni server for MOSS-TTS-Nano
#
# Usage:
#   ./run_server.sh
#   CUDA_VISIBLE_DEVICES=0 PORT=8091 ./run_server.sh

set -e

MODEL="${MODEL:-OpenMOSS-Team/MOSS-TTS-Nano}"
PORT="${PORT:-8091}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "Starting MOSS-TTS-Nano server with model: $MODEL"

FLASHINFER_DISABLE_VERSION_CHECK=1 \
vllm-omni serve "$MODEL" \
    --stage-configs-path "$REPO_ROOT/vllm_omni/model_executor/stage_configs/moss_tts_nano.yaml" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --enforce-eager \
    --omni
