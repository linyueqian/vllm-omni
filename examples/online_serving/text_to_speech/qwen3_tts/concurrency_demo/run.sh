#!/usr/bin/env bash
# Launch the concurrency demo against a running Qwen3-TTS-Base server.
#
# Prerequisite: start the server in another terminal:
#   vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni --port 8000
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8000}"
PORT="${PORT:-7860}"

if ! curl -sf "${API_BASE}/v1/models" > /dev/null; then
    echo "Server not reachable at ${API_BASE}. Start it with:"
    echo "  vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni --port 8000"
    exit 1
fi

exec python -m examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.app \
    --api-base "${API_BASE}" --port "${PORT}"
