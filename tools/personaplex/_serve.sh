#!/bin/bash
# Launch the PersonaPlex duplex WS server on H200 GPU2 (persistent via nohup).
set +e
WT=/home/yueqian/pplex-wt
PY=/home/yueqian/vllm-omni-main/.venv/bin/python
PORT=${1:-8124}
LOG=/home/yueqian/pplex_ws_server.log
export HF_TOKEN="$(cat ~/.hf_token 2>/dev/null)" CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$WT"
find "$WT" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null

# Free GPU2 of any stale duplex server / leftover procs.
pkill -9 -f serving.server 2>/dev/null
U=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2{print $2}')
for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$U" '$2==u{print $1}'); do kill -9 "$p" 2>/dev/null; done
sleep 2

cd "$WT" || { echo "cd failed"; exit 1; }
nohup "$PY" -m vllm_omni.experimental.fullduplex.personaplex.serving.server \
  --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 < /dev/null &
SV=$!
echo "server pid $SV on 127.0.0.1:$PORT (log $LOG)"
for i in $(seq 1 120); do
  if curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then echo "READY after $((i*2))s"; exit 0; fi
  if ! kill -0 "$SV" 2>/dev/null; then echo "SERVER_DIED"; tail -8 "$LOG"; exit 1; fi
  sleep 2
done
echo "TIMEOUT waiting for ready"; tail -8 "$LOG"; exit 1
