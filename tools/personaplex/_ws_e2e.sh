#!/bin/bash
# Duplex WS e2e on H200 GPU2: start server, wait health, run client, ASR, kill.
set +e
WT=/home/yueqian/pplex-wt
WAV=/home/yueqian/personaplex/assets/test/input_assistant.wav
PY=/home/yueqian/vllm-omni-main/.venv/bin/python
RES=/home/yueqian/pplex_ws_e2e_result.txt
: > "$RES"
export HF_TOKEN="$(cat ~/.hf_token 2>/dev/null)" CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$WT"
find "$WT" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null

U=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2{print $2}')
for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$U" '$2==u{print $1}'); do kill -9 "$p" 2>/dev/null; done
pkill -9 -f serving.server 2>/dev/null
sleep 2

cd "$WT" || { echo "cd failed" >> "$RES"; exit 1; }
nohup "$PY" -m vllm_omni.experimental.fullduplex.personaplex.serving.server --host 127.0.0.1 --port 8124 > /home/yueqian/pplex_ws_server.log 2>&1 < /dev/null &
SV=$!
echo "server pid $SV" >> "$RES"
for i in $(seq 1 100); do
  if curl -s http://127.0.0.1:8124/health 2>/dev/null | grep -q ok; then echo "READY after $((i*3))s" >> "$RES"; break; fi
  if ! kill -0 "$SV" 2>/dev/null; then echo "SERVER_DIED (see log)" >> "$RES"; tail -5 /home/yueqian/pplex_ws_server.log >> "$RES" 2>&1; break; fi
  sleep 3
done

"$PY" tools/personaplex/test_duplex_ws.py --input-wav "$WAV" --seconds 6 --tail 10 \
  >> "$RES" 2>&1
kill -9 "$SV" 2>/dev/null
echo "DONE" >> "$RES"
