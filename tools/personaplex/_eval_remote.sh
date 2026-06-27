#!/bin/bash
# Self-contained PersonaPlex e2e eval on H200 GPU2: clear pycache, run native omni
# pipeline, ASR the output, write a clean result line. Avoids fragile inline ssh.
set +e
SNAP=/home/yueqian/hf_cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309
WAV=/home/yueqian/personaplex/assets/test/input_assistant.wav
OUT=/home/yueqian/pplex_eval.wav
RES=/home/yueqian/pplex_eval_result.txt
LOG=/home/yueqian/pplex_eval.log
: > "$RES"

find /home/yueqian/pplex-wt -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
U=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2{print $2}')
for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$U" '$2==u{print $1}'); do kill -9 "$p" 2>/dev/null; done
sleep 3
rm -f "$OUT"

cd /home/yueqian/pplex-wt || exit 1
export HF_TOKEN="$(cat ~/.hf_token 2>/dev/null)" VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/yueqian/pplex-wt
/home/yueqian/vllm-omni-main/.venv/bin/python tools/personaplex/end2end_native_omni.py \
  --model "$SNAP" --input-wav "$WAV" --seconds 6 --out "$OUT" "$@" > "$LOG" 2>&1

echo "run_exit=$? $(grep -iE 'wrote [0-9].*->|E2E (PASS|FAIL)' "$LOG" | tail -1)" >> "$RES"

/home/yueqian/vllm-omni-main/.venv/bin/python - "$OUT" >> "$RES" 2>&1 <<'PY'
import sys, soundfile as sf, numpy as np, tempfile, os, whisper
p = sys.argv[1]
try:
    w, sr = sf.read(p); w = np.asarray(w, dtype=np.float32)
except Exception as e:
    print(f"ASR_ERR {e}"); sys.exit(0)
print("dur %.1fs rms=%.4f peak=%.3f" % (len(w)/sr, float(np.sqrt((w**2).mean())), float(np.abs(w).max())))
m = whisper.load_model("small", device="cuda")
for a, b in [(0, 12), (0, 30)]:
    seg = w[int(a*sr):int(b*sr)]
    tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); sf.write(tf.name, seg, sr)
    r = m.transcribe(tf.name, language="en", fp16=True); os.unlink(tf.name)
    print("ASR[%d:%ds]=%r" % (a, b, r["text"].strip()[:200]))
PY
echo "DONE" >> "$RES"
