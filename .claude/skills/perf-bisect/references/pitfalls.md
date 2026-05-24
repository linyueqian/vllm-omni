# Mechanical pitfalls (caught in real bisects)

Reference for debug-time lookup. The pre-flight red-flag list lives in `SKILL.md`;
this file is what to consult when a bench misbehaves.

## 1. `pytest -k` selects 0 tests silently

Parametrised IDs may not contain the literal keyword the bisecter assumes
(e.g. `test_qwen3_tts_base`'s parametrised IDs don't contain "customvoice",
so `-k "not customvoice"` matches everything).

```bash
# ALWAYS run this first to confirm the filter matches anything
pytest --collect-only -q -k "<expr>" | head
```

## 2. Venv PATH not inherited by pytest subprocess

`.venv/bin/pytest` runs fine for the test, but subprocess calls to `ninja`
(for kernel compilation) raise `FileNotFoundError`.

```bash
# Belt-and-suspenders activation
source <venv>/bin/activate
export PATH="<venv>/bin:$PATH"
```

## 3. Stale server PID

`kill_server` must `pkill -9` AND verify the port is free; otherwise the
next iteration binds a different port, the curl probe matches the wrong
PID, and the bench reports `0/N completed`.

```bash
# Inside the bisect loop
pkill -9 -f "vllm.*serve|StageEngineCore|OmniServer"
sleep 8
ss -lntp | grep ":$PORT " && echo "PORT STILL BOUND — investigate" && exit 1
```

## 4. Multi-tenant GPUs

Another tenant's container may have pinned GPUs that the configured
`CUDA_VISIBLE_DEVICES` mask covers. The bench still runs but allocates
into a contended memory pool, doubling absolute numbers without changing
relative deltas.

```bash
# Probe first
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# Pick GPUs whose memory.used is < 1 GiB
```

## 5. `/v1/models` ≠ bench-ready

The API server registers the model entry as soon as it loads, but CUDA
graphs may still be compiling. Bench requests in this window block on
graph-capture and report inflated TTFP.

```bash
# After /v1/models returns the model entry
sleep 30   # settle window
# OR scan server.log for the warmup-complete marker
```

## 6. Cold model download

Several minutes via `hf-mirror`, longer on cold disk. The default
server-ready timeout may fire before the model finishes downloading.

```bash
# Pre-download into $HF_HOME BEFORE the bisect loop
hf download <model> --cache-dir "$HF_HOME"
# OR bump the server-ready timeout to 30 min in run_bisect.sh
```
