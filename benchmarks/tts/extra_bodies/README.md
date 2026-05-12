# TTS bench `extra_body` files

`bench_tts_continuity.py --extra-body-file <path>` merges the file's JSON dict
into every `/v1/audio/speech` POST. Two are needed for the standard sweeps.

## `qwen3_default_voice.json` (shipped)

71-byte body that selects a built-in CustomVoice. Drives the
`custom_default_voice` task in `run_continuity_variant.sh`.

## `qwen3_voice_clone.json` (not shipped — generate locally)

For Qwen3-TTS-Base voice-clone, the server expects a base64-encoded reference
audio inline. The resulting JSON is ~1 MB, so we don't keep it in the repo.

Generate one from any short reference clip:

```bash
python - <<'PY' > benchmarks/tts/extra_bodies/qwen3_voice_clone.json
import base64, json, sys
wav = open("/path/to/reference_16k.wav", "rb").read()
print(json.dumps({
    "task_type": "Base",
    "ref_audio": "data:audio/wav;base64," + base64.b64encode(wav).decode(),
    "ref_text": "The reference transcript exactly matching the reference audio.",
}))
PY
```

The exact key names must match what the deployed Qwen3-TTS-Base build accepts
(`task_type`, `ref_audio`, `ref_text`). Check `vllm_omni/entrypoints/openai/`
if you're on a different build.
