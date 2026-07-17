# vLLM-Omni · MiniCPM-o 4.5 Online Demo

Gradio-based web UI for **MiniCPM-o 4.5** served via `vllm-omni`'s
OpenAI-compatible endpoints.

The UI supports:

- **Inputs**: text prompt + optional image, audio (file or mic), video.
- **Outputs**: text + speech (WAV player).

## 1. Start the backend server

Pick a deploy config that matches your GPU layout:

| config | GPUs | TP | Notes |
|---|---|---|---|
| `minicpmo_4_5.yaml` | 2 | 1 | Thinker on GPU0, talker+t2w on GPU1. |
| `minicpmo_4_5_3gpu.yaml` | 3 | 2 | Thinker 2-way TP on GPU0/1, talker+t2w share GPU2. |
| `minicpmo_4_5_8x4090.yaml` | 8 | 4 | Thinker 4-way TP on GPU0-3, talker+t2w on GPU4. |
| `minicpmo_4_5_3gpu_stage1_replicas.yaml` | 3 | 1 | Thinker on GPU0, two talker+Token2wav replicas on GPU1/2 for concurrent text+audio serving. |
| `minicpmo_4_5_4gpu_stage1_replicas.yaml` | 4 | 1 | Thinker on GPU0, three talker+Token2wav replicas on GPU1/2/3. |
| `minicpmo_4_5_8x4090_stage1_replicas.yaml` | 8 | 4 | Thinker 4-way TP on GPU0-3, four talker+Token2wav replicas on GPU4-7. |

Then:

```bash
vllm-omni serve openbmb/MiniCPM-o-4_5 \
    --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

For production or internal networks where Hugging Face downloads are slow, pass
a local ModelScope-downloaded checkpoint path instead of `openbmb/MiniCPM-o-4_5`.

### TTS throughput notes

MiniCPM-o 4.5's remote-code `MiniCPMTTS.generate()` currently runs as a
single-request whole-waveform path, so the deploy configs keep Stage1
`max_num_seqs: 1`. Use the `*_stage1_replicas.yaml` configs to scale concurrent
text+audio throughput horizontally.

```bash
vllm-omni serve /path/to/MiniCPM-o-4_5 \
    --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_4gpu_stage1_replicas.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

Talker/token2wav runtime behavior uses checked-in defaults rather than
MiniCPM-specific environment variables. If these knobs need to be exposed, add
them through a first-class stage/model config so deployments have one clear
configuration surface.

Request-level reference audio is cached by content hash before it is passed to
Token2wav. This keeps repeated requests with the same voice prompt from
thrashing Token2wav's prompt cache while still resetting the cache when the
reference audio changes.

### Experimental native duplex

MiniCPM-o 4.5 supports the experimental `/v1/duplex` and
`/v1/realtime?duplex=1` WebSocket entry points. The native path streams audio
through a resumable scheduler data-plane request and forwards Stage0 output
through the existing Stage1/TTS pipeline.

Start the server with the duplex-specific deploy config. The regular
`minicpmo_4_5.yaml` deploy does not opt into duplex sessions and keeps the
non-streaming Stage1 token budget.

```bash
vllm-omni serve openbmb/MiniCPM-o-4_5 \
    --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_duplex.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

Clients enable the model-native path explicitly:

```json
{"extra_body": {"minicpmo45_native_duplex": true}}
```

The normal vLLM model runner still owns attention metadata, sampling, and
request KV. The current append path is not a scheduler-native append primitive
or persistent KV lease. MiniCPM advertises neither automatic/VAD barge-in nor
production multi-session concurrency. See
[`docs/design/minicpmo45_duplex_runtime_architecture.md`](../../../docs/design/minicpmo45_duplex_runtime_architecture.md)
for the active runtime path, lifecycle invariants, capability boundary, and
validation scope.

## 2. Launch the Gradio demo

```bash
bash examples/online_serving/minicpmo/run_gradio_demo.sh

# Or run the python entry point directly:
python examples/online_serving/minicpmo/gradio_demo.py \
    --minicpmo45-api-base http://localhost:8099/v1 \
    --minicpmo45-model openbmb/MiniCPM-o-4_5 \
    --port 7862
```

Open `http://<host>:7862` in a browser.

## 3. Run the Realtime duplex scenario demo

After the server is running, use the scenario client to validate the native
duplex semantics end to end:

```bash
python examples/online_serving/minicpmo/realtime_duplex_demo.py \
    --url ws://localhost:8099/v1/realtime?duplex=1 \
    --model openbmb/MiniCPM-o-4_5 \
    --input-wav /path/to/input_16k_mono_pcm16.wav \
    --output-dir /tmp/minicpmo_realtime_duplex_demo
```

The script streams sequential clean speech turns and relies on `auto_response`;
it never sends `response.create` or a serving-side barge-in signal. Pass a
different `--turn-input-wav` for each later turn and use
`--require-distinct-inputs` to reject repeated audio content and cross-turn
transcript tail reuse. It fails on incomplete response/audio lifecycle events,
stale output, cancellation, transcript delta/done mismatch, or missing audio
when `--require-audio` is set.

## 4. Open the experimental browser client

The canonical browser UI lives with the experimental runtime. It serves the
page and proxies the same-origin Realtime WebSocket to the backend:

```bash
python -m vllm_omni.experimental.fullduplex.web \
    --port 7862 \
    --ws-backend ws://127.0.0.1:8099 \
    --ref-audio /path/to/MiniCPM-o-Demo/assets/ref_audio/ref_minicpm_signature.wav
```

Open `http://<host>:7862/`. When using a reverse proxy, open the proxy URL that
maps to port `7862`. The browser derives its WebSocket endpoint relative to
that URL, preserving any proxy path prefix.

Client behavior and options:

- **Prompt presets**: the system prompt defaults to the official
  MiniCPM-o-Demo presets — `Streaming Omni Conversation.` (omni preset, for
  camera + voice) with the audio-call personas (中文通话 / English Call)
  selectable, or fully custom text.
- **Reference voice**: `--ref-audio` points at a wav whose voice the TTS
  clones (the official demo defaults to its signature voice at
  `assets/ref_audio/ref_minicpm_signature.wav`). Without it the model's
  built-in timbre is used.
- **Camera**: the **Camera** button streams ~1 fps JPEG frames riding the
  audio appends (`video_frames` on `input_audio_buffer.append`, the official
  omni contract) so the model sees while it listens.
- **Auto-commit (client VAD)**: on by default; commits the turn after ~0.5 s
  of post-speech silence. If the runtime does not start a response after a
  commit (its auto-response currently only fires on the first turn), the
  client requests one with `response.create` after a short fallback window.

## Notes

- **TTS trigger**: the demo sets
  `extra_body.chat_template_kwargs.use_tts_template=True`, which appends
  `<|tts_bos|>` to the assistant prefix.
- Uncheck **"Generate speech output (TTS)"** to get text-only responses
  (faster).
- The audio output is the raw WAV returned by the stage-1 talker +
  Token2Wav; sample rate is 24 kHz.
- Video input is forwarded as a base64 `video_url` entry; the server needs
  decord/torchvision to decode it.
