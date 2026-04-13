# MOSS-TTS-Nano

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/moss_tts_nano>.


## Installation

Please refer to [README.md](https://github.com/vllm-project/vllm-omni/tree/main/README.md)

## Supported Models

| Model | Parameters | Description |
|-------|-----------|-------------|
| `OpenMOSS-Team/MOSS-TTS-Nano` | 0.1B | AR LM + MOSS-Audio-Tokenizer-Nano codec, 48 kHz stereo, ZH/EN/JA |

## Gradio Demo

!!! note "Gradio is an optional dependency"
    The Gradio demo requires the `[demo]` extras. Install them first:

    ```bash
    pip install 'vllm-omni[demo]'
    ```

An interactive Gradio demo is available with multilingual voice presets, custom voice cloning, and gapless AudioWorklet streaming.

```bash
# Option 1: Launch server + Gradio together
./run_gradio_demo.sh

# Option 2: If server is already running
python gradio_demo.py --api-base http://localhost:8091
```

Then open http://localhost:7860 in your browser.

Features:

- 15 built-in voice presets (6 ZH, 4 EN, 5 JA)
- Custom voice cloning from uploaded reference audio
- Gapless AudioWorklet streaming with TTFP/RTF metrics
- Non-streaming mode with WAV/MP3/FLAC download

## Run examples (MOSS-TTS-Nano)

### Launch the Server

```bash
vllm-omni serve OpenMOSS-Team/MOSS-TTS-Nano \
    --stage-configs-path vllm_omni/model_executor/stage_configs/moss_tts_nano.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager
```

Or use the convenience script:

```bash
./run_server.sh
```

### Using curl

```bash
# Built-in voice preset (non-streaming)
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, this is MOSS-TTS-Nano.",
        "voice": "Ava",
        "response_format": "wav"
    }' --output output.wav

# Chinese voice
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "你好，这是语音合成测试。",
        "voice": "Junhao",
        "response_format": "wav"
    }' --output output_zh.wav

# Streaming (raw PCM)
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, streaming output from MOSS-TTS-Nano.",
        "voice": "Ava",
        "stream": true,
        "response_format": "pcm"
    }' --no-buffer | play -t raw -r 48000 -e signed -b 16 -c 2 -
```

### Using Python

```python
import httpx

response = httpx.post(
    "http://localhost:8091/v1/audio/speech",
    json={
        "input": "Hello, this is MOSS-TTS-Nano.",
        "voice": "Ava",
        "response_format": "wav",
    },
    timeout=300.0,
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## Built-in Voice Presets

| Voice | Language | Description |
|-------|----------|-------------|
| `Junhao` | ZH | Male, standard Mandarin |
| `Zhiming` | ZH | Male |
| `Weiguo` | ZH | Male |
| `Xiaoyu` | ZH | Female |
| `Yuewen` | ZH | Female |
| `Lingyu` | ZH | Female |
| `Ava` | EN | Female, American English |
| `Bella` | EN | Female |
| `Adam` | EN | Male |
| `Nathan` | EN | Male |
| `Sakura` | JA | Female |
| `Yui` | JA | Female |
| `Aoi` | JA | Female |
| `Hina` | JA | Female |
| `Mei` | JA | Female |

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | **required** | Text to synthesize (ZH / EN / JA) |
| `voice` | string | `"Junhao"` | Built-in voice name (see table above) |
| `response_format` | string | `"wav"` | Audio format: wav, mp3, flac, pcm |
| `stream` | bool | false | Stream raw PCM chunks |

## Output Format

WAV files: 48 kHz mono. PCM streaming: 48 kHz stereo (2-channel interleaved int16).

## Stage Config

The stage config is at `vllm_omni/model_executor/stage_configs/moss_tts_nano.yaml`. Key settings:

```yaml
engine_args:
  gpu_memory_utilization: 0.3   # ~2 GB VRAM; increase for faster init
  max_num_seqs: 4               # concurrent requests
  max_model_len: 4096
```

## Troubleshooting

1. **`libnvrtc.so.13: cannot open shared object file`**: torchaudio 2.10+ defaults to torchcodec which requires NVRTC. vLLM-Omni patches this automatically at model load time to use soundfile instead.
2. **Connection refused**: Ensure the server is running on the correct port.
3. **Flashinfer version mismatch**: Set `FLASHINFER_DISABLE_VERSION_CHECK=1` if you see version warnings.
4. **Out of memory**: The default `gpu_memory_utilization=0.3` is conservative. Increase it in the stage config if you have more VRAM available.

## Known Limitations

- **Streaming TTFP**: Single-stage generation models currently run the full inference before returning audio. True progressive streaming (low TTFP) requires multi-step scheduling support in GPUGenerationModelRunner.
