# Fish Speech S2 Pro

## Summary

- Vendor: FishAudio
- Model: `fishaudio/s2-pro`
- Task: Text-to-speech synthesis with optional voice cloning
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API
- Maintainer: Community

## When to use this recipe

Use this recipe as a practical baseline for running `fishaudio/s2-pro` for
high-quality text-to-speech synthesis. Fish Speech S2 Pro outputs 44.1 kHz
audio and supports voice cloning from reference audio.

## References

- User guide: [`docs/user_guide/examples/online_serving/fish_speech.md`](../../docs/user_guide/examples/online_serving/fish_speech.md)
- Example guide: [`examples/online_serving/fish_speech/README.md`](../../examples/online_serving/fish_speech/README.md)
- Related issue or discussion:
  [RFC: add recipes folder](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware Support

This recipe documents reference GPU configuration for Fish Speech S2 Pro.
Other hardware and configurations are welcome as community validation lands.

## GPU

### 1x A800 80GB

#### Environment

- OS: Linux
- Python: 3.10+
- Required package: `fish-speech` (for DAC codec)
- CUDA 12.8
- vLLM version: 0.19.0
- vLLM-Omni version or commit: c93359bb354a6aa5c14d062430cb85b2c4db251e

```bash
# Install PortAudio dependency if missing (required by pyaudio which is a dependency of fish-speech)
# For Ubuntu/Debian:
if ! dpkg -l | grep -q libportaudio2; then
    sudo apt-get update && sudo apt-get install -y libportaudio2 portaudio19-dev
fi
pip install fish-speech
```

#### Command

```bash
vllm serve fishaudio/s2-pro --omni --port 8091
```

Notes:

- `--omni` is required.
- The default deploy config `vllm_omni/deploy/fish_qwen3_omni.yaml` is loaded
  automatically by model registry (HF `model_type=fish_qwen3_omni`).


#### Verification

Basic TTS:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "default",
        "response_format": "wav"
    }' --output output.wav
```
[output.wav](https://github.com/user-attachments/files/27134970/output.wav)

Voice cloning:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, this is a cloned voice.",
        "voice": "default",
        "ref_audio": "https://example.com/reference.wav",
        "ref_text": "Transcript of the reference audio."
    }' --output cloned.wav
```
[reference.wav](https://github.com/user-attachments/files/27134971/reference.wav) <br>
[cloned.wav](https://github.com/user-attachments/files/27134969/cloned.wav)



#### Notes

- Key flags: `--omni` is required.
- Known limitations: Output audio is 44.1 kHz mono WAV. Voice cloning requires
  both `ref_audio` and `ref_text` parameters.
- Memory usage: Model loads at ~48.3 GiB, peaks at ~48.9 GiB during inference
  headroom for video frames and audio caches.

### 2x H100 80GB

#### Environment

- OS: Linux
- Python: 3.10+
- Required package: `fish-speech` (for DAC codec)
- CUDA 12.8
- vLLM version: 0.19.0+

#### Command

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve fishaudio/s2-pro --omni --port 8091 \
    --deploy-config vllm_omni/deploy/fish_qwen3_omni_2gpu.yaml
```

This profile pins Stage0 (Slow/Fast AR) to GPU 0 and Stage1 (DAC decoder) to
GPU 1 to remove AR/DAC contention. It also enlarges Stage0 batching
(`max_num_seqs=8`) and lets Stage1 consume DAC chunks in parallel
(`max_num_seqs=4`), and reduces `connector_get_sleep_s` from 10 ms to 1 ms.

#### Throughput (H100x2)

| concurrency | req/s | audio/s | mean E2E ms | p99 E2E ms | mean TTFP ms | p99 TTFP ms | mean RTF | p99 RTF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.63 | 2.95  | 1578.70 | 1922.99 | 97.47  | 110.82  | 0.339 | 0.362 |
| 2  | 1.17 | 5.46  | 1710.66 | 2248.20 | 121.16 | 202.01  | 0.367 | 0.395 |
| 4  | 2.11 | 9.83  | 1839.65 | 2520.31 | 138.05 | 298.55  | 0.395 | 0.451 |
| 8  | 3.60 | 16.93 | 2095.23 | 2807.10 | 163.04 | 242.78  | 0.445 | 0.459 |
| 10 | 3.67 | 17.26 | 2538.12 | 3933.99 | 598.81 | 1744.61 | 0.549 | 0.933 |

Source: vllm-project/vllm-omni#2515 (comment by @Sy0307).
