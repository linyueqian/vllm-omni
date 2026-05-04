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

This profile pins Stage0 (Slow/Fast AR) to GPU 0 and Stage1 (DAC decoder)
to GPU 1 to remove AR/DAC contention, sets `max_num_seqs=64` on both
stages so concurrencies above 8 don't queue, and reduces
`connector_get_sleep_s` from 10 ms to 1 ms so DAC chunks aren't held in
the connector once they're ready.

#### Throughput (H20x2, 100 unique prompts)

| concurrency | req/s | audio/s | mean E2E ms | p99 E2E ms | mean TTFP ms | p99 TTFP ms | mean RTF |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  1 | 0.86 |  4.06 | 1166 | 2109 |  67 |   74 | 0.246 |
|  4 | 2.88 | 13.59 | 1368 | 1884 |  94 |  127 | 0.290 |
| 16 | 4.92 | 22.74 | 3151 | 8241 | 312 | 5615 | 0.689 |
| 32 | 7.06 | 32.92 | 4249 | 5325 | 528 |  781 | 0.926 |
| 64 | 8.34 | 39.47 | 6377 | 9254 | 732 | 1356 | 1.361 |

For H100x2 numbers under an earlier configuration of this profile, see
the original benchmark in vllm-project/vllm-omni#2515 (comment by
@Sy0307).
