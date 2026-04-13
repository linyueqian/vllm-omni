# MOSS-TTS-Nano

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/moss_tts_nano>.


Single-stage offline TTS pipeline using the 0.1B MOSS-TTS-Nano AR LM and MOSS-Audio-Tokenizer-Nano codec. Outputs 48 kHz stereo WAV.

## Quick Start

```bash
python end2end.py --text "Hello, this is MOSS-TTS-Nano."
```

The first run downloads `OpenMOSS-Team/MOSS-TTS-Nano` and `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` from Hugging Face.

## Usage

```
python end2end.py [OPTIONS]

Options:
  --text TEXT               Text to synthesize (default: "Hello, this is MOSS-TTS-Nano speaking.")
  --voice VOICE             Built-in voice preset (default: Junhao)
  --mode MODE               continuation (default) or voice_clone
  --prompt-audio PATH       Reference WAV/MP3 for custom voice cloning
  --prompt-text TEXT        Reference transcript (continuation mode)
  --max-new-frames N        Max AR frames, default 375 (~14 s audio)
  --seed INT                Random seed for reproducibility
  --batch                   Run a built-in batch of diverse samples (ZH/EN)
  --output-dir DIR          Directory for WAV outputs (default: /tmp/moss_tts_nano_output)
  --stage-config PATH       Path to stage config YAML
```

## Examples

```bash
# Built-in Chinese voice
python end2end.py --text "你好，这是MOSS-TTS-Nano的语音合成演示。" --voice Junhao

# Built-in English voice
python end2end.py --text "Hello from MOSS-TTS-Nano." --voice Ava

# Batch synthesis
python end2end.py --batch --output-dir /tmp/batch_output

# Reproducible output
python end2end.py --text "Deterministic test." --seed 42
```

## Built-in Voice Presets

| Voice | Language |
|-------|----------|
| `Junhao` | ZH |
| `Zhiming` | ZH |
| `Weiguo` | ZH |
| `Xiaoyu` | ZH |
| `Yuewen` | ZH |
| `Lingyu` | ZH |
| `Ava` | EN |
| `Bella` | EN |
| `Adam` | EN |
| `Nathan` | EN |
| `Sakura` | JA |
| `Yui` | JA |
| `Aoi` | JA |
| `Hina` | JA |
| `Mei` | JA |

## Stage Config

The stage config is at `vllm_omni/model_executor/stage_configs/moss_tts_nano.yaml`.

```yaml
engine_args:
  gpu_memory_utilization: 0.3   # ~2 GB VRAM
  max_num_seqs: 4               # concurrent requests
  max_model_len: 4096
```
