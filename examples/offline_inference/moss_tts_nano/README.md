# MOSS-TTS-Nano Offline Inference

## Overview

Single-stage offline TTS pipeline using the 0.1B MOSS-TTS-Nano AR LM and MOSS-Audio-Tokenizer-Nano codec. Outputs 48 kHz stereo WAV.

> **Voice cloning only.** MOSS-TTS-Nano has no built-in speaker presets.
> Every request needs `--prompt-audio` (a reference clip) and `--prompt-text`
> (the exact transcript of that clip). Sample reference clips ship in the
> upstream repo under
> [`assets/audio/`](https://github.com/OpenMOSS/MOSS-TTS-Nano/tree/main/assets/audio)
> (e.g. `zh_1.wav`, `en_2.wav`, `jp_2.wav`).

## Quick Start

```bash
# Fetch a sample reference clip from upstream (one-off, user-scoped cache).
REF_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/moss-tts-nano"
mkdir -p "$REF_DIR"
[ -s "$REF_DIR/zh_1.wav" ] || \
    curl -L -o "$REF_DIR/zh_1.wav" https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS-Nano/main/assets/audio/zh_1.wav

python end2end.py \
    --text "你好，这是MOSS-TTS-Nano的语音合成演示。" \
    --prompt-audio "$REF_DIR/zh_1.wav" \
    --prompt-text "欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。"
```

The first run downloads `OpenMOSS-Team/MOSS-TTS-Nano` and `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` from Hugging Face.

## Usage

```
python end2end.py [OPTIONS]

Required:
  --prompt-audio PATH       Reference WAV/MP3 for voice cloning
  --prompt-text TEXT        Exact transcript of --prompt-audio

Options:
  --text TEXT               Text to synthesize (default: "Hello, this is MOSS-TTS-Nano speaking.")
  --mode MODE               voice_clone (default) or continuation
  --max-new-frames N        Max AR frames, default 375 (~14 s audio)
  --seed INT                Random seed for reproducibility
  --audio-temperature F     Audio sampling temperature (default: 0.8)
  --audio-top-k N           Audio top-k sampling (default: 25)
  --audio-top-p F           Audio top-p sampling (default: 0.95)
  --text-temperature F      Text layer temperature (default: 1.0)
  --output-dir DIR          Directory for WAV outputs (default: $XDG_CACHE_HOME/moss_tts_nano_output, falls back to ~/.cache/...)
  --deploy-config PATH      Override deploy YAML (defaults to vllm_omni/deploy/moss_tts_nano.yaml)
  --stage-init-timeout INT  Timeout in seconds for stage init (default: 120)
```

## Examples

```bash
REF_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/moss-tts-nano"

# Chinese reference clip → Chinese synthesis
python end2end.py \
    --text "你好，这是 MOSS-TTS-Nano 的语音合成测试。" \
    --prompt-audio "$REF_DIR/zh_1.wav" \
    --prompt-text "欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。"

# Reproducible output
python end2end.py \
    --text "Deterministic test." \
    --prompt-audio "$REF_DIR/en_2.wav" \
    --prompt-text "Transcript of the English reference clip." \
    --seed 42
```

## Deploy Config

Runtime knobs live in `vllm_omni/deploy/moss_tts_nano.yaml` (auto-loaded;
override with `--deploy-config PATH`). Key stage-level settings:

```yaml
stages:
  - stage_id: 0
    gpu_memory_utilization: 0.3   # ~2 GB VRAM; increase for faster init
    max_num_seqs: 4               # concurrent requests
    max_model_len: 4096
```

## Output Format

WAV files, 48 kHz, stereo (2-channel). The codec interleaves stereo as `[L, R, L, R, ...]` in the flat tensor returned by the model.

## Troubleshooting

- **`libnvrtc.so.13: cannot open shared object file`**: torchaudio 2.10+ torchcodec backend requires NVRTC. The model patches `torchaudio.load/save` automatically at load time to fall back to soundfile.
- **`flash_attn not installed`**: The model falls back to `sdpa` attention automatically.
- **Empty audio**: Check that `--text` is non-empty and the model loaded successfully (look for "MOSS-TTS-Nano LM loaded" in logs).
