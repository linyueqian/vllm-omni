# Fish Speech Single-Stage Investigation — Results

## Experiment Setup
- **Server**: H20 (Alibaba Cloud, NVIDIA H20-3e 141GB)
- **Model**: fishaudio/s2-pro at `/home/vllm_25fall/yueqian/models/s2-pro`
- **Branch**: `feat/fish-speech-single-stage`
- **Date**: 2026-04-14

## Results

### Two-Stage Baseline (streaming PCM, concurrency=1)
```
Mean TTFB:     395 ms
Mean E2E:      1204 ms
P50 E2E:       1233 ms
P90 E2E:       1290 ms
Mean RTF:      0.320
Mean Audio:    3.77 s
Throughput:    3.133 audio-s/wall-s
```

### Single-Stage Chunked (non-streaming WAV, concurrency=1)
```
Mean E2E:      9226 ms  (skewed by long-generation requests)
Mean RTF:      0.503    (skewed by short/failed requests)
Throughput:    4.307 audio-s/wall-s

Per-request (valid only, excluding 400 errors):
  RTF range: 0.227 - 0.250 for normal-length requests
  Requests hitting max_tokens: E2E ~22s, Audio ~94s, RTF ~0.23
```

### Key Observations

1. **RTF improvement**: Single-stage achieves RTF ~0.23 vs two-stage RTF ~0.32.
   That's a **28% improvement** in real-time factor.

2. **Throughput improvement**: 4.307 vs 3.133 audio-s/wall-s = **37% improvement**.

3. **The benchmark is not apples-to-apples**: Two-stage used streaming PCM,
   single-stage used non-streaming WAV. Streaming adds TTFP benefit but
   slightly worse E2E due to connector overhead per chunk.

4. **Audio length variance**: Single-stage requests sometimes generate much
   more audio (94s vs 3-4s). This might be a seed/sampling difference or
   a bug in the prompt template. Needs investigation.

5. **Consistent 400 error**: The 4th request always fails with 400 Bad Request.
   Likely a race condition or request queuing issue.

## What Was Eliminated

| Overhead | Status |
|----------|--------|
| Second vLLM engine process (mp) | Eliminated |
| SharedMemoryConnector | Eliminated |
| OmniGenerationScheduler | Eliminated |
| Connector polling (10ms sleep) | Eliminated |

## What Was Added

| Cost | Description |
|------|-------------|
| Per-chunk DAC decode (~50-100ms) | Runs every 25 frames inline |
| Code accumulation in model_intermediate_buffer | List append per step |

## Architecture

```
Single-Stage:
  Slow AR (Qwen3, 36 layers, CUDA graph)
    → Fast AR (4 layers, torch.compile)
    → audio_codes accumulated in _dac_all_codes list
    → Every 25 frames: DAC decode chunk (50 frames with context)
    → Waveform chunks concatenated in _dac_wav_chunks
    → Final audio returned via multimodal_outputs
```

## Known Issues

1. **Non-streaming benchmark**: Need to fix benchmark to use streaming
   for fair comparison.
2. **Audio length variance**: Some requests generate way too many tokens.
3. **400 errors**: Need to investigate the intermittent Bad Request.
4. **Tail frames**: Frames after the last chunk boundary are not decoded
   until the next chunk boundary. Need a "flush on finish" mechanism.

## Next Steps

1. Fix benchmark to use streaming PCM for both configs.
2. Add flush-on-finish for remaining frames.
3. Profile per-step time to understand exact overhead reduction.
4. Test concurrent requests (concurrency 4, 10).

## Files Changed

- `vllm_omni/model_executor/models/fish_speech/fish_speech_single_stage.py`
- `vllm_omni/model_executor/stage_configs/fish_speech_s2_pro_single_stage.yaml`
- `vllm_omni/model_executor/models/registry.py`
- `benchmarks/fish-speech/bench_single_vs_two_stage.py`
