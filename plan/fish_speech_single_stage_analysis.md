# Fish Speech Single-Stage Investigation

## Motivation

Issue #2515 shows baseline benchmarks for Fish Speech S2 Pro. While vllm-omni
already beats sglang on E2E latency and throughput, the **per-step latency gap**
(73ms/step vllm-omni vs 11ms/step sglang) suggests significant overhead in the
multi-stage pipeline.

## Current Architecture (Two-Stage)

```
Stage 0 (Slow AR)                    Stage 1 (DAC Decoder)
─────────────────                    ─────────────────────
FishSpeechSlowARForConditional  →  SharedMemoryConnector  →  FishSpeechDACDecoder
  - Qwen3 backbone (36 layers)       codec_streaming=true       - DAC codec.pth
  - Fast AR (4 layers)                connector_get_sleep_s=0.01 - 44.1kHz output
  - engine_output_type: latent        codec_chunk_frames=25      - engine_output_type: audio
  - worker_type: ar                                              - worker_type: generation
  - distributed_executor_backend: mp                             - distributed_executor_backend: mp
```

### Overhead Sources

| Source | Description | Impact |
|--------|-------------|--------|
| Second vLLM engine | Separate process with its own scheduler, model runner | Memory + startup time + process overhead |
| `distributed_executor_backend: "mp"` | Multiprocessing for BOTH stages | IPC overhead per step |
| SharedMemoryConnector | Serialise/deserialise codec frames | Copy + polling (10ms sleep) |
| OmniGenerationScheduler | Scheduling for DAC decode stage | Per-chunk scheduling overhead |
| Connector polling | `connector_get_sleep_s: 0.01` | 10ms minimum per chunk wait |
| Async chunk framing | 25-frame chunks with 25-frame left context overlap | DAC decodes 50 frames to get 25 new ones |

## Proposed Architecture (Single-Stage)

```
Stage 0 (Single Stage)
──────────────────────
FishSpeechSingleStageForConditionalGeneration
  - Qwen3 backbone (36 layers)
  - Fast AR (4 layers)
  - DAC codec (inline decode in make_omni_output)
  - engine_output_type: audio
  - worker_type: ar
  - NO connectors, NO second engine, NO mp
```

### What's Eliminated

1. **Second vLLM engine process** — no more mp overhead for stage 1
2. **SharedMemoryConnector** — no serialisation, no 10ms polling sleep
3. **OmniGenerationScheduler** — no per-chunk scheduling
4. **Connector framing overhead** — no 25-frame chunks with left-context overlap
5. **GPU memory fragmentation** — one engine instead of two (gpu_memory_utilization: 0.6 + 0.1 → 0.7)

### What's Changed

- `make_omni_output` now runs DAC decode inline after each AR step
- DAC codec loaded lazily on first decode (same as current stage 1)
- Audio is re-decoded from all accumulated codes each step (O(N²) pattern, same as VoxCPM2)
- `async_chunk: false` — no streaming chunking (full decode each step)

### Trade-offs

| Aspect | Two-Stage | Single-Stage |
|--------|-----------|--------------|
| Per-step latency | ~73ms (AR) + connector + DAC scheduling | ~73ms (AR) + ~Xms (DAC inline) |
| Streaming TTFA | Good (first 25 frames → audio quickly) | Worse (audio only on finish or per-step) |
| GPU memory | 0.6 + 0.1 = 0.7 split across 2 engines | 0.7 in one engine |
| Concurrent requests | DAC stage can batch independently | DAC runs serially per AR step |
| Complexity | 2 processes, connectors, async chunk | 1 process, simpler |

## Expected Speedup

### Optimistic Scenario
If the per-step overhead is dominated by connector/scheduling:
- **Connector polling**: 10ms per chunk → eliminated
- **Scheduling overhead**: ~2-5ms per step → eliminated
- **SharedMemory serialisation**: ~1-2ms per chunk → eliminated
- **Potential per-step saving**: 13-17ms → from 73ms to ~56-60ms/step

### Realistic Scenario
If the 73ms/step includes significant GPU compute (forward pass):
- Per-step saving of 5-10ms from eliminating IPC
- Main bottleneck may be the Slow AR forward itself
- Need profiling to determine actual breakdown

### E2E Latency Impact (concurrency=1)
From #2515 baseline: E2E = 2779ms (concurrency=1)
- ~2048 max tokens → ~28 AR steps (if generating ~1400 tokens)
- If saving ~10ms/step → ~280ms saving → **~10% E2E improvement**
- If saving ~15ms/step → ~420ms saving → **~15% E2E improvement**

## How to Benchmark

### Setup

```bash
# Two-stage (baseline)
python -m vllm_omni.entrypoints.openai.api_server \
  --model fishaudio/s2-pro \
  --stage-config fish_speech_s2_pro

# Single-stage (experimental)
python -m vllm_omni.entrypoints.openai.api_server \
  --model fishaudio/s2-pro \
  --stage-config fish_speech_s2_pro_single_stage
```

### Metrics to Compare

1. **Per-step latency**: Profile AR forward + make_omni_output time
2. **TTFP (Time to First Phone)**: Time to first audio output
3. **E2E latency**: Total request time
4. **Audio quality**: Compare WAV outputs (should be identical given same seeds)
5. **Memory usage**: `nvidia-smi` during serving
6. **Throughput at concurrency 1, 4, 10**: Requests/sec

### Profiling Commands

```bash
# Quick single-request test
python benchmarks/fish-speech/bench_voice_cache.py \
  --backend vllm-omni \
  --concurrency 1

# Full benchmark suite
# (use existing fish-speech benchmark scripts from #2515)
```

## Future Optimisations

1. **Incremental DAC decode**: Instead of re-decoding all codes each step,
   only decode new frames. Requires DAC state caching (sliding window).
2. **CUDA graph for DAC**: The DAC decode is a fixed computation graph
   that could benefit from CUDA graph capture.
3. **Batched DAC decode**: When multiple requests are active, batch their
   DAC decode calls.

## Files Changed

- `vllm_omni/model_executor/models/fish_speech/fish_speech_single_stage.py` — New model class
- `vllm_omni/model_executor/stage_configs/fish_speech_s2_pro_single_stage.yaml` — New config
- `vllm_omni/model_executor/models/registry.py` — Register new model arch
