---
name: add-tts-model
description: "Integrate a new text-to-speech model into vLLM-Omni from HuggingFace reference implementation through production-ready serving with streaming and CUDA graph acceleration. Use when adding a new TTS model, wiring stage separation for speech synthesis, enabling online voice generation serving, debugging TTS integration behavior, or building audio output pipelines."
---

# TTS Model Integration Workflow

## Overview

```
HF Reference -> Stage Separation -> Online Serving -> Async Chunk -> CUDA Graph -> Pre-commit/DCO
   (Phase 1)      (Phase 2)          (Phase 3)        (Phase 4)     (Phase 5)      (Phase 6)
```

Two architecture patterns are supported:

- **Two-stage pipeline** (e.g. Qwen3-TTS, Fish Speech): AR code-predictor → audio decoder, connected via async_chunk for low-latency streaming. Use this for maximum performance.
- **Single-stage AR** (e.g. MOSS-TTS-Nano): entire model runs inside one AR worker, streaming audio chunks directly via a per-request generator. Use this when the upstream model bundles AR + codec inseparably or when a two-stage split is not feasible.

The single-stage variant skips Phase 4 (async_chunk) but Phase 5 (CUDA graph) is still encouraged — capture the inner AR loop as a CUDA graph for a significant speedup. The VoxCPM-style streaming pattern is described in Phase 2.

## Phase 1: HuggingFace Reference

**Goal**: Understand the reference implementation and verify it produces correct audio.

### Steps

1. **Run the reference model** end-to-end using the official HuggingFace / GitHub code
2. **Document the architecture**:
   - What are the sub-models? (AR decoder, codec decoder, vocoder, etc.)
   - What is the token vocabulary? (semantic codes, RVQ codebooks, special tokens)
   - What is the output format? (sample rate, channels, codec type)
3. **Capture reference outputs** for comparison during integration
4. **Identify the config structure**: `config.json` fields, `model_type`, sub-model configs

### Key Questions

- How many codebooks? What are the codebook sizes?
- What special tokens exist? (`<|voice|>`, `<|audio_start|>`, `<|im_end|>`, etc.)
- What is the token-to-ID mapping for codec codes?
- What is the hop length / frame rate of the codec?
- Does the model support voice cloning? How? (reference audio encoding, speaker embeddings, etc.)

### Deliverables

- Working reference script that produces audio
- Architecture diagram / notes
- Token vocabulary mapping
- Reference audio samples for regression testing

## Phase 2: Stage Separation (Offline Inference)

**Goal**: Split the model into vLLM-Omni stages and get offline inference working.

### Steps

1. **Register the model** in `vllm_omni/model_executor/models/registry.py`
2. **Create config classes** (`configuration_<model>.py`) with `model_type` registration
3. **Implement Stage 0** (AR model):
   - Subclass appropriate base (e.g., wrap Qwen3 decoder layers)
   - Implement `forward()` for autoregressive token generation
   - Handle special token logic (start/stop tokens, codec token mapping)
   - If dual-AR (like Fish Speech), implement Fast AR as a nested module
4. **Implement Stage 1** (Decoder):
   - Load codec weights (may need lazy loading from separate checkpoint)
   - Implement `forward()`: codec codes -> audio waveform
   - Return `OmniOutput` with `multimodal_outputs`
5. **Create stage config YAML** defining both stages, memory allocation, and model paths
6. **Create stage input processor** for prompt building
7. **Write end2end.py** test script

### Critical Parameters to Get Right

| Parameter | Impact if Wrong |
|-----------|----------------|
| Hop length | Audio duration wrong, streaming noise |
| Token ID mapping | Garbage codes -> noise output |
| Codebook count/size | Shape mismatch crashes |
| Stop token | Generation never stops or stops too early |
| dtype / autocast | Numerical issues, silent quality degradation |
| Repetition penalty | Must match reference (often 1.0 for TTS) |

### Debugging Priority (from experience)

When audio output is wrong, check in this order:

1. **RoPE / attention**: Are position encodings correct? Is the attention mask right?
2. **Normalization**: RMSNorm epsilon, layer norm placement (pre vs post)
3. **Hop length**: Product of all upsample rates in the codec decoder
4. **Token mapping**: Are codec IDs correctly offset from the vocabulary base?
5. **Sampling parameters**: Temperature, top_k, top_p, repetition_penalty
6. **Tensor layout**: Codebook-major vs frame-major ordering
7. **dtype**: Float32 for codec decoders (autocast can corrupt audio)

### Streaming Correctness Rules (single-stage and two-stage)

These bugs appear in almost every new TTS PR. Check all before the first push:

- **Accumulate codes across AR steps** — each `forward()` appends new codes; do not reset between steps or audio will be truncated (fish speech: `fix: accumulate audio_codes across steps`)
- **Emit delta audio, not full waveform** — in streaming mode yield only the new chunk per step, not the re-decoded full waveform from step 0 (fish speech: `fix: emit delta audio not full waveform`)
- **All return paths must emit `model_outputs`** — if any early-return branch skips setting `model_outputs`, the serving layer silently drops that step's audio (fish speech: `fix: ensure ALL return paths emit model_outputs`)
- **Per-request state isolation** — for batched concurrent requests, key all state by request ID; a shared buffer corrupts audio across requests (fish speech: `fix: per-request vocode + delta emission`)
- **Codec tensor device** — move codec codes to the codec decoder's device before calling decode; mismatches cause silent CPU fallback or crashes (fish speech: `fix: use model device for CUDA stream`)
- **Stage 1 `max_num_seqs`** — set to at least 4 in production configs; default of 1 causes audio gaps under concurrency because Stage 1 round-robins codec windows across requests (RFC #2568)

### Optional Dependency Handling

Models that rely on `torchaudio`, `torchcodec`, `soundfile`, or other optional packages
must handle the missing-package case at import time, not at call time. Failing to do this
causes cryptic errors only on environments without the optional package — after the model
is already deployed.

Pattern (used in MOSS-TTS-Nano):

```python
def _patch_torchaudio_load() -> None:
    """Fallback torchaudio.load/save to soundfile if torchcodec is unavailable."""
    try:
        import torchcodec  # noqa: F401
        return  # torchcodec present, torchaudio works as-is
    except ImportError:
        pass

    import soundfile as sf
    import torchaudio

    def _sf_load(path, **kwargs):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data).T, sr

    torchaudio.load = _sf_load
    # patch .save similarly if needed
```

Call this patch function at the top of `load_weights()` before loading any audio assets.

### Single-Stage AR Pattern (alternative to two-stage)

When the upstream model cannot be cleanly split into an AR stage and a separate decoder,
use the VoxCPM-style single-stage pattern instead:

1. **Single model file** — load both AR LM and codec inside `modeling_<model>.py`
2. **Load weights in `load_weights()`**, not `__init__()` — vLLM initializes distributed
   state before any CUDA allocations
3. **Stream via a per-request generator** stored in `self._stream_gens`:

```python
class YourModelForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self._lm = None                   # populated in load_weights()
        self._stream_gens: dict = {}      # request_key → generator

    def load_weights(self, weights):
        # Load self._lm here, after vLLM distributed init
        ...

    def forward(self, input_ids, positions, intermediate_tensors=None, **kwargs):
        key = self._get_request_key(kwargs)
        if key not in self._stream_gens:
            self._stream_gens[key] = self._create_stream_gen(kwargs)
        try:
            chunk, is_last = next(self._stream_gens[key])
        except StopIteration:
            chunk, is_last = torch.zeros(0), True
        if is_last:
            del self._stream_gens[key]
        return OmniOutput(multimodal_outputs={"waveform": chunk, "is_last": is_last})

    def _create_stream_gen(self, kwargs):
        for event in self._lm.inference_stream(...):
            if event["type"] == "audio":
                yield event["waveform"], False
        yield torch.zeros(0), True

    def compute_logits(self, hidden_states, sampling_metadata):
        # Emit EOS only after the last chunk so the AR scheduler ends the request
        ...
```

4. **Stage config** — single stage with `worker_type: ar`, `engine_output_type: audio`,
   `final_output: true`
5. **Only extract params you forward** — any variable extracted from `additional_information`
   but not passed to the model call will fail `ruff F841` in pre-commit

Reference: `vllm_omni/model_executor/models/moss_tts_nano/modeling_moss_tts_nano.py`

### Deliverables

- Model files in `vllm_omni/model_executor/models/<model_name>/`
- Stage config YAML
- Working `end2end.py` with correct audio output
- README.md in the example directory

## Phase 3: Online Serving

**Goal**: Expose the model via `/v1/audio/speech` API endpoint.

### Steps

1. **Register in `serving_speech.py`** — add all 5 points in a **single commit**;
   partial integration causes hard-to-debug failures. This file is modified by every
   model PR and is the most common source of rebase conflicts — see conflict note below.

   **Point 1** — stage constant (~line 50):
   ```python
   _YOUR_MODEL_TTS_MODEL_STAGES = {"your_stage_key"}
   ```

   **Point 2** — union into `_TTS_MODEL_STAGES` (~line 57):
   ```python
   _TTS_MODEL_STAGES: set[str] = (
       ...
       | _YOUR_MODEL_TTS_MODEL_STAGES
   )
   ```

   **Point 3** — model type detection in `_get_tts_model_type()`:
   ```python
   if model_stage in _YOUR_MODEL_TTS_MODEL_STAGES:
       return "your_model"
   ```

   **Point 4** — validation dispatch in `_validate_tts_request()`:
   ```python
   if self._tts_model_type == "your_model":
       return self._validate_your_model_request(request)
   ```

   **Point 5** — validation + parameter-builder methods:
   ```python
   def _validate_your_model_request(self, request) -> str | None:
       if not request.input or not request.input.strip():
           return "Input text cannot be empty"
       return None

   def _build_your_model_params(self, request) -> dict:
       params = {"text": [request.input]}
       if request.voice is not None:
           params["voice"] = [request.voice]
       return params
   ```
   Wire `_build_your_model_params` into `_create_tts_request()` alongside the other
   model-specific param builders.

   > **Unused variable rule**: only extract fields in `_build_your_model_params` that
   > are actually forwarded to the model. Unused extractions fail `ruff F841`.

   **Rebase conflict note**: when rebasing onto `main` after another model was merged,
   `serving_speech.py` will conflict. Resolution: always keep *both* the upstream
   model's additions and your own — never discard either side.

2. **Handle model-specific parameters**:
   - Voice cloning: `ref_audio` encoding and prompt injection
   - `max_new_tokens` override in sampling params
   - Model-specific default values
3. **Create client scripts**: `speech_client.py`, `run_server.sh`
4. **Test all response formats**: wav, mp3, flac, pcm
5. **Add Gradio demo**: Interactive web UI with streaming support

### Voice Cloning Pattern

```python
import base64
from pathlib import Path

def build_voice_clone_prompt(ref_audio_path: str, text: str, codec) -> list:
    """Build prompt with reference audio for voice cloning in serving_speech.py."""
    audio_bytes = Path(ref_audio_path).read_bytes()
    codes = codec.encode(audio_bytes)  # Encode on CPU using model's codec (e.g., DAC)
    token_ids = [code + codec.vocab_offset for code in codes.flatten().tolist()]
    return [
        {"role": "system", "content": f"<|voice|>{''.join(chr(t) for t in token_ids)}"},
        {"role": "user", "content": text},
    ]
```

### Deliverables

- Updated `serving_speech.py` with all 5 integration points (single commit)
- Client scripts and server launcher
- Gradio demo with streaming and voice cloning UI
- E2E online serving test (`tests/e2e/online_serving/test_<model>.py`)
- Buildkite CI entry in `.buildkite/test-merge.yml`
- Documentation (offline + online serving docs)

## Phase 4: Async Chunk (Streaming)

**Goal**: Enable inter-stage streaming so audio chunks are produced while AR generation continues.

### Steps

1. **Update stage config YAML**:
   ```yaml
   async_chunk: true
   codec_chunk_frames: 25      # frames per chunk
   codec_left_context_frames: 25  # overlap for smooth boundaries
   ```
2. **Implement chunk handling in Stage 1**:
   - Accept partial input (chunk of codec codes)
   - Handle left context for smooth audio boundaries
   - Return partial audio in `OmniOutput`
3. **Test streaming**:
   - Verify audio quality matches non-streaming output
   - Check for artifacts at chunk boundaries
   - Measure TTFA (time to first audio)
4. **Update online serving** to support `stream=true` with PCM output

### Streaming Architecture

```
Stage 0 (AR)                    Stage 1 (Decoder)
  |                                |
  |-- chunk 0 (25 frames) ------> decode -> audio chunk 0 -> client
  |-- chunk 1 (25 frames) ------> decode -> audio chunk 1 -> client
  |-- chunk 2 (25 frames) ------> decode -> audio chunk 2 -> client
  ...
```

### Key Considerations

- **Left context overlap**: Prevents audible artifacts at chunk boundaries
- **Hop length matters**: `context_audio_samples = context_frames * hop_length`
- **First chunk latency**: Can use larger initial chunk for better quality, then smaller chunks

### Deliverables

- Updated stage config with async_chunk enabled
- Smooth streaming audio without boundary artifacts
- TTFA metrics

## Phase 5: CUDA Graph Acceleration

**Goal**: Capture the AR loop as a CUDA graph for significant speedup.

### Steps

1. **Identify the hot loop**: The AR decoding loop that runs N steps per token
2. **Create static buffers**:
   - KV caches with fixed max sequence length
   - Pre-built causal masks and position tensors per step
   - Static input/output tensors
3. **Implement graph capture**:
   - Warm up with real data
   - Capture the forward pass
   - Replay with updated inputs
4. **Handle constraints**:
   - Use `torch.argmax` instead of `torch.multinomial` (graph-safe)
   - Fixed batch size (fall back to eager for other sizes)
   - No dynamic control flow inside the graph

### Example: Code Predictor CUDA Graph (Qwen3-TTS)

```python
import torch

class CodePredictorGraph:
    """Captures the 16-step code predictor AR loop as a single CUDA graph."""

    def setup_graph(self, device: torch.device, kv_heads: int = 4, head_dim: int = 64):
        self.num_steps = 16
        self.kv_cache = torch.zeros(1, kv_heads, self.num_steps, head_dim, device=device)
        self.positions = torch.arange(self.num_steps, device=device)
        self.causal_mask = torch.tril(torch.ones(self.num_steps, self.num_steps, device=device))
        self.input_buf = torch.zeros(1, 1, kv_heads * head_dim, device=device)
        self.output_buf = torch.zeros(1, self.num_steps, device=device, dtype=torch.long)
        # Warm up, then: self.graph = torch.cuda.CUDAGraph(); self.graph.capture(...)

    def run_graph(self, initial_input: torch.Tensor) -> torch.Tensor:
        self.input_buf.copy_(initial_input)
        self.graph.replay()
        return self.output_buf.clone()
```

### Performance Expectations

Based on Qwen3-TTS code predictor experience:
- **3-5x speedup** for the graphed component
- Only effective for fixed batch sizes (typically batch_size=1)
- Falls back to eager mode for unsupported configurations

### Deliverables

- CUDA graph implementation for the AR hot loop
- Benchmark script comparing eager vs graph performance
- Documentation of constraints and fallback behavior

## Phase 6: Pre-commit and DCO

**Goal**: Ensure every commit passes CI linting checks and carries the required
Developer Certificate of Origin sign-off before pushing.

### Pre-commit

Install hooks once: `pre-commit install`. Run before every commit:

```bash
pre-commit run --files \
  vllm_omni/model_executor/models/<model_name>/*.py \
  vllm_omni/entrypoints/openai/serving_speech.py \
  vllm_omni/model_executor/models/registry.py \
  tests/e2e/offline_inference/test_<model_name>.py \
  tests/e2e/online_serving/test_<model_name>.py
```

When pre-commit **modifies files** (ruff format auto-fix), it exits non-zero but the
changes are correct — stage the modified files and re-commit.

| Failure | Root cause | Fix |
|---------|-----------|-----|
| `ruff F841` | Variable extracted but never forwarded to model call | Remove the extraction or wire it through |
| `ruff E402` | Import added below function definitions | Move to top-level import block |
| `ruff format` | Line length, spacing, quote style | Accept auto-fix, stage, re-commit |

### DCO sign-off

Every commit must carry `Signed-off-by: Your Name <your@email.com>`. Use `-s`:

```bash
git commit -s -m "feat(<model>): add <Model> TTS support"
```

Or set permanently: `git config format.signOff true`

The DCO check verifies that the commit author email matches the `Signed-off-by` line.
Confirm `git config user.email` matches your GitHub account email before committing.

To fix a missing or mismatched sign-off on the latest commit:

```bash
git commit --amend -s --no-edit
git push origin <branch> --force-with-lease
```

## Integration Checklist

Use this checklist when integrating a new TTS model:

### Phase 1: HF Reference
- [ ] Reference model runs and produces correct audio
- [ ] Architecture documented (stages, codebooks, tokens, sample rate)
- [ ] Reference audio samples saved for comparison

### Phase 2: Stage Separation
- [ ] Model registered in `registry.py`
- [ ] Config classes created with `model_type` registration
- [ ] Stage 0 (AR) implemented and generates correct tokens
- [ ] Stage 1 (Decoder) produces correct audio from tokens — dtype float32 for codec decoder
- [ ] Stage 1 `max_num_seqs` ≥ 4 in production config (default 1 causes gaps under concurrency)
- [ ] Optional dependency fallbacks handled at `load_weights()` time (torchaudio/soundfile/etc.)
- [ ] Streaming: codec codes accumulated across AR steps (not reset per step)
- [ ] Streaming: delta audio emitted per chunk, not full re-decoded waveform
- [ ] Streaming: all `forward()` return paths emit `model_outputs`
- [ ] Streaming: per-request state keyed by request ID (not shared across requests)
- [ ] Streaming: codec tensors moved to codec decoder device before decode
- [ ] Stage config YAML created
- [ ] `end2end.py` produces audio matching reference quality
- [ ] README.md written

### Phase 3: Online Serving
- [ ] All 5 `serving_speech.py` integration points added in one commit
- [ ] Only extract params in `_build_*_params` that are forwarded to the model call (ruff F841)
- [ ] Prompt builder handles text input correctly
- [ ] Voice cloning works (if supported)
- [ ] All response formats work (wav, mp3, flac, pcm)
- [ ] Client scripts and server launcher created
- [ ] E2E online serving test written (`tests/e2e/online_serving/test_<model>.py`)
- [ ] Buildkite CI entry added to `.buildkite/test-merge.yml`
- [ ] Gradio demo working
- [ ] Documentation added (offline + online docs, nav, supported models)

### Phase 4: Async Chunk
- [ ] Stage config updated with `async_chunk: true`
- [ ] Stage 1 handles partial chunks correctly
- [ ] No audio artifacts at chunk boundaries
- [ ] Streaming via API (`stream=true`) works
- [ ] TTFA measured and acceptable

### Phase 5: CUDA Graph
- [ ] Hot loop identified and profiled
- [ ] Static buffers allocated
- [ ] Graph captured and replays correctly
- [ ] Benchmark shows meaningful speedup
- [ ] Fallback to eager works for unsupported configs

### Phase 6: Pre-commit and DCO
- [ ] `pre-commit run` passes on all changed files before every push
- [ ] No `ruff F841` (unused variables) — all extracted params forwarded to model call
- [ ] No `ruff E402` (import ordering) — imports at top-level
- [ ] Every commit has `Signed-off-by` matching author email (`git commit -s`)
- [ ] `git log --format="%ae"` shows only your registered GitHub account email
- [ ] `serving_speech.py` rebase conflicts resolved by keeping both sides

## References

- [TTS audio skill](../vllm-omni-audio-tts/SKILL.md) -- supported models and usage
- [Fish Speech integration](../vllm-omni-audio-tts/references/fish-speech.md) -- complete example of Phases 1-3
- [Qwen3-TTS reference](../vllm-omni-audio-tts/references/qwen-tts.md) -- complete example of all 5 phases
- [Adding a TTS model (developer guide)](https://github.com/vllm-project/vllm-omni/blob/main/docs/contributing/model/adding_tts_model.md)
