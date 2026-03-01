# Adding a TTS Model

This guide walks through the process of adding a new TTS model to vLLM-Omni, using **Qwen3-TTS**
as a comprehensive example. Qwen3-TTS demonstrates the standard two-stage TTS pipeline and the
key optimizations all TTS models in this repo should follow.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Key Components](#key-components)
5. [Model Registration](#model-registration)
6. [Stage Configuration](#stage-configuration)
7. [Stage Input Processors](#stage-input-processors)
8. [Testing](#testing)
9. [Summary](#summary)

## Overview

vLLM-Omni supports TTS models as multi-stage pipelines where each stage runs independently
and can be placed on different devices. Qwen3-TTS exemplifies this with two stages:

1. **AR Stage (Code Predictor)**: Text input → intermediate audio representations (e.g. codec codes, latent patches)
2. **Decoder Stage (Code2Wav)**: Intermediate representations → audio waveform

Each stage is implemented as a separate model class configured independently via YAML.

The two stages are connected by the `async_chunk` framework, which enables inter-stage streaming
for low first-packet latency (see [Async Chunk Design](../../design/feature/async_chunk_design.md)).

**Qwen3-TTS stage overview:**

| Stage | Name | Input | Output |
|-------|------|-------|--------|
| 0 | Code Predictor (AR) | Text tokens | Discrete RVQ codec codes |
| 1 | Code2Wav (Decoder) | RVQ codec codes | Audio waveform |

**Without `async_chunk` (batch mode):** Stage 0 runs to completion before Stage 1 starts,
resulting in long first-packet latency.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-tts-non-async-chunk.drawio.png">
    <img alt="TTS pipeline without async_chunk" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-tts-non-async-chunk.drawio.png" width=80%>
  </picture>
</p>

**With `async_chunk` (streaming mode):** Stage 0 sends codec codes to Stage 1 every
`chunk_size=25` tokens. Stage 1 begins decoding audio immediately, reducing first-packet
latency from the full AR generation time down to just the first chunk.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-tts-async-chunk.drawio.png">
    <img alt="TTS pipeline with async_chunk" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-tts-async-chunk.drawio.png" width=80%>
  </picture>
</p>

## Directory Structure

When adding a new TTS model, create the following structure:

```
vllm_omni/model_executor/models/
└── your_model_name/
    ├── __init__.py
    ├── your_model.py                    # Unified class (stage dispatch)
    ├── your_model_ar_stage.py           # Stage 0: AR stage
    └── your_model_decoder.py            # Stage 1: audio decoder

vllm_omni/model_executor/stage_input_processors/
└── your_model_name.py                   # Stage 0 -> Stage 1 transition

vllm_omni/model_executor/stage_configs/
└── your_model_name.yaml
└── your_model_name_async_chunk.yaml
```

**Qwen3-TTS reference files:**

| File | Purpose |
|------|---------|
| `models/qwen3_tts/qwen3_tts.py` | Unified model class |
| `models/qwen3_tts/qwen3_tts_code_predictor_vllm.py` | Stage 0 - optimized AR |
| `models/qwen3_tts/qwen3_tts_code2wav.py` | Stage 1 - decoder |
| `stage_configs/qwen3_tts.yaml` | Batch mode config |
| `stage_configs/qwen3_tts_batch.yaml` | Batch mode (alternative) |
| `stage_input_processors/qwen3_tts.py` | Stage transition processors |

## Step-by-Step Implementation

### Step 1: Implement Stage 0 - AR Stage

Stage 0 is the autoregressive stage that generates intermediate audio representations.
**It must use vLLM's native decoder layers with fused ops and PagedAttention** for the LLM
backbone - this is the primary source of speedup over HuggingFace inference.

#### 1.1 Use vLLM Decoder Layers Directly

In Qwen3-TTS, the code predictor builds its transformer layers from `Qwen3DecoderLayer`
(vLLM's implementation with fused `QKVParallelLinear`) rather than wrapping the HF model:

```python
# qwen3_tts_code_predictor_vllm.py

from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer

class Qwen3TTSTalkerCodePredictorVllm(nn.Module):
    """
    AR code predictor built directly on vLLM's Qwen3DecoderLayer.
    Uses fused QKVParallelLinear and PagedAttention for fast inference.
    """

    def __init__(self, config, vllm_config, prefix):
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                config, vllm_config=vllm_config, prefix=f"{prefix}.layers.{i}"
            )
            for i in range(config.num_hidden_layers)
        ])
        self.lm_head = ParallelLMHead(config.codec_size, config.hidden_size)
```

If your model's AR backbone is based on a different architecture (e.g. Qwen2, LLaMA), use the
corresponding vLLM decoder layer class. Avoid wrapping the HuggingFace model directly as that
bypasses PagedAttention and fused kernels.

#### 1.2 Forward Pass

The AR stage forward pass is managed by vLLM's scheduler. Implement `forward()` to return an
`OmniOutput` with any intermediate data needed by Stage 1:

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs,
) -> OmniOutput:
    hidden_states = self.run_layers(input_ids, positions, intermediate_tensors, inputs_embeds)
    logits = self.lm_head(hidden_states)

    return OmniOutput(
        text_hidden_states=hidden_states,
        multimodal_outputs={
            "your_intermediate_key": self.extract_intermediate(hidden_states),
        },
    )
```

#### 1.3 Custom Stop Condition (if no EOS token)

Some TTS models use a learned stop head rather than an EOS token to terminate generation.
If your model does this, implement it inside `sample()`:

```python
def sample(
    self,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> SamplerOutput | None:
    output = self.sampler(logits, sampling_metadata)

    # Check learned stop head and mark request as finished when triggered
    if self._stop_head_fired():
        output = mark_as_finished(output)

    return output
```

### Step 2: Implement Stage 1 - Decoder

Stage 1 decodes the output of Stage 0 into audio. It runs outside the scheduler (no
PagedAttention needed). Implement `chunked_decode_streaming()` to support the async_chunk
streaming path with low-latency chunk-by-chunk decoding:

```python
# qwen3_tts_code2wav.py

class Qwen3TTSCode2Wav(nn.Module):
    """Code2Wav stage: RVQ codes -> audio waveform."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # Initialize your audio decoder here
        # (SpeechTokenizer, HiFiGAN, AudioVAE, etc.)

    def forward(self, codes: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.decoder(codes)

    def chunked_decode_streaming(
        self,
        codes: torch.Tensor,
        chunk_size: int = 25,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        """
        Decode with a sliding context window for smooth chunk boundaries.
        The left context is trimmed from the output after decoding.
        """
        end_index = codes.shape[-1]
        context_size = 0 if end_index <= chunk_size else left_context_size
        wav_chunk = self(codes)
        # Trim left context from output to avoid duplicate audio
        return wav_chunk[..., context_size * self.total_upsample:]
```

`chunk_size=25` and `left_context_size=25` are the validated defaults from Qwen3-TTS and
Qwen3-Omni. Use the same values unless your model has a strong reason not to.

### Step 3: Implement the Unified Model Class

The unified class dispatches to the correct stage based on `model_stage` in the config,
following the same pattern as `Qwen3TTSModelForGeneration` in `qwen3_tts.py`:

```python
# your_model.py

class YourTTSModelForConditionalGeneration(nn.Module, SupportsPP):
    """
    Unified TTS model combining AR stage and decoder stage.

    Set `model_stage` in vllm_config to one of: "ar_stage", "decoder"
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "ar_stage":
            ar_vllm_config = vllm_config.with_hf_config(
                vllm_config.model_config.hf_config.ar_config,
                architectures=["YourTTSARStageForConditionalGeneration"],
            )
            self.ar_stage = init_vllm_registered_model(
                vllm_config=ar_vllm_config,
                prefix=maybe_prefix(prefix, "ar"),
                hf_config=ar_vllm_config.model_config.hf_config,
                architectures=["YourTTSARStageForConditionalGeneration"],
            )
            self.model = self.ar_stage

        elif self.model_stage == "decoder":
            self.decoder = YourTTSDecoder(vllm_config=vllm_config, prefix=prefix)
            self.model = self.decoder

        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage!r}. "
                "Must be one of: 'ar_stage', 'decoder'"
            )
```

### Step 4: Create `__init__.py`

Export the main model class:

```python
# vllm_omni/model_executor/models/your_model_name/__init__.py
from .your_model import YourTTSModelForConditionalGeneration

__all__ = ["YourTTSModelForConditionalGeneration"]
```

## Key Components

### 1. AR Stage Optimization (Critical)

The most important optimization is using **vLLM's native decoder layers** with fused QKV
projections. In `load_weights()`, pack the HF `q_proj` / `k_proj` / `v_proj` into the fused
`qkv_proj` using `stacked_params_mapping`:

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    stacked_params_mapping = [
        # (fused_param_name, hf_param_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(self.named_parameters())
    loaded_weights = set()

    for name, loaded_weight in weights:
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
        loaded_weights.add(name)

    return loaded_weights
```

This is taken directly from `qwen3_tts_code_predictor_vllm.py` and should be reused as-is
for any Qwen-family backbone.

### 2. Output Format

Use `OmniOutput` for stage outputs so the orchestrator can route intermediate data to Stage 1:

```python
from vllm_omni.model_executor.models.output_templates import OmniOutput

return OmniOutput(
    text_hidden_states=hidden_states,
    multimodal_outputs={
        "codec_codes": codec_codes,    # or whatever your model produces
    },
)
```

The keys in `multimodal_outputs` are what your stage input processor will read via
`output.multimodal_output["codec_codes"]`.

### 3. Weight Loading

If both stages load from a single checkpoint, separate them by prefix in the unified class:

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    ar_weights, decoder_weights = [], []
    for name, tensor in weights:
        if name.startswith("decoder."):
            decoder_weights.append((name, tensor))
        else:
            ar_weights.append((name, tensor))

    if self.model_stage == "ar_stage":
        return self.ar_stage.load_weights(ar_weights)
    elif self.model_stage == "decoder":
        return self.decoder.load_weights(decoder_weights)
```

## Model Registration

Register all stage classes in `vllm_omni/model_executor/models/registry.py`:

```python
_OMNI_MODELS = {
    # ... existing models ...

    "YourTTSModelForConditionalGeneration": (
        "your_model_name",
        "your_model",
        "YourTTSModelForConditionalGeneration",
    ),
    "YourTTSARStageForConditionalGeneration": (
        "your_model_name",
        "your_model_ar_stage",
        "YourTTSARStageForConditionalGeneration",
    ),
    "YourTTSDecoder": (
        "your_model_name",
        "your_model_decoder",
        "YourTTSDecoder",
    ),
}
```

The registry uses lazy loading, so model classes are only imported when needed.

## Stage Configuration

### Batch mode

```yaml
# stage_configs/your_model_name.yaml

stage_args:
  - stage_id: 0
    stage_type: llm
    runtime:
      devices: "0"
      max_batch_size: 64
    engine_args:
      model_stage: ar_stage
      model_arch: YourTTSModelForConditionalGeneration
      worker_type: ar
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      engine_output_type: latent
    default_sampling_params:
      temperature: 0.9
      top_k: 50
      max_tokens: 2048

  - stage_id: 1
    stage_type: llm
    runtime:
      devices: "0"
    engine_args:
      model_stage: decoder
      model_arch: YourTTSModelForConditionalGeneration
      worker_type: generation
      scheduler_cls: vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
      engine_output_type: audio
    engine_input_source: [0]
    final_output: true
    final_output_type: audio
```

### Streaming mode (async_chunk)

Set `async_chunk: true` and provide `custom_process_next_stage_input_func` on Stage 0:

```yaml
# stage_configs/your_model_name_async_chunk.yaml

async_chunk: true

stage_args:
  - stage_id: 0
    stage_type: llm
    runtime:
      devices: "0"
      max_batch_size: 64
    engine_args:
      model_stage: ar_stage
      model_arch: YourTTSModelForConditionalGeneration
      worker_type: ar
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      engine_output_type: latent
      custom_process_next_stage_input_func: >
        vllm_omni.model_executor.stage_input_processors.your_model_name.ar2decoder_async_chunk
    default_sampling_params:
      temperature: 0.9
      top_k: 50
      max_tokens: 2048

  - stage_id: 1
    stage_type: llm
    runtime:
      devices: "0"
    engine_args:
      model_stage: decoder
      model_arch: YourTTSModelForConditionalGeneration
      worker_type: generation
      scheduler_cls: vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
      engine_output_type: audio
    engine_input_source: [0]
    final_output: true
    final_output_type: audio
```

## Stage Input Processors

Stage input processors convert Stage 0 outputs into Stage 1 inputs. They follow the pattern
established in `stage_input_processors/qwen3_tts.py`. Create yours in
`vllm_omni/model_executor/stage_input_processors/your_model_name.py`.

### Batch mode (non-streaming)

```python
def ar2decoder(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Collect all Stage 0 outputs and forward them to Stage 1 in one shot."""
    source_id = engine_input_source[0]
    decoder_inputs = []

    for output in stage_list[source_id].engine_outputs:
        result = output.outputs[0]
        # Extract the intermediate key set in your model's forward() OmniOutput
        intermediate = result.multimodal_output["your_intermediate_key"].cpu()

        decoder_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=intermediate.reshape(-1).tolist(),
            )
        )

    return decoder_inputs
```

### Streaming mode (async_chunk)

Buffer Stage 0 outputs in the connector and forward a chunk to Stage 1 as soon as
`chunk_size` steps are ready. Use `chunk_size=25` and `left_context_size=25` (validated
defaults from Qwen3-TTS and Qwen3-Omni) unless your decoder has a strong reason to differ:

```python
def ar2decoder_async_chunk(
    connector: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> dict[str, Any] | None:
    """
    Forward a chunk of intermediate outputs to Stage 1 as soon as chunk_size steps
    have accumulated. Mirrors talker2code2wav_async_chunk in qwen3_tts.py.
    """
    if "your_intermediate_key" not in pooling_output:
        return None

    intermediate = pooling_output["your_intermediate_key"]
    if intermediate is None or intermediate.numel() == 0:
        return None

    request_id = request.external_req_id
    chunk_size = left_context_size = 25

    connector.code_prompt_token_ids[request_id].append(
        intermediate.cpu().reshape(-1).tolist()
    )

    length = len(connector.code_prompt_token_ids[request_id])
    chunk_remainder = length % chunk_size

    if chunk_remainder != 0 and not request.is_finished():
        return None

    context_len = chunk_remainder if chunk_remainder != 0 else chunk_size
    end_idx = min(length, left_context_size + context_len)

    return {
        "your_intermediate_key": (
            torch.tensor(connector.code_prompt_token_ids[request_id][-end_idx:])
            .reshape(-1).tolist()
        ),
        "finished": torch.tensor(request.is_finished(), dtype=torch.bool),
    }
```

## Testing

For general testing conventions, see [tests_style.md](../ci/tests_style.md).

Recommended test cases for a new TTS model:

1. **Single request**: verify waveform output shape and sample rate
2. **Batched requests**: verify each request in the batch finishes independently
3. **async_chunk streaming**: verify audio chunks arrive incrementally and decode correctly
4. **Speaker conditioning** (if applicable): verify different speaker inputs produce different outputs
5. **Model variants** (dense vs MoE, if applicable): verify weight loading works for each

Reference test location for Qwen3-TTS:
`tests/model_executor/stage_input_processors/test_qwen3_tts_async_chunk.py`

## Adding a Model Recipe

After implementing and testing your model, add a model recipe to the
[vllm-project/recipes](https://github.com/vllm-project/recipes) repository so users can
get started quickly. See [Adding an Omni-Modality Model](./adding_omni_model.md#adding-a-model-recipe)
for the expected format.

## Summary

Adding a TTS model to vLLM-Omni involves:

1. **Create model directory** with AR stage, decoder stage, and unified class
2. **AR stage**: use vLLM's native decoder layers and fused QKV - do not wrap the HF model directly
3. **Decoder stage**: thin wrapper around your audio decoder; implement `chunked_decode_streaming()`
4. **Unified class**: dispatches on `model_stage`; identical structure to `Qwen3TTSModelForGeneration`
5. **Register** all stage classes in `registry.py`
6. **YAML configs**: provide both batch and `async_chunk` variants following RFC #1225 conventions
7. **Stage input processor**: buffer Stage 0 outputs and forward in chunks of 25
8. **Tests**: cover single request, batching, and async_chunk streaming
9. **Model recipe**: add to [vllm-project/recipes](https://github.com/vllm-project/recipes)

### Qwen3-TTS Reference Files

| File | Purpose |
|------|---------|
| `models/qwen3_tts/qwen3_tts.py` | Unified model class |
| `models/qwen3_tts/qwen3_tts_code_predictor_vllm.py` | AR stage with vLLM fused ops (key reference) |
| `models/qwen3_tts/qwen3_tts_code2wav.py` | Decoder stage with `chunked_decode_streaming()` |
| `stage_configs/qwen3_tts.yaml` | Stage configuration |
| `stage_input_processors/qwen3_tts.py` | Stage transition processors |
| `tests/model_executor/stage_input_processors/test_qwen3_tts_async_chunk.py` | Test reference |

For more information, see:

* [Architecture Overview](../../design/architecture_overview.md)
* [Async Chunk Design](../../design/feature/async_chunk_design.md)
* [Stage Configuration Guide](../../configuration/stage_configs.md)
* [RFC #1225 - General TTS Model Implementation](https://github.com/vllm-project/vllm-omni/issues/1225)
