# Qwen3-Omni on vLLM-Omni: Performance Optimizations
## Summary

Qwen3-Omni is a native multimodal model that can understand **text, audio, image, and video** inputs, and generate both **text** and **speech** outputs. In production serving, this architecture is naturally split into three stages:

- **Thinker**: multimodal understanding and text generation
- **Talker (+ Talker-MTP / code predictor path)**: converts semantic/text representations into codec tokens
- **Code2Wav**: decodes codec tokens into waveform audio

vLLM-Omni now supports running this full Qwen3-Omni pipeline end-to-end, and more importantly, supports stacking multiple latency/throughput optimizations that work together:

1. **Batching** improves GPU utilization stage by stage and increases overall throughput.
2. **CUDA Graph** reduces CPU launch overhead and decode-time jitter on stable shapes.
3. **Async Chunk and Streaming Output** overlap compute and communication across stages and emit audio incrementally, improving both TTFP and E2E.

Compared with **Hugging Face Transformers** (offline, single request), vLLM-Omni with the full optimization stack (Batching + CUDA Graph + Async Chunk + Streaming Output) delivers much lower latency and higher efficiency:

<table><tr>
<td><img src="figures/E2EL_s_vllm_omni_vs_transformers.png" alt="E2E Latency: vLLM-Omni vs HF transformers" width="100%"/></td>
<td><img src="figures/TTFP_s_vllm_omni_vs_transformers.png" alt="TTFP: vLLM-Omni vs HF transformers" width="100%"/></td>
<td><img src="figures/RTF_vllm_omni_vs_transformers.png" alt="RTF: vLLM-Omni vs HF transformers" width="100%"/></td>
</tr></table>

- **E2E latency**: 23.78 s (vLLM-Omni) vs 336.10 s (transformers) — **~93%** reduction
- **Time to first audio (TTFP)**: 0.934 s vs 336.10 s — **~99.7%** reduction
- **Real-time factor (RTF)**: 0.32 vs 3.776 — **~91%** reduction (~12× faster)

Compared with the baseline (vLLM-Omni without these optimizations), the stacked setup (Batching + CUDA Graph + Async Chunk + Streaming Output) achieves:

- **E2E latency (E2EL) reduction**: **~91%** at concurrency 10 (1,523,135 ms → 130,682 ms); **~81%** at concurrency 1 (325,865 ms → 60,436 ms)
- **Audio TTFP reduction**: **~99.2%** at concurrency 10 (1,522,804 ms → 12,262 ms); **~99.6%** at concurrency 1 (325,517 ms → 1,263 ms)
- **Real-time factor (RTF)**: **~89%** reduction at concurrency 10 (6.94 → 0.74), i.e. **~9×** higher effective throughput; **~78%** at concurrency 1 (1.52 → 0.33)

*Final = Batching + CUDA Graph + Async Chunk + Streaming Output.*

**Benchmark environment:** All performance data and figures in this post were measured on **A100** GPUs.

All experiments can be reproduced with the following software stack:

- **vLLM**: v0.16.0  
- **vllm-omni**: commit 070ea0dd  
- **CUDA**: 12.8  

Below is an example command used to generate these benchmarks:

```bash
vllm bench serve \
  --dataset-name random \
  --port ${PORT} \
  --model ${MODEL_PATH} \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --max-concurrency ${MAX_CONCURRENCY} \
  --num-prompts ${NUM_PROMPTS} \
  --random-input-len 2500 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
  --random-output-len 900 \
  --extra_body '{"modalities": ["text","audio"]}'
```

This post walks through each optimization in the same order they are typically enabled in practice, then ends with a deployment playbook you can directly apply.

---

## Pipeline Batching Across Three Stages

### How stage-wise batching works

For Qwen3-Omni, batching is not a single switch at one model boundary. It is a pipeline-level optimization:

- requests are grouped per stage using `runtime.max_batch_size`
- each stage executes batch inference with its own scheduler/worker
- stage outputs are routed to downstream stages with per-request mapping preserved

**Batching strategy by stage:** Stage-0 (Thinker) and Stage-1 (Talker) use **continuous batching**: requests can join and leave the batch over time as they progress, so the scheduler can pack work efficiently. Stage-2 (Code2Wav) uses **static batching**: once a batch is formed, the stage runs inference on the whole batch and only starts the next batch after every request in the current batch has finished. This matches the decode pattern of Code2Wav and keeps implementation simple while still improving throughput over no batching.

This is especially important for multimodal speech generation, where bottlenecks may move between Thinker, Talker, and Code2Wav depending on concurrency and output length.

### Stage-level batching results (Baseline vs. Batch)

Batching alone greatly reduces E2EL and RTF across all concurrencies (e.g. at concurrency 10, E2EL drops from ~1,523 s to ~262 s; at concurrency 1, from ~326 s to ~259 s).

<table><tr>
<td><img src="figures/Mean_E2EL_ms_Baseline_vs_Batch.png" alt="E2EL: Baseline vs Batch" width="100%"/></td>
<td><img src="figures/Mean_AUDIO_TTFP_ms_Baseline_vs_Batch.png" alt="TTFP: Baseline vs Batch" width="100%"/></td>
<td><img src="figures/Mean_AUDIO_RTF_Baseline_vs_Batch.png" alt="RTF: Baseline vs Batch" width="100%"/></td>
</tr></table>

---

## CUDA Graph on the Critical Decode Path

### Why CUDA Graph helps here

In decode-heavy serving, repeatedly launching many small kernels from CPU can become a visible overhead. CUDA Graph reduces this overhead by capturing and replaying stable execution graphs.
In stage configs, this is typically represented by `enforce_eager: false` for stages where graph capture is desired (commonly Thinker/Talker), while Code2Wav may keep eager mode depending on stage behavior.

### CUDA Graph results on top of batching

Adding CUDA Graph on the decode path further cuts E2EL and TTFP (e.g. at concurrency 1, E2EL drops from ~259 s to ~67 s; at concurrency 10, from ~262 s to ~153 s) and lowers RTF at every concurrency.

<table><tr>
<td><img src="figures/Mean_E2EL_ms_Batch_vs_Batch_CUDA_Graph.png" alt="E2EL: Batch vs Batch + CUDA Graph" width="100%"/></td>
<td><img src="figures/Mean_AUDIO_TTFP_ms_Batch_vs_Batch_CUDA_Graph.png" alt="TTFP: Batch vs Batch + CUDA Graph" width="100%"/></td>
<td><img src="figures/Mean_AUDIO_RTF_Batch_vs_Batch_CUDA_Graph.png" alt="RTF: Batch vs Batch + CUDA Graph" width="100%"/></td>
</tr></table>

---

## Async Chunk and Streaming Output: Earlier Audio and Cross-Stage Overlap

### Why this step matters for first-packet latency

Two mechanisms work together to improve user-visible latency:

- **Streaming output**: **audio streaming** emits audio chunks as soon as they are decoded (lower **TTFP**). Without streaming, the client waits for larger buffers or end-of-sequence.
- **Async chunk** is the main enabler for *earlier* audio: instead of handing off whole-request results between stages, each stage forwards **chunks** so the next stage can start as soon as the first chunk is ready. Thinker → Talker forwards hidden-state chunks; Talker → Code2Wav forwards codec chunks; Code2Wav decodes and emits packets incrementally. This **overlaps compute and communication** across stages and directly reduces time-to-first-audio-packet (TTFP) and end-to-end latency (E2EL).

So in practice: streaming output defines *how* bytes are sent to the client; async chunk defines *when* the pipeline can produce the first bytes.

**Dependency between the two:** Async chunk and audio streaming output are mutually dependent. Without async chunk, **audio streaming output cannot truly take effect**. Without audio streaming output, async chunk’s **TTFP advantage is not fully realized**: the client would still wait for larger buffers or end-of-sequence instead of hearing the first packet as soon as it is ready. We therefore recommend enabling **both** on top of batching + CUDA Graph; the benchmarks in this post use both.

Enabling **async chunk and streaming output on top of batching + CUDA Graph** yields large gains for TTFP (first-packet latency) and improves E2EL and RTF as well in our benchmarks.

### Results: Batch + CUDA Graph vs. Batch + CUDA Graph + Async Chunk + Streaming Output

Stacking assumptions: batching and CUDA Graph enabled. The figures below compare **without** vs **with** async chunk and streaming output. Enabling both brings **TTFP down sharply** (e.g. at concurrency 1: 67,121 ms → 1,263 ms, **~98% reduction**; at concurrency 4: 98,679 ms → 3,175 ms, **~97% reduction**), so users hear the first audio much sooner. E2EL and RTF also improve at every concurrency (e.g. E2EL 67,381 → 60,436 ms at concurrency 1; 153,352 → 130,682 ms at concurrency 10). The stacked setup consistently outperforms the baseline across all metrics and concurrencies.

<table><tr>
<td><img src="figures/Mean_E2EL_ms_Batch_CUDA_Graph_vs_Async_Chunk.png" alt="E2EL: Batch+CG vs +Async Chunk + Streaming" width="100%"/></td>
<td><img src="figures/Mean_AUDIO_TTFP_ms_Batch_CUDA_Graph_vs_Async_Chunk.png" alt="TTFP: Batch+CG vs +Async Chunk + Streaming" width="100%"/></td>
<td><img src="figures/Mean_AUDIO_RTF_Batch_CUDA_Graph_vs_Async_Chunk.png" alt="RTF: Batch+CG vs +Async Chunk + Streaming" width="100%"/></td>
</tr></table>

---

## Deployment Playbook: Enabling Qwen3-Omni Optimizations in vLLM-Omni

### 1) Serve Qwen3-Omni with the default 3-stage config

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091
```

Notes:

- `runtime.max_batch_size` controls stage-level batching.
- Thinker/Talker commonly use `enforce_eager: false` for CUDA Graph paths.
- Code2Wav often remains eager (`enforce_eager: true`) depending on runtime behavior.

### 2) Enable async chunk

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091 \
  --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_omni_moe_async_chunk.yaml
```

The async chunk config enables:

- top-level `async_chunk: true`
- async stage handoff processors (`thinker2talker_async_chunk`, `talker2code2wav_async_chunk`)

### 3) Key config knobs (quick reference)

```yaml
async_chunk: true
stage_args:
  - stage_id: 0 # thinker
    runtime:
      max_batch_size: 64
    engine_args:
      enforce_eager: false
      max_num_batched_tokens: 32768
      custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker_async_chunk

  - stage_id: 1 # talker
    runtime:
      max_batch_size: 64
    engine_args:
      enforce_eager: false
      max_num_batched_tokens: 32768
      custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav_async_chunk

  - stage_id: 2 # code2wav
    runtime:
      max_batch_size: 64
    engine_args:
      enforce_eager: true
      max_num_batched_tokens: 51200
```
---
