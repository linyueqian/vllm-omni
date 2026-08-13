# Model-local KV caches

Several vLLM-Omni model implementations keep their own attention KV state instead of
using the engine's paged KV manager. This page inventories them, because the
question "which caches exist and what do they cost" currently has no written answer,
and three separate RFCs ([#4366](https://github.com/vllm-project/vllm-omni/issues/4366),
[#5244](https://github.com/vllm-project/vllm-omni/issues/5244),
[#4855](https://github.com/vllm-project/vllm-omni/issues/4855) K3) each assume a
different one.

Line numbers are against `dbc0dd6d8`.

## What is actually there

They are not ad-hoc lists. Every one of them is a HuggingFace `transformers`
cache object, held in model code, allocated per call:

| Model | Type | Allocated | Scope |
|-------|------|-----------|-------|
| `mimo_audio` | `DynamicCache` | `mimo_audio_llm.py:827` | local, re-bound each step from `output.past_key_values` |
| `ming_flash_omni` talker | `StaticCache` | `talker_module.py:843` via `_init_kv_cache` | local to the generate call |
| `minicpmo_4_5` | `EncoderDecoderCache(DynamicCache(), DynamicCache())` | `minicpmo_4_5_omni_llm.py:2765` | local, 5 allocation sites |
| `nemotron_voicechat` | HF cache held in a session dict | `nemotron_voicechat_talker.py:343` | per session, lives across turns |

`DynamicCache` grows a `list[Tensor]` per layer as tokens are appended;
`StaticCache` preallocates the full extent up front. Both are ordinary torch
allocations on the same device as the model.

## The problem is accounting, not lifetime

The obvious worry is leaks. That is mostly not the issue — `_init_kv_cache`
returns a local, and the `DynamicCache` sites are re-bound or dropped when the
call returns, so refcounting frees them. `nemotron_voicechat` is the exception
worth watching, since its cache hangs off a session that outlives a single
request.

The real problems are that this memory is invisible and arbitrarily sized:

**Invisible.** None of these allocations pass through
`determine_available_memory()` or any per-stage budget. A stage's reported
footprint is weights plus graphs plus vLLM's own KV pool; these caches are on
top of that, allocated after the profiling run that decided how much KV memory
the engine could claim. That is the cross-stage OOM class that
[#4855](https://github.com/vllm-project/vllm-omni/issues/4855) K2 and
[#6071](https://github.com/vllm-project/vllm-omni/pull/6071) are about, arriving
from a direction neither of them currently covers.

**Arbitrarily sized.** `ming_flash_omni` hardcodes `max_cache_len = 2048` with
`max_batch_size=1` (`talker_module.py:840-848`). The constant has no relation to
the request's actual length or to `max_model_len`; a `StaticCache` at that extent
is preallocated in full whether the request needs 50 tokens or 2000. Each model
picks its own number this way.

**Per-model, so per-model bugs.** Four models, four cache disciplines, no shared
test surface. A fix for one does not carry.

## What this does not propose

Deliberately no design here. Unifying these onto a paged manager is
[#4855](https://github.com/vllm-project/vllm-omni/issues/4855) K3, and it should
be scoped with the owners of
[#5244](https://github.com/vllm-project/vllm-omni/issues/5244) rather than
decided TTS-side, especially now that
[#6094](https://github.com/vllm-project/vllm-omni/pull/6094) has landed a
scheduler-managed block allocator for the diffusion path. The useful next step is
agreeing which of the three RFCs owns this surface; that argument is easier with
the inventory written down than without it.

Two things that are worth doing regardless of how that lands, in rough order of
cost:

1. Report these allocations per stage, so the footprint numbers stop being wrong.
   [#5180](https://github.com/vllm-project/vllm-omni/pull/5180) is adding
   per-stage memory observability and is the natural place.
2. Derive `max_cache_len` from the stage's `max_model_len` instead of a
   per-model constant, so the preallocation is at least proportionate.

## Adding to this page

If you add a model that keeps its own KV state, add a row. The check that
produced the table is a `git grep` for
`DynamicCache|StaticCache|EncoderDecoderCache|SlidingWindowCache|HybridCache`
under `vllm_omni/model_executor/models/`.
