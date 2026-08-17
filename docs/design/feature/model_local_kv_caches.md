# Model-local KV caches

Some models keep their attention KV as HuggingFace `transformers` cache objects
instead of using the engine's paged KV manager. That memory is allocated after
the profiling run that sizes the KV pool, so no per-stage footprint includes it.

Each model declares what it holds by implementing `model_local_kv_specs()`, and
`OmniGPUModelRunner.load_model` logs the total after load. The numbers below are
what that reporting emits on an H200; they are produced by the shipped code, not
transcribed into this page.

## What the four models declare

| Model | Cache | Bounded by | Per instance | Resident | Per request |
|---|---|---|---|---|---|
| Qwen3-TTS codec decoder (stage 1) | sliding `DynamicCache` | `sliding_window - 1 = 71` | 4.44 MiB | 22.19 MiB (5 captures) | 4.44 MiB |
| MiMo-Audio local transformer | `DynamicCache` | `group_size + max(delay_pattern) = 11` | 704 KiB | 179.44 MiB (261 rows) | 704 KiB |
| MiniCPM-o 4.5 Whisper encoder | `EncoderDecoderCache` | `embed_positions` rows = 1500 | 140.62 MiB | — | per session |
| ming_flash_omni talker | `StaticCache` | hardcoded `max_cache_len = 2048` | 24.00 MiB [^1] | — | full extent |

**No cache is bounded by `max_model_len`.** Each has its own mechanism: a
sliding window, a decode-loop trip count, an encoder frame limit, a constant.
Sizing any of them by sequence length overstates them by orders of magnitude --
Qwen3-TTS at stage 1's `max_model_len` would read 256 MiB/request instead of
4.44 MiB.

**ming's geometry does not exist in this repo.** `self._llm_config` is a
`Qwen2Config` from the checkpoint, so layers, kv-heads, head-dim and dtype are
only knowable after load. A static table has a hole here; a post-load query does
not. This is the case that decides the shape of the whole thing.

[^1]: 24 layers, 2 kv-heads, head-dim 64 (derived from `hidden_size 896 /
    num_attention_heads 14`), bfloat16 -- every one of those read from
    `Jonathan1909/Ming-flash-omni-2.0`'s `talker/llm/config.json`, and none of
    them present anywhere in this repository. The 24.00 MiB was checked against
    a real `StaticCache` built from that config.

## Two multiplicities, not one

Peak memory needs both of these, and they scale differently:

`max_live_instances` is **model-side replication**, and the model knows it. Both
Qwen3-TTS and MiMo copy a cache into every captured CUDA-graph bucket, where it
stays for the process lifetime. MiMo's per-call cache is 704 KiB and its graph
pool is 179.44 MiB of the same thing.

Engine concurrency is **not** something the model can know, so it must not
appear in a declaration. `scope` decides whether the engine multiplies: a
`MODEL`-scoped cache is captured once and shared, so it costs the same at
`max_num_seqs=64` as at 1, while everything else is per request, session, or
call. `ModelLocalKVSpec.peak_bytes(concurrency)` applies that rule; the runner
passes `max_num_seqs`.

This is where the number gets interesting. Qwen3-TTS stage 1 at
`max_num_seqs=64` declares 284 MiB of per-request cache on top of 22 MiB
resident -- roughly 306 MiB living entirely outside the paged manager, on a
stage whose weights are under 4 GiB.

## Declaring instead of tabulating

`vllm_omni/model_executor/models/model_local_kv.py` defines `ModelLocalKVSpec`
and `HasModelLocalKV`. It describes and never allocates -- allocation for the
diffusion path is owned by RFC #5244 / PR #6094, and this must not become a
second allocator.

`collect_model_local_kv_specs()` walks the module tree to find owners, so a
model does not have to forward anything from its registered class down to
whichever inner module holds the cache. Two wrinkles it handles, both found by
running it rather than by reading:

- The runner's `self.model` may be a `CUDAGraphWrapper`, which is a plain
  callable holding `.runnable` and not an `nn.Module`. Walking it directly finds
  nothing, and every declaration silently reports zero. The collector unwraps
  first.
- A cache owner that is not itself a submodule (Qwen3-TTS's graph wrapper,
  ming's `MingAudioGenerator`) is reached by one explicit hop from the module
  that holds it.

A table like the one above goes stale the moment someone changes
`sliding_window`. A declaration does not.

## Adding a model

Return one entry per distinct *lifetime*, not per cache object: a retained
per-request cache and a short-lived working copy of the same geometry are two
entries. `capacity_source` is diagnostic text -- never branch on it. Prefer
`spec_from_hf_config()` and pass the same config object the cache is built from;
it raises rather than guessing when an attribute is missing, because a silently
wrong geometry is the failure this is meant to prevent.

## Known limits

Counts are taken from live state after warmup, so a cache that `transformers`
has not yet materialized reports zero. Qwen3-TTS's xvec-path captures are in
that state at load, which makes the 22.19 MiB resident figure a floor rather
than a ceiling for that path.

MiMo's graph-pool cache has no reachable Python object at all -- it is a local
inside the captured frame, and only the pool holds its memory -- so its 179.44
MiB is derived from the captured bucket list rather than measured from tensors.

MiniCPM-o's encoder cache exists only on the streaming path, so a text-only run
declares 140.62 MiB and allocates none of it. The declaration is the ceiling
that path reaches, not a claim about every request.

MiMo captures every bucket in `MIMO_CUDAGRAPH_BATCH_SIZES` gated only on
`torch.cuda.is_available()` (`mimo_audio_llm.py:670`), so those 179.44 MiB stay
resident even when the deploy config sets `enforce_eager: true`. Not fixed here:
gating it is a one-line change, but verifying it needs a model boot, and an
unverified startup-behaviour change is not worth bundling into this PR.
