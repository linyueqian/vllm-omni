# Model-local KV caches

Some models keep their attention KV as HuggingFace `transformers` cache objects
instead of using the engine's paged KV manager. That memory is allocated after
the profiling run that sizes the KV pool, so no per-stage footprint includes it.

Measured against `596c16a55`. Every figure is derived from the checkpoint config
plus the code that bounds the cache.

| Model | Cache | Bounded by | Per instance |
|---|---|---|---|
| Qwen3-TTS codec decoder (stage 1) | sliding `DynamicCache` | `sliding_window - 1 = 71` | 4.44 MiB |
| MiMo-Audio local transformer | `DynamicCache` | `group_size + max(delay_pattern) = 11` | 704 KiB |
| MiMo-Audio graph pool | same, captured | 261 retained batch rows | 179.44 MiB [^1] |
| MiniCPM-o 4.5 Whisper encoder | `EncoderDecoderCache` | 1500 encoder frames | 140.62 MiB |
| ming_flash_omni talker | `StaticCache`, preallocated | hardcoded `max_cache_len = 2048` | not statically determinable |

Three things this table makes obvious, none of which were before.

**No cache is bounded by `max_model_len`.** Each has its own mechanism: a
sliding window, a decode-loop trip count, an encoder frame limit, a constant.
Sizing any of them by sequence length overstates them by orders of magnitude --
Qwen3-TTS at stage 1's `max_model_len` would read 256 MiB/request instead of
4.44 MiB.

**Lifetime and multiplicity are independent.** MiMo's per-call cache is 704 KiB,
but it is captured once per bucket in `MIMO_CUDAGRAPH_BATCH_SIZES`, so 261 batch
rows stay resident for the process. Peak memory needs both numbers.

**ming's geometry does not exist in this repo.** `self._llm_config` is a
`Qwen2Config` from the checkpoint, so layers, kv-heads, head-dim and dtype are
only knowable after load. Any static table has a hole here; a post-load query
does not.

## Declaring instead of tabulating

`vllm_omni/model_executor/models/model_local_kv.py` defines
`ModelLocalKVSpec` + `HasModelLocalKV`: a model declares what it holds, the
engine sums it. It describes and never allocates -- allocation for the diffusion
path is owned by RFC #5244 / PR #6094 and this must not become a second
allocator.

A table like the one above is stale the moment someone changes `sliding_window`.
A declaration is not.

## Adding a model

Return one entry per distinct *lifetime*, not per cache object: a retained
per-request cache and a short-lived working copy of the same geometry are two
entries. `capacity_source` is diagnostic text -- never branch on it.

[^1]: MiMo captures every bucket in `MIMO_CUDAGRAPH_BATCH_SIZES` gated only on
`torch.cuda.is_available()` (`mimo_audio_llm.py:670`), so these stay resident
even when the deploy config sets `enforce_eager: true`. Not fixed here --
gating it is a one-line change but verifying it needs a model boot, and an
unverified startup-behaviour change is not worth bundling into this PR.
