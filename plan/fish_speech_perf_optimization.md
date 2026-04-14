# Fish Speech Two-Stage Perf Optimization Plan

## Goal
Beat sgl-omni's RTF on Fish Speech S2-Pro.

## Baseline Targets
- **sgl-omni claim**: RTF 0.34, 63.3 tok/s on H200 (single batch)
- **vllm-omni current** (H20, two-stage, c=1): RTF 0.30, 3.36 audio-s/wall-s
- **Need to measure**: sgl-omni on H20-server-0 same hardware

## Branch
`feat/fish-speech-perf` (from origin/main, worktree at `../vllm-omni-fish-perf`)

## Architecture Comparison

| Aspect | sgl-omni | vllm-omni current |
|--------|----------|-------------------|
| Slow AR + Fast AR fusion | One CUDA graph | Two separate graphs |
| DAC vocoder | Stage 3 (separate) | Stage 1 (separate) ✓ |
| KV cache | Paged + radix prefix | Paged ✓ |
| Attention | FA3 (forced) | flashinfer/FA3 (?) |
| BF16 RoPE truncation | Yes | TBD check |
| Fast AR batching | Per-request sequential | Per-request sequential |

## Optimization Targets (ranked by expected gain)

### High impact (>5%)
1. **Fuse Slow AR + Fast AR into single CUDA graph** — eliminates 2 graph launches per step + Python orchestration
2. **Batch Fast AR across concurrent requests** — sgl-omni admits this is a future weakness
3. **Reduce SharedMemoryConnector overhead** — polling, serialization (~50-100ms per request)

### Medium impact (1-5%)
4. **Force FlashAttention 3** if not already
5. **BF16 RoPE truncation** if not already
6. **Reduce per-step Python overhead** in `_preprocess` for-loop

### Low impact (<1%)
7. CUDA graph capture sizes tuning
8. Better sampling kernels

## Phases

### Phase 0: Baseline + sgl-omni same-hardware comparison (today)
- Run sgl-omni server on h20-server-0
- Run unified benchmark script across both
- Document exact numbers

### Phase 1: Connector overhead reduction
- Profile SharedMemoryConnector
- Reduce polling intervals where safe
- Measure delta

### Phase 2: Slow+Fast AR CUDA graph fusion
- Move talker_mtp call into model.forward()
- Use persistent buffers for inter-step state
- Re-capture combined CUDA graph
- Measure delta

### Phase 3: Batched Fast AR
- Fast AR currently runs per-request sequentially
- Batch across concurrent requests
- Measure delta at concurrency 4, 8

### Phase 4: Misc optimizations
- FA3 verification, BF16 RoPE, etc.

## Success Criteria
- Match sgl-omni RTF on same hardware
- Bonus: beat at higher concurrency

## Same-Hardware Baseline (H20, measured 2026-04-14)

| Metric | sgl-omni | vllm-omni | gap |
|--------|----------|-----------|-----|
| c=1 throughput | 3.88 | 3.31 | -15% |
| c=1 RTF | 0.258 | 0.298 | +15% |
| c=1 audio | 5.03s | 4.68s | -7% (truncation) |
| c=4 throughput | 11.72 | 9.26 | -21% |
| c=4 RTF | 0.329 | 0.415 | +26% |
| c=8 throughput | 18.00 | 10.20 | -43% |
| c=8 RTF | 0.371 | 0.698 | +88% |

The gap explodes with concurrency — points to a serialization/locking
issue in the inter-stage pipeline that doesn't scale.

## Profile Results (PR #2472 torch profiler, H20, c=1, 3 reqs)

**CPU time (2.97s) > CUDA time (2.03s)** — CPU-bound at orchestration.

| Op | CPU time | % | Calls |
|----|----------|---|-------|
| aten::copy_ | 1.213s | 40.82% | 7910 |
| aten::sort | 56ms | 1.90% | 247 |
| aten::mm | 26ms | 0.88% | 679 |
| aten::index | 21ms | 0.70% | 494 |

**Top GPU kernels (already optimal):**
- nvjet matmul (transformer attn/MLP): 423ms (20.86%)
- Flash Attention 3 fwd: 123ms (6.07%)
- _vllm_fa3_C::fwd: 1.5ms (FA3 confirmed)

## Bottleneck Identified
The 1.21s spent on `aten::copy_` is the single biggest opportunity.
Sources of per-step copies in `_preprocess`:
- 4 per-request talker_mtp buffer copies (lines 1258-1261)
- 1 per-request inputs_embeds copy (line 1269)
- 1 per-request input_ids copy (line 1271)
- KV cache writes (in graph, but plumbing copies before)
- Prefill prompt embed overlay (line 1061)
- inputs_embeds.gpu copy (line 1138)

For batch_size=N at 100 steps: ~6×N×100 = 600×N small copies.
Each copy has ~20-30μs Python overhead. At c=8: 4800 copies × 25μs ≈ 120ms.
Profile shows 1.2s — there's more overhead than just direct copies.

## Next Action: Batch the per-request preprocess loop
Replace per-request `.gpu[i:i+1].copy_(...)` with batched
`torch.cat(per_req_data)` + single `.gpu[:n].copy_(...)`.
Expected gain: 5-10% reduction in CPU time → similar improvement
in throughput at high concurrency where CPU-bound.
