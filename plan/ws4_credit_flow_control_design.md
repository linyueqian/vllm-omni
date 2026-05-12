# WS-4: credit-based producer-consumer flow control

Owner: @linyueqian. Tracking: RFC issue #3535 (WS-4).

## Problem

At c≥64 on Qwen3-TTS, Stage 0 (talker) emits codec chunks for every running stream every LM step. Stage 1 (code2wav) processes one stream per step (`max_num_seqs=1`). Stage 0 has no backpressure: it keeps oversubscribing the SHM connector regardless of how far behind Stage 1 is, and at `chunk_transfer_adapter.py:357` every running request beyond `max_num_seqs` is preempted each step — so 127 streams churn between WAITING_FOR_CHUNK and PREEMPTED while 1 is served.

Symptoms in bench (H20, default config):

- SHM queue unbounded (puts - gets grows linearly in time per stream until Stage 1 drains).
- Stage 0 spends LM steps producing chunks for streams whose previous chunks have not been consumed yet (wasted compute on backpressured streams).
- per-stream undr_p99 = 8–18 s.

The behavior matches a producer running open-loop against a slow consumer. Stage 0 should *skip* a stream in its LM step when that stream already has too many unconsumed chunks downstream.

## Non-goal

Fixing the underrun SLO itself. Bounded-K *scheduling* (servicing fewer streams concurrently to give each one continuous-enough chunks for realtime playback) is a separate RFC. WS-4 only matches rates so Stage 0 stops wasting work; it does not change which stream Stage 1 picks next.

## Design

### Concept

Each stream `s` carries `outstanding(s) = puts(s) - gets(s)`. A static cap `K` (config: `stage_credit_per_stream`). Stage 0's scheduler skips `s` in the current LM step when `outstanding(s) >= K`. When Stage 1 consumes a chunk, `outstanding(s)` drops, unblocking `s` on the next step.

`K=1` means strict ping-pong (Stage 0 never gets ahead). `K=4` means up to 4 unconsumed chunks per stream — leaves enough buffer to mask Stage 1 jitter without unbounded growth. Default off (no cap → current behavior).

### Cross-process credit return

Stage 0 and Stage 1 run as separate engines (separate processes) under the orchestrator. `put_req_chunk` (Stage 0's adapter) and `get_req_chunk` (Stage 1's adapter) are local to their owners. We need a cheap cross-process counter.

Option A — separate SHM counter segment (chosen):

- One mmap-backed segment per `(external_req_id)`, fixed size (e.g. 16 bytes: u64 puts, u64 gets). Created on first put.
- Stage 0 atomically increments `puts` on each successful `connector.put(...)` (chunk_transfer_adapter.py:258).
- Stage 1 atomically increments `gets` on each successful `_poll_single_request` (chunk_transfer_adapter.py:153).
- Stage 0's scheduler reads `outstanding = puts - gets` before deciding to schedule that stream's LM step.
- Cleanup: unlink on `cleanup_sender`.

Why not piggyback existing SHM keys: the keys are per-chunk and torn down on consume. There is no persistent per-stream registry today, and adding one to the connector keeps the credit mechanism local to the transport layer where it belongs.

Option B — reverse SHM messages (rejected): higher latency, polling overhead in Stage 0's hot path.
Option C — RPC over orchestrator control plane (rejected): cross-process latency too high relative to per-chunk timescale (~50 ms).

### Integration points

1. **`OmniChunkTransferAdapter.__init__`**: open/create a `StreamCreditRegistry` keyed by `external_req_id`. Read `stage_credit_per_stream` from `model_config.async_chunk` extras; default `0` = unlimited.
2. **`_send_single_request` (Stage 0 side, line 258)**: on success, `registry.inc_put(external_req_id)`.
3. **`_poll_single_request` (Stage 1 side, line 153)**: on success, `registry.inc_get(req_id_external)`.
4. **`OmniARScheduler.schedule` (Stage 0)**: before letting a running request consume `token_budget`, query `registry.outstanding(external_req_id)`. If `≥ K`, leave the request in `running` but skip allocation (same shape as the existing `WAITING_FOR_CHUNK` skip in `OmniGenerationScheduler.schedule:104`).
5. **`cleanup_sender`**: `registry.drop(external_req_id)`.

The skip in step 4 must not change request state — just defer this stream's LM step until next tick. Track a counter `stage0_credit_skipped` for stats.

### Knobs

```yaml
async_chunk:
  enable: true
  stage_credit_per_stream: 4  # 0 = unlimited (legacy)
  stage_credit_registry_path: /dev/shm/vllm_omni_credits  # opt; default /dev/shm
```

K must be ≥1. K=1 is correct but pessimistic (no buffer for Stage 1 jitter). Start with K=4.

### Failure modes & fallback

- **Stage 1 dies**: `gets` never advances, all streams' outstanding grows to K and Stage 0 blocks forever. Mitigation: existing request abort/timeout machinery in the orchestrator already kills the request, which calls `cleanup_sender` → `registry.drop`. No new logic needed.
- **Registry segment missing/corrupted**: treat as `outstanding=0` (degrade to legacy unlimited behavior), log warning. Never block the scheduler on a registry read failure.
- **Process crash leaves segment behind**: registry cleans up matching its key prefix on `OmniChunkTransferAdapter.__init__` startup (same pattern as `SharedMemoryConnector.close`).
- **`K=0` config**: registry calls become no-ops; behavior identical to today. This is the rollout default.

### Static-K vs adaptive K

Adaptive K (size K from measured Stage 1 service rate / realtime rate) is a follow-up. Static-K is correct, predictable, and easy to bisect. Adaptive moves to a separate PR after we have throughput data from WS-5 multi-replica.

### Metrics

- `stage0_credit_skipped_total{external_req_id}` — per-stream skip count
- `chunk_queue_depth{external_req_id}` — gauge sampled at `_send_single_request`
- both surfaced through existing stats path (`make_stats`)

## Scope & rollout

PR 1 (WS-4-a, scaffolding):
- `StreamCreditRegistry` class, mmap-backed
- Adapter wiring (no scheduler skip yet)
- Off by default
- Adds `chunk_queue_depth` metric only

PR 2 (WS-4-b, gate):
- Wire Stage 0 scheduler skip path
- Add `stage0_credit_skipped` metric
- Default K still 0 (legacy); CI runs a smoke test at K=4

PR 3 (WS-4-c, validation):
- Bench rerun (H20, c=64/128/256, 2-GPU) at K∈{off, 2, 4, 8}
- Update RFC #3535 acceptance with results
- Recommend default K once data is in

Adaptive K is explicit follow-up after WS-5 lands.

## What this does not do

- Does not improve underrun unless Stage 1 happens to choose stream order more fairly — see scheduler RFC for that.
- Does not help when Stage 0 is the bottleneck (e.g. voice_clone at c=4) — the gate never fires because outstanding stays low.
- Does not interact with PR #3322 (cross-request batched code2wav). They are orthogonal — credit cap bounds the queue, batched decode increases drain rate.

## Open questions

- Should `K` be config-time or per-request (`extra_body`)? Per-request lets users tune low-latency vs throughput inline. Defer to PR 2 if simple.
- Should we gate the cleanup of the registry segment on `_pending_keys` being empty, in case Stage 1 is still draining when Stage 0 has finished? Yes — drop only on `cleanup_sender` AND `outstanding == 0`, else leak-then-cleanup on next startup.
