# Task Plan: PersonaPlex Milestone C — native omni serve pipeline

## Goal
Make `vllm serve` run PersonaPlex (Moshi finetune) natively on vLLM's omni staged
engine (paged-KV incremental decode), replacing the embed/pooling re-prefill harness.
Tasks 2-6 of `plan/personaplex_vllm_native_plan.md`.

## State at start (2026-06-26)
PRESENT (verified earlier): configs (Task 1), `personaplex_code2wav.py` (Task 4),
`pipeline.py`, `deploy/personaplex.yaml`, `tts_adapters/personaplex.py`,
`pipeline_registry.py`+`arg_utils.py` edits, `modeling_helium.py` (HeliumForCausalLM,
moshi-parity verified via `tools/personaplex/parity_vllm.py`).

MISSING (the remaining work):
- [ ] `personaplex_depformer.py` (Task 3) — custom per-codebook code predictor [#1 RISK]
- [ ] `personaplex_talker.py` (Task 2) — PersonaPlexTalkerForConditionalGeneration
- [ ] `stage_input_processors/personaplex.py` (Task 5)
- [ ] model-arch entries in `vllm_omni/model_executor/models/registry.py` (Task 5)
- [ ] serving_speech.py wiring (`_detect_tts_model_type`, `_build_personaplex_request`)

## Phases (this session)
- [x] Phase A: pulled moshi depformer reference from H200; ported depformer faithfully
- [x] Phase B: `parity_depformer.py` — VERIFIED on H200 GPU2 (commit 5ac58f7c).
      Teacher-forced logit max-abs-diff <=0.47 across seeds 0/1/7/13 x B=1/4;
      every argmax flip a proven bf16 tie (ref top-2 gap 0.0625). Bug found+fixed:
      moshi forces depformer context=None (full inner attention), not window 8.
- [ ] Phase C: native input embeddings (embed_codes: emb.0..15 + text_emb) + parity
- [ ] Phase D: wire the talker (Task 2) around HeliumForCausalLM + embed_codes + depformer
      + talker_mtp/preprocess; registry + stage input processor
- [ ] Phase E: boot on omni engine + offline parity vs moshi.offline (heavier; may extend)

## Decisions Made
- Depformer first: it is the #1 ranked risk and is independently verifiable against
  moshi WITHOUT the full engine (pure torch + checkpoint). De-risks the whole port.
- Do this directly (not subagent fan-out): focused deep-port work, single context.
- Verify on H200 GPU 2/3 (free); moshi + helium_hf already present.

## Key Constraints (from plan Global Constraints)
- Depformer: dim 1024, 6 layers, 16 heads, ctx 8, depformer_pos_emb="none" (NO RoPE),
  weights_per_step=True (per-codebook attn+MLP weights), multi_linear=True (per-cb in-proj),
  SiLU gating, dep_q=16 (only cb 0..7 decoded to PCM), KV reset each frame.
- Weights: depformer.layers.N.{self_attn,gating}.STEP.*, depformer_in.N (16),
  depformer_emb.N (15) + depformer_text_emb, linears.N (16).

## Grounded depformer facts (from moshi loaders.py/lm.py/transformer.py, 2026-06-26)
- dep_q=16 (loader override of the 8 in _lm_kwargs), depformer_dim=1024, 6 layers,
  16 heads (head_dim 64), depformer_context=8 (causal sliding KV over the inner steps),
  depformer_gating="silu", depformer_pos_emb="none" (NO rope/sin), weights_per_step=True
  (16 weight sets), multi_linear=True (16 depformer_in), norm="rms_norm_f32" (eps 1e-8, alpha).
- gating hidden = (2*int(4.125*1024))//3 = 2816 (ActivationGating: linear_in[2*2816,1024],
  silu(x0)*x1, linear_out[1024,2816]).
- Per frame: KV reset; for cb in 0..15: x=depformer_in[cb](transformer_out)
  + (cb==0 ? text_emb(prev) : audio_emb[cb-1](prev)); 6 layers per-step weights; logits=linears[cb];
  greedy argmax; prev=teacher-forced-or-sampled. (lm.py depformer_step + forward_depformer.)
- ScaledEmbedding zero_idx=-1 (->0 vector); card 2048 (table 2049); text table 32001.
- Checkpoint stores 8 steps; loader expands self_attn 8->16 (concat copy) and copies
  cb0..7 -> cb8..15 for gating/linears/depformer_in/depformer_emb. Port load_weights mirrors this.
- moshi ref pulled to ~/Downloads/pplex_moshi_ref/{lm,loaders,transformer}.py

## Status
**Phase A done; Phase B** — writing personaplex_depformer.py + parity_depformer.py (keystone gate).
