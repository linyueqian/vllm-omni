# PersonaPlex vLLM-Native Integration — Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or executing-plans to implement task-by-task. Steps use `- [ ]` checkboxes.

**Goal:** Run PersonaPlex (Moshi finetune) on vLLM-Omni's real inference engine — paged attention + continuous batching — first as turn-based offline generation (Phase 1, a pure model port), then as frame-clocked full-duplex sessions (Phase 2, engine work gated on RFC #3745).

**Architecture:** A Qwen3-TTS-shaped staged pipeline. Stage 0 (`LLM_AR`) = a "talker" that wraps a stock vLLM sliding-window decoder (the Helium temporal transformer) fed custom `inputs_embeds` built by `embed_codes`, with the Moshi depformer as an in-talker per-step code predictor. Stage 1 (`LLM_GENERATION`) = a Mimi code2wav decode stage. Registration is dict edits; the per-step depformer runner glue is free.

**Tech Stack:** vLLM 0.23 paged attention; vllm-omni stage/pipeline/registry; `CodePredictorWrapper` (RFC #2967) outer loop; moshi weights (`nvidia/personaplex-7b-v1`), Mimi codec.

## Global Constraints
- Temporal transformer (MEASURED from the checkpoint via the Phase-0 harness, supersedes
  the `_lm_kwargs` guesses): dim 4096, 32 layers, 32 heads, head_dim 128, **intermediate
  11264** (SwiGLU: `gating.linear_in=[2*11264,4096]` gate+up, `linear_out=[4096,11264]`),
  RMSNorm in **float32** eps 1e-8 (`norm="rms_norm_f32"`, weight named `alpha`), RoPE θ=10000
  **`is_neox_style=False`** (interleaved), **causal sliding-window context=3000**, no biases,
  fused QKV `in_proj_weight=[12288,4096]` (MHA, 32 kv heads), **lm_head vocab 32000**,
  text embedding 32001 rows (32000 + 1 special), audio card 2048, n_q=16.
- Depformer: dim 1024, 6 layers, 16 heads, ctx 8, **`depformer_pos_emb="none"` (NO RoPE)**, **`weights_per_step=True` (per-codebook attention+MLP weights)**, `multi_linear=True` (per-codebook in-proj), SiLU gating, dep_q=16 (loader override; only cb 0..7 decoded to PCM).
- Mimi: 24 kHz, 12.5 Hz, 1920 samples/frame, 8 active codebooks, bf16. Per-frame column `[B,17,1]`: row0 text, 1-8 agent audio, 9-16 user audio. Acoustic delay `[0, 0,1×7, 0,1×7]`.
- `config.json` is EMPTY (`model_type: personaplex`) — all hyperparams are hardcoded; the port must supply a real HF config class.
- No engine/scheduler edits in Phase 1. Phase 2 is the only place engine changes are allowed.
- Parity bar: greedy logits/codes vs `python -m moshi.offline` within fp tolerance for the shared deterministic prefix (cross-process bit-parity is impossible — moshi greedy is nondeterministic, cudnn.deterministic=False).

---

## Phase 0 — De-risk: temporal-transformer numerical parity (FOUNDATION)

Proves the 7B Helium backbone is a standard decoder whose weights map onto vLLM's
Llama/Mistral math. If this matches, the whole port is real.

### Task 0.1: Weight-mapping coverage + one-step parity harness
**Files:**
- Create: `tools/personaplex/parity_temporal.py` (standalone, not shipped)
**Interfaces:**
- Produces: a script that (a) maps every `transformer.*`/`text_emb`/`text_linear`/`out_norm`/`emb.*` checkpoint tensor to a vLLM-Llama param layout with shape assertions (100% coverage, no orphans), (b) runs moshi `lm.forward_codes(seq)` for a real frame column to get reference `(transformer_out, text_logits)`, (c) runs an HF-standard-ops reimplementation (SDPA causal+sliding-window, HF RoPE θ=10000, fp32 RMSNorm eps 1e-8, SwiGLU) with the mapped weights, (d) asserts max-abs-diff < 1e-2 on hidden + text_logits.
- [ ] Step 1: enumerate checkpoint tensors; build the name map (`in_proj_weight`→split q/k/v; `gating.linear_in`→gate_up, `linear_out`→down; `norm{1,2}.alpha`→layernorm weight; `out_norm.alpha`→final norm). Assert every temporal tensor is consumed.
- [ ] Step 2: run on H200, capture max-abs-diff. Expected: < 1e-2 (bf16). If a layer diverges, bisect RoPE convention (interleaved vs half) and gating split order.
- [ ] Step 3: write the resolved mapping + tolerances into this plan's Phase 1 Task 1 notes; commit the harness.

> Decision gate: parity PASS → proceed to Phase 1. FAIL → the port needs a custom attention/RoPE; re-scope Task 1.

**RESOLVED conventions (de-risk, 2026-06-25):**
- RoPE = **interleaved / GPT-J** (`moshi rope.py:67-89` views `[D//2,2]`, rotates adjacent pairs) ⇒ build vLLM rope with **`is_neox_style=False`**; NO weight permutation. (Default Llama uses neox=True — must override.)
- Norm = fp32 RMSNorm eps 1e-8 (`transformer.py:143`); weight param is named `alpha`.
- MLP = SwiGLU via `gating.linear_in`(→gate_up) / `gating.linear_out`(→down); verify gate/up half-order in the parity run.
- Attention = combined `in_proj_weight` `[3*4096,4096]` = vLLM fused `qkv_proj` directly (MHA, 32 kv heads); verify per-head pairing under neox=False.
- Net: temporal transformer = a vLLM **Mistral-class sliding-window decoder + is_neox_style=False**, weights drop in (no surgery). Keystone risk reduced to gate/up + per-head-pairing checks in Task 0.1.

---

## Phase 1 — Turn-based native pipeline (a MODEL PORT, no engine changes)

Validation target: `examples/.../personaplex/end2end_native.py` produces audio whose
greedy text matches `moshi.offline` on the shared prefix, driven through vLLM paged attention.

### Task 1: HF config classes
**Files:** Create `vllm_omni/model_executor/models/personaplex/configuration_personaplex.py`; Edit `vllm_omni/engine/arg_utils.py:63-78` (add `AutoConfig.register("personaplex", PersonaPlexConfig)`).
**Interfaces:** Produces `PersonaPlexConfig(model_type="personaplex")` with `sub_configs={"temporal_config","depformer_config","mimi_config"}`, each carrying the Global-Constraints hyperparams (the empty checkpoint config means these are defaults baked into the class).
- [ ] Map `moshi loaders.py::_lm_kwargs` → config fields; unit-test that the config round-trips and exposes sliding_window=3000, rope_theta=10000, rms_norm_eps=1e-8.

### Task 2: Temporal transformer as a vLLM-native talker (stage 0, LLM_AR)
**Files:** Create `.../personaplex/personaplex_talker.py`. Analog: `qwen3_tts_talker.py:265,322,507-515,610-781,1041-1107`.
**Interfaces:**
- Consumes: `PersonaPlexConfig`; vLLM `LlamaModel`/`MistralModel` (sliding-window) as `self.model`.
- Produces: `PersonaPlexTalkerForConditionalGeneration(nn.Module)` with `forward(inputs_embeds passthrough)`, `compute_logits` (text_linear → 32000, mask non-text), `preprocess()` (build `inputs_embeds` via `embed_codes`), `talker_mtp()` (drive depformer + build next summed embedding), `load_weights` + `hf_to_vllm_mapper` (Phase-0 mapping), and the flags `mtp_hidden_size`, `talker_mtp_output_key=("codes","audio")`, `have_multimodal_outputs=True`, `has_preprocess=True`.
- [ ] Step 1: wrap the stock decoder; load weights via the Phase-0 mapper; verify a forward runs on the engine (1 request) without error.
- [ ] Step 2: implement `embed_codes` summed-embedding in `preprocess` (prefill) — sum 16 audio cb embeds (rows 1-16) + text emb (row 0). For Phase 1, the **user-audio rows (9-16) come from a precomputed Mimi-encode of the input WAV** (offline; not live).
- [ ] Step 3: copy Qwen3-TTS's per-step write-back (`gpu_model_runner.py:1875`) — `talker_mtp` returns `(summed_inputs_embeds, agent_codes)`; runner glue auto-activates.

### Task 3: Moshi depformer as a CUSTOM code predictor
**Files:** Create `.../personaplex/personaplex_depformer.py`. Reuse the OUTER loop of `common/qwen3_code_predictor.py:404 CodePredictorWrapper` (per-step AR over codebooks, no paged KV, CUDA-graph per bucket) but a CUSTOM inner model (the shared `CodePredictorBaseModel` uses RoPE+shared weights; Moshi needs **no pos-emb + per-codebook weights**).
**Interfaces:** Produces `PersonaPlexDepformer` exposing the same call signature as `CodePredictorWrapper.forward(layer0_code, layer0_embed, last_talker_hidden, ...) -> [B, dep_q]`.
- [ ] Step 1: implement the per-codebook-weight attention/MLP (`weights_per_step`): index layer weights by codebook step; drop RoPE. Load `depformer.*`, `depformer_in.N`, `depformer_emb.N`+`depformer_text_emb`, `linears.N`.
- [ ] Step 2: unit-test one frame: feed a known `transformer_out`, assert the 16 sampled codes match moshi `depformer_step` (greedy) within tolerance.

### Task 4: Mimi codec decode stage (stage 1, LLM_GENERATION)
**Files:** Create `.../personaplex/personaplex_code2wav.py`. Analog: `qwen3_tts_code2wav.py:52,280`.
**Interfaces:** Produces `PersonaPlexCode2Wav(nn.Module, execution_type=LLM_GENERATION)`: build Mimi decoder from config, `forward(codes[:,1:9] → PCM)` returning `OmniOutput({"model_outputs": audio, "sr": 24000})`; streaming left-context cache + inner CUDA-graph.
- [ ] Step 1: load Mimi weights (the `tokenizer-*.safetensors`); decode a known code sequence; compare PCM to `mimi.decode` reference.

### Task 5: Pipeline + stage processor + registration + deploy
**Files:** Create `.../personaplex/pipeline.py` (`PERSONAPLEX_PIPELINE`, two stages); Create `vllm_omni/model_executor/stage_input_processors/personaplex.py` (talker→code2wav code-window producer, analog `stage_input_processors/qwen3_tts.py:152`); Edit `registry.py:117` (+2 arch entries), `pipeline_registry.py:89` (+`"personaplex"`); Create `vllm_omni/deploy/personaplex.yaml` (analog `deploy/qwen3_tts.yaml`: `codec_chunk_frames`, left-context, `subtalker_sampling_params`); Create `vllm_omni/entrypoints/openai/tts_adapters/personaplex.py` (`@register_tts_adapter`).
- [ ] Step 1: registration resolves (pipeline → 2 stages, DAG ok); engine boots the model on H200.
- [ ] Step 2: e2e offline `end2end_native.py`: feed input WAV (Mimi-encoded to user codes), run the pipeline, write agent WAV; assert greedy text matches `moshi.offline` shared prefix.

### Task 6: Persona + voice prompt injection
**Files:** Edit `personaplex_talker.py preprocess`. Replicate `step_system_prompts` as a prefill: force voice-clone agent codes (rows 1-8) + persona text (row 0) into the initial `inputs_embeds` prefix before the user frames (moshi `lm.py:1010-1104`).
- [ ] Step 1: persona/voice in the offline example match `moshi.offline --voice-prompt --text-prompt` behavior.

---

## Phase 2 — Frame-clocked full-duplex (ENGINE WORK, gated on RFC #3745)

Not a model port. Requires new engine surface. Outline (each is its own sub-plan):

- **2a. Frame-budgeted park/wake scheduler mode.** Extend the AR append/preserve path (`omni_ar_scheduler._update_request_as_session:668-706` — KV-preserving, already on main) with a "decode exactly k columns for k delivered frames, then park" budget (the missing piece per Agent B; today it's stop/append-driven, not frame-budgeted). Ties to #4385 (per-segment max_tokens bug) and #4383 (per-frame preprocess cost).
- **2b. Streaming Mimi ENCODE stage.** New stage type holding Mimi causal-conv ring state per session (mic PCM → codes each frame). No in-tree template (all codec stages decode-only). Feeds the user-audio rows (9-16) of `embed_codes` live.
- **2c. Conversation-lifetime session.** The RFC #3745 `DuplexSession` primitive (KV lease, epoch, ring buffer). `experimental/fullduplex/` has zero commits on main — land the session primitive first.
- **2d. WS protocol.** `serving_duplex.py` OpenAI-Realtime WS ↔ the session (the existing PersonaPlex orchestration adapter on this branch can wrap the native engine once 2a-2c land).

Barge-in is native (model always consumes the user stream); no flush needed — the engine work is the clock + encode + session, not turn logic.

## Self-review notes
- Spec coverage: temporal→Task 2, embed_codes→Task 2, depformer→Task 3 (custom, not 4-line — corrected), Mimi→Task 4, persona/voice→Task 6, registration→Task 5, duplex→Phase 2. Covered.
- Biggest risks, ranked: (1) depformer per-codebook-weights custom predictor (Task 3); (2) RoPE/gating convention in the weight map (Phase 0); (3) frame-budget scheduler (2a) — the only true engine change.

## Phase-0 RESULT (ran on H200 2026-06-25 via tools/personaplex/parity_temporal.py)
- **195/195 temporal tensors map 1:1 onto a vLLM Llama layout; 0 unmapped, all shapes
  consistent** (only the expected text_emb 32001 vs lm_head 32000 asymmetry). Mechanically
  complete port surface — no weight surgery.
- Exact vLLM config PINNED (see Global Constraints). Depformer = 248 tensors + 16 audio
  codebook embeddings = the talker-wrapper/code-predictor parts (not vLLM Llama), confirming
  Task 3 (custom depformer) is the real custom work.
- Conventions confirmed: fused QKV (MHA), SwiGLU gate+up, fp32 RMSNorm, interleaved RoPE.
- Reference `forward_codes` captured as the future numerical-parity oracle.
- VERDICT: the keystone (does the 7B Helium backbone fit vLLM's engine?) is GREEN at the
  weight/structure/convention level. Remaining Phase-0 work = one-step numerical parity
  (gate/up half-order + per-head pairing under neox=False) during Task 2.

## EXECUTION RESULTS (2026-06-25, verified on H200)
- **Milestone A — temporal transformer on vLLM: DONE + VERIFIED.** Approach: instead of a
  custom vLLM model class, export the Moshi temporal weights to a STOCK HF Llama
  (`tools/personaplex/export_helium_hf.py`) with the interleaved->rotate-half q/k permute,
  SwiGLU gate/up, fp32 RMSNorm. HF-Llama parity vs moshi = **100% argmax**; loaded into
  `vllm.LLM`, greedy token matches moshi (1929==1929). The 7B runs on vLLM paged attention.
- **Milestone B — full generation on vLLM: DONE + VERIFIED.** `tools/personaplex/generate_vllm.py`
  drives Moshi's depformer/delay/Mimi loop with the temporal forward swapped to vLLM (raw
  hidden via embed pooler `use_activation=False`, cosine 0.9998). vLLM-driven generation
  matches pure-Moshi token-for-token over 73 frames: **100% inner-monologue text, 98.8%
  agent-code agreement**. PersonaPlex generates correctly on vLLM.
- **Milestone C — production omni serving pipeline: NOT built (the remaining work).** This is
  the mechanical Qwen3-TTS-template integration (Tasks 1-6 above): a talker wrapping the
  Helium model + the depformer as a custom code-predictor + Mimi code2wav stage + registration
  + deploy yaml + adapter, so `vllm serve` exposes PersonaPlex. The core (does the 7B run +
  generate correctly on vLLM) is now PROVEN; C is productionization on top of it.
- KEY engineering finding for C: the per-step hidden state the depformer needs comes out of
  the omni AR engine's `talker_mtp` runner hook (gpu_model_runner.py:1860/1875) — the
  production analog of the embed-pooler trick used in the Milestone-B harness.

## Status
Phase-0 de-risk GREEN; **Milestones A+B (PersonaPlex runs + generates on vLLM) VERIFIED**;
Milestone C (serve pipeline) = the remaining bounded model-port integration. Phase 2 (duplex)
= engine work (frame-budget scheduler + streaming Mimi encode), gated on RFC #3745.
Inputs: `/tmp/pplex/{vllm_tts_template,vllm_native_duplex_state,moshi_architecture_spec}.md`.
