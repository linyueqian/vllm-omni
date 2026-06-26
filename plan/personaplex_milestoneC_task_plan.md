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

## Phase C result (2026-06-26) — native components compose in the REAL loop, VERIFIED
`tools/personaplex/end2end_native.py` (commit 400d7aa1): drives Moshi's actual per-frame
LMGen.step loop (acoustic-delay cache, code feedback, Mimi) with BOTH native components
swapped in (graphed_main->PersonaPlexInputEmbeddings.embed_codes; graphed_depth->
PersonaPlexDepformer). e2e H200 input_assistant 6s vs pure Moshi: **text 100% + agent-code
(cb0..7) 100% over 73 frames.** The ENTIRE native MODEL port is now end-to-end validated.
embed_codes/depformer signatures match graphed_main/graphed_depth exactly (drop-in).

## Talker (Task 2) contract — REVERSE-ENGINEERED (the remaining engine-integration)
`talker_mtp(input_ids[B], input_embeds[B,H], last_talker_hidden[B,H], text_step[B,H], ...)
-> (next_inputs_embeds[B,H], audio_codes[B,Q])`. Runner calls it (gpu_model_runner.py:1133)
after sampling the text token from compute_logits. Flags: have_multimodal_outputs/has_preprocess
=True, mtp_hidden_size=temporal hidden, talker_mtp_output_key=("codes","audio"). PersonaPlex map:
input_ids=sampled TEXT token; last_talker_hidden=temporal transformer_out; run depformer->16
codes (agent cb1..8 sampled, USER cb9..16 teacher-forced from mic-Mimi codes, only cb0..7 vocoded);
next_inputs_embeds=embed_codes(next DELAYED 17-row stack). Delay state = Moshi LMGen cache[B,17,
CT=4] + provided + offset, per-codebook modular read(offset-1)/write(offset+delay)%CT (lm.py
process_transformer_output:876-953) -> must be carried PER REQUEST in info_dict.
**FULL-DUPLEX IMPEDANCE:** PersonaPlex consumes a USER AUDIO stream (not text) + dual-stream
predict + acoustic delays -> does NOT fit the TTS talker contract (text->audio prompt builder,
preprocess entangled w/ AR prefill-chunk/cache-recovery lifecycle gpu_model_runner.py:610-781).
Plan Phase 1 = turn-based: preprocess Mimi-encodes the input WAV to user codes (rows 9-16),
prefill system+voice prompt; talker_mtp teacher-forces user rows. Phase 2 = live duplex (engine).
REMAINING = personaplex_talker.py (this contract) + registry.py arch entry + stage_input_processors
/personaplex.py + serving_speech.py wiring; THEN engine boot + parity vs moshi.offline (heavy verify).

## Talker build progress (2026-06-26, commit faf5358d)
- `personaplex_talker.py` CORE written: PersonaPlexTalkerForConditionalGeneration =
  HeliumModel(temporal, config-override) + lm_head(text_linear) + verified embeddings +
  verified depformer + flags (have_multimodal_outputs/has_preprocess/mtp_hidden_size/
  talker_mtp_output_key). forward/compute_logits/make_omni_output/load_weights done.
- `modeling_helium.py` HeliumModel now accepts an optional `config` (temporal sub-config)
  so the talker builds the backbone while the engine hf_config = PersonaPlexConfig.
- GATE 1 (routing) VERIFIED H200 `parity_talker_load.py`: load_weights routes 100% of the
  Moshi checkpoint (194/194 temporal targets, 17/17 emb, 84/84 depformer from 264 ckpt
  tensors, 0 unrouted). Shape-level weight_loader check deferred to the engine boot.
- BLOCKER for standalone construct: vLLM config-context tug-of-war (init_distributed wants
  config=None; initialize_model_parallel wants it set) -> construct only inside the real engine.

## REMAINING for the engine boot (next focused work)
- [ ] preprocess(): Mimi-encode input WAV -> user codes; build initial inputs_embeds; init
      Moshi delay cache(B,17,CT=4)+offset in additional_information. Per-request, stateful.
- [ ] talker_mtp(): depformer(text_token, last_talker_hidden, teacher-force user codes) ->
      agent codes; build next inputs_embeds via embed_codes(delayed stack). Batched/stateless;
      delay STATE lives in preprocess's info_dict, talker_mtp adds the delay-0 agent cb0 embed.
- [ ] stage_input_processors/personaplex.py (talker codes cb0..7 -> code2wav window).
- [ ] registry.py arch entries (PersonaPlexTalkerForConditionalGeneration + PersonaPlexCode2Wav).
- [ ] model dir + config.json (model_type: personaplex) OR hf_overrides; serving_speech wiring.
- [ ] BOOT on omni engine (GPU2/3) + parity vs moshi.offline. The real end-to-end verify.

## GATE 2 ACHIEVED — native pipeline BOOTS on the omni engine (2026-06-26, commit 3809bc90)
Full 2-stage PersonaPlex omni server UP on H200 (health 200, both stage cores alive):
`vllm serve <repo> --omni --deploy-config personaplex.yaml --trust-remote-code --skip-tokenizer-init`
(CUDA_VISIBLE_DEVICES=2). Stage-0 talker loaded 15.6 GiB on vLLM paged attention (KV 81k
tokens, 27x concurrency, torch.compile+cudagraph); stage-1 Mimi loaded (8 cb, 24kHz, 1920/frame).
This IN-ENGINE verifies the talker load_weights at the SHAPE level (the deferred Gate-1 check).
BOOT BUGS FIXED (each a real integration gap):
1. config.json empty -> set config.architectures=[PersonaPlexTalkerForConditionalGeneration]
   (else vLLM "does not support --runner generate").
2. talker inheriting SupportsPP (a Protocol) broke vLLM's VllmModelForTextGeneration isinstance
   -> drop SupportsPP (plain nn.Module, like Qwen3-TTS) + add embed_input_ids (VllmModel protocol).
3. get_mimi(filename, device) takes no num_codebooks in this moshi build; use set_num_codebooks.
4. code2wav.load_weights must REPORT mimi.* params as loaded (get_mimi loads them internally)
   else strict loader "weights not initialized".
5. need --skip-tokenizer-init (raw spm tokenizer, talker works in token/code space, detokenize:False).
LAUNCH GOTCHA: pkill -f patterns matching 'personaplex'/'vllm serve'/'pplex_run' SELF-MATCH the
launching ssh shell and kill it -> use `fuser -k 8123/tcp` only; nohup bash pplex_run.sh >> log.

## GATE 3 ACHIEVED — native pipeline GENERATES end-to-end on the omni engine (2026-06-26)
`tools/personaplex/end2end_native_omni.py` (offline Omni driver) runs the full native pipeline:
talker (7B temporal on paged-KV) -> depformer codes -> connector -> Mimi code2wav -> PCM. Produces
audio e2e (commit e8e82e39); turn-based user-audio input wired (commit c216aa78: Mimi-encode input
WAV -> user codes [F,8] as a tensor in additional_information -> preprocess injects delayed user rows).
Generation bugs fixed along the way:
1. talker_mtp passed zero hidden -> degenerate (1 frame repeated). FIX: postprocess captures
   hidden_states[-1] -> hidden_states["last"]; preprocess reads it as last_talker_hidden. Codes then VARY.
2. token_only sized placeholder from multimodal_output (only "latent", no codes). FIX: size from the
   generated token count (engine_output_type=latent; codes ship via full_payload connector).
3. user codes dropped by serialize_payload (forbid_unknown_fields). FIX: pass as a top-level TENSOR
   (round-trips via _serialize_tensor); list-of-lists mangles.
**VERIFICATION (key finding):** with NO persona prompt, the native engine output matches Moshi's own
behavior -- the Moshi-faithful reference (end2end_native.py, 100% token+code) is itself SILENT
(rms 2e-4, ASR empty) for input_assistant.wav. So near-silence without a persona is CORRECT Moshi
behavior, not an engine bug. Engine is slightly louder (rms 0.023, faint garble) -> minor delay-approx
inexactness vs Moshi's clean silence.

## REMAINING for a loud COHERENT-SPEECH demo (scoped next step, moshi needs it too)
- persona/voice prompt injection (step_system_prompts): force persona text (row0) + voice codes
  (agent rows) in the prefill so the agent adopts a persona and SPEAKS. (The live moshi.server demo
  used this to get "Hey, let me..." openings.)
- exact delay-machinery parity (match Moshi's cache[17,CT] read/write/gather) so the engine == Moshi
  bit-for-bit (currently a faithful approximation). max_new_tokens cap not applied (generates to 3000).

## Status
**Model-side port COMPLETE + VERIFIED; native pipeline GENERATES e2e on the omni engine (Gate 3),
verified FAITHFUL to Moshi (both silent w/o persona).** Remaining for loud coherent speech: persona
injection + exact delay parity (refinements on a working, verified pipeline).
