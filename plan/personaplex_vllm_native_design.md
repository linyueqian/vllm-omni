# Design: PersonaPlex / Moshi as a vLLM-NATIVE model in vLLM-Omni

## Goal
Run PersonaPlex (Moshi finetune) on vLLM-Omni's actual inference engine — paged
attention + continuous batching — so we get the real serving win (many concurrent
sessions batched on one GPU), eventually with frame-clocked full-duplex sessions.
NOT the orchestration-layer wrapper (that calls raw moshi LMGen.step); this is the
deep integration.

## Phasing (the honest multi-phase reality)
- **Phase 0 (de-risk spike, TONIGHT):** prove the 7B Helium temporal transformer runs
  on vLLM paged attention with custom `inputs_embeds` + sliding window, logits matching
  moshi's reference `forward_codes`. If this matches, the whole thing is real.
- **Phase 1:** full offline/turn-based generation as a vllm-omni staged pipeline
  (temporal transformer LLM_AR + depformer code-predictor + Mimi codec), parity vs
  `moshi.offline`. NO duplex yet. This is a normal "add a staged AR-TTS model".
- **Phase 2:** frame-clocked duplex session (conversation-lifetime KV + frame-budgeted
  park/wake + streaming Mimi encode). Gated on the duplex session primitive (RFC #3745).

## Model-side mapping (GROUNDED in moshi source, /tmp/personaplex)
Moshi RQ-Transformer decomposes cleanly into vllm-omni's existing staged-TTS shape
(thinker AR LM + code predictor + codec) — this is the key feasibility insight.

| Moshi piece | What it is (file:line) | vLLM-Omni home |
|---|---|---|
| Temporal transformer | StreamingTransformer 4096d/32L, causal **RoPE + sliding window context=3000**, RMSNorm, combined QKV no-bias (`transformer.py:317` MHA, `:232` RingKVCache) | **vLLM-native AR decoder** (LLM_AR stage) fed `inputs_embeds`; RingKVCache(capacity=context) = vLLM **sliding-window paged KV** |
| `embed_codes` | sum of 8 agent + 8 user audio cb embeddings + text emb -> [B,S,4096] (`lm.py:425-439`) | model-specific **input-embedding builder** -> `inputs_embeds` (NOT token ids) |
| `forward_embeddings` | transformer -> out_norm -> `text_linear` (32001) (`lm.py:447-455`) | main forward + text head |
| Depformer | per-cb AR head, per-codebook weights (`weights_per_step`), ctx=8, KV reset each frame; `forward_depformer` (`lm.py:457-493`), `depformer_step` (`:1129`) | **code predictor** custom op (per-step, after main step); NOT paged |
| Mimi codec | 24kHz/12.5Hz, 8 cb, streaming conv (`compression.py`) | **codec stage(s)**: decode (codes->PCM) + duplex: streaming **encode** (PCM->codes) |

### Why this is tractable
The temporal transformer is a vanilla sliding-window RoPE decoder — vLLM serves these
(Mistral-class). The ONLY non-standard input is `inputs_embeds` (vLLM supports this via
the multimodal embedding-merge path). The depformer is a small custom op (like existing
TTS code predictors). Mimi is a codec stage (like code2wav). So PersonaPlex ≈ "a staged
AR-TTS model with a custom per-step input embedding + an extra audio-ENCODE codec stage".

### Hardest mismatches (to design around)
1. **Custom summed multi-stream `inputs_embeds` per step** — must feed the vLLM model a
   precomputed embedding, not input_ids. (Check: vLLM inputs_embeds / mm-merge path.)
2. **Streaming Mimi ENCODE** (user mic -> codes) — existing models only DECODE; duplex
   needs a per-frame encode stage with persistent conv state.
3. **Frame-clocked lockstep + conversation-lifetime KV** — Phase 2; gated on RFC #3745.
4. **vLLM-native Moshi may already exist upstream** — CHECK vllm core for a Moshi/Kyutai
   model class (huge shortcut if present).

## Open (filled by codebase research, /tmp/pplex/vllm_tts_template.md + vllm_native_duplex_state.md)
- Exact vllm-omni model-registration seam + a worked staged-AR-TTS template (Qwen3-TTS).
- Does the AR engine accept inputs_embeds? sliding-window support in the AR scheduler?
- Current duplex session machinery: landed vs stub (post-#3907).

## Checkpoint weight structure (model.safetensors, 475 tensors) — the port surface
Temporal (32L) -> map to a vLLM Llama-variant:
- `transformer.layers.N.self_attn.in_proj_weight` (combined QKV 3*4096x4096) -> vLLM `qkv_proj` (split)
- `...self_attn.out_proj.weight` -> `o_proj`
- `...gating.linear_in.weight` / `gating.linear_out.weight` -> gated SiLU MLP (gate_up / down)
- `...norm1.alpha` / `norm2.alpha` -> input/post RMSNorm (weight is "alpha")
- `out_norm.alpha` -> final norm; `text_emb.weight` -> embed; `text_linear.weight` -> text head
- `emb.N.weight` (16) -> audio codebook embeddings (used in embed_codes, model-specific)
Depformer (6L, per-codebook): `depformer.layers.N.gating.N.*` (96=6x16 weights_per_step),
`depformer_in.N` (16), `depformer_emb.N` (15) + `depformer_text_emb`, `linears.N` (16) -> code predictor.
NOTE: config.json is EMPTY (`{model_type: personaplex}`) — all hyperparams hardcoded in
`loaders.py::_lm_kwargs` (dim4096/32L/32H/head128/context3000/RoPE/RMSNorm/silu/text_card32000).
Port differences from vanilla Llama: combined in_proj (split QKV), "alpha" RMSNorm naming,
gating split order, inputs_embeds (summed, not embed_tokens), sliding_window=3000, no biases.

vLLM core has NO moshi/helium model (checked, v0.23) BUT vLLM `llama.py` supports `inputs_embeds`
(the critical capability). So: register a Helium/Llama-variant + feed precomputed inputs_embeds.

## Phase 0 spike (decided): NUMERICAL PARITY harness
Build a vLLM-served Helium temporal transformer from the mapped weights, run a forward with
`inputs_embeds` built by `embed_codes` on a real frame sequence, compare hidden states +
text_logits vs moshi `lm.forward_codes`. Match (fp tol) => the 7B backbone runs on vLLM =>
Phase 1 is real. This isolates the riskiest unknown without the full omni pipeline.

## Status
**Phase 0 design** — model mapping + weight surface grounded; awaiting vllm-omni-side seam
(registration, inputs_embeds-through-omni-runner, sliding window, duplex state) from 2 agents.
