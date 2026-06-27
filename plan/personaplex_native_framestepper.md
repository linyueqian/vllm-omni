# Native PersonaPlex FrameStepper + WS endpoint

## Goal
Serve PersonaPlex duplex S2S on the NATIVE components (not the moshi LMGen wrapper):
a `FrameStepper` (one 80ms user PCM frame -> one agent PCM frame + text) driving the
verified native pieces (PersonaPlexInputEmbeddings + Helium temporal + PersonaPlexDepformer
+ acoustic de-delay + Mimi), then a WebSocket endpoint on `personaplex/session.py`.
User chose this approach 2026-06-27.

## Seam (from engine.py)
FrameStepper Protocol: `sample_rate`, `frame_size`, `open_session(voice_prompt, persona)`,
`step(user_pcm)->FrameOutput(audio,text)`. Owns all conversation state. The moshi
`PersonaPlexEngine` is the reference impl (LMGen.step). The native impl must match this
interface so `session.py` + a WS server work unchanged.

## Verified building blocks (already done, on the omni engine)
- PersonaPlexInputEmbeddings (embed_codes) — bit-identical to moshi.
- PersonaPlexDepformer — parity-verified.
- Helium temporal — HeliumForCausalLM (parity_vllm, eager) matches moshi cos 0.999.
- Per-frame delay assembly (input): cb0=gen[t-1], cb1..7=gen[t-2], init=card; user cb0=enc[t-1],
  cb1..7=enc[t-2]. (talker preprocess.)
- Acoustic DE-DELAY (output->Mimi): tokens[t]=[gen[t][0], gen[t+1][1:8]] (derived agree=1.00).
- Mimi encode/decode (24kHz, 8 codebooks).

## Plan
- [ ] Phase 1: native streaming temporal. Decide KV strategy: HeliumForCausalLM needs a
      streaming KV cache (frame-by-frame). Option (i) build a manual KV cache on the eager
      sliding-window attention (parity_vllm style); option (ii) reuse moshi lm.forward_embeddings
      (streaming KV, verified) with native embeddings+depformer as the FIRST working version,
      then swap to (i). Start with whichever yields a WORKING coherent stepper fastest.
- [ ] Phase 2: NativePersonaPlexEngine implementing FrameStepper (open_session: persona prefill
      via step_system_prompts analog + voice prompt; step: encode user -> delayed input frame ->
      temporal -> sample text -> depformer -> de-delay -> Mimi decode -> FrameOutput).
- [ ] Phase 3: verify standalone — feed input_assistant.wav frame-by-frame, ASR the output,
      expect coherent (match the omni-engine greedy result "Thank you.").
- [ ] Phase 4: WS endpoint on session.py (PCM in/out frames), wire into api_server.
- [ ] Phase 5: e2e serve test (client sends user wav frames over WS, gets coherent agent audio).

## Status — DONE (e2e verified 2026-06-27)
Native-component duplex serve VERIFIED end-to-end over WebSocket:
- Phase 1-2: native embed_codes + depformer swapped into LMGen's per-frame seam
  (use_native_components). [commit 72d2eef0]
- Phase 3: standalone FrameStepper test -> coherent detailed reply ("...rinse the rice...").
- Phase 4-5: serving/server.py (FastAPI WS /v1/audio/duplex) + test_duplex_ws.py client;
  e2e WS: server READY 24s, client streams user wav, gets 16s coherent agent audio,
  ASR "You can rinse the rice a few times until the water runs clear. Then use a pot that
  has a tight fit. Bring it to a boil, then lower the heat and let it simmer without a lid."
  [commit 39663183]
REMAINING (refinement, not blocker): temporal still lm.forward_embeddings (moshi); swap to
HeliumForCausalLM streaming for full-native temporal. Multi-session (currently 1 conn at a
time). Sampling (greedy default; sampling needs bit-exact temporal).

## Notes / gotchas
- Remote eval: tool-held ssh (not nohup/setsid). pycache must be cleared per code change.
- Greedy is reliably coherent; sampling diverges (temporal cos 0.999 not bit-exact).
- The omni-engine offline path (talker+code2wav) stays as the verified batched path; this
  FrameStepper is the real-time duplex serving path.
