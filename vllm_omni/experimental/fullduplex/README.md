# Full-duplex interaction framework

A model-agnostic framework for real-time, full-duplex (streaming-in / streaming-out)
model interaction, plus the JoyVL implementation built on it.

To **run the JoyVL model**, see the recipe:
[`recipes/JD/JoyAI-VL-Interaction.md`](../../../recipes/JD/JoyAI-VL-Interaction.md).
This README covers the framework itself and how to add a new model.

## Layout

```
vllm_omni/experimental/fullduplex/
  core/      generic full-duplex framework (model-agnostic): DuplexRuntime (event
             loop + epoch barge-in), DuplexSession, DuplexAdapter (ABC), protocol
  joyvl/     JoyVL implementation (model-specific):
             adapter.py            JoyVLDuplexAdapter (implements core.DuplexAdapter)
             decision/             policy + output_parser + prompts (speak/silence/delegate)
             memory/               InteractionBrain — 3-tier summary memory (async)
             serving/              OpenAI-compatible HTTP orchestrator
             bridges/              model backend + delegation
```

`core/` is the only part shared across models; data planes differ by model and are
intentionally not shared.

## Scope

The runnable serving path is `joyvl/serving/` driving `decision/` + `memory/` directly.
`joyvl/adapter.py` + `core/` are a **demonstration** of how a model plugs into the
generic full-duplex framework (exercised by `tests/fullduplex/`), not the serving path
the HTTP orchestrator currently uses — a fused-audio model (e.g. MiniCPM-o) is the case
`core/` is built for.

`personaplex/` is that fused-audio case made concrete: a Moshi-class, pure-lockstep
speech-to-speech model (see `recipes/NVIDIA/PersonaPlex.md`). It is the **second** model
on `core/`, and the one that justified promoting a small lifecycle mode into it: the
default-off `continuous` flag (`DuplexCapability` / `DuplexSessionConfig`) makes
`DuplexRuntime` run ONE eternal, frame-clocked response that drains on close, instead of
the turn-style start/cancel-per-trigger lifecycle. Turn-based adapters (JoyVL,
MiniCPM-o) are unaffected. Its runnable path is `personaplex/session.py` (lockstep
driver); `personaplex/adapter.py` is the `core.DuplexAdapter` demonstration.

## Adding a full-duplex model

The seam is `core.DuplexAdapter`. `core/` owns the session lifecycle, epoch-based
barge-in, playback cursor, and the event protocol — you implement only model policy.

1. Create a sibling package `vllm_omni/experimental/fullduplex/<model>/` next to `joyvl/`; keep
   model-specific code there and do not touch `core/`.
2. Implement one `DuplexAdapter` (three required methods; the rest have defaults):

   ```python
   from collections.abc import AsyncIterator
   from vllm_omni.experimental.fullduplex.core.adapter import DuplexAdapter, DuplexCapability, OutputChunk
   from vllm_omni.experimental.fullduplex.core.session import DuplexSession

   class MyModelAdapter(DuplexAdapter):
       def capabilities(self) -> DuplexCapability:
           return DuplexCapability(
               input_modalities=frozenset({"audio", "text"}),
               output_modalities=frozenset({"audio", "text"}),
               proactive=True,            # speak without being asked?
           )

       async def on_input(self, session: DuplexSession, modality: str, data) -> None:
           ...                            # buffer/route an incoming chunk

       async def respond(self, session: DuplexSession) -> AsyncIterator[OutputChunk]:
           async for piece in self._model_stream(session):
               yield OutputChunk(modality="audio", data=piece)   # runtime drops stale
               #                                                   chunks after a barge-in

       # optional: should_respond / on_barge_in / on_playback_ack
   ```

3. Run it through the shared runtime — no new control-plane code:

   ```python
   from vllm_omni.experimental.fullduplex.core.runtime import DuplexRuntime
   from vllm_omni.experimental.fullduplex.core.session import DuplexSession, DuplexSessionConfig

   rt = DuplexRuntime(DuplexSession("sid", DuplexSessionConfig()), MyModelAdapter())
   await rt.run(input_events, emit)
   ```

`joyvl/adapter.py` is the worked demonstration (currently exercised by tests, not the
HTTP serving path). Promote a helper from a model package up into `core/` only once a
second model actually needs it.
