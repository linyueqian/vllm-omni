# PersonaPlex (full-duplex speech-to-speech)

[`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1) is a
**Moshi finetune**: a pure-lockstep, full-duplex speech-to-speech model with persona
(role text) and voice (zero-shot clone) control. It is the first Moshi-class model
wired into vllm-omni's experimental full-duplex framework
(`vllm_omni/experimental/fullduplex/personaplex/`).

## Why it is different from MiniCPM-o / JoyVL duplex

| | MiniCPM-o 4.5 (#3907) | JoyVL | **PersonaPlex (this)** |
|---|---|---|---|
| Cadence | 1 s chunk groups | ~1 fps frames | **80 ms lockstep (12.5 Hz)** |
| Turn control | learned `⟨listen⟩`/`⟨speak⟩` | `</silence>`/`</response>` | **none — pure lockstep** |
| Per step | variable-length token group | text decision | **1 user frame in → 1 agent frame + 1 text token** |
| Barge-in | at chunk boundary | n/a | **native (model always hears the user)** |
| Session state | chunk-group KV | per-tick HTTP | **persistent Mimi + LMGen streaming KV** |

This is the "parallel-frame joint" adapter pattern from [RFC #3745](https://github.com/vllm-project/vllm-omni/issues/3745)
and the [#1335](https://github.com/vllm-project/vllm-omni/issues/1335) full-duplex target.
It is also the framework's lockstep "acid test": the model declares
`DuplexCapability.continuous = True`, and `core.DuplexRuntime` runs ONE eternal
response that consumes input frames as they arrive and drains on close — added as a
small, default-off lifecycle mode so turn-based adapters (JoyVL, MiniCPM-o) are
unaffected.

## Architecture (Moshi RQ-Transformer)

- **Mimi codec** 24 kHz, 12.5 Hz, 1920 samples/frame, 8 active codebooks (card 2048), bf16.
- **Temporal transformer** (Helium backbone) 4096-d / 32 layers / 32 heads / context 3000 frames.
- **Depformer** 1024-d / 6 layers / 16 heads, autoregressive over codebooks (reset each frame).
- Per-frame token column `[B,17,1]`: row 0 = inner-monologue text, rows 1–8 = agent audio,
  rows 9–16 = user audio. The cleanest single-frame seam is `LMGen.step(user_codes)`.
- Persona + voice are injected once at session open by *forcing* tokens through the same
  `step()`: voice clone forces the agent stream from reference-audio Mimi codes; persona
  forces the text stream from `<system> … <system>` tokens (0.5 s silence spacers).

## Install

```bash
# the PersonaPlex vendored moshi fork
git clone https://github.com/NVIDIA/personaplex && pip install ./personaplex/moshi/.
sudo apt install libopus-dev        # opus for the live server (offline needs only sphn)
# accept the (auto) gate at https://huggingface.co/nvidia/personaplex-7b-v1
export HF_TOKEN=<your token>
```

## Offline run (through the vllm-omni backend)

```bash
python examples/offline_inference/personaplex/personaplex_offline.py \
    --input-wav input_assistant.wav --output-wav out.wav \
    --voice-prompt NATF2.pt --persona "You are a wise and friendly teacher." \
    --seed 42424242 --greedy
```

Bundled voices (`voices.tgz`): `NATF*/NATM*` (natural f/m), `VARF*/VARM*` (varied).
This driver goes through `PersonaPlexEngine` + `PersonaPlexSession`; a matching run via
upstream `python -m moshi.offline` is the reference for parity.

## Package layout

```
vllm_omni/experimental/fullduplex/personaplex/
  config.py    PersonaPlexConfig (frozen): voice / persona / sampling
  engine.py    PersonaPlexEngine — real moshi LMGen + 2× Mimi (FrameStepper); the only
               module importing `moshi`. FrameStepper / FrameOutput live here.
  session.py   PersonaPlexSession — lockstep driver (frames PCM → 80 ms windows). Runnable path.
  adapter.py   PersonaPlexDuplexAdapter — core.DuplexAdapter in continuous mode (framework demo)
```

`core/` is untouched except a default-off `continuous` flag (`DuplexCapability` +
`DuplexSessionConfig`) and the matching start-once / drain-on-close branch in
`DuplexRuntime`. GPU-free tests in `tests/fullduplex/test_personaplex_*.py` exercise
the lockstep contract with a stub `FrameStepper`.
