# PersonaPlex full-duplex online serving

Serve [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1)
(a Moshi-based full-duplex speech-to-speech model) with the native vLLM-Omni engine.

The server hosts the **official PersonaPlex web client** at `/` (auto-downloaded from
the model repo) and implements its WebSocket protocol at `/api/chat`, so you talk to the
model live in the browser — mic in, agent speech + inner-monologue text out.

> Experimental. Requires a GPU and Hugging Face access to the gated repo
> (`HF_TOKEN` with access to `nvidia/personaplex-7b-v1`).

## Start the server

```bash
HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 bash run_server.sh
# or directly:
HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 python -m \
    vllm_omni.experimental.fullduplex.personaplex.serving.server --port 8124
```

## Talk to it

- **Browser (recommended):** open `http://localhost:8124/` and allow the microphone.
  Use headphones so the agent does not hear itself.
- **Headless:** stream a 24 kHz mono WAV and save the reply:

  ```bash
  python duplex_client.py --url ws://localhost:8124/api/chat --input user.wav --out reply.wav
  ```

## Protocol (`/api/chat`)

Binary WebSocket messages, first byte is a tag (identical to `moshi.server`):

| Direction | Tag | Payload |
| --- | --- | --- |
| server → client | `\x00` | ready handshake (after the system-prompt prefill) |
| client → server | `\x01` | Opus-encoded mic audio (24 kHz) |
| server → client | `\x01` | Opus-encoded agent audio |
| server → client | `\x02` | UTF-8 inner-monologue text |

Connect with query params `text_prompt` (persona) and `voice_prompt` (voice file).
A raw-PCM endpoint (`/v1/audio/duplex`, JSON `open` + float32 frames) is also available
for clients that do not want an Opus dependency.

## Notes

- **Run the client near the server.** Real-time 80 ms audio is sensitive to network
  latency/jitter; over a high-latency remote link playback can stutter regardless of
  engine speed. On localhost it is smooth.
- Greedy decoding by default; one conversation per server instance.
