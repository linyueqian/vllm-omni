# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the OFFICIAL Moshi-protocol endpoint (/api/chat) e2e with Opus.

Mimics what the official PersonaPlex web client does: connect, wait for the
``\\x00`` handshake, stream Opus mic audio (``\\x01``), collect agent Opus
(``\\x01``) + text (``\\x02``), then ASR-check coherence.

    HF_TOKEN=... python tools/personaplex/test_official_ws.py \
        --url ws://127.0.0.1:8124/api/chat --input-wav <wav> --seconds 6 --tail 10
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
import soundfile as sf
import sphn


async def run(url: str, pcm: np.ndarray, sr: int) -> tuple[np.ndarray, str]:
    import websockets

    reader = sphn.OpusStreamReader(sr)  # decode agent audio
    writer = sphn.OpusStreamWriter(sr)  # encode our mic audio
    pieces: list[str] = []
    ready = asyncio.Event()
    last_audio = {"t": 0.0}

    async with websockets.connect(url, max_size=None) as ws:

        async def receiver() -> None:
            loop = asyncio.get_event_loop()
            while True:
                try:
                    msg = await ws.recv()
                except Exception:
                    return
                if not isinstance(msg, (bytes, bytearray)) or len(msg) == 0:
                    continue
                tag, payload = msg[0], msg[1:]
                if tag == 0:
                    ready.set()
                elif tag == 1:
                    reader.append_bytes(bytes(payload))
                    last_audio["t"] = loop.time()
                elif tag == 2:
                    pieces.append(payload.decode("utf8", errors="replace"))

        recv_task = asyncio.create_task(receiver())
        await asyncio.wait_for(ready.wait(), timeout=120)  # system prompts can take a moment

        frame = 1920
        for i in range(0, len(pcm), frame):
            chunk = pcm[i : i + frame]
            if len(chunk) < frame:
                chunk = np.concatenate([chunk, np.zeros(frame - len(chunk), dtype=np.float32)])
            writer.append_pcm(chunk)
            data = writer.read_bytes()
            if data:
                await ws.send(b"\x01" + data)
            await asyncio.sleep(frame / sr)  # pace at real time

        # Drain: keep the socket open until the agent stops talking (>2.5s quiet).
        loop = asyncio.get_event_loop()
        last_audio["t"] = loop.time()
        while loop.time() - last_audio["t"] < 2.5:
            await asyncio.sleep(0.1)
        recv_task.cancel()

    # Flush the decoder.
    chunks: list[np.ndarray] = []
    while True:
        out = reader.read_pcm()
        if out is None or out.shape[-1] == 0:
            break
        chunks.append(np.asarray(out, dtype=np.float32))
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return audio, "".join(pieces)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8124/api/chat")
    ap.add_argument("--input-wav", required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--tail", type=float, default=10.0)
    ap.add_argument("--out", default="/home/yueqian/pplex_official_ws.wav")
    args = ap.parse_args()

    from moshi.models.lm import load_audio

    sr = 24000
    wav = np.asarray(load_audio(args.input_wav, sr), dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    wav = wav[: int(args.seconds * sr)]
    wav = np.concatenate([wav, np.zeros(int(args.tail * sr), dtype=np.float32)])

    audio, text = asyncio.run(run(f"{args.url}?text_prompt=&voice_prompt=NATF2.pt", wav, sr))
    sf.write(args.out, audio if audio.size else np.zeros(sr, dtype=np.float32), sr)
    rms = float(np.sqrt((audio**2).mean())) if audio.size else 0.0
    print(f"agent audio {audio.size / sr:.2f}s rms={rms:.4f} -> {args.out}")
    print(f"inner-monologue text: {text[:200]!r}")

    import os
    import tempfile

    import whisper

    m = whisper.load_model("small", device="cuda")
    seg = audio[: 30 * sr]
    if seg.size:
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tf.name, seg, sr)
        r = m.transcribe(tf.name, language="en", fp16=True)
        os.unlink(tf.name)
        print(f"ASR={r['text'].strip()[:200]!r}")


if __name__ == "__main__":
    main()
