# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WebSocket client e2e test for the PersonaPlex duplex server.

Streams a user WAV (+ trailing silence so the agent can reply) to the duplex WS
endpoint, collects the agent PCM + inner-monologue text, then ASR-checks coherence.

    HF_TOKEN=... python tools/personaplex/test_duplex_ws.py \
        --url ws://localhost:8124/v1/audio/duplex --input-wav <wav> --seconds 6 --tail 10
"""

from __future__ import annotations

import argparse
import asyncio
import json

import numpy as np
import soundfile as sf


async def run(url: str, pcm: np.ndarray, sr: int, persona: str | None, voice: str) -> tuple[np.ndarray, str]:
    import websockets

    agent: list[np.ndarray] = []
    pieces: list[str] = []
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "open", "persona": persona, "voice": voice}))

        async def receiver() -> None:
            while True:
                try:
                    msg = await ws.recv()
                except Exception:
                    return
                if isinstance(msg, bytes):
                    agent.append(np.frombuffer(msg, dtype=np.float32).copy())
                else:
                    obj = json.loads(msg)
                    if obj.get("type") == "text":
                        pieces.append(obj["text"])
                    elif obj.get("type") == "done":
                        return

        recv_task = asyncio.create_task(receiver())
        # wait briefly for "ready" is folded into the receiver; just stream the PCM.
        chunk = sr  # 1s chunks
        for i in range(0, len(pcm), chunk):
            await ws.send(pcm[i : i + chunk].astype(np.float32).tobytes())
            await asyncio.sleep(0.01)
        await ws.send(json.dumps({"type": "close"}))
        with __import__("contextlib").suppress(asyncio.TimeoutError):
            await asyncio.wait_for(recv_task, timeout=120)
    audio = np.concatenate(agent) if agent else np.zeros(0, dtype=np.float32)
    return audio, "".join(pieces)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8124/v1/audio/duplex")
    ap.add_argument("--input-wav", required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--tail", type=float, default=10.0)
    ap.add_argument("--voice", default="NATF2.pt")
    ap.add_argument("--persona", default=None)
    ap.add_argument("--out", default="/home/yueqian/pplex_duplex_ws.wav")
    args = ap.parse_args()

    from moshi.models.lm import load_audio

    sr = 24000
    wav = np.asarray(load_audio(args.input_wav, sr), dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    wav = wav[: int(args.seconds * sr)]
    wav = np.concatenate([wav, np.zeros(int(args.tail * sr), dtype=np.float32)])

    audio, text = asyncio.run(run(args.url, wav, sr, args.persona, args.voice))
    sf.write(args.out, audio if audio.size else np.zeros(sr, dtype=np.float32), sr)
    rms = float(np.sqrt((audio**2).mean())) if audio.size else 0.0
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    print(f"agent audio {audio.size / sr:.2f}s rms={rms:.4f} peak={peak:.3f} -> {args.out}")
    print(f"inner-monologue text: {text[:200]!r}")

    import os
    import tempfile

    import whisper

    m = whisper.load_model("small", device="cuda")
    for a, b in [(0, 30)]:
        seg = audio[int(a * sr) : int(b * sr)]
        if seg.size == 0:
            continue
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tf.name, seg, sr)
        r = m.transcribe(tf.name, language="en", fp16=True)
        os.unlink(tf.name)
        print(f"ASR[{a}:{b}s]={r['text'].strip()[:200]!r}")


if __name__ == "__main__":
    main()
