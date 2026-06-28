# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex full-duplex server, compatible with the OFFICIAL Moshi web client.

Serves the official PersonaPlex web UI (``dist.tgz`` from ``nvidia/personaplex-7b-v1``,
the same bundle ``moshi.server`` downloads) and implements its WebSocket protocol at
``/api/chat`` -- but driven by our native-component engine instead of moshi's server.

Moshi protocol (binary WS messages, first byte = tag):
  server -> client  ``\\x00``                 handshake (ready, after system prompts)
  client -> server  ``\\x01`` + opus bytes    mic audio (Opus @ 24kHz)
  server -> client  ``\\x01`` + opus bytes    agent audio (Opus @ 24kHz)
  server -> client  ``\\x02`` + utf8          inner-monologue text
Query params on connect: ``text_prompt`` (persona), ``voice_prompt`` (voice file).

Run (auto-downloads the official UI; open http://<host>:8124/):
    HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 python -m \\
        vllm_omni.experimental.fullduplex.personaplex.serving.server --port 8124

A raw-PCM endpoint (``/v1/audio/duplex``, JSON ``open`` + float32 frames) remains for
tooling/tests that do not want an Opus dependency.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import tarfile
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sphn
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download

from vllm_omni.experimental.fullduplex.personaplex.config import PersonaPlexConfig
from vllm_omni.experimental.fullduplex.personaplex.engine import PersonaPlexEngine
from vllm_omni.experimental.fullduplex.personaplex.session import PersonaPlexSession

logger = logging.getLogger(__name__)
_HF_REPO = "nvidia/personaplex-7b-v1"


def _official_web_dir() -> Path | None:
    """Download + extract the official PersonaPlex web client (dist.tgz)."""
    try:
        tgz = Path(hf_hub_download(_HF_REPO, "dist.tgz"))
        dist = tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(tgz, "r:gz") as tar:
                tar.extractall(path=tgz.parent)  # noqa: S202 - trusted first-party bundle
        return dist if dist.exists() else None
    except Exception as exc:  # network / auth / layout
        logger.warning("official web client unavailable (%s); / will return an error note", exc)
        return None


class DuplexServer:
    """Owns the loaded engine; serves one duplex conversation at a time."""

    def __init__(self, config: PersonaPlexConfig) -> None:
        self.config = config
        self.engine = PersonaPlexEngine(config)

    def load(self) -> None:
        self.engine.load()

    # ---- official Moshi protocol (Opus) -------------------------------------

    async def handle_chat(self, ws: WebSocket) -> None:
        await ws.accept()
        q = ws.query_params
        persona = (q.get("text_prompt") or "").strip() or None
        voice = (q.get("voice_prompt") or "").strip() or None
        sr = self.engine.sample_rate
        session = PersonaPlexSession(self.engine, self.config)
        session.open(voice_prompt=voice, persona=persona)  # system prompts (sync)

        reader = sphn.OpusStreamReader(sr)
        writer = sphn.OpusStreamWriter(sr)
        state = {"close": False}
        feed_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="personaplex-feed")
        loop = asyncio.get_running_loop()
        pending_pcm = np.zeros(0, dtype=np.float32)
        await ws.send_bytes(b"\x00")  # ready handshake

        async def recv_loop() -> None:
            try:
                while not state["close"]:
                    data = await ws.receive_bytes()
                    if data and data[0] == 1:
                        reader.append_bytes(data[1:])
            except (WebSocketDisconnect, RuntimeError):
                pass
            finally:
                state["close"] = True

        async def proc_loop() -> None:
            nonlocal pending_pcm
            frame_size = session.frame_size
            while not state["close"]:
                await asyncio.sleep(0.001)
                pcm = reader.read_pcm()
                if pcm is not None and pcm.shape[-1] != 0:
                    samples = np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1)
                    pending_pcm = np.concatenate([pending_pcm, samples]) if pending_pcm.size else samples
                if pending_pcm.shape[0] < frame_size:
                    continue
                frame, pending_pcm = pending_pcm[:frame_size], pending_pcm[frame_size:]
                outputs = await loop.run_in_executor(
                    feed_executor,
                    session.feed,
                    np.ascontiguousarray(frame, dtype=np.float32),
                )
                for frame_out in outputs:
                    if frame_out.audio is not None:
                        writer.append_pcm(np.ascontiguousarray(frame_out.audio, dtype=np.float32))
                    if frame_out.text:
                        await ws.send_bytes(b"\x02" + frame_out.text.encode("utf8"))

        async def send_loop() -> None:
            while not state["close"]:
                await asyncio.sleep(0.001)
                msg = writer.read_bytes()
                if msg:
                    await ws.send_bytes(b"\x01" + msg)

        tasks = [asyncio.create_task(recv_loop()), asyncio.create_task(proc_loop()), asyncio.create_task(send_loop())]
        try:
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
        finally:
            feed_executor.shutdown(wait=False, cancel_futures=True)
            with contextlib.suppress(Exception):
                await ws.close()

    # ---- simple raw-PCM protocol (no Opus, for tests) -----------------------

    async def handle_raw(self, ws: WebSocket) -> None:
        await ws.accept()
        session = PersonaPlexSession(self.engine, self.config)
        opened = False
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if (text := msg.get("text")) is not None:
                    req = json.loads(text)
                    if req.get("type") == "open":
                        session.open(voice_prompt=req.get("voice"), persona=req.get("persona"))
                        opened = True
                        await ws.send_json({"type": "ready"})
                    elif req.get("type") == "close":
                        if opened:
                            await self._emit(ws, session.flush())
                        await ws.send_json({"type": "done"})
                        break
                    continue
                data = msg.get("bytes")
                if data is None:
                    continue
                if not opened:
                    session.open()
                    opened = True
                await self._emit(ws, session.feed(np.frombuffer(data, dtype=np.float32)))
        except WebSocketDisconnect:
            pass
        finally:
            with contextlib.suppress(Exception):
                await ws.close()

    @staticmethod
    async def _emit(ws: WebSocket, outputs: list) -> None:
        for frame_out in outputs:
            if frame_out.audio is not None:
                await ws.send_bytes(np.ascontiguousarray(frame_out.audio, dtype=np.float32).tobytes())
            if frame_out.text:
                await ws.send_json({"type": "text", "text": frame_out.text})


def create_app(config: PersonaPlexConfig) -> FastAPI:
    server = DuplexServer(config)
    web_dir = _official_web_dir()

    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        logger.info("PersonaPlex duplex server: loading engine")
        server.load()
        logger.info("PersonaPlex duplex server: ready (official web: %s)", bool(web_dir))
        yield

    app = FastAPI(title="vLLM-Omni PersonaPlex Duplex Server", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/api/chat")
    async def chat(ws: WebSocket) -> None:
        await server.handle_chat(ws)

    @app.websocket("/v1/audio/duplex")
    async def duplex(ws: WebSocket) -> None:
        await server.handle_raw(ws)

    # Mount the official PersonaPlex web client LAST so explicit routes win.
    if web_dir is not None:
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    else:

        @app.get("/")
        async def index() -> dict[str, str]:
            return {"error": f"official web client unavailable; check access to {_HF_REPO} (dist.tgz)"}

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8124)
    ap.add_argument("--persona", default=None)
    ap.add_argument("--voice", default="NATF2.pt")
    ap.add_argument("--greedy", action="store_true", default=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = PersonaPlexConfig(
        use_native_components=True,
        greedy=args.greedy,
        seed=42,
        voice_prompt=args.voice,
        **({"persona": args.persona} if args.persona else {}),
    )
    uvicorn.run(create_app(cfg), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
