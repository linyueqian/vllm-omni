# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex full-duplex server, compatible with the OFFICIAL Moshi web client.

Serves the official PersonaPlex web UI (``dist.tgz`` from ``nvidia/personaplex-7b-v1``,
the same bundle ``moshi.server`` downloads) and implements its WebSocket protocol at
``/api/chat`` -- but driven by our native-component engine instead of moshi's server.

Built on **aiohttp**, mirroring ``moshi.server`` exactly: aiohttp writes+drains each
WebSocket frame promptly, whereas uvicorn's ``websockets`` backend coalesces many small
(~80 ms Opus) frames into a send buffer, which makes browser playback choppy even when
the engine is real-time. aiohttp ships as a moshi dependency, so this adds nothing new.

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
from pathlib import Path

import numpy as np
import sphn
from aiohttp import WSMsgType, web
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

    # ---- official Moshi protocol (Opus over aiohttp WS) ---------------------

    async def handle_chat(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        persona = (request.query.get("text_prompt") or "").strip() or None
        voice = (request.query.get("voice_prompt") or "").strip() or None
        sr = self.engine.sample_rate
        session = PersonaPlexSession(self.engine, self.config)
        session.open(voice_prompt=voice, persona=persona)  # system-prompt prefill

        reader = sphn.OpusStreamReader(sr)
        writer = sphn.OpusStreamWriter(sr)
        state = {"close": False}
        await ws.send_bytes(b"\x00")  # ready handshake

        async def recv_loop() -> None:
            try:
                async for msg in ws:
                    if msg.type == WSMsgType.BINARY:
                        data = msg.data
                        if data and data[0] == 1:
                            reader.append_bytes(data[1:])
                    elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                        break
            finally:
                state["close"] = True

        async def proc_loop() -> None:
            # Mirror moshi.server's opus_loop: drain decoded PCM and step the model,
            # appending agent audio to the Opus writer; the send loop streams it out.
            while not state["close"]:
                await asyncio.sleep(0.001)
                pcm = reader.read_pcm()
                if pcm is None or pcm.shape[-1] == 0:
                    continue
                for frame_out in session.feed(np.asarray(pcm, dtype=np.float32)):
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
            with contextlib.suppress(Exception):
                await ws.close()
        return ws

    # ---- simple raw-PCM protocol (no Opus, for tests) ----------------------

    async def handle_raw(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        session = PersonaPlexSession(self.engine, self.config)
        opened = False
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    req = json.loads(msg.data)
                    if req.get("type") == "open":
                        session.open(voice_prompt=req.get("voice"), persona=req.get("persona"))
                        opened = True
                        await ws.send_json({"type": "ready"})
                    elif req.get("type") == "close":
                        if opened:
                            await self._emit(ws, session.flush())
                        await ws.send_json({"type": "done"})
                        break
                elif msg.type == WSMsgType.BINARY:
                    if not opened:
                        session.open()
                        opened = True
                    await self._emit(ws, session.feed(np.frombuffer(msg.data, dtype=np.float32)))
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                    break
        finally:
            with contextlib.suppress(Exception):
                await ws.close()
        return ws

    @staticmethod
    async def _emit(ws: web.WebSocketResponse, outputs: list) -> None:
        for frame_out in outputs:
            if frame_out.audio is not None:
                await ws.send_bytes(np.ascontiguousarray(frame_out.audio, dtype=np.float32).tobytes())
            if frame_out.text:
                await ws.send_json({"type": "text", "text": frame_out.text})


def create_app(config: PersonaPlexConfig) -> web.Application:
    server = DuplexServer(config)
    web_dir = _official_web_dir()

    async def _startup(_app: web.Application) -> None:
        logger.info("PersonaPlex duplex server: loading engine")
        server.load()
        logger.info("PersonaPlex duplex server: ready (official web: %s)", bool(web_dir))

    app = web.Application()
    app.on_startup.append(_startup)

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    app.router.add_get("/health", health)
    app.router.add_get("/api/chat", server.handle_chat)
    app.router.add_get("/v1/audio/duplex", server.handle_raw)

    # Serve the official PersonaPlex web client (explicit routes are matched first).
    if web_dir is not None:

        async def index(_request: web.Request) -> web.FileResponse:
            return web.FileResponse(web_dir / "index.html")

        app.router.add_get("/", index)
        app.router.add_static("/", path=str(web_dir), follow_symlinks=True, name="web")
    else:

        async def index_missing(_request: web.Request) -> web.Response:
            return web.json_response({"error": f"official web client unavailable; check access to {_HF_REPO}"})

        app.router.add_get("/", index_missing)

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
    web.run_app(create_app(cfg), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
