# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex full-duplex WebSocket server (PCM in -> agent PCM + text out).

Thin async wrapper over the verified serving primitive (PersonaPlexSession driving
the native-component FrameStepper). A client streams 24kHz mono float32 PCM frames
as binary WS messages; the server replies, per consumed 80ms frame, with a binary
agent-PCM message and (optionally) a JSON inner-monologue text message.

Single conversation at a time (the engine holds one lockstep KV state). Run:

    HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 python -m vllm_omni.experimental.fullduplex.personaplex.serving.server \
        --host 0.0.0.0 --port 8124

Protocol (per connection):
  client -> {"type":"open","persona":"...","voice":"NATF2.pt"}   (text/JSON, optional)
  client -> <binary float32 PCM>  (any length; server frames it)
  server -> <binary float32 agent PCM>  (one per consumed frame, may be empty-skipped)
  server -> {"type":"text","text":"..."}  (inner-monologue pieces)
  client -> {"type":"close"}  -> server flushes the tail and closes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
from collections.abc import AsyncIterator

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from vllm_omni.experimental.fullduplex.personaplex.config import PersonaPlexConfig
from vllm_omni.experimental.fullduplex.personaplex.engine import PersonaPlexEngine
from vllm_omni.experimental.fullduplex.personaplex.session import PersonaPlexSession

logger = logging.getLogger(__name__)


class DuplexServer:
    """Owns the loaded engine; serves one duplex conversation at a time."""

    def __init__(self, config: PersonaPlexConfig) -> None:
        self.config = config
        self.engine = PersonaPlexEngine(config)

    def load(self) -> None:
        self.engine.load()

    async def handle(self, ws: WebSocket) -> None:
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
                    kind = req.get("type")
                    if kind == "open":
                        session.open(voice_prompt=req.get("voice"), persona=req.get("persona"))
                        opened = True
                        await ws.send_json({"type": "ready"})
                    elif kind == "close":
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
                pcm = np.frombuffer(data, dtype=np.float32)
                await self._emit(ws, session.feed(pcm))
        except WebSocketDisconnect:
            pass
        finally:
            with contextlib.suppress(Exception):
                await ws.close()

    @staticmethod
    async def _emit(ws: WebSocket, outputs: list) -> None:
        for fo in outputs:
            if fo.audio is not None:
                await ws.send_bytes(np.ascontiguousarray(fo.audio, dtype=np.float32).tobytes())
            if fo.text:
                await ws.send_json({"type": "text", "text": fo.text})


def create_app(config: PersonaPlexConfig) -> FastAPI:
    server = DuplexServer(config)

    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        logger.info("PersonaPlex duplex server: loading engine")
        server.load()
        logger.info("PersonaPlex duplex server: ready")
        yield

    app = FastAPI(title="vLLM-Omni PersonaPlex Duplex Server", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/v1/audio/duplex")
    async def duplex(ws: WebSocket) -> None:
        await server.handle(ws)

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
