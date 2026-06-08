"""Static host for the MiniCPM-o 4.5 continuous duplex voice-chat web UI.

A minimal FastAPI + uvicorn app that serves index.html and the two AudioWorklet
JS files. It does NOT proxy the WebSocket: the browser connects directly to the
vLLM duplex endpoint (``ws://<host>:8099/v1/realtime?duplex=1``). Gradio is
intentionally avoided because it interferes with AudioWorklet registration.

The page derives the WS host from ``location.hostname`` (so it works both via an
SSH ``localhost`` tunnel and direct ``10.137.72.57``). ``--ws-base`` may inject an
explicit base if needed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("realtime_web")

STATIC_DIR = Path(__file__).parent / "static"


def build_app(ws_base: str = "") -> FastAPI:
    """Build the FastAPI app serving the static UI.

    Args:
        ws_base: Optional explicit duplex WS base (e.g. ``ws://10.137.72.57:8099``).
            Empty string -> the client derives it from ``location.hostname:8099``.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="MiniCPM-o Realtime Web")
    index_path = STATIC_DIR / "index.html"

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = index_path.read_text(encoding="utf-8")
        html = html.replace("__DUPLEX_WS_BASE__", ws_base)
        return HTMLResponse(html)

    @app.get("/healthz")
    def healthz() -> Response:
        return Response(content="ok", media_type="text/plain")

    # serve worklet JS + app.js under /static
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniCPM-o realtime web UI host")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument(
        "--ws-base",
        default="",
        help="Explicit duplex WS base, e.g. ws://10.137.72.57:8099. "
        "Empty -> client derives from location.hostname:8099.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Serving realtime web UI on %s:%d (ws_base=%r)", args.host, args.port, args.ws_base)
    app = build_app(ws_base=args.ws_base)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
