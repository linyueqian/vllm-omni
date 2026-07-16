"""Official MiniCPM-o-Demo audio-duplex UI on vLLM-Omni's realtime endpoint.

Serves the audio-duplex page of the official OpenBMB/MiniCPM-o-Demo frontend
directly against ``/v1/realtime?duplex=1`` — no official gateway, no worker,
no protocol bridge. At startup this script builds a runtime overlay of the
official static tree (audio-duplex page only), drops in the bundled
``realtime-duplex-session.js`` transport (same hook surface as the official
``duplex-session.js``, but speaking the realtime duplex protocol from the
browser), and byte-proxies ``/v1/realtime`` to the vLLM-Omni server so the
page works same-origin. The official demo checkout itself is never modified.

Run (after the vLLM-Omni duplex server is up on --ws-backend):
  python serve.py --port 8006 \
      --demo-root /path/to/MiniCPM-o-Demo \
      --ws-backend ws://127.0.0.1:8099

Then open http://<host>:8006 (mic capture needs localhost or HTTPS).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import logging
import os
import shutil
import sys
import tempfile

import numpy as np
import soundfile as sf
import uvicorn
import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("official_ui_duplex")

ARGS: argparse.Namespace | None = None
app = FastAPI(title="Official MiniCPM-o duplex UI on vLLM realtime")

_presets_cache: dict | None = None
_default_ref_cache: dict | None = None

# Static subtrees the audio-duplex page depends on. Other demo pages
# (omni, half-duplex, turn-based, mobile, admin) are intentionally not served.
_STATIC_SUBTREES = ("audio-duplex", "omni", "duplex", "shared", "lib", "assets", "faq")

_OLD_IMPORT = "from '../duplex/lib/duplex-session.js'"
_NEW_IMPORT = "from '../duplex/lib/realtime-duplex-session.js'"


def build_overlay(demo_root: str) -> str:
    """Copy the audio-duplex subtree and swap its transport module."""
    src = os.path.join(demo_root, "static")
    if not os.path.isdir(src):
        raise SystemExit(f"--demo-root has no static/ dir: {demo_root}")

    overlay = tempfile.mkdtemp(prefix="minicpmo_official_ui_")
    for name in _STATIC_SUBTREES:
        path = os.path.join(src, name)
        if os.path.isdir(path):
            shutil.copytree(path, os.path.join(overlay, name))
    # The pages also load a few loose top-level scripts (e.g. the RefAudioPlayer
    # component at /static/ref-audio-player.js); copy them so imports resolve.
    for entry in os.listdir(src):
        entry_path = os.path.join(src, entry)
        if os.path.isfile(entry_path) and entry.endswith((".js", ".css")):
            shutil.copy(entry_path, os.path.join(overlay, entry))

    here = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(
        os.path.join(here, "realtime-duplex-session.js"),
        os.path.join(overlay, "duplex", "lib", "realtime-duplex-session.js"),
    )

    # Swap the transport module in each served page: the official pages
    # import DuplexSession from duplex-session.js (the gateway /ws/duplex
    # dialect); realtime-duplex-session.js is a drop-in that speaks the
    # realtime duplex protocol (and forwards omni camera frames + injects an
    # end-of-utterance commit). The pages are otherwise unmodified.
    page_apps = (
        ("audio-duplex", "audio-duplex-app.js"),
        ("omni", "omni-app.js"),
    )
    patched_any = False
    for subdir, filename in page_apps:
        app_js = os.path.join(overlay, subdir, filename)
        if not os.path.exists(app_js):
            continue
        with open(app_js, encoding="utf-8") as f:
            content = f.read()
        if _OLD_IMPORT not in content:
            logger.warning("%s does not import duplex-session.js as expected; skipping", filename)
            continue
        with open(app_js, "w", encoding="utf-8") as f:
            f.write(content.replace(_OLD_IMPORT, _NEW_IMPORT))
        patched_any = True
    if not patched_any:
        raise SystemExit(
            "no served page imports duplex-session.js as expected; "
            "pin the demo checkout to a compatible commit (see README)."
        )

    logger.info("overlay built at %s", overlay)
    return overlay


# ---------------------------------------------------------------------------
# Gateway API stubs the page expects
# ---------------------------------------------------------------------------


@app.get("/status")
async def status() -> dict:
    return {"idle_workers": 1, "total_workers": 1, "queue_length": 0}


@app.get("/api/frontend_defaults")
async def frontend_defaults() -> dict:
    return {"playback_delay_ms": 200}


def _load_wav_f32_16k(path: str) -> tuple[np.ndarray, float]:
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != 16000:
        n_out = int(round(len(mono) * 16000 / rate))
        x_out = np.linspace(0.0, len(mono) - 1.0, num=n_out, dtype=np.float64)
        mono = np.interp(x_out, np.arange(len(mono), dtype=np.float64), mono).astype(np.float32)
    return mono, len(mono) / 16000.0


@app.get("/api/default_ref_audio")
async def default_ref_audio() -> dict:
    global _default_ref_cache
    if _default_ref_cache is not None:
        return _default_ref_cache
    assert ARGS is not None
    path = ARGS.ref_audio
    if not os.path.isabs(path):
        path = os.path.join(ARGS.demo_root, path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"ref audio not found: {path}")
    mono, duration = _load_wav_f32_16k(path)
    _default_ref_cache = {
        "name": os.path.basename(path),
        "duration": round(duration, 1),
        "base64": base64.b64encode(mono.astype("<f4").tobytes()).decode("ascii"),
    }
    return _default_ref_cache


def _gateway_presets() -> dict:
    """Reuse the official gateway's preset loader against its assets dir."""
    global _presets_cache
    if _presets_cache is not None:
        return _presets_cache
    assert ARGS is not None
    sys.path.insert(0, ARGS.demo_root)
    try:
        import gateway  # noqa: PLC0415 - official demo loader, path added above

        _presets_cache = gateway._load_presets_from_dir(ARGS.demo_root)
    except Exception as exc:  # noqa: BLE001 - presets are optional
        logger.warning("preset loading failed (%s); serving empty presets", exc)
        _presets_cache = {}
    return _presets_cache


@app.get("/api/presets")
async def presets() -> dict:
    return _gateway_presets()


@app.get("/api/presets/{mode}/{preset_id}/audio")
async def preset_audio(mode: str, preset_id: str) -> dict:
    assert ARGS is not None
    cache = _gateway_presets()
    preset = next((p for p in cache.get(mode, []) if p.get("id") == preset_id), None)
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset not found: {mode}/{preset_id}")
    import gateway  # noqa: PLC0415 - path added by _gateway_presets

    result: dict = {}
    if "system_content" in preset:
        audio_items = []
        for item in preset["system_content"]:
            if item.get("type") == "audio" and item.get("path"):
                loaded = gateway._load_audio_base64(item["path"], ARGS.demo_root)
                audio_items.append(loaded or {"data": None, "name": item.get("name", ""), "duration": 0})
        result["system_content_audio"] = audio_items
    if preset.get("ref_audio") and preset["ref_audio"].get("path"):
        loaded = gateway._load_audio_base64(preset["ref_audio"]["path"], ARGS.demo_root)
        result["ref_audio"] = loaded or {
            "data": None,
            "name": preset["ref_audio"].get("name", ""),
            "duration": 0,
        }
    return result


@app.post("/api/sessions/{session_id}/comment")
async def session_comment(session_id: str) -> dict:
    return {"ok": True}


@app.post("/api/sessions/{session_id}/upload-recording")
async def session_upload(session_id: str) -> dict:
    return {"ok": True}


# ---------------------------------------------------------------------------
# Realtime WS byte-proxy (same-origin -> vLLM backend)
# ---------------------------------------------------------------------------


async def _pump_client_to_backend(client: WebSocket, backend) -> None:
    try:
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                with contextlib.suppress(Exception):
                    await backend.close()
                return
            if message.get("text") is not None:
                await backend.send(message["text"])
            elif message.get("bytes") is not None:
                await backend.send(message["bytes"])
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        with contextlib.suppress(Exception):
            await backend.close()


async def _pump_backend_to_client(client: WebSocket, backend) -> None:
    try:
        async for message in backend:
            if isinstance(message, bytes):
                await client.send_bytes(message)
            else:
                await client.send_text(message)
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        pass
    with contextlib.suppress(Exception):
        await client.close()


@app.websocket("/v1/realtime")
async def realtime_proxy(ws: WebSocket) -> None:
    assert ARGS is not None
    query = str(ws.url.query or "")
    backend_url = ARGS.ws_backend.rstrip("/") + "/v1/realtime" + (f"?{query}" if query else "")
    await ws.accept()
    try:
        backend = await websockets.connect(backend_url, max_size=64 * 1024 * 1024)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client
        logger.error("backend connect failed: %s", exc)
        with contextlib.suppress(Exception):
            await ws.close(code=1013, reason=f"backend unavailable: {exc}")
        return
    try:
        await asyncio.gather(
            _pump_client_to_backend(ws, backend),
            _pump_backend_to_client(ws, backend),
        )
    finally:
        with contextlib.suppress(Exception):
            await backend.close()


@app.get("/")
async def index() -> RedirectResponse:
    return RedirectResponse(url="/static/audio-duplex/audio_duplex.html")


def main() -> None:
    global ARGS  # noqa: PLW0603 - simple CLI singleton
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument(
        "--demo-root",
        required=True,
        help="Path to an OpenBMB/MiniCPM-o-Demo checkout (static UI, presets, ref audio)",
    )
    parser.add_argument("--ws-backend", default="ws://127.0.0.1:8099")
    parser.add_argument(
        "--ref-audio",
        default="assets/ref_audio/ref_minicpm_signature.wav",
        help="Default voice reference wav, relative to --demo-root unless absolute",
    )
    ARGS = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    overlay = build_overlay(ARGS.demo_root)
    app.mount("/static", StaticFiles(directory=overlay), name="static")
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="info")


if __name__ == "__main__":
    main()
