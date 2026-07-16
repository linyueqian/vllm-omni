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
import json
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

    here = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(
        os.path.join(here, "realtime-duplex-session.js"),
        os.path.join(overlay, "duplex", "lib", "realtime-duplex-session.js"),
    )

    app_js = os.path.join(overlay, "audio-duplex", "audio-duplex-app.js")
    with open(app_js, encoding="utf-8") as f:
        content = f.read()
    if _OLD_IMPORT not in content:
        raise SystemExit(
            "audio-duplex-app.js does not import duplex-session.js as expected; "
            "pin the demo checkout to a compatible commit (see README)."
        )
    with open(app_js, "w", encoding="utf-8") as f:
        f.write(content.replace(_OLD_IMPORT, _NEW_IMPORT))

    # The omni page's RealtimeSession speaks a near-Realtime dialect (result-
    # shaped response.output_audio.delta, f32 appends without a format field,
    # no commit). Point it at the translating proxy route; the page itself is
    # otherwise unmodified.
    rt_js = os.path.join(overlay, "duplex", "lib", "realtime-session.js")
    if os.path.exists(rt_js):
        with open(rt_js, encoding="utf-8") as f:
            rt_content = f.read()
        rt_patched = rt_content.replace("{location.host}/v1/realtime", "{location.host}/v1/realtime/omni")
        if rt_patched == rt_content:
            logger.warning("realtime-session.js WS URL not found; omni page may bypass the translating proxy")
        with open(rt_js, "w", encoding="utf-8") as f:
            f.write(rt_patched)

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


# ---------------------------------------------------------------------------
# Omni-page translating proxy
#
# The official omni page's RealtimeSession dialect differs from the runtime's
# realtime protocol in four ways; this route adapts both directions so the
# page itself stays unmodified:
#   client->backend: f32 appends gain format/sample_rate_hz; ref_audio_base64
#     maps to extra_body.ref_audio; a commit is injected after end-of-utterance
#     silence (the runtime schedules responses on commit); max_slice_nums is
#     stripped (HD slicing unsupported).
#   backend->client: response.audio.delta + transcript deltas are aggregated
#     into result-shaped response.output_audio.delta events (f32le audio),
#     with end_of_turn derived from response.done.
# ---------------------------------------------------------------------------

_OMNI_SILENCE_COMMIT_MS = 500
_OMNI_SPEECH_RMS = 0.015


def _f32b64_to_wav_data_uri(audio_b64: str, sample_rate: int = 16000) -> str:
    import io
    import wave

    f32 = np.frombuffer(base64.b64decode(audio_b64), dtype=np.float32)
    pcm16 = np.clip(f32 * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _omni_translate_client_event(event: dict, state: dict) -> list[dict]:
    etype = event.get("type")
    if etype == "session.update":
        session = event.get("session") if isinstance(event.get("session"), dict) else {}
        extra_body: dict = {"auto_response": True, "minicpmo45_native_duplex": True}
        ref_b64 = session.get("tts_ref_audio_base64") or session.get("ref_audio_base64")
        if isinstance(ref_b64, str) and ref_b64:
            extra_body["ref_audio"] = _f32b64_to_wav_data_uri(ref_b64)
        out_session: dict = {
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm_f32le",
            "output_audio_format": "pcm16",
            "extra_body": extra_body,
        }
        if isinstance(session.get("instructions"), str):
            out_session["instructions"] = session["instructions"]
        return [{"type": "session.update", "session": out_session}]
    if etype == "input_audio_buffer.append":
        out = {
            "type": "input_audio_buffer.append",
            "audio": event.get("audio"),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
        }
        if event.get("force_listen"):
            out["force_listen"] = True
        frames = event.get("video_frames")
        if isinstance(frames, list) and frames:
            out["video_frames"] = frames
        events = [out]
        # End-of-utterance commit injection (the omni page never commits).
        audio_b64 = event.get("audio")
        rms = 0.0
        duration_ms = 1000
        if isinstance(audio_b64, str) and audio_b64:
            try:
                f32 = np.frombuffer(base64.b64decode(audio_b64), dtype=np.float32)
                if f32.size:
                    rms = float(np.sqrt(np.mean(np.square(f32, dtype=np.float64))))
                    duration_ms = int(f32.size * 1000 / 16000)
            except Exception:  # noqa: BLE001 - treat undecodable audio as silence
                pass
        if rms > _OMNI_SPEECH_RMS:
            state["had_speech"] = True
            state["silence_ms"] = 0
        elif state.get("had_speech"):
            state["silence_ms"] = state.get("silence_ms", 0) + duration_ms
            if state["silence_ms"] >= _OMNI_SILENCE_COMMIT_MS:
                state["had_speech"] = False
                state["silence_ms"] = 0
                events.append({"type": "input_audio_buffer.commit", "final": True})
        return events
    return [event]


def _pcm16_b64_to_f32_b64(delta_b64: str) -> str:
    pcm16 = np.frombuffer(base64.b64decode(delta_b64), dtype="<i2")
    f32 = (pcm16.astype(np.float32) / 32768.0).astype("<f4")
    return base64.b64encode(f32.tobytes()).decode("ascii")


def _omni_translate_backend_event(event: dict, state: dict) -> list[dict]:
    etype = event.get("type")
    if etype in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
        delta = event.get("delta")
        if isinstance(delta, str):
            state["pending_text"] = state.get("pending_text", "") + delta
        return []
    if etype == "response.audio.delta":
        audio_b64 = event.get("delta") or ""
        fmt = str(event.get("format") or "pcm16").lower()
        if isinstance(audio_b64, str) and audio_b64 and "f32" not in fmt:
            try:
                audio_b64 = _pcm16_b64_to_f32_b64(audio_b64)
            except Exception:  # noqa: BLE001 - pass through undecodable audio
                pass
        text = state.pop("pending_text", "")
        return [{"type": "response.output_audio.delta", "audio": audio_b64, "text": text, "end_of_turn": False}]
    if etype == "response.done":
        text = state.pop("pending_text", "")
        return [{"type": "response.output_audio.delta", "audio": "", "text": text, "end_of_turn": True}]
    if etype == "response.output_audio.delta":
        # Internal dialect duplicate of response.audio.delta; drop it so the
        # page does not double-play audio.
        return []
    return [event]


@app.websocket("/v1/realtime/omni")
async def realtime_omni_proxy(ws: WebSocket) -> None:
    assert ARGS is not None
    backend_url = ARGS.ws_backend.rstrip("/") + "/v1/realtime?duplex=1"
    await ws.accept()
    try:
        backend = await websockets.connect(backend_url, max_size=64 * 1024 * 1024)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client
        logger.error("backend connect failed: %s", exc)
        with contextlib.suppress(Exception):
            await ws.close(code=1013, reason=f"backend unavailable: {exc}")
        return
    state: dict = {}

    async def client_to_backend() -> None:
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    with contextlib.suppress(Exception):
                        await backend.close()
                    return
                text = message.get("text")
                if text is None:
                    continue
                try:
                    event = json.loads(text)
                except ValueError:
                    await backend.send(text)
                    continue
                for out in _omni_translate_client_event(event, state):
                    await backend.send(json.dumps(out))
        except (WebSocketDisconnect, websockets.ConnectionClosed):
            with contextlib.suppress(Exception):
                await backend.close()

    async def backend_to_client() -> None:
        try:
            async for message in backend:
                if isinstance(message, bytes):
                    await ws.send_bytes(message)
                    continue
                try:
                    event = json.loads(message)
                except ValueError:
                    await ws.send_text(message)
                    continue
                for out in _omni_translate_backend_event(event, state):
                    await ws.send_text(json.dumps(out))
        except (WebSocketDisconnect, websockets.ConnectionClosed):
            pass
        with contextlib.suppress(Exception):
            await ws.close()

    try:
        await asyncio.gather(client_to_backend(), backend_to_client())
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
