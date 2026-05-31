"""Background asyncio worker loop and ref-audio loading.

Gradio click handlers call submit() from the UI thread; the burst executes on
this background loop. Snapshots remain thread-safe via MetricsAggregator's lock.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from concurrent.futures import Future
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
REF_AUDIO_PATH = REPO_ROOT / "tests" / "assets" / "qwen3_tts" / "clone_2.wav"

# Transcript of REF_AUDIO_PATH. Qwen3-TTS-Base rejects requests without it
# unless x_vector_only_mode is set. Same constant used by
# tests/e2e/online_serving/test_qwen3_tts_base.py.
REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)


def load_ref_audio_b64() -> str:
    """Read the in-repo ref-audio fixture and return base64-encoded WAV bytes."""
    if not REF_AUDIO_PATH.is_file():
        raise FileNotFoundError(
            f"Ref audio not found at {REF_AUDIO_PATH}. The demo expects the vendored "
            "tests/assets/qwen3_tts/clone_2.wav. Re-clone or pull the repo to restore it."
        )
    return base64.b64encode(REF_AUDIO_PATH.read_bytes()).decode("ascii")


class WorkerLoop:
    """Owns a thread + asyncio event loop. submit() schedules a coroutine."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ccd-worker")
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def submit(self, coro) -> Future:
        if self._loop is None:
            raise RuntimeError("Worker loop not initialised")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
