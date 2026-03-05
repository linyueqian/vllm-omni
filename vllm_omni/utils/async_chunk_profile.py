# SPDX-License-Identifier: Apache-2.0
"""Lightweight profiling for Async Chunk code paths (Qwen3-Omni etc.).

Enable by setting env: VLLM_OMNI_ASYNC_CHUNK_PROFILE=1
Then grep logs for [ASYNC_CHUNK_PROFILE] to analyze timings (ms).
"""

import os
import threading
import time
from contextlib import contextmanager
from typing import Generator


def _profile_enabled() -> bool:
    return os.environ.get("VLLM_OMNI_ASYNC_CHUNK_PROFILE", "").strip().lower() in ("1", "true", "yes")


def log_async_chunk_profile(tag: str, ms: float, extra: str = "") -> None:
    """Log a single timing. tag: short id (e.g. thinker2talker, code2wav_streaming)."""
    if not _profile_enabled():
        return
    from vllm.logger import init_logger
    logger = init_logger(__name__)
    msg = f"[ASYNC_CHUNK_PROFILE] {tag} {ms:.2f} ms"
    if extra:
        msg += f" | {extra}"
    logger.info(msg)


def log_async_chunk_event(tag: str, phase: str, extra: str = "") -> None:
    """Log an event marker for overlap analysis.

    The timestamp uses time.perf_counter_ns() so start/end events can be
    correlated across threads in the same process.
    """
    if not _profile_enabled():
        return
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    msg = (
        f"[ASYNC_CHUNK_EVENT] {tag} {phase} "
        f"ts_ns={time.perf_counter_ns()} tid={threading.get_ident()}"
    )
    if extra:
        msg += f" | {extra}"
    logger.info(msg)


@contextmanager
def async_chunk_timer(tag: str, extra: str = "") -> Generator[None, None, None]:
    """Context manager to time a block and log with [ASYNC_CHUNK_PROFILE]."""
    if not _profile_enabled():
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - t0) * 1000
        log_async_chunk_profile(tag, ms, extra)
