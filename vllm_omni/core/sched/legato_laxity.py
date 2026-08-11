# SPDX-License-Identifier: Apache-2.0
"""Legato controller v2: laxity-ordered scheduling for async-chunk codec stages.

Stage-1 (Code2Wav) requests are per-stream and long-lived: each client stream
holds one request whose prompt is replaced chunk-by-chunk by the transfer
adapter.  Under sustained load the stage is deadline-blind FCFS at two points
in ``OmniGenerationScheduler.schedule()``:

  * admission: which waiting streams get one of ``max_num_seqs`` slots, and
  * allocation: which running streams' chunk tokens fit the step token budget.

This module ranks streams by playback laxity, computable stage-locally:

    laxity = frames_decoded * SECONDS_PER_FRAME - (now - first_decode_time)

``frames_decoded`` counts NEW audio frames dispatched to the codec (the
left-context frames re-sent with every chunk window are excluded via the
``meta.left_context_size`` the producer attaches to each chunk payload).
Streams that have never decoded rank at laxity 0.  Lowest laxity first.

Enabled only when the ``VLLM_OMNI_LEGATO_LAXITY`` environment variable is
truthy; otherwise the scheduler behaves exactly as before (tracker is None).
"""

from __future__ import annotations

import os
import time
from typing import Any

from vllm.logger import init_logger
from vllm.v1.request import Request

logger = init_logger(__name__)

_ENV_FLAG = "VLLM_OMNI_LEGATO_LAXITY"
_TRUTHY = ("1", "true", "yes", "on")

# 12.5 Hz codec: one frame is 80 ms of audio.
_SECONDS_PER_FRAME = 0.08
# Codebook-major flat codec stream: tokens per frame.  Resolved from the
# model config when available; Qwen3-TTS-12Hz uses 16 code groups.
_DEFAULT_NUM_CODE_GROUPS = 16


def _meta_int(value: Any) -> int:
    """Coerce a chunk-meta scalar (int or 0-d/1-el tensor) to int."""
    if value is None:
        return 0
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return int(value.item())
        except (ValueError, RuntimeError):
            return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class LegatoLaxityTracker:
    """Per-stream laxity state for one async-chunk consumer scheduler."""

    @classmethod
    def maybe_create(cls, model_config: Any, adapter: Any) -> "LegatoLaxityTracker | None":
        if os.environ.get(_ENV_FLAG, "").lower() not in _TRUTHY:
            return None
        if adapter is None or not getattr(adapter, "receives_chunks", False):
            return None
        return cls(model_config, adapter)

    def __init__(self, model_config: Any, adapter: Any):
        self._adapter = adapter
        self._q = self._resolve_num_code_groups(model_config)
        # request_id -> monotonic time of first decode dispatch.
        self._first_sched: dict[str, float] = {}
        # request_id -> cumulative NEW audio frames dispatched to the codec.
        self._frames: dict[str, int] = {}
        # request_id -> adapter chunk counter at last credit (dedup guard).
        self._seen_chunk: dict[str, int] = {}
        logger.info(
            "[Legato] Stage-local laxity scheduling ENABLED (num_code_groups=%d, %.0f ms/frame)",
            self._q,
            _SECONDS_PER_FRAME * 1000.0,
        )

    @staticmethod
    def _resolve_num_code_groups(model_config: Any) -> int:
        hf = getattr(model_config, "hf_config", None)
        for holder in (hf, getattr(hf, "talker_config", None)):
            if holder is None:
                continue
            for attr in ("num_code_groups", "num_quantizers"):
                value = getattr(holder, attr, None)
                if isinstance(value, int) and value > 0:
                    return value
        return _DEFAULT_NUM_CODE_GROUPS

    def laxity(self, request: Request, now: float) -> float:
        """Playback lead of this stream in seconds; 0.0 with no history."""
        start = self._first_sched.get(request.request_id)
        if start is None:
            return 0.0
        frames = self._frames.get(request.request_id, 0)
        return frames * _SECONDS_PER_FRAME - (now - start)

    def order_queues(self, running: list[Request], waiting_queue: Any = None) -> None:
        """Sort scheduling queues lowest-laxity-first (stable; FCFS ties).

        ``running`` order decides step-batch allocation under the token
        budget (and, at end of schedule, which stream the adapter's
        over-cap tail preemption evicts: the last = highest laxity).
        ``waiting_queue`` order decides admission under ``max_num_seqs``.
        """
        now = time.monotonic()
        if len(running) > 1:
            running.sort(key=lambda r: self.laxity(r, now))
        if waiting_queue is not None and len(waiting_queue) > 1:
            reqs = list(waiting_queue)
            ordered = sorted(reqs, key=lambda r: self.laxity(r, now))
            if ordered != reqs:
                waiting_queue.remove_requests(reqs)
                for req in ordered:
                    waiting_queue.add_request(req)

    def note_scheduled(self, req_ids: Any, requests: dict[str, Request]) -> None:
        """Credit newly dispatched chunks after a schedule() pass.

        Called with the step's scheduled request ids.  A chunk is credited
        once (guarded by the adapter's per-request chunk counter) with
        ``len(prompt)//Q - left_context_size`` new frames.
        """
        now = time.monotonic()
        get_req_chunk = getattr(self._adapter, "get_req_chunk", {})
        for rid in req_ids:
            request = requests.get(rid)
            if request is None:
                continue
            chunk_idx = get_req_chunk.get(rid, 0)
            if self._seen_chunk.get(rid) == chunk_idx:
                continue
            self._seen_chunk[rid] = chunk_idx
            self._first_sched.setdefault(rid, now)
            prompt = request.prompt_token_ids or ()
            frames = len(prompt) // self._q
            new_frames = frames - self._left_context_frames(request)
            if new_frames > 0:
                self._frames[rid] = self._frames.get(rid, 0) + new_frames

    @staticmethod
    def _left_context_frames(request: Request) -> int:
        info = getattr(request, "additional_information", None)
        if not isinstance(info, dict):
            return 0
        meta = info.get("meta")
        if not isinstance(meta, dict):
            return 0
        return max(0, _meta_int(meta.get("left_context_size")))

    def free(self, request_id: str) -> None:
        """Drop per-stream state once the request is freed."""
        self._first_sched.pop(request_id, None)
        self._frames.pop(request_id, None)
        self._seen_chunk.pop(request_id, None)
