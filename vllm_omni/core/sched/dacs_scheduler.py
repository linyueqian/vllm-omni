"""Deadline-Aware Chunk Scheduler (DACS) for streaming TTS serving.

Extends OmniGenerationScheduler with gap-urgency-based priority ordering.
When multiple requests have ready chunks at the Code2Wav stage, DACS
prioritizes the request whose last audio emission was longest ago,
preventing temporal continuity SLO violations.

Reference: StreamSched spec §4 (DACS algorithm).
"""

import time
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

logger = init_logger(__name__)


class DACSScheduler(OmniGenerationScheduler):
    """Gap-urgency-aware scheduler for the Code2Wav stage.

    Tracks per-request last-emit timestamps and reorders the running queue
    so that the request with the highest gap urgency is scheduled first.
    Falls back to FIFO when gap urgencies are equal.

    Config (via env vars for now):
        DACS_GAP_BUDGET: perceptual gap tolerance in seconds (default: 0.5)
        DACS_TTFA_WEIGHT: weight for first-chunk urgency (default: 0.8)
        DACS_GAP_WEIGHT: weight for gap urgency (default: 1.0)
        DACS_AGE_WEIGHT: weight for chunk age (default: 0.1)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        import os

        self.gap_budget = float(os.environ.get("DACS_GAP_BUDGET", "0.5"))
        self.ttfa_weight = float(os.environ.get("DACS_TTFA_WEIGHT", "0.8"))
        self.gap_weight = float(os.environ.get("DACS_GAP_WEIGHT", "1.0"))
        self.age_weight = float(os.environ.get("DACS_AGE_WEIGHT", "0.1"))

        # Per-request tracking
        self._last_emit_time: dict[str, float] = {}
        self._first_chunk_emitted: set[str] = set()
        self._request_enqueue_time: dict[str, float] = {}

        logger.info(
            "DACS scheduler initialized: gap_budget=%.3f, weights=(gap=%.2f, ttfa=%.2f, age=%.2f)",
            self.gap_budget,
            self.gap_weight,
            self.ttfa_weight,
            self.age_weight,
        )

    def _compute_priority(self, request: Request) -> float:
        """Compute scheduling priority for a request. Higher = scheduled first."""
        now = time.monotonic()
        req_id = request.request_id

        # Gap urgency: how long since last audio emission for this request
        gap_urgency = 0.0
        if req_id in self._last_emit_time:
            elapsed = now - self._last_emit_time[req_id]
            gap_urgency = min(1.0, elapsed / self.gap_budget)

        # TTFA urgency: first chunk not yet emitted
        ttfa_urgency = 0.0 if req_id in self._first_chunk_emitted else 1.0

        # Age: how long since request was enqueued
        age = 0.0
        if req_id in self._request_enqueue_time:
            age = min(1.0, (now - self._request_enqueue_time[req_id]) / 2.0)

        return self.gap_weight * gap_urgency + self.ttfa_weight * ttfa_urgency + self.age_weight * age

    def schedule(self) -> SchedulerOutput:
        """Override to reorder running queue by DACS priority before scheduling."""
        # Track enqueue time for new requests
        now = time.monotonic()
        for request in self.waiting:
            if request.request_id not in self._request_enqueue_time:
                self._request_enqueue_time[request.request_id] = now

        # Reorder running queue by priority (highest first)
        if len(self.running) > 1:
            self.running.sort(
                key=lambda r: self._compute_priority(r),
                reverse=True,
            )

        return super().schedule()

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: Any,
    ) -> dict[int, Any]:
        """Override to track per-request emit timestamps."""
        now = time.monotonic()

        # Record emit time for all scheduled requests (they just produced output)
        for req_id in scheduler_output.num_scheduled_tokens:
            self._last_emit_time[req_id] = now
            self._first_chunk_emitted.add(req_id)

        result = super().update_from_output(scheduler_output, model_runner_output)

        # Clean up finished requests
        if scheduler_output.finished_req_ids:
            for req_id in scheduler_output.finished_req_ids:
                self._last_emit_time.pop(req_id, None)
                self._first_chunk_emitted.discard(req_id)
                self._request_enqueue_time.pop(req_id, None)

        return result
