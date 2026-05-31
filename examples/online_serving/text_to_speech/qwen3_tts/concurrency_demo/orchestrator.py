"""Async orchestrator that fires N concurrent /v1/audio/speech streams.

Reads PCM bytes from each response, emits StreamEvents into the aggregator,
and computes a c=1 reference latency for the serial-ETA badge.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Literal

import httpx

from .metrics import MetricsAggregator, StreamEvent

PayloadKind = Literal["reference", "parallel"]


@dataclass(frozen=True)
class StreamConfig:
    stream_id: int
    text: str
    payload_kind: PayloadKind


class Orchestrator:
    """Owns an httpx.AsyncClient and a worker asyncio loop."""

    def __init__(
        self,
        api_base: str,
        ref_audio_b64: str,
        *,
        timeout_s: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._ref_audio_b64 = ref_audio_b64
        self._timeout_s = timeout_s
        self._transport = transport

    def _build_payload(self, text: str) -> dict:
        return {
            "input": text,
            "response_format": "pcm",
            "stream": True,
            "task_type": "Base",
            "ref_audio": f"data:audio/wav;base64,{self._ref_audio_b64}",
        }

    async def _run_one(self, cfg: StreamConfig, aggregator: MetricsAggregator) -> float:
        """Runs a single streaming request. Returns wall-clock duration.

        Emits events into the aggregator; on error, emits a single error event
        and re-raises only if it is not a transport-level failure.
        """
        sent_at = time.perf_counter()
        aggregator.mark_request_sent(stream_id=cfg.stream_id, ts=sent_at)
        payload = self._build_payload(cfg.text)
        first_seen = False
        bytes_total = 0
        async with httpx.AsyncClient(
            timeout=self._timeout_s,
            transport=self._transport,
        ) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self._api_base}/v1/audio/speech",
                    json=payload,
                    headers={"Authorization": "Bearer EMPTY"},
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        msg = f"HTTP {response.status_code}: {body[:200]!r}"
                        aggregator.apply(StreamEvent.error(cfg.stream_id, time.perf_counter(), msg))
                        return time.perf_counter() - sent_at
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        now = time.perf_counter()
                        if not first_seen:
                            aggregator.apply(StreamEvent.first(cfg.stream_id, now))
                            first_seen = True
                        aggregator.apply(StreamEvent.chunk(cfg.stream_id, now, byte_count=len(chunk)))
                        bytes_total += len(chunk)
            except (httpx.HTTPError, asyncio.CancelledError) as exc:
                aggregator.apply(StreamEvent.error(cfg.stream_id, time.perf_counter(), str(exc)))
                return time.perf_counter() - sent_at
        aggregator.apply(StreamEvent.done(cfg.stream_id, time.perf_counter()))
        return time.perf_counter() - sent_at
