# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex duplex adapter — the lockstep model on the generic framework seam.

PersonaPlex is the canonical *continuous* (frame-clocked, no turn-taking) consumer
of ``core``: one eternal ``respond()`` that pulls one 80 ms user frame at a time and
emits one agent frame + text token. It declares ``continuous=True`` so
``DuplexRuntime`` runs the start-once / drain-on-close lifecycle instead of the
turn-style start/cancel-per-trigger one. Stepping happens off the event loop
(``asyncio.to_thread``, as Moshi's own server does) so the input side stays live —
that is what makes barge-in work without any explicit flush.

Depends only on the ``FrameStepper`` protocol, so it is exercised by GPU-free tests
with a stub stepper; ``PersonaPlexEngine`` is the real backend.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from vllm_omni.experimental.fullduplex.core.adapter import DuplexAdapter, DuplexCapability, OutputChunk
from vllm_omni.experimental.fullduplex.core.session import DuplexSession
from vllm_omni.experimental.fullduplex.personaplex.engine import FrameStepper

_CLOSE = object()  # sentinel: drain the inbox and end the eternal response


class PersonaPlexDuplexAdapter(DuplexAdapter):
    """Drive a :class:`FrameStepper` as a continuous full-duplex response."""

    def __init__(
        self,
        stepper: FrameStepper,
        *,
        voice_prompt: str | None = None,
        persona: str | None = None,
    ) -> None:
        self._stepper = stepper
        self._voice_prompt = voice_prompt
        self._persona = persona
        self._frame_size = stepper.frame_size
        self._inbox: asyncio.Queue[Any] = asyncio.Queue()
        self._pending = np.zeros(0, dtype=np.float32)
        self._opened = False

    def capabilities(self) -> DuplexCapability:
        return DuplexCapability(
            input_modalities=frozenset({"audio"}),
            output_modalities=frozenset({"audio", "text"}),
            proactive=True,
            continuous=True,
        )

    async def on_input(self, session: DuplexSession, modality: str, data: Any) -> None:
        """Frame incoming PCM into exact 80 ms windows and queue them for the loop."""
        if modality != "audio":
            return
        samples = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
        self._pending = np.concatenate([self._pending, samples]) if self._pending.size else samples
        n = self._frame_size
        while self._pending.shape[0] >= n:
            frame, self._pending = self._pending[:n], self._pending[n:]
            await self._inbox.put(frame)

    async def respond(self, session: DuplexSession) -> AsyncIterator[OutputChunk]:
        """The eternal lockstep loop: one user frame in, one agent frame + text out."""
        if not self._opened:
            await asyncio.to_thread(self._stepper.open_session, self._voice_prompt, self._persona)
            self._opened = True
        while True:
            frame = await self._inbox.get()
            if frame is _CLOSE:
                break
            out = await asyncio.to_thread(self._stepper.step, frame)
            if out.audio is not None:
                yield OutputChunk("audio", out.audio)
            if out.text:
                yield OutputChunk("text", out.text)

    async def on_close(self, session: DuplexSession) -> None:
        await self._inbox.put(_CLOSE)

    async def on_barge_in(self, session: DuplexSession) -> None:
        # No-op: Moshi-class models perceive the user continuously, so an interrupt
        # needs no downstream flush — the next frames simply steer the model.
        return
