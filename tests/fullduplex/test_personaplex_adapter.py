# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex adapter driven through the real DuplexRuntime in continuous mode.

Proves the generic framework expresses pure lockstep: the adapter declares
``continuous=True``, the runtime starts ONE response, streams one agent frame per
user frame, and drains on close — all with a GPU-free stub stepper.
"""

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.core import protocol as ev
from vllm_omni.experimental.fullduplex.core.runtime import DuplexRuntime
from vllm_omni.experimental.fullduplex.core.session import DuplexSession, DuplexSessionConfig
from vllm_omni.experimental.fullduplex.personaplex.adapter import PersonaPlexDuplexAdapter
from vllm_omni.experimental.fullduplex.personaplex.config import FRAME_SIZE
from vllm_omni.experimental.fullduplex.personaplex.engine import FrameOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class StubStepper:
    """Deterministic FrameStepper: frame i -> audio filled with i, text 't{i}' on odds."""

    sample_rate = 24000
    frame_size = FRAME_SIZE

    def __init__(self) -> None:
        self.opened: tuple[str | None, str | None] | None = None
        self.steps = 0

    def open_session(self, voice_prompt=None, persona=None) -> None:
        self.opened = (voice_prompt, persona)
        self.steps = 0

    def step(self, user_pcm):
        assert user_pcm.shape[0] == self.frame_size
        self.steps += 1
        i = self.steps
        return FrameOutput(
            audio=np.full(self.frame_size, float(i), dtype=np.float32),
            text=(f"t{i}" if i % 2 else None),
        )


def _collector():
    out: list[dict] = []

    async def emit(event: dict) -> None:
        out.append(event)

    return out, emit


async def _feed(events):
    for e in events:
        yield e


def _continuous_session():
    cfg = DuplexSessionConfig(
        input_modalities=("audio",),
        output_modalities=("audio", "text"),
        continuous=True,
    )
    return DuplexSession("s", cfg)


@pytest.mark.asyncio
async def test_lockstep_one_agent_frame_per_user_frame():
    stub = StubStepper()
    adapter = PersonaPlexDuplexAdapter(stub, voice_prompt="NATF2.pt", persona="be terse")
    rt = DuplexRuntime(_continuous_session(), adapter)
    out, emit = _collector()

    frame = np.zeros(FRAME_SIZE, dtype=np.float32)
    events = [{"type": ev.INPUT_APPEND, "modality": "audio", "data": frame} for _ in range(3)]
    events.append({"type": ev.CLOSE})
    await rt.run(_feed(events), emit)

    types = [e["type"] for e in out]
    # exactly one response for the whole session, drained cleanly on close
    assert types.count(ev.RESPONSE_CREATED) == 1
    assert types.count(ev.RESPONSE_DONE) == 1
    audio = [e for e in out if e["type"] == ev.RESPONSE_DELTA and e["modality"] == "audio"]
    text = [e["data"] for e in out if e["type"] == ev.RESPONSE_DELTA and e["modality"] == "text"]
    assert len(audio) == 3  # one agent frame per user frame
    assert [int(e["data"][0]) for e in audio] == [1, 2, 3]  # lockstep order
    assert text == ["t1", "t3"]
    assert stub.steps == 3
    assert stub.opened == ("NATF2.pt", "be terse")


@pytest.mark.asyncio
async def test_unaligned_chunks_are_reframed_to_80ms():
    stub = StubStepper()
    adapter = PersonaPlexDuplexAdapter(stub)
    rt = DuplexRuntime(_continuous_session(), adapter)
    out, emit = _collector()

    # 2.5 frames in one chunk + 0.5 frame in another => 3 whole frames
    events = [
        {"type": ev.INPUT_APPEND, "modality": "audio", "data": np.zeros(FRAME_SIZE * 5 // 2, dtype=np.float32)},
        {"type": ev.INPUT_APPEND, "modality": "audio", "data": np.zeros(FRAME_SIZE // 2, dtype=np.float32)},
        {"type": ev.CLOSE},
    ]
    await rt.run(_feed(events), emit)
    audio = [e for e in out if e["type"] == ev.RESPONSE_DELTA and e["modality"] == "audio"]
    assert len(audio) == 3
    assert stub.steps == 3


@pytest.mark.asyncio
async def test_close_without_input_is_clean():
    adapter = PersonaPlexDuplexAdapter(StubStepper())
    rt = DuplexRuntime(_continuous_session(), adapter)
    out, emit = _collector()
    await rt.run(_feed([{"type": ev.CLOSE}]), emit)
    # no input -> response never started -> no created/done, no error
    assert not any(e["type"] == ev.ERROR for e in out)
    assert not any(e["type"] == ev.RESPONSE_CREATED for e in out)
