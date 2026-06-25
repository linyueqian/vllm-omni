# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex full-duplex backend for the experimental fullduplex framework.

PersonaPlex (``nvidia/personaplex-7b-v1``) is a Moshi finetune: a pure-lockstep
speech-to-speech model. This package plugs it into the model-agnostic
``fullduplex/core`` seam without touching the engine:

- :class:`PersonaPlexConfig`   immutable session config (voice / persona / sampling)
- :class:`PersonaPlexEngine`   real Moshi/Mimi backend (``FrameStepper``)
- :class:`PersonaPlexSession`  lockstep driver (the runnable serving primitive)
- :class:`PersonaPlexDuplexAdapter`  the ``core.DuplexAdapter`` (continuous mode)
"""

from vllm_omni.experimental.fullduplex.personaplex.adapter import PersonaPlexDuplexAdapter
from vllm_omni.experimental.fullduplex.personaplex.config import PersonaPlexConfig
from vllm_omni.experimental.fullduplex.personaplex.engine import (
    FrameOutput,
    FrameStepper,
    PersonaPlexEngine,
)
from vllm_omni.experimental.fullduplex.personaplex.session import PersonaPlexSession

__all__ = [
    "FrameOutput",
    "FrameStepper",
    "PersonaPlexConfig",
    "PersonaPlexDuplexAdapter",
    "PersonaPlexEngine",
    "PersonaPlexSession",
]
