# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex full-duplex backend for the experimental fullduplex framework.

PersonaPlex (``nvidia/personaplex-7b-v1``) is a Moshi finetune: a pure-lockstep
speech-to-speech model. This package plugs it into the model-agnostic
``fullduplex/core`` seam without touching the engine:

- :class:`PersonaPlexConfig`   immutable session config (voice / persona / sampling)
- :class:`PersonaPlexEngine`   real Moshi/Mimi backend (``FrameStepper``)
- :class:`PersonaPlexSession`  lockstep driver (the runnable serving primitive)
- :class:`PersonaPlexDuplexAdapter`  the ``core.DuplexAdapter`` for lockstep audio
- :class:`PersonaPlexDuplexRuntime`  the model-owned lockstep lifecycle
  (start-once / drain-on-close; ``core`` stays verbatim upstream)
- :class:`NativePersonaPlexEngine`  the moshi-free lockstep engine (single
  session and B concurrent slots with elastic per-slot recycle)
"""

from vllm_omni.experimental.fullduplex.personaplex.adapter import (
    PersonaPlexDuplexAdapter,
    PersonaPlexDuplexRuntime,
)
from vllm_omni.experimental.fullduplex.personaplex.config import PersonaPlexConfig
from vllm_omni.experimental.fullduplex.personaplex.engine import (
    FrameOutput,
    FrameStepper,
    PersonaPlexEngine,
)
from vllm_omni.experimental.fullduplex.personaplex.runtime import (
    NativePersonaPlexEngine,
    PrefillStep,
)
from vllm_omni.experimental.fullduplex.personaplex.session import PersonaPlexSession

__all__ = [
    "NativePersonaPlexEngine",
    "FrameOutput",
    "FrameStepper",
    "PersonaPlexConfig",
    "PersonaPlexDuplexAdapter",
    "PersonaPlexDuplexRuntime",
    "PersonaPlexEngine",
    "PersonaPlexSession",
    "PrefillStep",
]
