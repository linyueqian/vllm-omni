# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the PersonaPlex full-duplex backend.

PersonaPlex (``nvidia/personaplex-7b-v1``) is a Moshi finetune: a pure-lockstep
full-duplex speech-to-speech model running at the Mimi codec frame rate
(12.5 Hz / 80 ms). One config drives both the offline driver and the duplex
adapter. All values mirror PersonaPlex's ``offline.py`` defaults so the wrapper
stays bit-for-bit faithful to the reference loop.
"""

from __future__ import annotations

from dataclasses import dataclass

# Mimi codec constants (loaders.py: SAMPLE_RATE / FRAME_RATE). Fixed by the
# pretrained checkpoint; exposed here only so the rest of the package never
# hard-codes them.
SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)  # 1920 samples per 80 ms frame

# Default assistant persona shipped with PersonaPlex (offline.py:335).
DEFAULT_PERSONA = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."


@dataclass(frozen=True)
class PersonaPlexConfig:
    """Immutable session configuration for a PersonaPlex conversation.

    Attributes:
        hf_repo: HuggingFace repo holding the weights, Mimi codec and tokenizer.
        voice_prompt: Voice-clone reference. Either a bundled basename
            (``"NATF2.pt"`` / ``"NATM1.pt"`` from ``voices.tgz``) or a path to a
            ``.pt`` embedding bundle or a reference ``.wav``.
        persona: System role text; injected as ``<system> ... <system>`` into the
            inner-monologue stream at session start.
        device: Torch device for the backend (``"cuda"`` / ``"cpu"``).
        cpu_offload: Offload LM layers to CPU when GPU memory is tight (needs
            ``accelerate``).
        greedy: Disable sampling (deterministic decoding).
        temp_audio / temp_text: Sampling temperatures for the audio depformer /
            inner-monologue text head.
        topk_audio / topk_text: Top-k for the audio / text heads.
        seed: RNG seed; ``None`` leaves global RNG untouched.
    """

    hf_repo: str = "nvidia/personaplex-7b-v1"
    voice_prompt: str = "NATF2.pt"
    persona: str = DEFAULT_PERSONA

    device: str = "cuda"
    cpu_offload: bool = False

    greedy: bool = False
    temp_audio: float = 0.8
    temp_text: float = 0.7
    topk_audio: int = 250
    topk_text: int = 25
    seed: int | None = None

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def frame_size(self) -> int:
        return FRAME_SIZE
