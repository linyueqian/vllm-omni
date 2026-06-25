# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex inference backend (the real Moshi/Mimi wrapper).

This is the only module that imports the external ``moshi`` package (PersonaPlex's
vendored fork, ``pip install moshi/.`` from the NVIDIA/personaplex repo). It wraps
the exact load + per-frame lockstep loop from ``moshi.offline`` behind a narrow,
GPU-free-testable seam: ``FrameStepper`` — *one 80 ms user frame in, one agent
frame + text token out*. The duplex adapter and the session driver depend only on
that protocol, so they import without ``moshi``, ``torch`` or a GPU.

Reference: ``moshi/moshi/offline.py::run_inference`` (the canonical PersonaPlex loop).
"""

from __future__ import annotations

import logging
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from vllm_omni.experimental.fullduplex.personaplex.config import PersonaPlexConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Inner-monologue token ids that carry no displayable text (offline.py:287).
_NON_TEXT_TOKEN_IDS = frozenset({0, 3})  # EPAD, PAD


@dataclass(frozen=True)
class FrameOutput:
    """Result of stepping the model by exactly one 80 ms frame.

    Attributes:
        audio: Agent PCM for this frame (``frame_size`` mono float32 samples), or
            ``None`` during the model's initial delay/warmup frames.
        text: Inner-monologue text piece for this frame, or ``None`` when the
            model emitted a padding token (silence between words).
    """

    audio: NDArray[np.float32] | None
    text: str | None


@runtime_checkable
class FrameStepper(Protocol):
    """The single seam between PersonaPlex and the duplex framework.

    A stepper consumes one user audio frame and returns one agent frame. It owns
    all conversation state (KV caches / streaming buffers); callers never re-feed
    history. ``PersonaPlexEngine`` is the real implementation; tests pass a stub.
    """

    sample_rate: int
    frame_size: int

    def open_session(self, voice_prompt: str | None = None, persona: str | None = None) -> None:
        """Reset streaming state and inject the voice clone + persona prompt."""
        ...

    def step(self, user_pcm: NDArray[np.float32]) -> FrameOutput:
        """Advance the conversation by one frame of exactly ``frame_size`` samples."""
        ...


class PersonaPlexEngine:
    """Real PersonaPlex backend wrapping ``moshi`` LMGen + two Mimi codecs.

    Lifecycle mirrors ``offline.py``: ``load()`` (weights + warmup) once, then
    ``open_session()`` per conversation, then ``step()`` per frame. Construction is
    cheap; the heavy work happens in ``load()`` so the class can be imported and
    introspected without a GPU.
    """

    def __init__(self, config: PersonaPlexConfig | None = None) -> None:
        self.config = config or PersonaPlexConfig()
        self.sample_rate = self.config.sample_rate
        self.frame_size = self.config.frame_size
        self._loaded = False
        # moshi handles, populated by load(); typed loosely to avoid importing moshi here.
        self._mimi = None
        self._other_mimi = None
        self._lm_gen = None
        self._tokenizer = None

    # -- load -------------------------------------------------------------

    def load(self) -> PersonaPlexEngine:
        """Load Mimi, the Moshi LM and tokenizer, build LMGen, and warm up graphs."""
        if self._loaded:
            return self
        try:
            import sentencepiece
            import torch
            from huggingface_hub import hf_hub_download
            from moshi.models import LMGen, loaders
        except ImportError as exc:  # pragma: no cover - exercised only without moshi
            raise ImportError(
                "PersonaPlexEngine needs the PersonaPlex 'moshi' fork. Install it with "
                "`pip install moshi/.` from a clone of https://github.com/NVIDIA/personaplex"
            ) from exc

        cfg = self.config
        if cfg.seed is not None and cfg.seed >= 0:
            _seed_all(cfg.seed)

        hf_hub_download(cfg.hf_repo, "config.json")  # bump the HF download counter (offline.py:187)

        logger.info("PersonaPlex: loading Mimi codec")
        mimi_weight = hf_hub_download(cfg.hf_repo, loaders.MIMI_NAME)
        self._mimi = loaders.get_mimi(mimi_weight, cfg.device)
        self._other_mimi = loaders.get_mimi(mimi_weight, cfg.device)

        tokenizer_path = hf_hub_download(cfg.hf_repo, loaders.TEXT_TOKENIZER_NAME)
        self._tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

        logger.info("PersonaPlex: loading Moshi LM")
        moshi_weight = hf_hub_download(cfg.hf_repo, loaders.MOSHI_NAME)
        lm = loaders.get_moshi_lm(moshi_weight, device=cfg.device, cpu_offload=cfg.cpu_offload)
        lm.eval()

        self._lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * self._mimi.frame_rate),
            sample_rate=self._mimi.sample_rate,
            device=cfg.device,
            frame_rate=self._mimi.frame_rate,
            save_voice_prompt_embeddings=False,
            use_sampling=not cfg.greedy,
            temp=cfg.temp_audio,
            temp_text=cfg.temp_text,
            top_k=cfg.topk_audio,
            top_k_text=cfg.topk_text,
        )
        self._mimi.streaming_forever(1)
        self._other_mimi.streaming_forever(1)
        self._lm_gen.streaming_forever(1)

        logger.info("PersonaPlex: warming up CUDA graphs")
        self._warmup(torch)
        self._loaded = True
        return self

    def _warmup(self, torch) -> None:
        """Prime CUDA graphs + streaming state (offline.py:warmup)."""
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.config.device)
            codes = self._mimi.encode(chunk)
            self._other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self._lm_gen.step(codes[:, :, c : c + 1])
                if tokens is not None:
                    self._mimi.decode(tokens[:, 1:9])
                    self._other_mimi.decode(tokens[:, 1:9])
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # -- session ----------------------------------------------------------

    def open_session(self, voice_prompt: str | None = None, persona: str | None = None) -> None:
        """Reset all streaming state and force the voice clone + persona into KV.

        Replays ``offline.py`` steps 6-7: load the voice prompt (forces the agent
        audio stream), set the persona text tokens, reset streaming, then
        ``step_system_prompts`` (voice -> 0.5 s silence -> persona -> 0.5 s silence).
        """
        if not self._loaded:
            self.load()

        voice = voice_prompt or self.config.voice_prompt
        persona_text = self.config.persona if persona is None else persona

        voice_path = self._resolve_voice_prompt(voice)
        if voice_path.endswith(".pt"):
            self._lm_gen.load_voice_prompt_embeddings(voice_path)
        else:
            self._lm_gen.load_voice_prompt(voice_path)
        self._lm_gen.text_prompt_tokens = (
            self._tokenizer.encode(_wrap_with_system_tags(persona_text)) if persona_text else None
        )

        self._mimi.reset_streaming()
        self._other_mimi.reset_streaming()
        self._lm_gen.reset_streaming()
        self._lm_gen.step_system_prompts(self._mimi)
        self._mimi.reset_streaming()  # drop the voice-encode state before the live loop

    # -- step -------------------------------------------------------------

    def step(self, user_pcm: NDArray[np.float32]) -> FrameOutput:
        """One lockstep frame: user PCM in, agent PCM + text out (offline.py:267-295)."""
        if not self._loaded:
            raise RuntimeError("PersonaPlexEngine.open_session() must be called before step()")
        import torch

        pcm = np.ascontiguousarray(user_pcm, dtype=np.float32).reshape(-1)
        if pcm.shape[0] != self.frame_size:
            raise ValueError(f"expected {self.frame_size} samples per frame, got {pcm.shape[0]}")

        chunk = torch.from_numpy(pcm).to(self.config.device).reshape(1, 1, self.frame_size)
        codes = self._mimi.encode(chunk)

        audio: NDArray[np.float32] | None = None
        text: str | None = None
        for c in range(codes.shape[-1]):
            tokens = self._lm_gen.step(codes[:, :, c : c + 1])
            if tokens is None:  # initial delay frames prime the cache
                continue
            pcm_out = self._mimi.decode(tokens[:, 1:9])
            self._other_mimi.decode(tokens[:, 1:9])
            audio = pcm_out.detach().cpu().numpy()[0, 0].astype(np.float32)
            text = self._decode_text(int(tokens[0, 0, 0].item()))
        return FrameOutput(audio=audio, text=text)

    def _decode_text(self, token_id: int) -> str | None:
        if token_id in _NON_TEXT_TOKEN_IDS:
            return None
        return self._tokenizer.id_to_piece(token_id).replace("▁", " ")

    # -- helpers ----------------------------------------------------------

    def _resolve_voice_prompt(self, voice: str) -> str:
        """Resolve a voice basename to a path, downloading ``voices.tgz`` if needed."""
        if os.path.exists(voice):
            return voice
        from huggingface_hub import hf_hub_download

        voices_tgz = Path(hf_hub_download(self.config.hf_repo, "voices.tgz"))
        voices_dir = voices_tgz.parent / "voices"
        if not voices_dir.exists():
            with tarfile.open(voices_tgz, "r:gz") as tar:
                tar.extractall(path=voices_tgz.parent)
        resolved = voices_dir / voice
        if not resolved.exists():
            raise FileNotFoundError(f"voice prompt '{voice}' not found under {voices_dir}")
        return str(resolved)


def _wrap_with_system_tags(text: str) -> str:
    """Wrap a persona string in the ``<system> ... <system>`` tags (offline.py:83)."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def _seed_all(seed: int) -> None:
    import random

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
