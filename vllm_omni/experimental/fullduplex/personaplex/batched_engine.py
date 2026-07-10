# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched PersonaPlex engine: B concurrent conversations on one model.

Extends :class:`PersonaPlexEngine` with a slot-oriented API on top of the elastic
moshi patches (``elastic.py``). All slots advance in strict lockstep (one shared
80 ms tick); a finished slot is recycled for a new caller by masking its KV
window and replaying its system prompt (voice embeddings + persona tokens)
per-row while the rest of the batch keeps conversing.

Slot lifecycle driven by the caller (see ``serving/batched.py``):

1. ``open_batch()`` once at boot: batch-wide system-prompt prefill (default voice
   + default persona), so pristine slots are live-ready immediately.
2. On connect with a custom persona (or to a previously used slot):
   ``recycle_slot(b)`` then feed the frames from ``prefill_steps(...)`` through
   ``step_batch`` one per tick; ``voice_end`` entries are executed in place (no
   tick). When exhausted, ``end_prefill(b)`` and the slot is live.
3. Every tick: ``step_batch(user_pcm, prefill)`` where ``user_pcm[b]`` is the
   live user frame (zeros when the client is silent / the slot idle).

The voice prompt must be a ``.pt`` embedding bundle (the format shipped in
``voices.tgz``); raw ``.wav`` voice prompts would need a Mimi encode inside the
live batch state and are not supported in batched mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm_omni.experimental.fullduplex.personaplex.engine import (
    FrameOutput,
    PersonaPlexEngine,
    _wrap_with_system_tags,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrefillStep:
    """One tick of a recycled slot's system-prompt replay.

    kind:
        ``"sacrifice"``: the mandatory first post-recycle tick. Its KV write is
            masked by ``reset_slot`` (+1); it aligns the recycled row with a fresh
            conversation's ``o == 0`` skipped tick.
        ``"voice"``: replay one stored voice-prompt embedding frame.
        ``"voice_end"``: not a tick; restores the staging cache row (ring-phase
            aligned) exactly like the stock embeddings path does at boot.
        ``"tokens"``: token-forced frame (silence / persona text).
    """

    kind: str
    embedding: Any = None  # [1, 1, H] torch tensor for kind == "voice"
    moshi_tokens: Any = None  # [8, 1] torch tensor for kind == "tokens"
    text_token: int | None = None
    user_sine: bool = False  # sine user channel (silence/persona); else dummy


class BatchedPersonaPlexEngine(PersonaPlexEngine):
    """Drive ``config.batch_size`` lockstep conversations on one PersonaPlex model."""

    def __init__(self, config=None) -> None:
        super().__init__(config)
        if self.config.batch_size < 2:
            raise ValueError("BatchedPersonaPlexEngine requires config.batch_size >= 2")
        self._voice_embeddings = None  # [T, 1, 1, H]
        self._voice_cache = None  # [1, K, CT]
        self._sine_frame = None  # [1, 8, 1]
        self._zero_frame = None  # [1, 8, 1] (silence)
        self._dummy_frame = None  # [1, 8, 1] (initial token)
        self._opened = False

    # -- boot ---------------------------------------------------------------

    def open_batch(self) -> None:
        """Reset streaming and prefill the default voice + persona on ALL rows."""
        import torch

        if not self._loaded:
            self.load()
        cfg = self.config
        B = cfg.batch_size
        lm_gen = self._lm_gen

        voice_path = self._resolve_voice_prompt(cfg.voice_prompt)
        if not voice_path.endswith(".pt"):
            raise ValueError(f"batched mode needs a .pt voice-embedding bundle (voices.tgz); got {voice_path!r}")
        lm_gen.load_voice_prompt_embeddings(voice_path)
        lm_gen.text_prompt_tokens = self._tokenizer.encode(_wrap_with_system_tags(cfg.persona)) if cfg.persona else None

        self._mimi.reset_streaming()
        self._other_mimi.reset_streaming()
        lm_gen.reset_streaming()
        if cfg.use_native_components:
            self._install_native_swap()

        self._sine_frame = lm_gen._encode_sine_frame()
        self._zero_frame = lm_gen._encode_zero_frame()
        initial = self._lm._get_initial_token()
        self._dummy_frame = initial[:, 1:9, :].clone()  # [1, 8, 1]
        self._voice_embeddings = lm_gen.voice_prompt_embeddings
        self._voice_cache = lm_gen.voice_prompt_cache

        # Batch-wide system prompt, mirroring LMGen.step_system_prompts at B:
        # voice embeddings -> staging-cache restore -> silence -> persona -> silence.
        with torch.no_grad():
            for next_embed in self._voice_embeddings:
                lm_gen.step_embeddings(next_embed.expand(B, -1, -1))
            lm_gen._streaming_state.cache.copy_(self._voice_cache)
            sine = self._sine_frame.expand(B, -1, -1)
            zero = self._zero_frame.expand(B, -1, -1)
            for _ in range(lm_gen.audio_silence_frame_cnt):
                lm_gen.step(input_tokens=sine, moshi_tokens=zero, text_token=lm_gen.zero_text_code)
            for tok in lm_gen.text_prompt_tokens or []:
                lm_gen.step(input_tokens=sine, moshi_tokens=zero, text_token=tok)
            for _ in range(lm_gen.audio_silence_frame_cnt):
                lm_gen.step(input_tokens=sine, moshi_tokens=zero, text_token=lm_gen.zero_text_code)

        self._mimi.reset_streaming()  # drop boot state before the live loop (offline.py parity)
        self._opened = True
        logger.info("PersonaPlex batched: %d slots prefilled (default voice + persona)", B)

    # -- slot lifecycle -------------------------------------------------------

    def prefill_steps(self, persona: str | None = None) -> list[PrefillStep]:
        """The per-tick system-prompt replay for one recycled slot."""
        lm_gen = self._lm_gen
        zero = self._zero_frame[0]  # [8, 1]
        silence = [
            PrefillStep("tokens", moshi_tokens=zero, text_token=lm_gen.zero_text_code, user_sine=True)
            for _ in range(lm_gen.audio_silence_frame_cnt)
        ]
        steps: list[PrefillStep] = [PrefillStep("sacrifice")]
        for e in self._voice_embeddings:
            steps.append(PrefillStep("voice", embedding=e))
        steps.append(PrefillStep("voice_end"))
        steps.extend(silence)
        text = persona if persona is not None else self.config.persona
        if text:
            for tok in self._tokenizer.encode(_wrap_with_system_tags(text)):
                steps.append(PrefillStep("tokens", moshi_tokens=zero, text_token=int(tok), user_sine=True))
        steps.extend(silence)
        return steps

    def recycle_slot(self, b: int) -> None:
        """Mask slot ``b``'s history so a new caller can reuse it (between ticks)."""
        self._lm_gen.reset_slot(b)

    def voice_end(self, b: int) -> None:
        """Restore the voice-prompt staging cache for row ``b``, ring-phase aligned.

        The stock embeddings path ends with ``state.cache.copy_(voice_prompt_cache)``
        at boot, where phys phase == the saved phase. A recycled row replays at an
        arbitrary phys offset, so the saved [K, CT] ring content is rolled by the
        row's phys-local phase difference before the row-wise copy.
        """
        import torch

        state = self._lm_gen._streaming_state
        ct = state.cache.shape[2]
        shift = int((state.phys_offset - int(state.local_offset[b])) % ct)
        state.cache[b] = torch.roll(self._voice_cache[0], shifts=shift, dims=-1)

    def end_prefill(self, b: int) -> None:
        """Give slot ``b`` fresh Mimi encode/decode state before its live loop."""
        self._mimi.reset_slot(b)
        self._other_mimi.reset_slot(b)

    # -- lockstep tick ---------------------------------------------------------

    def step_batch(
        self, user_pcm: NDArray[np.float32], prefill: dict[int, PrefillStep] | None = None
    ) -> list[FrameOutput | None]:
        """Advance every slot by one 80 ms frame.

        Args:
            user_pcm: ``[B, frame_size]`` float32 user audio (zeros for idle rows;
                prefill rows are overridden internally).
            prefill: per-slot replay frames for slots mid-recycle. Callers must
                not pass ``voice_end`` entries here (execute them via
                :meth:`voice_end` without consuming a tick).

        Returns:
            One :class:`FrameOutput` per slot, ``None`` for rows still inside the
            model's delay warmup (freshly recycled).
        """
        import torch

        if not self._opened:
            raise RuntimeError("open_batch() must be called before step_batch()")
        cfg = self.config
        B = cfg.batch_size
        lm_gen = self._lm_gen

        pcm = np.ascontiguousarray(user_pcm, dtype=np.float32).reshape(B, self.frame_size)
        chunk = torch.from_numpy(pcm).to(cfg.device).view(B, 1, self.frame_size)
        codes = self._mimi.encode(chunk)[:, :, 0:1]  # [B, 8, 1]

        if prefill:
            device = codes.device
            force_mask = torch.zeros(B, dtype=torch.bool, device=device)
            embed_rows = torch.zeros(B, dtype=torch.bool, device=device)
            emb_full = None
            moshi_forced = self._dummy_frame.expand(B, -1, -1).clone()
            text_forced = torch.full((B,), lm_gen.zero_text_code, dtype=torch.long, device=device)
            for b, stp in prefill.items():
                if stp.kind == "voice_end":
                    raise ValueError("voice_end is not a tick; call engine.voice_end(b) instead")
                codes[b] = (self._sine_frame if stp.user_sine else self._dummy_frame)[0]
                if stp.kind == "voice":
                    embed_rows[b] = True
                    if emb_full is None:
                        e0 = stp.embedding
                        emb_full = torch.zeros(B, 1, e0.shape[-1], dtype=e0.dtype, device=device)
                    emb_full[b] = stp.embedding[0]
                else:  # "tokens" or "sacrifice" (sacrifice keeps the dummy/zero forcing)
                    force_mask[b] = True
                    if stp.kind == "tokens":
                        moshi_forced[b] = stp.moshi_tokens
                        text_forced[b] = int(stp.text_token)
            if bool(embed_rows.any()):
                tokens = lm_gen.step_prefill_embedding(
                    emb_full,
                    embed_rows,
                    codes,
                    moshi_tokens=moshi_forced,
                    text_token=text_forced,
                    force_rows=force_mask,
                )
            else:
                tokens = lm_gen.step(codes, moshi_forced, text_forced, force_rows=force_mask)
        else:
            tokens = lm_gen.step(codes)

        if tokens is None:
            return [None] * B

        # Rows still in warmup (or mid-prefill) carry forced placeholder tokens;
        # the audio "initial" token (== card) is a valid LM input but OUT OF RANGE
        # for the Mimi codebook (card entries), so sanitize those rows to silence
        # tokens before the batched codec decode. Their audio is discarded anyway.
        valid = lm_gen.valid_mask()
        emit = [bool(valid[b]) and (not prefill or b not in prefill) for b in range(B)]
        audio_tokens = tokens[:, 1:9]
        if not all(emit):
            hide = torch.tensor([not e for e in emit], device=audio_tokens.device).view(B, 1, 1)
            audio_tokens = torch.where(hide, self._zero_frame.expand(B, -1, -1), audio_tokens)
        pcm_out = self._mimi.decode(audio_tokens)  # [B, 1, frame]
        self._other_mimi.decode(audio_tokens)
        outs: list[FrameOutput | None] = []
        for b in range(B):
            if not emit[b]:
                outs.append(None)
                continue
            audio = pcm_out[b, 0].detach().cpu().numpy().astype(np.float32)
            outs.append(FrameOutput(audio=audio, text=self._decode_text(int(tokens[b, 0, 0].item()))))
        return outs
