# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One AuK DiT denoise step over several requests at once.

Each request keeps its own conditioning, time grid and Euler state, so
requests of different lengths, guidance, step counts and progress share a
forward. Rows are laid out request by request, the cond row then, under CFG,
the uncond row; every sequence (text, reference, target) is right-padded to
the batch's bucket and masked, and each row reads the adaLN modulations of its
own request's current step.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

from vllm_omni.diffusion.models.auk.auk_transformer import (
    AuKStepContext,
    AuKTransformer,
    _key_padding_bias,
    _rope_cos_sin,
    joint_key_mask,
)

logger = init_logger(__name__)

# Guidance below this runs the cond branch only, as in the single-request sampler.
CFG_EPSILON = 1e-5
# Batch rows are padded up to one of these so a few graphs cover every batch size.
_ROW_BUCKETS = (1, 2, 4, 6, 8, 12, 16, 24, 32)
_TEXT_ALIGNMENT = 32
_REF_ALIGNMENT = 50
_TARGET_ALIGNMENT = 32

_request_uids = itertools.count()


@dataclass
class AuKBatchRequest:
    """A request's conditioning and ODE state, built once when it is admitted.

    ``text`` and ``ref`` are the raw conditioning the single-request graph
    path consumes; ``c``, ``prompt`` and ``prompt_uncond`` are their DiT
    embeddings for the batched path. ``latents`` is ``[1, frames, latent_dim]``
    and advances along ``grid`` one Euler step at a time.
    """

    uid: int
    text: torch.Tensor
    ref: torch.Tensor
    c: torch.Tensor
    prompt: torch.Tensor | None
    prompt_uncond: torch.Tensor | None
    cfg: float
    grid: torch.Tensor
    modulation: torch.Tensor
    latents: torch.Tensor
    step_index: int = 0

    @property
    def guided(self) -> bool:
        return self.cfg >= CFG_EPSILON

    @property
    def rows(self) -> int:
        return 2 if self.guided else 1

    @property
    def num_steps(self) -> int:
        return int(self.grid.numel()) - 1

    @property
    def done(self) -> bool:
        return self.step_index >= self.num_steps

    @property
    def target_frames(self) -> int:
        return int(self.latents.shape[1])

    @property
    def ref_frames(self) -> int:
        return int(self.ref.shape[1])

    def advance(self, velocity: torch.Tensor) -> None:
        """One Euler step along the request's own grid."""
        i = self.step_index
        self.latents = self.latents + (self.grid[i + 1] - self.grid[i]) * velocity
        self.step_index = i + 1


def row_bucket(rows: int) -> int:
    for bucket in _ROW_BUCKETS:
        if rows <= bucket:
            return bucket
    return round_up(rows, _ROW_BUCKETS[-1])


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_x: torch.Tensor
    static_ctx: AuKStepContext
    static_out: torch.Tensor


class AuKBatchedStepRunner:
    """Velocities of several requests from one batched DiT step.

    By default the step runs eagerly (through the compiled blocks) at the
    batch's exact padded shape: a batched step does enough work to hide the
    kernel launches, while a continuous batch changes shape too often for
    per-shape graphs to pay back their capture. With ``enabled`` it instead
    replays a CUDA graph per ``(rows, text, reference, target)`` bucket.
    Either way the per-request conditioning is assembled only when the
    batch's membership changes, and each step copies just the latents and
    the modulation rows.
    """

    def __init__(self, dit: AuKTransformer, *, enabled: bool = True, max_graphs: int = 32) -> None:
        self.dit = dit
        self.enabled = bool(enabled)
        self.max_graphs = max(1, int(max_graphs))
        self._cache: OrderedDict[tuple[int, int, int, int], _GraphEntry] = OrderedDict()
        self._pool_handle: int | None = None
        # Membership whose context is loaded, and where (a graph key or eager).
        self._loaded: tuple[tuple[int, ...], tuple[int, int, int, int] | None] | None = None
        self._eager_ctx: AuKStepContext | None = None

    def make_request(
        self,
        *,
        text: torch.Tensor,
        ref: torch.Tensor,
        cfg: float,
        grid: torch.Tensor,
        latents: torch.Tensor,
    ) -> AuKBatchRequest:
        """Embed a request's conditioning once. Call under the DiT's autocast."""
        dit = self.dit
        guided = cfg >= CFG_EPSILON
        prompt = prompt_uncond = None
        if ref.shape[1] > 0:
            prompt = dit._embed_prompt(ref, None, False)
            if guided:
                prompt_uncond = dit._embed_prompt(ref, None, True)
        return AuKBatchRequest(
            uid=next(_request_uids),
            text=text,
            ref=ref,
            c=dit.project_text(text),
            prompt=prompt,
            prompt_uncond=prompt_uncond,
            cfg=float(cfg),
            grid=grid,
            modulation=dit.modulation_table(grid[:-1]),
            latents=latents,
        )

    @staticmethod
    def batch_shape(requests: Sequence[AuKBatchRequest]) -> tuple[int, int, int, int]:
        """``(rows, text, reference, target)``: the longest of each sequence in the batch."""
        return (
            sum(r.rows for r in requests),
            max(int(r.text.shape[1]) for r in requests),
            max(r.ref_frames for r in requests),
            max(r.target_frames for r in requests),
        )

    @staticmethod
    def bucket_key(requests: Sequence[AuKBatchRequest]) -> tuple[int, int, int, int]:
        """:meth:`batch_shape` rounded up to the CUDA graph buckets."""
        ref = max(r.ref_frames for r in requests)
        return (
            row_bucket(sum(r.rows for r in requests)),
            round_up(max(int(r.text.shape[1]) for r in requests), _TEXT_ALIGNMENT),
            round_up(ref, _REF_ALIGNMENT) if ref else 0,
            round_up(max(r.target_frames for r in requests), _TARGET_ALIGNMENT),
        )

    def assemble(self, requests: Sequence[AuKBatchRequest], key: tuple[int, int, int, int]) -> AuKStepContext:
        """The padded, masked step context of ``requests`` at bucket ``key``."""
        dit = self.dit
        rows, text_len, ref_len, target_len = key
        first = requests[0]
        device, dtype = first.c.device, first.c.dtype
        c = torch.zeros(rows, text_len, first.c.shape[-1], device=device, dtype=dtype)
        # Padding rows keep every position valid so their softmax stays finite.
        c_mask = torch.ones(rows, text_len, dtype=torch.bool, device=device)
        target_mask = torch.ones(rows, target_len, dtype=torch.bool, device=device)
        prompt = ref_mask = None
        if ref_len:
            prompt = torch.zeros(rows, ref_len, first.c.shape[-1], device=device, dtype=dtype)
            ref_mask = torch.ones(rows, ref_len, dtype=torch.bool, device=device)
        row = 0
        for request in requests:
            nt, np_, n = int(request.text.shape[1]), request.ref_frames, request.target_frames
            branches = [(request.c, request.prompt)]
            if request.guided:
                # The uncond branch drops the text (a zeroed projection) and the reference audio.
                branches.append((None, request.prompt_uncond))
            for text_embed, prompt_embed in branches:
                if text_embed is not None:
                    c[row, :nt] = text_embed[0]
                c_mask[row, nt:] = False
                target_mask[row, n:] = False
                if prompt is not None:
                    if prompt_embed is not None:
                        prompt[row, :np_] = prompt_embed[0]
                    ref_mask[row, np_:] = False
                row += 1

        audio_mask = target_mask if ref_mask is None else torch.cat([ref_mask, target_mask], dim=1)
        single_mask = torch.cat([c_mask, audio_mask], dim=1)
        audio_len = ref_len + target_len
        return AuKStepContext(
            c=c,
            prompt=prompt,
            target_mask=target_mask,
            c_mask=c_mask,
            audio_mask=audio_mask,
            single_mask=single_mask,
            joint_bias=_key_padding_bias(joint_key_mask(audio_mask, c_mask, text_len), dtype),
            single_bias=_key_padding_bias(single_mask, dtype),
            rope_audio=_rope_cos_sin(dit.rotary_embed(audio_len, audio_mask)),
            rope_text=_rope_cos_sin(dit.rotary_embed(text_len, c_mask)),
            rope_single=_rope_cos_sin(dit.rotary_embed(text_len + audio_len, single_mask)),
            branches=1,
            modulation=torch.zeros(rows, first.modulation.shape[1], device=device, dtype=first.modulation.dtype),
            modulation_per_row=True,
        )

    @staticmethod
    def _fill_step_inputs(requests: Sequence[AuKBatchRequest], x: torch.Tensor, modulation: torch.Tensor) -> None:
        """Write each row's current latents and modulation into the step buffers."""
        x.zero_()
        row = 0
        for request in requests:
            n = request.target_frames
            mod = request.modulation[request.step_index]
            for _ in range(request.rows):
                x[row, :n] = request.latents[0]
                modulation[row] = mod
                row += 1

    def _step(self, x: torch.Tensor, ctx: AuKStepContext) -> torch.Tensor:
        # The modulation rows carry the time, so the time argument is unused.
        return self.dit.step(x, x.new_zeros(()), ctx)

    @torch.no_grad()
    def velocities(self, requests: Sequence[AuKBatchRequest]) -> list[torch.Tensor]:
        """Guided velocity ``[1, frames, latent_dim]`` of each request at its current step."""
        if not self.dit.attn_mask_enabled:
            raise ValueError("Batching AuK requests needs attn_mask_enabled: padding must be masked out.")
        members = tuple(r.uid for r in requests)
        first = requests[0]
        use_graph = self.enabled and first.latents.is_cuda and not torch.cuda.is_current_stream_capturing()
        key = self.bucket_key(requests) if use_graph else self.batch_shape(requests)
        if use_graph:
            out = self._replay(requests, key, members)
        else:
            if self._loaded != (members, None) or self._eager_ctx is None:
                self._eager_ctx = self.assemble(requests, key)
                self._loaded = (members, None)
            x = torch.zeros(key[0], key[3], first.latents.shape[-1], device=first.latents.device)
            self._fill_step_inputs(requests, x, self._eager_ctx.modulation)
            out = self._step(x, self._eager_ctx)

        velocities: list[torch.Tensor] = []
        row = 0
        for request in requests:
            n = request.target_frames
            conditional = out[row : row + 1, :n]
            if request.guided:
                unconditional = out[row + 1 : row + 2, :n]
                conditional = conditional + (conditional - unconditional) * request.cfg
            else:
                # The graph's output buffer is overwritten by the next replay.
                conditional = conditional.clone()
            velocities.append(conditional)
            row += request.rows
        return velocities

    def _replay(
        self, requests: Sequence[AuKBatchRequest], key: tuple[int, int, int, int], members: tuple[int, ...]
    ) -> torch.Tensor:
        entry = self._cache.get(key)
        if entry is None:
            if len(self._cache) >= self.max_graphs:
                # Retire every graph together so none outlives the shared pool's workspaces.
                self._cache.clear()
            try:
                entry = self._capture(requests, key)
            except Exception:
                self._cache.clear()
                self.enabled = False
                self._loaded = None
                logger.exception("Disabling batched AuK DiT CUDA graphs after capture failure for key=%s", key)
                raise
            self._cache[key] = entry
        else:
            self._cache.move_to_end(key)
            if self._loaded != (members, key):
                entry.static_ctx.copy_(self.assemble(requests, key))
        self._loaded = (members, key)
        self._fill_step_inputs(requests, entry.static_x, entry.static_ctx.modulation)
        entry.graph.replay()
        return entry.static_out

    def _capture(self, requests: Sequence[AuKBatchRequest], key: tuple[int, int, int, int]) -> _GraphEntry:
        first = requests[0]
        static_ctx = self.assemble(requests, key)
        static_x = torch.zeros(key[0], key[3], first.latents.shape[-1], device=first.latents.device)
        self._fill_step_inputs(requests, static_x, static_ctx.modulation)
        # Warm-up runs any lazy torch.compile outside the capture.
        for _ in range(3):
            self._step(static_x, static_ctx)
        if self._pool_handle is None:
            self._pool_handle = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool_handle):
            static_out = self._step(static_x, static_ctx)
        logger.info(
            "Captured batched AuK DiT step CUDA graph: rows=%d text_tokens=%d ref_frames=%d target_frames=%d",
            *key,
        )
        return _GraphEntry(graph=graph, static_x=static_x, static_ctx=static_ctx, static_out=static_out)


__all__ = ["AuKBatchRequest", "AuKBatchedStepRunner", "CFG_EPSILON", "row_bucket"]
