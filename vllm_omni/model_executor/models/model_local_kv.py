# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declaration protocol for attention KV kept outside the paged manager.

Several models hold their own attention KV as HuggingFace ``transformers`` cache
objects rather than using the engine's paged KV manager. That memory is
allocated after the profiling run that decides how much KV the engine may claim,
so no per-stage footprint number accounts for it, and nothing catches a cache
that silently stops working.

This module lets a model *declare* what it holds. It describes; it never
allocates. Allocation for the diffusion path is owned by the DiT KV manager
(RFC #5244 / PR #6094) and this protocol must not grow into a second allocator.

Why a runtime query rather than a static table: geometry is frequently not
knowable until weights are loaded. ``ming_flash_omni``'s talker builds its cache
from a ``Qwen2Config`` that comes from the checkpoint, so layers, kv-heads,
head-dim and dtype simply do not exist in this repo. A table would have a hole
in it; a post-load query does not.

Why capacity is a resolved number rather than a formula: the four known caches
are each bounded by a different mechanism -- a sliding window, a decode-loop
trip count, an encoder frame limit, a hardcoded constant -- and not one of them
is ``max_model_len``. Only the model knows its own bound.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import torch

__all__ = [
    "ModelLocalKVScope",
    "ModelLocalKVSpec",
    "HasModelLocalKV",
    "total_declared_bytes",
]


class ModelLocalKVScope(str, Enum):
    """How long one cache instance lives.

    Separate from multiplicity: ``scope`` says how long an instance survives,
    ``max_live_instances`` says how many can exist at once. Peak memory needs
    both, and collapsing them into a single "preallocated vs grows" flag cannot
    express a cache that is short-lived but replicated per CUDA-graph bucket.
    """

    INVOCATION = "invocation"
    """Dies when the call returns (e.g. a per-step working copy)."""

    REQUEST = "request"
    """Retained across steps of one request, released when it finishes."""

    SESSION = "session"
    """Outlives a request; belongs to a duplex/streaming session."""

    MODEL = "model"
    """Lives as long as the model (e.g. captured into a CUDA-graph pool)."""


@dataclass(frozen=True)
class ModelLocalKVSpec:
    """One declared KV allocation.

    A model returns one entry per *distinct lifetime*, not per cache object: if
    the same geometry exists as both a retained per-request cache and a
    short-lived working copy, that is two entries.
    """

    name: str
    layers: int
    kv_heads: int
    head_dim: int
    dtype: torch.dtype

    physical_capacity_positions: int
    """Positions the tensor can physically hold.

    Not the logical sequence position. A sliding-window cache truncates on every
    write, so a stream of thousands of tokens may only ever occupy
    ``sliding_window - 1`` slots.
    """

    capacity_source: str
    """Free-text note on where the bound comes from, for diagnostics only.

    Never branch on this. It exists so a reader can tell "71 because sliding
    window" from "2048 because someone typed 2048", which is otherwise
    invisible.
    """

    scope: ModelLocalKVScope
    batch_capacity: int
    """Rows the tensor is sized for. Required: defaulting it to 1 silently
    under-reports every batched cache."""

    max_live_instances: int
    """How many instances can coexist. Usually 1; higher when a cache is
    replicated, e.g. one per captured CUDA-graph batch bucket."""

    allocation_note: str | None = None
    """Optional detail such as "rebuilt per text segment". Diagnostic only."""

    @property
    def bytes_per_instance(self) -> int:
        return (
            2  # K and V
            * self.layers
            * self.kv_heads
            * self.head_dim
            * self.physical_capacity_positions
            * self.batch_capacity
            * torch.empty((), dtype=self.dtype).element_size()
        )

    @property
    def max_live_bytes(self) -> int:
        return self.bytes_per_instance * self.max_live_instances


@runtime_checkable
class HasModelLocalKV(Protocol):
    """Implemented by the object visible to the runner after load.

    The cache-owning class is often an inner module, so the registered model may
    need to delegate to it rather than implement this directly.
    """

    def model_local_kv_specs(self) -> Sequence[ModelLocalKVSpec]: ...


def total_declared_bytes(model: object) -> int:
    """Sum declared peak bytes, or 0 for a model that declares nothing.

    Returning 0 rather than raising keeps this usable as an additive term in
    memory reporting before every model has been migrated.
    """
    if not isinstance(model, HasModelLocalKV):
        return 0
    return sum(spec.max_live_bytes for spec in model.model_local_kv_specs())
