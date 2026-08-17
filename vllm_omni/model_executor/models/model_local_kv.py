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
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "ModelLocalKVScope",
    "ModelLocalKVSpec",
    "HasModelLocalKV",
    "collect_model_local_kv_specs",
    "spec_from_hf_config",
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
    """Instances the *model* retains, at a concurrency of one.

    This is model-side replication only: 261 for a cache copied into every
    captured CUDA-graph bucket, 1 for almost everything else. A model cannot
    know how many requests are in flight, so it must not try to account for
    them here -- see ``scales_with_concurrency``.
    """

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
        """Model-side peak, at a concurrency of one.

        Use ``peak_bytes(concurrency)`` for the number an engine should
        actually budget against.
        """
        return self.bytes_per_instance * self.max_live_instances

    @property
    def scales_with_concurrency(self) -> bool:
        """Whether more in-flight requests mean more copies of this cache.

        Splitting the two multiplicities is the whole reason ``scope`` is not
        cosmetic. A ``MODEL``-scoped cache is captured once and shared, so it
        costs the same at concurrency 64 as at 1. Everything else is per
        request, session, or call, so the engine multiplies it -- and only the
        engine knows by how much.

        A batched cache that widens rather than replicating (one tensor with a
        larger batch dimension) belongs in ``batch_capacity`` instead, which is
        why that field has no default.
        """
        return self.scope is not ModelLocalKVScope.MODEL

    def peak_bytes(self, concurrency: int = 1) -> int:
        """Peak bytes at a given number of in-flight requests."""
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        if not self.scales_with_concurrency:
            return self.max_live_bytes
        return self.max_live_bytes * concurrency


def _resolve(config: object, candidates: Sequence[str], what: str) -> int:
    for attr in candidates:
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError(
        f"Cannot determine {what} from {type(config).__name__}: tried "
        f"{', '.join(candidates)}. Pass it explicitly to spec_from_hf_config()."
    )


def spec_from_hf_config(
    config: object,
    *,
    name: str,
    physical_capacity_positions: int,
    capacity_source: str,
    scope: ModelLocalKVScope,
    dtype: torch.dtype,
    batch_capacity: int = 1,
    max_live_instances: int = 1,
    allocation_note: str | None = None,
    layers: int | None = None,
    kv_heads: int | None = None,
    head_dim: int | None = None,
) -> ModelLocalKVSpec:
    """Build a spec from the same config object the cache is built from.

    Every known model-local cache is constructed from an HF config, so the
    geometry half of a declaration is the same three lookups each time. Pass
    ``config`` and this derives them; pass ``layers``/``kv_heads``/``head_dim``
    explicitly to override when the config uses encoder-decoder naming or when
    the built module is a more truthful source than the config.

    Raises rather than guessing when an attribute is absent: a silently wrong
    geometry would under-report, which is the failure this protocol exists to
    prevent.
    """
    if layers is None:
        layers = _resolve(config, ("num_hidden_layers", "encoder_layers", "num_layers"), "layer count")
    if kv_heads is None:
        kv_heads = _resolve(
            config,
            ("num_key_value_heads", "num_attention_heads", "encoder_attention_heads"),
            "kv head count",
        )
    if head_dim is None:
        explicit = getattr(config, "head_dim", None)
        if explicit is not None:
            head_dim = int(explicit)
        else:
            hidden = _resolve(config, ("hidden_size", "d_model"), "hidden size")
            heads = _resolve(config, ("num_attention_heads", "encoder_attention_heads"), "attention head count")
            head_dim = hidden // heads

    return ModelLocalKVSpec(
        name=name,
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        physical_capacity_positions=physical_capacity_positions,
        capacity_source=capacity_source,
        scope=scope,
        batch_capacity=batch_capacity,
        max_live_instances=max_live_instances,
        allocation_note=allocation_note,
    )


@runtime_checkable
class HasModelLocalKV(Protocol):
    """Implemented by whichever object owns the cache.

    Implement it on the owner, not on the registered model. The owner is
    usually several levels down (a codec decoder, a talker backbone), and
    ``collect_model_local_kv_specs`` finds it by walking the module tree, so
    no intermediate class has to forward anything.
    """

    def model_local_kv_specs(self) -> Sequence[ModelLocalKVSpec]: ...


def _unwrap_to_module(model: object) -> object:
    """Follow runnable-style wrappers down to something with a module tree.

    By the time the runner holds it, the model may be wrapped for CUDA-graph
    dispatch. ``vllm.compilation.cuda_graph.CUDAGraphWrapper`` is a plain
    callable holding ``.runnable``, not an ``nn.Module``, so walking it
    directly finds nothing and every declaration silently reports zero.
    """
    seen: set[int] = set()
    current = model
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if callable(getattr(current, "named_modules", None)):
            return current
        current = getattr(current, "runnable", None)
    return model


def collect_model_local_kv_specs(model: object) -> list[tuple[str, ModelLocalKVSpec]]:
    """Collect every declaration in a loaded model, with the owner's path.

    Walks ``named_modules()`` because the cache owner is an inner module in all
    known cases. Requiring each registered model to forward the call would put
    the burden on classes that do not own a cache and would silently report
    zero the moment someone forgot -- exactly the failure mode this is meant to
    surface.

    A raising declaration is logged and skipped rather than propagated:
    reporting memory must not be able to break model load.
    """
    owners: list[tuple[str, object]] = []
    seen_ids: set[int] = set()
    if isinstance(model, HasModelLocalKV):
        owners.append(("", model))
        seen_ids.add(id(model))

    root = _unwrap_to_module(model)
    if root is not model and isinstance(root, HasModelLocalKV) and id(root) not in seen_ids:
        owners.append(("", root))
        seen_ids.add(id(root))

    named_modules = getattr(root, "named_modules", None)
    if callable(named_modules):
        for path, module in named_modules():
            if id(module) in seen_ids or not isinstance(module, HasModelLocalKV):
                continue
            seen_ids.add(id(module))
            owners.append((path, module))

    collected: list[tuple[str, ModelLocalKVSpec]] = []
    for path, owner in owners:
        try:
            specs = owner.model_local_kv_specs()
        except Exception:
            logger.warning(
                "model_local_kv_specs() raised on %s; skipping its declaration",
                type(owner).__name__,
                exc_info=True,
            )
            continue
        collected.extend((path, spec) for spec in specs)
    return collected


def total_declared_bytes(model: object, concurrency: int = 1) -> int:
    """Sum declared peak bytes, or 0 for a model that declares nothing.

    Pass ``concurrency`` (the engine's ``max_num_seqs``) to include the
    per-request multiplicity the model cannot know. Returning 0 rather than
    raising keeps this usable as an additive term in memory reporting before
    every model has been migrated.
    """
    return sum(spec.peak_bytes(concurrency) for _, spec in collect_model_local_kv_specs(model))
