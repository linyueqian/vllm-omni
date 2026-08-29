# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Component weight loading for MiniMax Music 3.

The checkpoint is a multi-component repo. The Qwen3 backbone is a normal
``Qwen3ForCausalLM`` under ``language_model/`` and loads through vLLM. The
audio embedding table and the depth decoder live in ``rvq_depth_decoder/``,
and the acoustic stage's three components live in their own folders. None of
those are things vLLM's loader knows how to map, so they are read directly.

Every loader here is strict. A silently partial load would produce audio that
sounds plausible but is wrong, which is far worse than a startup failure.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from .constants import AUDIO_VOCAB_SIZE, NUM_CODEBOOKS

logger = init_logger(__name__)

_AUDIO_EMBEDDING_KEY = "audio_embeddings.weight"
_COMPONENT_WEIGHT_NAME = "diffusion_pytorch_model.safetensors"

# Component folders, relative to the repository root.
DEPTH_DECODER_DIR = "rvq_depth_decoder"
CONDITION_ENCODER_DIR = "condition_encoder"
TRANSFORMER_DIR = "transformer"
VOCODER_DIR = "vocoder"

_ROOT_MARKERS = (DEPTH_DECODER_DIR, TRANSFORMER_DIR, VOCODER_DIR)

# The subset a stage reading the repo ROOT needs on disk. The AR backbone under
# ``language_model/`` is pulled by vLLM's own loader, and the repo's top-level
# ``.pth`` files and ``qwen_7B/`` folder belong to the reference implementation,
# so neither is downloaded here.
_COMPONENT_PATTERNS = (
    "*.json",
    f"{DEPTH_DECODER_DIR}/*",
    f"{CONDITION_ENCODER_DIR}/*",
    f"{TRANSFORMER_DIR}/*",
    f"{VOCODER_DIR}/*",
)


def _snapshot_has_components(root: Path) -> bool:
    """True when every marker folder holds its component weights."""
    stem = Path(_COMPONENT_WEIGHT_NAME).stem
    return all(
        (root / marker / _COMPONENT_WEIGHT_NAME).is_file() or any((root / marker).glob(f"{stem}-*.safetensors"))
        for marker in _ROOT_MARKERS
    )


def _download_component_snapshot(model: str) -> Path:
    """Resolve an HF repo id to a snapshot holding the component folders.

    Cache-first, so a warm cache needs no Hub round trip. Depending on the
    huggingface_hub version, ``local_files_only=True`` either raises on a
    partial cache or returns the partial snapshot root unchanged; both
    variants must fall through to the online call, or a cold cache fails
    startup without ever consulting the Hub.
    """
    from vllm.transformers_utils.repo_utils import hf_api

    patterns = list(_COMPONENT_PATTERNS)
    try:
        cached = Path(hf_api().snapshot_download(model, allow_patterns=patterns, local_files_only=True))
    except Exception:
        cached = None
    if cached is not None and _snapshot_has_components(cached):
        return cached
    return Path(hf_api().snapshot_download(model, allow_patterns=patterns))


def _repo_id_from_cache_path(path: Path) -> str | None:
    """Recover the Hub repo id when *path* lies inside an HF cache snapshot.

    Stage init hands the talker the snapshot's ``language_model`` folder. On a
    cold cache the sibling component folders were never downloaded, so the
    marker walk fails even though the missing pieces are one snapshot_download
    away; the repo id is recoverable from the cache layout
    (``models--{org}--{name}/snapshots/{revision}/...``).
    """
    for part in path.parts:
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    return None


def resolve_repo_root(model_path: str | os.PathLike[str]) -> Path:
    """Find the multi-component repository root from a stage's model path.

    A stage that declares ``model_subdir`` sees the subfolder as its model
    path, so the root is usually the parent. Both are checked, and the search
    walks up a couple of levels for snapshot layouts.

    The acoustic stage declares no ``model_subdir`` because it reads the root
    itself, so on a hub-id deployment its model path is the repo id rather
    than a directory. Resolve that to a local snapshot instead of treating it
    as a relative path.

    A cold cache is recoverable too: stage init pre-downloads only
    ``language_model/`` and ``tokenizer/``, so the talker's model path exists
    while the component folders do not. Fetch them here instead of failing.

    Raises:
        FileNotFoundError: If no ancestor looks like the repository root.
    """
    path = Path(model_path).expanduser().resolve()
    candidates = [path, *path.parents[:3]]
    for candidate in candidates:
        if all((candidate / marker).is_dir() for marker in _ROOT_MARKERS):
            return candidate

    # No ancestor carries the markers. Two recoverable cases: the path is a
    # Hub repo id (the acoustic stage passes the id through), or it is a
    # cache-snapshot subfolder whose sibling component folders were never
    # downloaded. ``_download_component_snapshot`` is cache-first, so a warm
    # cache stays offline either way.
    reference = str(model_path) if not path.exists() else _repo_id_from_cache_path(path)
    if reference is not None:
        try:
            resolved = _download_component_snapshot(reference)
        except Exception as exc:
            raise FileNotFoundError(
                f"MiniMax Music 3 could not resolve {model_path!r} to a local repository root; "
                f"expected folders {_ROOT_MARKERS}"
            ) from exc
        if all((resolved / marker).is_dir() for marker in _ROOT_MARKERS):
            return resolved

    raise FileNotFoundError(
        f"MiniMax Music 3 could not locate the repository root from {path}; expected sibling folders {_ROOT_MARKERS}"
    )


def load_component_state(root: Path, component: str, *, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Read one component's safetensors, sharded or not.

    Raises:
        FileNotFoundError: If the component has no weight files.
    """
    from safetensors.torch import load_file

    folder = root / component
    single = folder / _COMPONENT_WEIGHT_NAME
    if single.exists():
        return load_file(str(single), device=device)

    shards = sorted(folder.glob(f"{Path(_COMPONENT_WEIGHT_NAME).stem}-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"MiniMax Music 3 component {component!r} has no weights under {folder}")
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(str(shard), device=device))
    return state


def load_depth_decoder_weights(
    vllm_config: VllmConfig,
    *,
    audio_embeddings: nn.Embedding,
    rvq_decoder: nn.Module,
) -> set[str]:
    """Load the audio embedding table and the RVQ depth decoder in place.

    Args:
        vllm_config: The stage's config, used to locate the checkpoint.
        audio_embeddings: The ``c1..c7`` embedding table to populate.
        rvq_decoder: The depth decoder to populate.

    Returns:
        The model-side parameter names that were loaded.

    Raises:
        ValueError: If the audio embedding is absent or the wrong shape.
    """
    root = resolve_repo_root(vllm_config.model_config.model)
    state = load_component_state(root, DEPTH_DECODER_DIR)

    weight = state.pop(_AUDIO_EMBEDDING_KEY, None)
    expected = (AUDIO_VOCAB_SIZE * (NUM_CODEBOOKS - 1), audio_embeddings.embedding_dim)
    if weight is None or tuple(weight.shape) != expected:
        raise ValueError(
            f"MiniMax Music 3 audio embedding mismatch: expected {expected}, got "
            f"{None if weight is None else tuple(weight.shape)}"
        )
    target = audio_embeddings.weight
    target.data.copy_(weight.to(device=target.device, dtype=target.dtype))

    rvq_decoder.load_checkpoint_state(state)
    logger.info(
        "MiniMax Music 3 loaded audio embedding and depth decoder from %s",
        root / DEPTH_DECODER_DIR,
    )
    loaded = {"audio_embeddings.weight"}
    loaded.update(f"rvq_decoder.{name}" for name, _ in rvq_decoder.named_parameters())
    return loaded


__all__ = [
    "CONDITION_ENCODER_DIR",
    "DEPTH_DECODER_DIR",
    "TRANSFORMER_DIR",
    "VOCODER_DIR",
    "load_component_state",
    "load_depth_decoder_weights",
    "resolve_repo_root",
]
