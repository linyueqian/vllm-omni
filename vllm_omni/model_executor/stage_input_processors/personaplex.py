# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Talker -> Code2Wav input processors for PersonaPlex.

The talker (stage 0) emits, per frame, the ``dep_q`` depformer audio codes under
``("codes","audio")``. Only the leading ``num_active_codebooks`` (the agent's
``cb 0..7``) are decoded to PCM by Mimi; the trailing codebooks are the user
stream and are not vocoded. These processors take the accumulated per-frame agent
codes ``[F, dep_q]``, keep ``cb 0..7``, and flatten them codebook-major
(``[8 * F]``) — the exact layout :class:`PersonaPlexCode2Wav` consumes.

Mirrors the Qwen3-TTS processors (sync ``full_payload`` + ``token_only``; an
async-chunk variant for the streaming path), but with PersonaPlex's agent-codebook
slice instead of Qwen3-TTS's residual layout.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayloadStruct,
)

_NUM_ACTIVE_CODEBOOKS = 8  # agent cb 0..7 (the PCM-bearing rows)


def _empty_finished_payload() -> OmniPayloadStruct:
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
        meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
    )


def _agent_codes_to_codebook_major(audio: torch.Tensor) -> torch.Tensor:
    """``[F, dep_q]`` agent+user codes -> flat codebook-major ``[8 * F]`` (agent only).

    Keeps the leading ``_NUM_ACTIVE_CODEBOOKS`` codebooks and drops frames that are
    out of range / all-zero padding, then transposes to ``[8, F]`` and flattens.
    """
    if audio.ndim != 2 or audio.shape[0] == 0:
        return torch.empty(0, dtype=torch.long)
    audio = audio.to(torch.long)
    k = min(_NUM_ACTIVE_CODEBOOKS, int(audio.shape[1]))
    agent = audio[:, :k]
    valid = (agent >= 0).all(dim=1)
    agent = agent[valid]
    if agent.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    # [F, k] -> [k, F] -> flat [k * F] (codebook-major), as Code2Wav expects.
    return agent.transpose(0, 1).contiguous().reshape(-1)


def talker2code2wav_token_only(transfer_manager: Any, multimodal_output: Any, request: Any) -> OmniPayloadStruct:
    """Sync placeholder: a length-only handle; the codec payload ships via full_payload."""
    del transfer_manager, multimodal_output, request
    return OmniPayloadStruct(codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)))


def talker2code2wav_full_payload(transfer_manager: Any, pooling_output: Any, request: Any) -> OmniPayloadStruct:
    """Producer: collect the talker's accumulated agent codes -> Code2Wav input."""
    del transfer_manager, request
    if not isinstance(pooling_output, dict):
        return _empty_finished_payload()
    audio = pooling_output.get("codes.audio")
    if audio is None:
        nested = pooling_output.get("codes")
        audio = nested.get("audio") if isinstance(nested, dict) else None
    if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
        return _empty_finished_payload()
    flat = _agent_codes_to_codebook_major(audio)
    if flat.numel() == 0:
        return _empty_finished_payload()
    return OmniPayloadStruct(
        codes=CodesStruct(audio=flat),
        meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
    )


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Streaming: accumulate per-frame agent codes, emit a codebook-major chunk.

    Minimal fixed-chunk variant (no left-context / ref-code complexity, which
    PersonaPlex does not use). Frames are buffered on the transfer manager until a
    chunk's worth is ready (or the request finishes), then flushed.
    """
    request_id = getattr(request, "external_req_id", getattr(request, "request_id", "?"))
    finished = bool(is_finished or (hasattr(request, "is_finished") and request.is_finished()))
    buf = getattr(transfer_manager, "_pplex_frames", None)
    if buf is None:
        buf = {}
        transfer_manager._pplex_frames = buf
    frames = buf.setdefault(request_id, [])

    if isinstance(multimodal_output, dict):
        audio = multimodal_output.get("codes", {}).get("audio")
        if isinstance(audio, torch.Tensor) and audio.ndim == 2 and audio.shape[0] > 0:
            frames.append(audio[-1].to(torch.long).cpu())

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk = int(cfg.get("codec_chunk_frames", 25))

    if len(frames) < chunk and not finished:
        return None
    if not frames:
        return _empty_finished_payload() if finished else None

    stacked = torch.stack(frames, dim=0)  # [F, dep_q]
    buf[request_id] = []
    flat = _agent_codes_to_codebook_major(stacked)
    return OmniPayloadStruct(
        codes=CodesStruct(audio=flat),
        meta=MetaStruct(finished=torch.tensor(bool(finished), dtype=torch.bool)),
    )


__all__ = [
    "talker2code2wav_token_only",
    "talker2code2wav_full_payload",
    "talker2code2wav_async_chunk",
]
