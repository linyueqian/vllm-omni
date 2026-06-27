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
    """``[F, dep_q]`` raw depformer agent codes -> de-delayed flat codebook-major.

    The talker emits the raw per-frame depformer codes ``gen[t]`` (cb 0..7). Mimi
    needs them ACOUSTICALLY DE-DELAYED to a common time step (Moshi agent delays
    ``[0, 1, 1, 1, 1, 1, 1, 1]``): acoustic frame t's cb_k is predicted at step
    ``t + delay[k]``, so output frame t = ``[gen[t][0], gen[t+1][1:8]]`` (cb0 from
    the current step, cb1..7 from the NEXT step, since the delayed codebooks lag).
    Without this the codebooks are misaligned by one frame and Mimi decodes garble.
    The last frame has no successor for cb1..7, so it is dropped (the delay warmup).
    """
    if audio.ndim != 2 or audio.shape[0] < 2:
        return torch.empty(0, dtype=torch.long)
    audio = audio.to(torch.long)
    k = min(_NUM_ACTIVE_CODEBOOKS, int(audio.shape[1]))
    agent = audio[:, :k]
    valid = (agent >= 0).all(dim=1)
    agent = agent[valid]
    if agent.shape[0] < 2:
        return torch.empty(0, dtype=torch.long)
    # De-delay: cb0 from frame t, cb1..7 from frame t+1 (drop the last frame).
    cb0 = agent[:-1, 0:1]  # [F-1, 1]
    cb_rest = agent[1:, 1:k]  # [F-1, k-1]
    dd = torch.cat([cb0, cb_rest], dim=1)  # [F-1, k] de-delayed
    # [F-1, k] -> [k, F-1] -> flat [k * (F-1)] (codebook-major), as Code2Wav expects.
    return dd.transpose(0, 1).contiguous().reshape(-1)


def talker2code2wav_token_only(
    source_outputs: list,
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list:
    """Sync ``process_engine_inputs``: build the code2wav placeholder inputs.

    Returns one :class:`OmniTokensPrompt` per finished talker request, with
    ``prompt_token_ids`` sized to the flat codebook-major codec length
    (``num_active_codebooks * num_agent_frames``). The actual codec ids are
    delivered via the worker connector payload from ``talker2code2wav_full_payload``.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    del prompt, _requires_multimodal_data
    inputs: list = []
    for talker_output in source_outputs:
        if not getattr(talker_output, "finished", False):
            continue
        output = talker_output.outputs[0]
        # The per-request output is the "latent" (engine_output_type="latent"); the
        # codes ship via the connector full_payload. Size the placeholder from the
        # generated token count (one AR token == one Mimi frame). prompt was 1 frame.
        token_ids = getattr(output, "cumulative_token_ids", None) or getattr(output, "token_ids", None) or []
        n_frames = max(len(token_ids) - 1, 0)
        prompt_len = _NUM_ACTIVE_CODEBOOKS * n_frames
        inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return inputs


def talker2code2wav_full_payload(
    transfer_manager: Any = None,
    multimodal_output: Any = None,
    request: Any = None,
    is_finished: bool = False,
    **_: Any,
) -> OmniPayloadStruct:
    """Producer: collect the talker's accumulated agent codes -> Code2Wav input.

    Called by the connector with (transfer_manager, multimodal_output, request,
    is_finished). The codes live in the request's additional_information under
    ("codes","audio") (talker_mtp_output_key), NOT in multimodal_output (which is
    the engine "latent" output); read them from the request.
    """
    del transfer_manager, is_finished
    info = getattr(request, "additional_information", None)
    audio = None
    if isinstance(info, dict):
        nested = info.get("codes")
        audio = nested.get("audio") if isinstance(nested, dict) else None
        if audio is None:
            audio = info.get("codes.audio")
    if audio is None and isinstance(multimodal_output, dict):
        nested = multimodal_output.get("codes")
        audio = nested.get("audio") if isinstance(nested, dict) else multimodal_output.get("codes.audio")
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

    # Codes live in the server-side request's additional_information under
    # ("codes","audio") (talker_mtp_output_key), not in multimodal_output (latent).
    def _codes_from(src: Any) -> torch.Tensor | None:
        if isinstance(src, dict):
            nested = src.get("codes")
            a = nested.get("audio") if isinstance(nested, dict) else None
            return a if a is not None else src.get("codes.audio")
        return None

    audio = _codes_from(getattr(request, "additional_information", None)) or _codes_from(multimodal_output)
    if isinstance(audio, torch.Tensor) and audio.numel() > 0:
        a = audio if audio.ndim == 2 else audio.reshape(1, -1)
        frames.append(a[-1].to(torch.long).cpu())  # latest frame's codes

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
