# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for VoxCPM2 (same logic as VoxCPM v1)."""
from __future__ import annotations

import logging
from typing import Any

import torch
from vllm.inputs import TextPrompt

logger = logging.getLogger(__name__)

from vllm_omni.inputs.data import OmniTokensPrompt


def latent2vae(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Sync path: Stage 0 latent output -> Stage 1 VAE input."""
    del prompt, requires_multimodal_data

    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    source_outputs = stage_list[source_stage_id].engine_outputs
    if source_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    vae_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        output = source_output.outputs[0]
        multimodal_output = getattr(output, "multimodal_output", None)
        logger.info(
            "latent2vae: multimodal_output type=%s keys=%s",
            type(multimodal_output).__name__,
            list(multimodal_output.keys()) if isinstance(multimodal_output, dict) else "N/A",
        )
        if not isinstance(multimodal_output, dict) or "latent_audio_feat" not in multimodal_output:
            raise ValueError(
                "VoxCPM2 latent stage output missing 'latent_audio_feat'. "
                f"request_id={getattr(source_output, 'request_id', None)}"
            )

        latent = multimodal_output["latent_audio_feat"]
        logger.info(
            "latent2vae: latent type=%s shape=%s",
            type(latent).__name__,
            latent.shape if hasattr(latent, "shape") else (
                len(latent) if isinstance(latent, list) else "?"),
        )
        # Native AR (Phase 2): output_processor concatenates per-step
        # patches [1, P*D] → [N, P*D]. Reshape to [D, N*P] for VAE.
        if isinstance(latent, torch.Tensor) and latent.ndim == 2:
            n, pd = latent.shape
            # Detect patch_size and feat_dim from the total elements
            # VoxCPM2: feat_dim=64, patch_size=4, so P*D=256
            d = 64  # feat_dim
            p = pd // d  # patch_size
            if p * d == pd:
                # [N, P*D] → [N, P, D] → [D, N*P]
                latent = latent.reshape(n, p, d).permute(2, 0, 1).reshape(d, -1)

        additional_information: dict[str, Any] = {
            "latent_audio_feat": latent,
        }
        if "sr" in multimodal_output:
            additional_information["sample_rate"] = [int(multimodal_output["sr"])]

        vae_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return vae_inputs


def latent2vae_async_chunk(
    transfer_manager: Any = None,
    pooling_output: dict[str, Any] | None = None,
    request: Any = None,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Async chunk path: Stage 0 latent chunk -> Stage 1 VAE connector payload."""
    finished_request = bool(is_finished)
    if callable(getattr(request, "is_finished", None)):
        finished_request = finished_request or bool(request.is_finished())

    if not isinstance(pooling_output, dict):
        if finished_request:
            return {"code_predictor_codes": [0], "finished": True}
        return None

    latent = pooling_output.get("latent_audio_feat")
    if isinstance(latent, torch.Tensor) and latent.numel() == 0:
        latent = None

    if latent is None:
        if finished_request:
            return {"code_predictor_codes": [0], "finished": True}
        return None

    sr = pooling_output.get("sr")
    out: dict[str, Any] = {
        "code_predictor_codes": [0],
        "latent_audio_feat": latent.detach().cpu().contiguous() if isinstance(latent, torch.Tensor) else latent,
        "finished": finished_request,
    }
    if isinstance(sr, torch.Tensor):
        out["sr"] = sr.detach().cpu().contiguous()
    return out
