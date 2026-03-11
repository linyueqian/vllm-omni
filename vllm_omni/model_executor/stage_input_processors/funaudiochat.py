from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

_MAX_COSYVOICE_TOKEN_ID = 6561
logger = init_logger(__name__)


def _validate_stage_inputs(stage_list: list[Any], engine_input_source: list[int]) -> Any:
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    stage_outputs = stage_list[source_stage_id].engine_outputs
    if stage_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")
    return stage_outputs


def _to_flat_audio_token_ids(audio_token_ids: Any) -> torch.Tensor:
    if not isinstance(audio_token_ids, torch.Tensor):
        audio_token_ids = torch.as_tensor(audio_token_ids, dtype=torch.long)
    audio_token_ids = audio_token_ids.to(dtype=torch.long)
    if audio_token_ids.ndim == 2:
        # Token id 0 is valid for code2wav. Only drop rows that are fully negative
        # placeholders, and preserve all-zero codec groups from stage-0.
        valid_rows = (audio_token_ids >= 0).any(dim=-1)
        audio_token_ids = audio_token_ids[valid_rows]
    return audio_token_ids.reshape(-1)


def funaudiochat2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Convert FunAudioChat stage-0 audio codec output into code2wav prompts."""
    del prompt, requires_multimodal_data

    stage_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs: list[OmniTokensPrompt] = []
    for stage_output in stage_outputs:
        output = stage_output.outputs[0]
        mm_output = getattr(output, "multimodal_output", None) or {}
        audio_token_ids = mm_output.get("audio_token_ids")
        if audio_token_ids is None:
            audio_token_ids = mm_output.get("speech_ids")
        if audio_token_ids is None:
            raise ValueError("Stage-0 FunAudioChat output does not contain `speech_ids` or `audio_token_ids`.")
        flat_audio_token_ids = _to_flat_audio_token_ids(audio_token_ids)
        filtered = flat_audio_token_ids[(flat_audio_token_ids >= 0) & (flat_audio_token_ids < _MAX_COSYVOICE_TOKEN_ID)]
        raw_min = int(flat_audio_token_ids.min().item()) if flat_audio_token_ids.numel() > 0 else None
        raw_max = int(flat_audio_token_ids.max().item()) if flat_audio_token_ids.numel() > 0 else None
        logger.info(
            "FunAudioChat stage0->stage1 audio tokens: raw_len=%d filtered_len=%d raw_min=%s raw_max=%s tail=%s",
            flat_audio_token_ids.numel(),
            filtered.numel(),
            raw_min,
            raw_max,
            flat_audio_token_ids[-8:].tolist() if flat_audio_token_ids.numel() > 0 else [],
        )
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=filtered.to(dtype=torch.long).reshape(-1).tolist(),
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs
