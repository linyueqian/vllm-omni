"""Apply the Legato controller v0 (adaptive cadence) patch to the worktree.

Anchored string replacements; fails loudly if any anchor is missing.
Run from ~/proj/vllm-omni-legato.
"""

import py_compile

UTILS = "vllm_omni/model_executor/stage_input_processors/chunk_size_utils.py"
PROC = "vllm_omni/model_executor/stage_input_processors/qwen3_tts.py"

utils_src = open(UTILS).read()
assert "compute_adaptive_emit" not in utils_src, "patch already applied"

utils_src += '''

FRAME_SECONDS = 0.08  # 12.5 Hz codec


def parse_chunk_adaptive(cfg: dict) -> dict | None:
    """Parse ``codec_chunk_adaptive`` from connector extra config.

    Expects a mapping with optional keys: ``ladder`` (ascending positive
    ints) and ``lead_low`` / ``lead_high`` (seconds, hysteresis band).
    Returns ``None`` when absent or invalid (adaptive cadence disabled).
    """
    raw = cfg.get("codec_chunk_adaptive")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        logger.warning("codec_chunk_adaptive must be a mapping; disabling.")
        return None
    try:
        ladder = [int(x) for x in (raw.get("ladder") or [4, 8, 16, 25])]
        lead_low = float(raw.get("lead_low", 0.6))
        lead_high = float(raw.get("lead_high", 1.5))
    except (TypeError, ValueError):
        logger.warning("codec_chunk_adaptive parse error for %r; disabling.", raw)
        return None
    if len(ladder) < 2 or any(x <= 0 for x in ladder) or sorted(ladder) != ladder:
        logger.warning(
            "codec_chunk_adaptive ladder must be >= 2 ascending positive ints; disabling."
        )
        return None
    if not 0.0 <= lead_low < lead_high:
        logger.warning("codec_chunk_adaptive needs 0 <= lead_low < lead_high; disabling.")
        return None
    return {"ladder": ladder, "lead_low": lead_low, "lead_high": lead_high}


class AdaptiveCadenceState:
    """Per-request state for closed-loop cadence control."""

    __slots__ = ("tier", "frames_emitted", "t_first_emit")

    def __init__(self) -> None:
        self.tier = 0
        self.frames_emitted = 0
        self.t_first_emit: float | None = None


def compute_adaptive_emit(
    length: int,
    state: AdaptiveCadenceState,
    ladder: list[int],
    lead_low: float,
    lead_high: float,
    finished: bool,
    now: float,
) -> tuple[bool, int]:
    """Closed-loop emission decision: chunk size follows banked buffer lead.

    Lead is conservative: playback is assumed to start at the first emission,
    so the real client lead is at least ``lead + client_buffer``. The tier
    moves at most one step per emitted chunk (hysteresis band
    ``[lead_low, lead_high]``), per the stability requirements.

    Returns ``(emit, context_length)`` with the same contract as
    ``compute_ramp_emit``.
    """
    pending = length - state.frames_emitted
    if pending <= 0:
        return (True, 0) if finished else (False, 0)
    target = ladder[state.tier]
    if not finished and pending < target:
        return False, 0
    emit_frames = pending if finished else target
    if state.t_first_emit is None:
        state.t_first_emit = now
    else:
        lead = state.frames_emitted * FRAME_SECONDS - (now - state.t_first_emit)
        if lead > lead_high and state.tier < len(ladder) - 1:
            state.tier += 1
        elif lead < lead_low and state.tier > 0:
            state.tier -= 1
    state.frames_emitted += emit_frames
    return True, emit_frames
'''

open(UTILS, "w").write(utils_src)

proc_src = open(PROC).read()
assert "compute_adaptive_emit" not in proc_src, "processor already patched"

old_import = """from collections.abc import Mapping
from typing import Any

import torch"""
new_import = """import time
from collections.abc import Mapping
from typing import Any

import torch"""
assert old_import in proc_src
proc_src = proc_src.replace(old_import, new_import, 1)

old_utils_import = """from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import (
    compute_dynamic_initial_chunk_size,
    compute_ramp_emit,
    max_ic_for_chunk_size,
    parse_chunk_ramp,
)"""
new_utils_import = """from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import (
    AdaptiveCadenceState,
    compute_adaptive_emit,
    compute_dynamic_initial_chunk_size,
    compute_ramp_emit,
    max_ic_for_chunk_size,
    parse_chunk_adaptive,
    parse_chunk_ramp,
)"""
assert old_utils_import in proc_src
proc_src = proc_src.replace(old_utils_import, new_utils_import, 1)

old_ramp_parse = """    if not hasattr(transfer_manager, "_ramp_parsed"):
        transfer_manager._ramp_parsed = parse_chunk_ramp(cfg, steady=chunk_size)
    ramp = transfer_manager._ramp_parsed"""
new_ramp_parse = old_ramp_parse + """

    # Adaptive cadence (Legato controller v0): closed-loop chunk sizing from
    # banked buffer lead. Supersedes the static ramp when configured.
    if not hasattr(transfer_manager, "_adaptive_parsed"):
        transfer_manager._adaptive_parsed = parse_chunk_adaptive(cfg)
    adaptive = transfer_manager._adaptive_parsed
    if adaptive is not None:
        ramp = None"""
assert old_ramp_parse in proc_src
proc_src = proc_src.replace(old_ramp_parse, new_ramp_parse, 1)

old_branch = """    if ramp is not None:
        chunk_index = transfer_manager.ramp_chunk_count.get(request_id, 0)"""
new_branch = """    if adaptive is not None:
        adaptive_states = getattr(transfer_manager, "_adaptive_states", None)
        if adaptive_states is None:
            adaptive_states = {}
            transfer_manager._adaptive_states = adaptive_states
        if request_id not in adaptive_states:
            adaptive_states[request_id] = AdaptiveCadenceState()
        emit, context_length = compute_adaptive_emit(
            length,
            adaptive_states[request_id],
            adaptive["ladder"],
            adaptive["lead_low"],
            adaptive["lead_high"],
            finished,
            time.time(),
        )
        if not emit:
            return None
        if context_length == 0:
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
    elif ramp is not None:
        chunk_index = transfer_manager.ramp_chunk_count.get(request_id, 0)"""
assert old_branch in proc_src
proc_src = proc_src.replace(old_branch, new_branch, 1)

open(PROC, "w").write(proc_src)

py_compile.compile(UTILS, doraise=True)
py_compile.compile(PROC, doraise=True)
print("PATCH OK: adaptive cadence v0 applied and compiled")
