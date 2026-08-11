# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

logger = logging.getLogger(__name__)


def max_ic_for_chunk_size(chunk_size: int) -> int:
    """Largest power of 2 strictly less than chunk_size."""
    if chunk_size <= 2:
        return 1
    return 1 << ((chunk_size - 1).bit_length() - 1)


def compute_dynamic_initial_chunk_size(
    active_requests: int,
    max_num_seqs: int,
    max_ic: int,
) -> int:
    """Select IC from power-of-2 steps [2, 4, ..., max_ic] based on load factor.

    - Low load: small IC (faster TTFA).
    - High load: large IC (amortise decode cost).
    """
    steps: list[int] = []
    v = 2
    while v <= max_ic:
        steps.append(v)
        v <<= 1
    if not steps:
        return max(1, max_ic)
    if max_num_seqs <= 0:
        return steps[0]
    load_factor = min(active_requests / max_num_seqs, 1.0)
    idx = int(round(load_factor * (len(steps) - 1)))
    return steps[idx]


def parse_chunk_ramp(cfg: dict, steady: int | None = None) -> list[int] | None:
    """Parse ``codec_chunk_ramp`` from connector extra config.

    Accepts a list of positive ints or a comma-separated string.
    Returns ``None`` when the key is absent or invalid (ramp disabled).
    Requires >= 2 entries; all entries must be > 0.

    When ``steady`` is provided, warns if ``ramp[-1] != steady`` (the last
    ramp entry should match ``codec_chunk_frames`` to avoid reintroducing
    the cliff the ramp is meant to eliminate).
    """
    raw = cfg.get("codec_chunk_ramp")
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            raw = [int(x.strip()) for x in raw.split(",")]
        ramp = [int(x) for x in raw]
    except (TypeError, ValueError):
        logger.warning("codec_chunk_ramp parse error for %r; disabling ramp.", raw)
        return None
    if len(ramp) < 2:
        logger.warning("codec_chunk_ramp needs >= 2 entries, got %d; disabling.", len(ramp))
        return None
    if any(x <= 0 for x in ramp):
        logger.warning("codec_chunk_ramp entries must be positive; disabling.")
        return None
    if steady is not None and ramp[-1] != steady:
        logger.warning(
            "codec_chunk_ramp[-1]=%d != codec_chunk_frames=%d; this reintroduces the chunk-size cliff at index %d.",
            ramp[-1],
            steady,
            len(ramp) - 1,
        )
    return ramp


def ramp_chunk_size(index: int, ramp: list[int], steady: int) -> int:
    """Return the target chunk size for chunk ``index`` under the ramp schedule.

    Indices within the ramp table use ``ramp[index]``; indices past the table
    fall back to ``steady`` (typically ``codec_chunk_frames``).
    """
    if index < len(ramp):
        return ramp[index]
    return steady


def ramp_cumulative(index: int, ramp: list[int], steady: int) -> int:
    """Cumulative frame count through chunk ``index`` (inclusive).

    O(len(ramp)) closed form: sum the ramp table once, then add
    ``(index + 1 - len(ramp)) * steady`` for indices past the table.
    """
    ramp_len = len(ramp)
    if index < ramp_len:
        return sum(ramp[: index + 1])
    return sum(ramp) + (index + 1 - ramp_len) * steady


def compute_ramp_emit(
    length: int,
    chunk_index: int,
    ramp: list[int],
    steady: int,
    finished: bool,
) -> tuple[bool, int]:
    """Decide whether to emit a chunk under the ramp schedule.

    Uses cumulative thresholds to determine emission boundaries.  Each chunk
    ``i`` covers frames ``[ramp_cumulative(i-1), ramp_cumulative(i))``.

    Returns:
        ``(emit, context_length)``

        - ``emit=False, context_length=0``: not enough frames yet, hold.
        - ``emit=True, context_length>0``: emit with this many new frames.
        - ``emit=True, context_length=0``: finished with no new frames
            (caller should emit an empty-finished sentinel).
    """
    threshold = ramp_cumulative(chunk_index, ramp, steady)
    prev_threshold = ramp_cumulative(chunk_index - 1, ramp, steady) if chunk_index > 0 else 0

    if not finished and length < threshold:
        return False, 0

    if finished and length <= prev_threshold:
        return True, 0

    return True, length - prev_threshold


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
