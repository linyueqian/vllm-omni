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

    __slots__ = ("tier", "frames_emitted", "t_first_emit", "last_emit")

    def __init__(self) -> None:
        self.tier = 0
        self.frames_emitted = 0
        self.t_first_emit: float | None = None
        self.last_emit: float | None = None


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


def parse_chunk_coordinated(cfg: dict) -> dict | None:
    """Parse ``codec_chunk_coordinated`` from connector extra config.

    Coordinated cadence (controller v1): under high system pressure the
    default tier moves UP (large chunks are ~2x cheaper per audio-second, so
    escalating safe streams deflates aggregate demand) and small tiers become
    a rationed budget for the poorest-lead streams. Under low pressure the
    per-stream lead ladder from adaptive v0 applies.
    """
    raw = cfg.get("codec_chunk_coordinated")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        logger.warning("codec_chunk_coordinated must be a mapping; disabling.")
        return None
    try:
        out = {
            "ladder": [int(x) for x in (raw.get("ladder") or [4, 8, 16, 25])],
            "lead_low": float(raw.get("lead_low", 0.6)),
            "lead_high": float(raw.get("lead_high", 1.5)),
            "pressure_high": float(raw.get("pressure_high", 0.85)),
            "pressure_low": float(raw.get("pressure_low", 0.70)),
            "small_frac": float(raw.get("small_frac", 0.25)),
            "escalate_lead": float(raw.get("escalate_lead", 1.0)),
        }
    except (TypeError, ValueError):
        logger.warning("codec_chunk_coordinated parse error for %r; disabling.", raw)
        return None
    ladder = out["ladder"]
    if len(ladder) < 2 or any(x <= 0 for x in ladder) or sorted(ladder) != ladder:
        logger.warning("codec_chunk_coordinated ladder invalid; disabling.")
        return None
    if not (0.0 < out["pressure_low"] < out["pressure_high"]):
        logger.warning("codec_chunk_coordinated pressure band invalid; disabling.")
        return None
    return out


class SystemPressureTracker:
    """Sliding-window aggregate demand/supply ratio with hysteresis."""

    def __init__(self, window_s: float = 2.0) -> None:
        self.window_s = window_s
        self.emissions: list[tuple[float, int]] = []
        self.high = False

    def record(self, now: float, frames: int) -> None:
        self.emissions.append((now, frames))
        cutoff = now - self.window_s
        while self.emissions and self.emissions[0][0] < cutoff:
            self.emissions.pop(0)

    def demand_ratio(self, now: float, active: int) -> float:
        cutoff = now - self.window_s
        frames = sum(f for t, f in self.emissions if t >= cutoff)
        supply = frames * FRAME_SECONDS / self.window_s
        if supply <= 0.0:
            return float("inf") if active > 0 else 0.0
        return active / supply

    def update(self, now: float, active: int, high_thr: float, low_thr: float) -> bool:
        ratio = self.demand_ratio(now, active)
        if self.high and ratio < low_thr:
            self.high = False
        elif not self.high and ratio > high_thr:
            self.high = True
        return self.high


def _stream_lead(state: "AdaptiveCadenceState", now: float) -> float:
    if state.t_first_emit is None:
        return 0.0
    return state.frames_emitted * FRAME_SECONDS - (now - state.t_first_emit)


def compute_coordinated_emit(
    length: int,
    request_id: str,
    states: dict,
    tracker: SystemPressureTracker,
    cfg: dict,
    finished: bool,
    now: float,
) -> tuple[bool, int]:
    """Coordinated emission decision (controller v1).

    Same return contract as ``compute_ramp_emit``. The states dict gives the
    cross-stream view; active streams are those that emitted recently or have
    not yet emitted.
    """
    state = states[request_id]
    pending = length - state.frames_emitted
    if pending <= 0:
        return (True, 0) if finished else (False, 0)

    ladder = cfg["ladder"]
    top = len(ladder) - 1
    active_states = [
        s for s in states.values()
        if s.last_emit is None or (now - s.last_emit) < 3.0
    ]
    active = max(1, len(active_states))
    high = tracker.update(now, active, cfg["pressure_high"], cfg["pressure_low"])

    if high:
        lead = _stream_lead(state, now)
        if lead >= cfg["escalate_lead"]:
            tier = top
        else:
            budget = max(1, int(cfg["small_frac"] * active))
            poorer = sum(
                1 for s in active_states
                if s is not state and _stream_lead(s, now) < lead
            )
            if poorer < budget:
                tier = 0
            else:
                tier = min(state.tier + 1, top)
        state.tier = tier
    else:
        # Low pressure: adaptive v0 behaviour (one-tier hysteresis on lead).
        if state.t_first_emit is not None:
            lead = _stream_lead(state, now)
            if lead > cfg["lead_high"] and state.tier < top:
                state.tier += 1
            elif lead < cfg["lead_low"] and state.tier > 0:
                state.tier -= 1

    target = ladder[state.tier]
    if not finished and pending < target:
        return False, 0
    emit_frames = pending if finished else target
    if state.t_first_emit is None:
        state.t_first_emit = now
    state.last_emit = now
    state.frames_emitted += emit_frames
    tracker.record(now, emit_frames)
    return True, emit_frames
