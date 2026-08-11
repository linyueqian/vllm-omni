"""Apply the Legato controller v1 (coordinated cadence) patch.

Anchored replacements on top of the v0 patch; fails loudly if anchors miss.
Run from ~/proj/vllm-omni-legato.
"""

import py_compile

UTILS = "vllm_omni/model_executor/stage_input_processors/chunk_size_utils.py"
PROC = "vllm_omni/model_executor/stage_input_processors/qwen3_tts.py"

utils_src = open(UTILS).read()
assert "compute_adaptive_emit" in utils_src, "v0 patch missing"
assert "compute_coordinated_emit" not in utils_src, "v1 already applied"

old_slots = '''    __slots__ = ("tier", "frames_emitted", "t_first_emit")

    def __init__(self) -> None:
        self.tier = 0
        self.frames_emitted = 0
        self.t_first_emit: float | None = None'''
new_slots = '''    __slots__ = ("tier", "frames_emitted", "t_first_emit", "last_emit")

    def __init__(self) -> None:
        self.tier = 0
        self.frames_emitted = 0
        self.t_first_emit: float | None = None
        self.last_emit: float | None = None'''
assert old_slots in utils_src
utils_src = utils_src.replace(old_slots, new_slots, 1)

utils_src += '''

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
'''

open(UTILS, "w").write(utils_src)

proc_src = open(PROC).read()
assert "compute_adaptive_emit" in proc_src, "v0 processor patch missing"
assert "compute_coordinated_emit" not in proc_src, "v1 processor already patched"

old_imp = """    AdaptiveCadenceState,
    compute_adaptive_emit,"""
new_imp = """    AdaptiveCadenceState,
    SystemPressureTracker,
    compute_adaptive_emit,
    compute_coordinated_emit,
    parse_chunk_coordinated,"""
assert old_imp in proc_src
proc_src = proc_src.replace(old_imp, new_imp, 1)

old_parse = """    if not hasattr(transfer_manager, "_adaptive_parsed"):
        transfer_manager._adaptive_parsed = parse_chunk_adaptive(cfg)
    adaptive = transfer_manager._adaptive_parsed
    if adaptive is not None:
        ramp = None"""
new_parse = old_parse + """

    # Coordinated cadence (Legato controller v1): supersedes adaptive and ramp.
    if not hasattr(transfer_manager, "_coordinated_parsed"):
        transfer_manager._coordinated_parsed = parse_chunk_coordinated(cfg)
    coordinated = transfer_manager._coordinated_parsed
    if coordinated is not None:
        adaptive = None
        ramp = None"""
assert old_parse in proc_src
proc_src = proc_src.replace(old_parse, new_parse, 1)

old_branch = """    if adaptive is not None:
        adaptive_states = getattr(transfer_manager, "_adaptive_states", None)"""
new_branch = """    if coordinated is not None:
        coord_states = getattr(transfer_manager, "_adaptive_states", None)
        if coord_states is None:
            coord_states = {}
            transfer_manager._adaptive_states = coord_states
        if request_id not in coord_states:
            coord_states[request_id] = AdaptiveCadenceState()
        tracker = getattr(transfer_manager, "_pressure_tracker", None)
        if tracker is None:
            tracker = SystemPressureTracker()
            transfer_manager._pressure_tracker = tracker
        emit, context_length = compute_coordinated_emit(
            length,
            request_id,
            coord_states,
            tracker,
            coordinated,
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
    elif adaptive is not None:
        adaptive_states = getattr(transfer_manager, "_adaptive_states", None)"""
assert old_branch in proc_src
proc_src = proc_src.replace(old_branch, new_branch, 1)

open(PROC, "w").write(proc_src)

py_compile.compile(UTILS, doraise=True)
py_compile.compile(PROC, doraise=True)
print("PATCH OK: coordinated cadence v1 applied and compiled")
