from __future__ import annotations

import time
from base64 import b64decode
from binascii import Error as BinasciiError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SessionMode(str, Enum):
    TURN = "turn"
    DUPLEX = "duplex"


class DuplexAdapterPattern(str, Enum):
    CHUNK_GROUP_APPEND = "chunk_group_append"
    PER_STEP_TENSOR_INJECT = "per_step_tensor_inject"
    EXPERIMENTAL_WORKER_CONTROL_RPC = "experimental_worker_control_rpc"
    PER_STEP_TENSOR_HANDOFF = "per_step_tensor_handoff"
    SCHEDULER_DATA_PLANE = "scheduler_data_plane"
    RUNNER_LOCAL_PAYLOAD_REF = "runner_local_payload_ref"
    PARALLEL_FRAME_JOINT = "parallel_frame_joint"


class DuplexInputMode(str, Enum):
    APPEND_TOKENS = "append_tokens"
    APPEND_AUDIO_CHUNK = "append_audio_chunk"
    APPEND_STAGE_HANDOFF = "append_stage_handoff"
    APPEND_TTS_HANDOFF = "append_tts_handoff"
    REPLACE_LATEST_CHUNK = "replace_latest_chunk"
    REENCODE_CONTEXT = "reencode_context"
    ROLLBACK_TO_CHECKPOINT = "rollback_to_checkpoint"
    TURN_COMMIT_ONLY = "turn_commit_only"


class DuplexSignalSource(str, Enum):
    MODEL_NATIVE = "model_native"
    EXTERNAL_VAD = "external_vad"
    CLIENT_EVENT = "client_event"
    SERVER_POLICY = "server_policy"
    DIALOGUE_STATE_MODEL = "dialogue_state_model"


@dataclass
class DuplexRuntimeCapabilities:
    """Model/runtime capabilities for a duplex session.

    This is intentionally capability-based. Full-duplex models differ in the
    unit they append or inject, and the engine must not force every model into
    token-append semantics.
    ``supports_core_kv_lease`` is scheduler-owned persistent KV. Model-native
    decoder, TTS, and token2wav state should be reported through
    ``supports_model_internal_state`` so clients and reviewers do not confuse
    internal state with a migratable core KV lease.
    """

    adapter_patterns: set[DuplexAdapterPattern] = field(default_factory=set)
    input_modes: set[DuplexInputMode] = field(default_factory=lambda: {DuplexInputMode.TURN_COMMIT_ONLY})
    signal_sources: set[DuplexSignalSource] = field(
        default_factory=lambda: {
            DuplexSignalSource.CLIENT_EVENT,
            DuplexSignalSource.SERVER_POLICY,
        }
    )
    supports_kv_lease: bool = False
    supports_core_kv_lease: bool = False
    supports_model_internal_state: bool = False
    supports_stage_resumption: bool = False
    supports_scheduler_native_append: bool = False
    supports_core_resumable_request: bool = False
    supports_stage_connector_handoff: bool = False
    supports_independent_io_streams: bool = False
    supports_realtime_endpoint: bool = False
    supports_multi_session: bool = False
    supports_multi_session_same_replica: bool = False
    supports_barge_in: bool = True
    supports_playback_ack: bool = True
    supports_audio_truncate: bool = False
    implementation_level: str = "serving_session_adapter"
    stage_handoff_transport: str | None = None
    chunk_period_ms: int | None = None
    target_barge_in_latency_ms: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter_patterns": sorted(pattern.value for pattern in self.adapter_patterns),
            "input_modes": sorted(mode.value for mode in self.input_modes),
            "signal_sources": sorted(source.value for source in self.signal_sources),
            "supports_kv_lease": self.supports_kv_lease,
            "supports_core_kv_lease": self.supports_core_kv_lease,
            "supports_model_internal_state": self.supports_model_internal_state,
            "supports_stage_resumption": self.supports_stage_resumption,
            "supports_scheduler_native_append": self.supports_scheduler_native_append,
            "supports_core_resumable_request": self.supports_core_resumable_request,
            "supports_stage_connector_handoff": self.supports_stage_connector_handoff,
            "supports_independent_io_streams": self.supports_independent_io_streams,
            "supports_realtime_endpoint": self.supports_realtime_endpoint,
            "supports_multi_session": self.supports_multi_session,
            "supports_multi_session_same_replica": self.supports_multi_session_same_replica,
            "supports_barge_in": self.supports_barge_in,
            "supports_playback_ack": self.supports_playback_ack,
            "supports_audio_truncate": self.supports_audio_truncate,
            "implementation_level": self.implementation_level,
            "stage_handoff_transport": self.stage_handoff_transport,
            "chunk_period_ms": self.chunk_period_ms,
            "target_barge_in_latency_ms": self.target_barge_in_latency_ms,
        }


@dataclass
class DuplexPlaybackCommitCursor:
    generated_ms: int = 0
    sent_ms: int = 0
    played_ms: int = 0
    committed_ms: int = 0

    def mark_generated(self, generated_ms: int) -> None:
        self.generated_ms = max(self.generated_ms, max(0, int(generated_ms)))

    def mark_sent(self, sent_ms: int) -> None:
        self.sent_ms = max(self.sent_ms, max(0, int(sent_ms)))

    def acknowledge(self, played_ms: int, committed_ms: int | None = None) -> None:
        self.played_ms = max(self.played_ms, max(0, int(played_ms)))
        if committed_ms is None:
            committed_ms = self.played_ms
        self.committed_ms = max(self.committed_ms, max(0, int(committed_ms)))

    def as_dict(self) -> dict[str, int]:
        return {
            "generated_ms": self.generated_ms,
            "sent_ms": self.sent_ms,
            "played_ms": self.played_ms,
            "committed_ms": self.committed_ms,
        }


@dataclass
class DuplexStageBinding:
    stage_id: int
    request_id: str
    replica_id: int | None = None
    lease_active: bool = False


@dataclass
class DuplexInputAppend:
    seq: int
    mode: DuplexInputMode
    payload_meta: dict[str, Any] = field(default_factory=dict)
    final: bool = False
    source: DuplexSignalSource = DuplexSignalSource.CLIENT_EVENT


@dataclass
class DuplexSessionRuntimeState:
    session_id: str
    session_mode: SessionMode = SessionMode.DUPLEX
    capabilities: DuplexRuntimeCapabilities = field(default_factory=DuplexRuntimeCapabilities)
    session_config: dict[str, Any] = field(default_factory=dict)
    epoch: int = 0
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    stage_bindings: dict[int, DuplexStageBinding] = field(default_factory=dict)
    input_seq: int = 0
    pending_inputs: list[DuplexInputAppend] = field(default_factory=list)
    playback: DuplexPlaybackCommitCursor = field(default_factory=DuplexPlaybackCommitCursor)
    closed: bool = False

    def touch(self) -> None:
        self.updated_at = time.monotonic()

    def bind_stage_request(self, stage_id: int, request_id: str, replica_id: int | None = None) -> None:
        self.stage_bindings[stage_id] = DuplexStageBinding(
            stage_id=stage_id,
            request_id=request_id,
            replica_id=replica_id,
            lease_active=self.capabilities.supports_core_kv_lease,
        )
        self.touch()

    def stage_request_ids(self) -> list[str]:
        return [binding.request_id for binding in self.stage_bindings.values()]

    def append_input(
        self,
        payload: Any,
        *,
        mode: DuplexInputMode,
        final: bool = False,
        source: DuplexSignalSource = DuplexSignalSource.CLIENT_EVENT,
    ) -> DuplexInputAppend:
        if mode not in self.capabilities.input_modes:
            raise ValueError(f"Duplex input mode {mode.value!r} is not supported by session {self.session_id}")
        self.input_seq += 1
        update = DuplexInputAppend(
            seq=self.input_seq,
            mode=mode,
            payload_meta=self._payload_metadata(payload),
            final=final,
            source=source,
        )
        self.pending_inputs.append(update)
        self.touch()
        return update

    @staticmethod
    def _payload_metadata(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            meta: dict[str, Any] = {
                "type": "dict",
                "keys": sorted(str(key) for key in payload.keys()),
            }
            audio = payload.get("audio") or payload.get("data")
            if isinstance(audio, str):
                try:
                    meta["audio_bytes"] = len(b64decode(audio, validate=True))
                except (BinasciiError, ValueError):
                    meta["audio_chars"] = len(audio)
            fmt = payload.get("format")
            if isinstance(fmt, str):
                meta["format"] = fmt
            sample_rate = payload.get("sample_rate_hz")
            if isinstance(sample_rate, int | float):
                meta["sample_rate_hz"] = int(sample_rate)
            return meta
        if isinstance(payload, str):
            return {"type": "str", "chars": len(payload)}
        if isinstance(payload, bytes | bytearray | memoryview):
            return {"type": type(payload).__name__, "bytes": len(payload)}
        if isinstance(payload, list | tuple):
            return {"type": type(payload).__name__, "items": len(payload)}
        return {"type": type(payload).__name__}

    def acknowledge_playback(self, played_ms: int, committed_ms: int | None = None) -> None:
        self.playback.acknowledge(played_ms, committed_ms)
        self.touch()

    def barge_in(self) -> tuple[int, list[str]]:
        # Stage0 is the single long-lived resumable request that owns the
        # conversation KV/context (see Orchestrator._duplex_stage_request_topology:
        # "stage0_long_lived_request": True). A barge-in must stop downstream
        # output but PRESERVE stage0 so conversation memory survives the
        # interruption, matching the official MiniCPM-o continuous design where
        # an interruption stops current speech but keeps context. Only
        # downstream (stage_id > 0) requests are torn down here.
        stale_request_ids = [binding.request_id for stage_id, binding in self.stage_bindings.items() if stage_id != 0]
        self.epoch += 1
        self.pending_inputs.clear()
        self.stage_bindings = {stage_id: binding for stage_id, binding in self.stage_bindings.items() if stage_id == 0}
        self.touch()
        return self.epoch, stale_request_ids

    def close(self) -> list[str]:
        self.closed = True
        stale_request_ids = self.stage_request_ids()
        self.stage_bindings.clear()
        self.pending_inputs.clear()
        self.touch()
        return stale_request_ids

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_mode": self.session_mode.value,
            "epoch": self.epoch,
            "input_seq": self.input_seq,
            "closed": self.closed,
            "stage_bindings": {
                stage_id: {
                    "request_id": binding.request_id,
                    "replica_id": binding.replica_id,
                    "lease_active": binding.lease_active,
                }
                for stage_id, binding in self.stage_bindings.items()
            },
            "playback": self.playback.as_dict(),
            "capabilities": self.capabilities.as_dict(),
        }


class DuplexSessionRuntimeManager:
    def __init__(self) -> None:
        self._sessions: dict[str, DuplexSessionRuntimeState] = {}

    def open_session(
        self,
        session_id: str,
        *,
        session_mode: SessionMode = SessionMode.DUPLEX,
        capabilities: DuplexRuntimeCapabilities | None = None,
        session_config: dict[str, Any] | None = None,
    ) -> DuplexSessionRuntimeState:
        if session_id in self._sessions:
            raise ValueError(f"Duplex session already exists: {session_id}")
        session = DuplexSessionRuntimeState(
            session_id=session_id,
            session_mode=session_mode,
            capabilities=capabilities or DuplexRuntimeCapabilities(),
            session_config=dict(session_config or {}),
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> DuplexSessionRuntimeState | None:
        return self._sessions.get(session_id)

    def require(self, session_id: str) -> DuplexSessionRuntimeState:
        session = self.get(session_id)
        if session is None:
            raise KeyError(f"Unknown duplex session: {session_id}")
        return session

    def close_session(self, session_id: str) -> DuplexSessionRuntimeState | None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()
        return session

    def close_sessions_for_request_ids(self, request_ids: list[str]) -> dict[str, list[str]]:
        """Close duplex sessions that own any of the given stage request ids."""
        request_id_set = set(request_ids)
        closed: dict[str, list[str]] = {}
        if not request_id_set:
            return closed
        for session_id, session in list(self._sessions.items()):
            stale_request_ids = session.stage_request_ids()
            if request_id_set.isdisjoint(stale_request_ids):
                continue
            self._sessions.pop(session_id, None)
            session.close()
            closed[session_id] = stale_request_ids
        return closed


def duplex_data_plane_request_info(result: dict[str, object]) -> tuple[str | None, int | None]:
    """Extract the scheduler data-plane request from a duplex control result."""
    stage_results = result.get("stage_results")
    if not isinstance(stage_results, list):
        return None, None
    for item in stage_results:
        if not isinstance(item, dict):
            continue
        inner = item.get("result")
        if not isinstance(inner, dict) or inner.get("data_plane_append") is not True:
            continue
        request_id = inner.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            continue
        response_stage_id = inner.get("response_stage_id")
        return request_id, response_stage_id if isinstance(response_stage_id, int) else None
    return None, None


# Audio framing contract shared with serving and the MiniCPM-o worker
# (MiniCPMO45DuplexPolicy): 1 s units at 16 kHz, one pooled audio embedding per
# 100 ms, plus <unit> and a </unit> closure per unit. Slot budgets must match
# the worker-built embeddings exactly: surplus slots become pad embeddings in
# the KV and measurably corrupt the model's listen/speak behavior.
_DUPLEX_CHUNK_SAMPLES = 16000
_DUPLEX_SAMPLES_PER_AUDIO_TOKEN = 1600


def _duplex_pcm_sample_count(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    audio = payload.get("audio") or payload.get("data")
    if payload.get("format") != "pcm_f32le" or not isinstance(audio, str):
        return None
    try:
        raw = b64decode(audio, validate=True)
    except (BinasciiError, ValueError):
        return None
    return len(raw) // 4


def duplex_payload_is_exact_chunks(payload: object) -> bool:
    """True when the payload is a whole number of model audio chunks."""
    sample_count = _duplex_pcm_sample_count(payload)
    return bool(sample_count) and sample_count % _DUPLEX_CHUNK_SAMPLES == 0


def duplex_scheduler_token_budget(payload: object, *, default: int = 64) -> int:
    """Scheduler token slots for a duplex data-plane input.

    vLLM schedulers still need token-shaped admission data. Keep that budget
    calculation inside the duplex data-plane layer instead of letting clients
    or serving adapters smuggle placeholder-token counts through request
    payloads.
    """
    sample_count = _duplex_pcm_sample_count(payload)
    if sample_count is None:
        return max(1, int(default))
    sample_count = max(1, sample_count)
    if sample_count % _DUPLEX_CHUNK_SAMPLES == 0:
        # Exact path used by the serving adapters: per unit, a </unit>
        # closure + <unit> + the pooled audio embeddings.
        units = sample_count // _DUPLEX_CHUNK_SAMPLES
        return units * (2 + _DUPLEX_CHUNK_SAMPLES // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN)
    # Legacy partial payloads: keep a small margin; the worker pads.
    return max(16, min(768, sample_count // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN + 8))


def duplex_first_append_context_reserve(session_config: object) -> int:
    """Extra scheduler token slots for the session-context prefix.

    The first data-plane append carries the system prompt and optional
    reference-audio embeddings in front of the first audio unit. Without this
    reserve the worker-built embeddings can exceed the scheduled prompt slots
    and the audio tail would be truncated.
    """
    if not isinstance(session_config, dict):
        return 48
    sources: list[dict[str, Any]] = [session_config]
    extra_body = session_config.get("extra_body")
    if isinstance(extra_body, dict):
        sources.append(extra_body)
    # Serving adapters precompute the exact context token count (system
    # template via the tokenizer + pooled reference-audio embeddings).
    for source in sources:
        exact = source.get("duplex_first_append_context_tokens")
        if isinstance(exact, int) and exact >= 0:
            return exact
    reserve = 48  # heuristic system prompt template tokens
    for source in sources:
        ref = source.get("ref_audio_data")
        if not isinstance(ref, str) or not ref:
            continue
        try:
            raw = b64decode(ref, validate=True)
        except (BinasciiError, ValueError):
            continue
        # pcm_f32le reference audio at one pooled token per 100 ms frame.
        reserve += max(0, (len(raw) // 4) // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN + 8)
        break
    return reserve
