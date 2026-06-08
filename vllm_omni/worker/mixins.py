from __future__ import annotations

import traceback
from typing import Any


class OmniWorkerMixin:
    """Mixin to ensure Omni plugins are loaded in worker processes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()

    def open_duplex_session_async(
        self,
        session_id: str,
        *,
        epoch: int,
        capabilities: dict[str, Any],
        session_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Default worker-side duplex hook for stages without native sessions."""
        target = self._get_native_duplex_target(capabilities)
        if target is not None:
            active_sessions = self._native_duplex_sessions()
            if (
                session_id not in active_sessions
                and active_sessions
                and not self._is_passive_native_duplex_stage(target)
                and not self._supports_multiple_native_duplex_sessions(target)
            ):
                return {
                    "supported": False,
                    "session_id": session_id,
                    "epoch": epoch,
                    "reason": "native_duplex_session_busy",
                    "active_session_ids": sorted(active_sessions),
                }
            try:
                result, session_target = self._open_native_duplex_session(
                    target,
                    session_id=session_id,
                    epoch=epoch,
                    capabilities=capabilities,
                    session_config=dict(session_config or {}),
                )
                self._native_duplex_sessions()[session_id] = session_target
                return result
            except Exception as exc:
                return {
                    "supported": False,
                    "session_id": session_id,
                    "epoch": epoch,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
        return {
            "supported": False,
            "session_id": session_id,
            "epoch": epoch,
            "capabilities": capabilities,
            "reason": "worker_duplex_session_not_implemented",
        }

    def append_duplex_input_async(
        self,
        session_id: str,
        *,
        epoch: int,
        seq: int,
        mode: str,
        payload: Any,
        final: bool,
    ) -> dict[str, Any]:
        """Default worker-side duplex input hook for non-append stages."""
        target = self._native_duplex_sessions().get(session_id)
        if target is not None:
            try:
                from vllm_omni.worker.native_duplex import append_native_duplex_input

                return append_native_duplex_input(
                    target,
                    session_id=session_id,
                    epoch=epoch,
                    seq=seq,
                    mode=mode,
                    payload=payload,
                    final=final,
                )
            except Exception as exc:
                return {
                    "supported": False,
                    "session_id": session_id,
                    "epoch": epoch,
                    "seq": seq,
                    "mode": mode,
                    "final": final,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
        return {
            "supported": False,
            "session_id": session_id,
            "epoch": epoch,
            "seq": seq,
            "mode": mode,
            "final": final,
            "reason": "worker_duplex_input_not_implemented",
        }

    def put_duplex_stage_payload_async(
        self,
        session_id: str,
        *,
        epoch: int,
        seq: int,
        payload_ref: str,
        payload: Any,
    ) -> dict[str, Any]:
        """Store a full duplex stage handoff payload on this worker.

        This mirrors the runner-local full-payload cache contract: the control
        plane only carries a stable reference after the payload has been staged
        on the target worker.
        """
        target = self._native_duplex_sessions().get(session_id)
        if target is None:
            return {
                "supported": False,
                "session_id": session_id,
                "epoch": epoch,
                "seq": seq,
                "payload_ref": payload_ref,
                "reason": "worker_duplex_session_not_open",
            }
        try:
            put_fn = getattr(target, "put_duplex_stage_payload", None)
            if callable(put_fn):
                native = put_fn(
                    session_id=session_id,
                    epoch=epoch,
                    seq=seq,
                    payload_ref=payload_ref,
                    payload=payload,
                )
            else:
                store = getattr(target, "_duplex_stage_payload_store", None)
                if not isinstance(store, dict):
                    store = {}
                    setattr(target, "_duplex_stage_payload_store", store)
                store[payload_ref] = payload
                native = {
                    "payload_ref": payload_ref,
                    "payload_cached": True,
                }
            model_runner = getattr(self, "model_runner", None)
            put_local_stage_payload = getattr(model_runner, "put_local_stage_payload", None)
            if callable(put_local_stage_payload):
                put_local_stage_payload(payload_ref, payload)
            return {
                "supported": True,
                "session_id": session_id,
                "epoch": epoch,
                "seq": seq,
                "payload_ref": payload_ref,
                "implementation_level": "model_native_duplex",
                "native_result": dict(native) if isinstance(native, dict) else {"value": native},
            }
        except Exception as exc:
            return {
                "supported": False,
                "session_id": session_id,
                "epoch": epoch,
                "seq": seq,
                "payload_ref": payload_ref,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

    def signal_duplex_turn_async(
        self,
        session_id: str,
        *,
        epoch: int,
        event: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Default worker-side duplex signal hook for stages without turn policy."""
        target = self._native_duplex_sessions().get(session_id)
        if target is not None:
            try:
                from vllm_omni.worker.native_duplex import as_native_result_dict

                signal_fn = getattr(target, "signal_duplex_turn", None)
                if callable(signal_fn):
                    native = signal_fn(
                        session_id=session_id,
                        epoch=epoch,
                        event=event,
                        payload=dict(payload or {}),
                    )
                    return {
                        "supported": True,
                        "session_id": session_id,
                        "epoch": epoch,
                        "event": event,
                        "payload": dict(payload or {}),
                        "implementation_level": "model_native_duplex",
                        "native_result": as_native_result_dict(native, target=target),
                    }
                if event in {"barge_in", "input.cancel", "response.cancel"}:
                    break_fn = getattr(target, "duplex_set_break", None) or getattr(target, "set_break", None)
                    if callable(break_fn):
                        break_fn()
                return {
                    "supported": True,
                    "session_id": session_id,
                    "epoch": epoch,
                    "event": event,
                    "payload": dict(payload or {}),
                    "implementation_level": "model_native_duplex",
                }
            except Exception as exc:
                return {
                    "supported": False,
                    "session_id": session_id,
                    "epoch": epoch,
                    "event": event,
                    "payload": dict(payload or {}),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
        return {
            "supported": False,
            "session_id": session_id,
            "epoch": epoch,
            "event": event,
            "payload": dict(payload or {}),
            "reason": "worker_duplex_signal_not_implemented",
        }

    def close_duplex_session_async(
        self,
        session_id: str,
        *,
        epoch: int,
        reason: str,
    ) -> dict[str, Any]:
        """Default worker-side duplex close hook for stages without native sessions."""
        target = self._native_duplex_sessions().get(session_id)
        if target is not None:
            cleanup_ok = False
            try:
                close_fn = getattr(target, "close_duplex_session", None)
                if callable(close_fn):
                    close_fn(session_id=session_id, epoch=epoch, reason=reason)
                else:
                    stop_fn = (
                        getattr(target, "duplex_stop", None)
                        or getattr(target, "stop", None)
                        or getattr(target, "set_session_stop", None)
                    )
                    if callable(stop_fn):
                        stop_fn()
                    reset_fn = getattr(target, "reset_session", None)
                    if callable(reset_fn):
                        reset_fn()
                cleanup_fn = getattr(target, "duplex_cleanup", None) or getattr(target, "cleanup", None)
                if callable(cleanup_fn):
                    cleanup_fn()
                cleanup_ok = True
                return {
                    "supported": True,
                    "session_id": session_id,
                    "epoch": epoch,
                    "reason": reason,
                    "implementation_level": "model_native_duplex",
                }
            except Exception as exc:
                return {
                    "supported": False,
                    "session_id": session_id,
                    "epoch": epoch,
                    "reason": reason,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            finally:
                if cleanup_ok:
                    self._native_duplex_sessions().pop(session_id, None)
        return {
            "supported": False,
            "session_id": session_id,
            "epoch": epoch,
            "reason": reason,
        }

    def _native_duplex_sessions(self) -> dict[str, Any]:
        sessions = getattr(self, "_omni_native_duplex_sessions", None)
        if sessions is None:
            sessions = {}
            self._omni_native_duplex_sessions = sessions
        return sessions

    def _get_native_duplex_target(self, capabilities: dict[str, Any] | None = None) -> Any | None:
        if capabilities is not None and capabilities.get("implementation_level") != "model_native_duplex":
            return None

        from vllm_omni.worker.native_duplex import get_native_duplex_target

        return get_native_duplex_target(self, capabilities)

    @staticmethod
    def _is_passive_native_duplex_stage(target: Any) -> bool:
        from vllm_omni.worker.native_duplex import is_passive_native_duplex_stage

        return is_passive_native_duplex_stage(target)

    @staticmethod
    def _supports_multiple_native_duplex_sessions(target: Any) -> bool:
        return bool(getattr(target, "supports_multiple_native_duplex_sessions", False))

    def _open_native_duplex_session(
        self,
        target: Any,
        *,
        session_id: str,
        epoch: int,
        capabilities: dict[str, Any],
        session_config: dict[str, Any],
    ) -> tuple[dict[str, Any], Any]:
        from vllm_omni.worker.native_duplex import open_native_duplex_session

        return open_native_duplex_session(
            target,
            session_id=session_id,
            epoch=epoch,
            capabilities=capabilities,
            session_config=session_config,
        )
