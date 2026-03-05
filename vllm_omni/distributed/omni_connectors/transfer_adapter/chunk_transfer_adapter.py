# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import os
import time
from collections import defaultdict, deque
from typing import Any

import torch
from vllm.v1.request import Request, RequestStatus

from vllm_omni.utils.async_chunk_profile import async_chunk_timer, log_async_chunk_event, log_async_chunk_profile

from ..factory import OmniConnectorFactory
from ..utils.config import ConnectorSpec
from ..utils.logging import get_connector_logger
from .base import OmniTransferAdapterBase

logger = get_connector_logger(__name__)


class OmniChunkTransferAdapter(OmniTransferAdapterBase):
    """Chunk-level transfer adapter for Omni connector pipelines.

    This class coordinates per-request chunk exchange between adjacent stages,
    and implements asynchronous get/put of chunks via background threads.
    It tracks per-request chunk indices for put/get, and accumulates
    payloads across chunks (concatenating tensors/lists in AR mode). It also
    caches prompt token ids and additional information for scheduler use.

    Scheduler integration is handled via WAITING_FOR_CHUNK transitions:
    requests are moved to waiting for chunk deque while polling, then restored
    to waiting/running queues once a chunk arrives. The requests will finish
    loading chunk util detecting the payload "finished" flag.

    The base class owns background recv/save loops; load/save only enqueue
    work and return immediately.
    """

    def __init__(self, vllm_config: Any):
        model_config = vllm_config.model_config
        self.scheduler_max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.connector = self.create_connector(model_config)
        super().__init__(model_config)
        self.model_mode = getattr(model_config, "worker_type", "ar")
        # State specific to Chunk management
        self.custom_process_next_stage_input_func = None
        custom_process_next_stage_input_func = getattr(model_config, "custom_process_next_stage_input_func", None)
        if custom_process_next_stage_input_func:
            module_path, func_name = custom_process_next_stage_input_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.custom_process_next_stage_input_func = getattr(module, func_name)
        # mapping for request id and chunk id
        self.put_req_chunk: dict[str, int] = defaultdict(int)
        self.get_req_chunk: dict[str, int] = defaultdict(int)
        self.finished_requests: set[str] = set()
        self.request_payload = {}
        self.code_prompt_token_ids: dict[str, list[list[int]]] = defaultdict(list)
        self.code_prompt_frame_counts: dict[str, int] = defaultdict(int)
        self.request_ids_mapping: dict[str, str] = {}

        self.waiting_for_chunk_waiting_requests: deque[Any] = deque()
        self.waiting_for_chunk_running_requests: deque[Any] = deque()
        self.requests_with_ready_chunks = set()

    @classmethod
    def create_connector(cls, model_config: Any):
        connector_config = getattr(model_config, "stage_connector_config", None)
        if connector_config is None:
            connector_config = {}
        elif not isinstance(connector_config, dict):
            connector_config = {
                "name": getattr(connector_config, "name", None),
                "extra": getattr(connector_config, "extra", {}),
            }

        connector_specs = ConnectorSpec(
            name=connector_config.get("name", "SharedMemoryConnector"),
            extra=connector_config.get("extra", {}),
        )
        return OmniConnectorFactory.create_connector(connector_specs)

    def _defer_code_window_on_load_enabled(self) -> bool:
        raw_cfg = getattr(self.connector, "config", {}) or {}
        cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
        if isinstance(cfg, dict) and "codec_defer_window_to_load" in cfg:
            return bool(cfg.get("codec_defer_window_to_load"))
        env_val = os.environ.get("VLLM_OMNI_ASYNC_CHUNK_DEFER_CODE_WINDOW", "")
        return env_val.strip().lower() in ("1", "true", "yes")

    def _build_code2wav_prompt_ids(
        self,
        payload_data: dict[str, Any],
        request_id: str,
    ) -> list[int] | None:
        """Build code2wav prompt ids from payload.

        Returns None when deferred-window mode needs more frames before emitting.
        """
        if self._defer_code_window_on_load_enabled() and "code_predictor_new_frames" in payload_data:
            new_frames = payload_data.get("code_predictor_new_frames", [])
            if new_frames:
                request_cache = self.code_prompt_token_ids[request_id]
                for frame in new_frames:
                    if isinstance(frame, torch.Tensor):
                        request_cache.append(frame.reshape(-1).tolist())
                    elif hasattr(frame, "_x"):
                        request_cache.append(list(frame._x))
                    elif isinstance(frame, list):
                        request_cache.append(frame)
                    else:
                        request_cache.append(list(frame))
                self.code_prompt_frame_counts[request_id] += len(new_frames)

            chunk_size = int(payload_data.get("codec_chunk_frames", 25))
            left_context_size = int(payload_data.get("codec_left_context_frames", 25))
            if chunk_size <= 0 or left_context_size < 0:
                return []

            request_cache = self.code_prompt_token_ids[request_id]
            total_frames = self.code_prompt_frame_counts[request_id]
            finished = bool(payload_data.get("finished"))
            if total_frames <= 0:
                return [] if finished else None
            chunk_length = total_frames % chunk_size
            if chunk_length != 0 and not finished:
                return None

            context_length = chunk_length if chunk_length != 0 else chunk_size
            end_index = min(len(request_cache), left_context_size + context_length)
            prompt_ids = torch.tensor(request_cache[-end_index:]).transpose(0, 1).reshape(-1).tolist()
            # Keep only required frames for future left-context assembly.
            max_keep = left_context_size + chunk_size
            if len(request_cache) > max_keep:
                del request_cache[: len(request_cache) - max_keep]
            return prompt_ids

        new_ids = payload_data.get("code_predictor_codes", [])
        if isinstance(new_ids, torch.Tensor):
            return new_ids.reshape(-1).tolist()
        if hasattr(new_ids, "_x"):
            return list(new_ids._x)
        return new_ids

    def load_async(self, request: Request):
        """Register a request for asynchronous chunk retrieval.

        This method does not read from the connector directly. It records
        request metadata and enqueues the request id for the background
        receive loop to poll.

        Stage-0 has no upstream producer, so this call is a no-op there.

        Args:
            request: The request object needing data.
        """
        stage_id = self.connector.stage_id

        if stage_id == 0:
            return
        if not hasattr(request, "additional_information"):
            request.additional_information = None
        request._async_chunk_load_enqueue_ts = time.perf_counter()
        self._pending_load_reqs.append(request)
        log_async_chunk_profile(
            "chunk_adapter_load_enqueue",
            0.0,
            extra=f"stage={stage_id} qlen={len(self._pending_load_reqs)}",
        )

    def save_async(
        self,
        pooling_output: torch.Tensor | None = None,
        request: Request | None = None,
    ):
        """Build and enqueue one chunk for asynchronous sending.

        Payload extraction happens in ``_send_single_request`` on the
        background save_loop thread.

        Args:
            pooling_output: Partial pooling output dictionary
            request: Request object
        """
        task = {
            "pooling_output": pooling_output,
            "request": request,
            "is_finished": request.is_finished(),
            "enqueue_ts": time.perf_counter(),
        }
        self._pending_save_reqs.append(task)
        log_async_chunk_profile(
            "chunk_adapter_save_enqueue",
            0.0,
            extra=f"stage={self.connector.stage_id} qlen={len(self._pending_save_reqs)}",
        )

    def _poll_single_request(self, request: Request):
        stage_id = self.connector.stage_id
        target_stage_id = stage_id - 1
        req_id = request.request_id
        chunk_id = self.get_req_chunk[req_id]
        external_req_id = self.request_ids_mapping.get(req_id, req_id)
        connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"

        # Use timeout=0 for non-blocking poll
        try:
            result = self.connector.get(
                str(target_stage_id),
                str(stage_id),
                connector_get_key,
            )
        except Exception as e:
            logger.error(f"SharedMemoryConnector get failed for req {connector_get_key}: {e}")
            return False

        if result is None:
            return False
        payload_data, size = result

        if payload_data:
            enqueue_ts = getattr(request, "_async_chunk_load_enqueue_ts", None)
            if enqueue_ts is not None:
                log_async_chunk_profile(
                    "chunk_adapter_load_queue_wait",
                    (time.perf_counter() - enqueue_ts) * 1000,
                    extra=f"stage={stage_id} req_id={str(req_id)[:12]} chunk={chunk_id}",
                )
            load_extra = f"stage={stage_id} req_id={str(req_id)[:12]} chunk={chunk_id}"
            log_async_chunk_event("chunk_adapter_load_work", "start", extra=load_extra)
            # Update connector state
            try:
                self.get_req_chunk[req_id] += 1

                if self.model_mode == "ar":
                    self._update_request_payload(external_req_id, payload_data)
                    request.additional_information = payload_data
                    if payload_data.get("finished"):
                        self.finished_requests.add(req_id)
                else:
                    if payload_data.get("finished"):
                        self.finished_requests.add(req_id)

                    new_ids = self._build_code2wav_prompt_ids(payload_data, external_req_id)
                    if new_ids is None:
                        return True
                    request.prompt_token_ids = new_ids
                    request.num_computed_tokens = 0

                    # Empty chunk with more data expected: keep polling.
                    if not new_ids and not payload_data.get("finished"):
                        return True

                # Mark as finished for consumption
                self._finished_load_reqs.add(req_id)
                if hasattr(request, "_async_chunk_load_enqueue_ts"):
                    delattr(request, "_async_chunk_load_enqueue_ts")
                logger.debug(f"[Stage-{stage_id}] Received one chunk for key {connector_get_key}")
                return True
            finally:
                log_async_chunk_event("chunk_adapter_load_work", "end", extra=load_extra)

        return False

    def _update_request_payload(self, req_id: str, payload_data: dict[str, Any]) -> dict[str, Any]:
        """Update the payload data for a request in the connector.

        Args:
            connector: OmniConnectorBase instance
            req_id: Request ID to update
            payload_data: New payload data to store
        """
        with async_chunk_timer(
            "chunk_adapter_update_request_payload",
            extra=f"req_id={str(req_id)[:12]}",
        ):
            if req_id not in self.request_payload:
                self.request_payload[req_id] = payload_data
                return payload_data
            origin_payload = self.request_payload[req_id]
            for key, value in payload_data.items():
                if key == "finished":
                    continue
                elif isinstance(value, torch.Tensor) and key in origin_payload:
                    payload_data[key] = torch.cat([origin_payload[key], value], dim=0)
                elif isinstance(value, list) and key in origin_payload:
                    payload_data[key] = origin_payload[key] + value

            self.request_payload[req_id] = payload_data
        return payload_data

    def _send_single_request(self, task: dict):
        pooling_output = task["pooling_output"]
        request = task["request"]
        is_finished = task["is_finished"]
        stage_id = self.connector.stage_id
        next_stage_id = stage_id + 1
        request_id = request.external_req_id
        chunk_id = self.put_req_chunk[request_id]
        enqueue_ts = task.get("enqueue_ts")
        if enqueue_ts is not None:
            log_async_chunk_profile(
                "chunk_adapter_save_queue_wait",
                (time.perf_counter() - enqueue_ts) * 1000,
                extra=f"stage={stage_id} req_id={str(request_id)[:12]} chunk={chunk_id}",
            )
        connector_put_key = f"{request_id}_{stage_id}_{chunk_id}"
        save_extra = f"stage={stage_id} req_id={str(request_id)[:12]} chunk={chunk_id}"
        log_async_chunk_event("chunk_adapter_save_work", "start", extra=save_extra)
        # Process payload in save_loop thread
        payload_data = None
        try:
            if self.custom_process_next_stage_input_func:
                try:
                    with async_chunk_timer(
                        "chunk_adapter_custom_process",
                        extra=f"stage={stage_id} chunk={chunk_id}",
                    ):
                        payload_data = self.custom_process_next_stage_input_func(
                            transfer_manager=self,
                            pooling_output=pooling_output,
                            request=request,
                            is_finished=is_finished,
                        )

                except Exception as e:
                    logger.error(f"Failed to use custom_process_input_func for payload extraction: {e}")

            if not payload_data:
                return

            with async_chunk_timer("chunk_adapter_connector_put", extra=f"stage={stage_id}"):
                success, size, metadata = self.connector.put(
                    from_stage=str(stage_id),
                    to_stage=str(next_stage_id),
                    put_key=connector_put_key,
                    data=payload_data,
                )

            if success:
                self.put_req_chunk[request_id] += 1
                logger.debug(f"[Stage-{stage_id}] Sent {connector_put_key}")
        finally:
            log_async_chunk_event("chunk_adapter_save_work", "end", extra=save_extra)

    ########################################################################
    # Cleanup
    ########################################################################

    def cleanup(
        self,
        request_id: str,
        external_req_id: str | None = None,
    ) -> None:
        """Reclaim all per-request state after a request finishes.

        Idempotent: calling with an already-cleaned or unknown id is safe.

        Args:
            request_id: Internal request id (receive / scheduler side key).
            external_req_id: External request id (send / payload side key).
                When *None*, looked up from ``request_ids_mapping``.
        """
        if external_req_id is None:
            external_req_id = self.request_ids_mapping.get(request_id, request_id)

        self.finished_requests.discard(request_id)
        self.get_req_chunk.pop(request_id, None)
        self.requests_with_ready_chunks.discard(request_id)
        self.request_ids_mapping.pop(request_id, None)

        remaining = deque(r for r in self._pending_load_reqs if getattr(r, "request_id", None) != request_id)
        self._pending_load_reqs = remaining
        self._finished_load_reqs.discard(request_id)

        self.put_req_chunk.pop(external_req_id, None)
        self.request_payload.pop(external_req_id, None)
        self.code_prompt_token_ids.pop(external_req_id, None)
        self.code_prompt_frame_counts.pop(external_req_id, None)

    ########################################################################
    # Schedule Helper
    ########################################################################

    def process_pending_chunks(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
    ) -> None:
        """
        Process pending chunks for waiting and running queues.
        """
        if self.connector.stage_id == 0:
            return
        self._process_chunk_queue(
            waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING, self._finished_load_reqs
        )
        self._process_chunk_queue(
            running_queue, self.waiting_for_chunk_running_requests, RequestStatus.RUNNING, self._finished_load_reqs
        )
        while len(running_queue) > self.scheduler_max_num_seqs:
            request = running_queue.pop()
            waiting_queue.prepend_requests([request])

    def restore_queues(self, waiting_queue: Any, running_queue: list[Request]) -> None:
        """
        Restore requests waiting for chunk to the waiting and running queues.
        """
        # Add request waiting for chunk to the waiting and running queue
        for request in self.waiting_for_chunk_waiting_requests:
            waiting_queue.add_request(request)
        self.waiting_for_chunk_waiting_requests = deque()

        if self.waiting_for_chunk_running_requests:
            running_queue.extend(self.waiting_for_chunk_running_requests)
        self.waiting_for_chunk_running_requests = deque()

    def postprocess_scheduler_output(
        self,
        scheduler_output: Any,
        requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Add additional info for cached requests and
        clean up ready chunks from scheduler output.
        """
        if requests is not None:
            self.attach_cached_additional_information(scheduler_output, requests)
        self._clear_chunk_ready(scheduler_output)

    @staticmethod
    def attach_cached_additional_information(scheduler_output: Any, requests: dict[str, Request]) -> None:
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if not cached_reqs:
            return
        if not hasattr(cached_reqs, "additional_information"):
            cached_reqs.additional_information = {}
        for req_id in cached_reqs.req_ids:
            request = requests.get(req_id) if req_id else None
            additional_info = getattr(request, "additional_information", None) if request else None
            cached_reqs.additional_information[req_id] = additional_info

    def _process_chunk_queue(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
        finished_load_reqs: set[str],
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    # Requests that have loaded chunk from last round
                    # of schedule, but have not scheduled
                    continue
                if request.request_id in self.finished_requests:
                    continue
                # Requests that waiting for chunk
                self.load_async(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in finished_load_reqs:
                    request.status = target_status
                    finished_load_reqs.remove(request.request_id)
                    self.requests_with_ready_chunks.add(request.request_id)
                    continue
            queue.remove(request)
            waiting_for_chunk_list.append(request)

    def _clear_chunk_ready(self, scheduler_output: Any) -> None:
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                if req_data.req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_data.req_id)

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_id)
