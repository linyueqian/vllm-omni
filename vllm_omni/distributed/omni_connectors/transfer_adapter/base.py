# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from collections import deque
from typing import Any

from ..utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


class OmniTransferAdapterBase:
    """Base class for managing data transfer via OmniConnector.

    This class handles the core loop logic and connector interactions, but
    leaves the specific data processing (chunks, KV cache, etc.) to subclasses.
    """

    def __init__(self, config: Any):
        self.config = config
        if not hasattr(self, "connector"):
            self.connector = None
        # Requests that are waiting to be polled
        self._pending_load_reqs = deque()
        # Requests that have successfully retrieved data
        self._finished_load_reqs = set()

        # Requests that are waiting to be saved
        self._pending_save_reqs = deque()
        # Requests that have successfully saved data
        self._finished_save_reqs = set()

        self.stop_event = threading.Event()
        self._load_cv = threading.Condition()
        self._save_cv = threading.Condition()

        self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.recv_thread.start()

        self.save_thread = threading.Thread(target=self.save_loop, daemon=True)
        self.save_thread.start()

    @classmethod
    def create_connector(cls, model_config: Any):
        raise NotImplementedError

    def recv_loop(self):
        """Loop to poll for incoming data."""
        while not self.stop_event.is_set():
            with self._load_cv:
                while not self._pending_load_reqs and not self.stop_event.is_set():
                    self._load_cv.wait(timeout=0.1)
                if self.stop_event.is_set():
                    return
                pending = [self._pending_load_reqs.popleft() for _ in range(len(self._pending_load_reqs))]

            requeue = []
            made_progress = False
            for request in pending:
                request_id = request.request_id
                self.request_ids_mapping[request_id] = request.external_req_id
                try:
                    is_success = self._poll_single_request(request)
                    if not is_success:
                        requeue.append(request)
                    else:
                        made_progress = True
                except Exception as e:
                    requeue.append(request)
                    logger.warning(f"Error receiving data for {request_id}: {e}")

            if requeue:
                with self._load_cv:
                    self._pending_load_reqs.extend(requeue)
            if requeue and not made_progress:
                self.stop_event.wait(timeout=0.001)

    def save_loop(self):
        """Loop to send outgoing data."""
        while not self.stop_event.is_set():
            with self._save_cv:
                while not self._pending_save_reqs and not self.stop_event.is_set():
                    self._save_cv.wait(timeout=0.1)
                if self.stop_event.is_set():
                    return
                pending = [self._pending_save_reqs.popleft() for _ in range(len(self._pending_save_reqs))]

            for task in pending:
                try:
                    self._send_single_request(task)
                except Exception as e:
                    logger.warning(f"Error saving data for {task.get('request_id')}: {e}")

    def _enqueue_load_request(self, request: Any) -> None:
        self._ensure_queue_sync()
        with self._load_cv:
            self._pending_load_reqs.append(request)
            self._load_cv.notify()

    def _enqueue_save_task(self, task: Any) -> None:
        self._ensure_queue_sync()
        with self._save_cv:
            self._pending_save_reqs.append(task)
            self._save_cv.notify()

    def _ensure_queue_sync(self) -> None:
        if not hasattr(self, "_load_cv"):
            self._load_cv = threading.Condition()
        if not hasattr(self, "_save_cv"):
            self._save_cv = threading.Condition()

    def _poll_single_request(self, *args, **kwargs):
        """Poll connector for a single request task.
        Subclasses should implement request-specific receive behavior."""
        raise NotImplementedError

    def _send_single_request(self, *args, **kwargs):
        """Send one pending save request task to the connector.
        Subclasses should implement task-specific handling logic."""
        raise NotImplementedError

    def load_async(self, *args, **kwargs):
        """Register a request to load data. To be implemented by subclasses."""
        raise NotImplementedError

    def save_async(self, *args, **kwargs):
        """Submit data to be saved. To be implemented by subclasses."""
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """Load request data from connector synchronously. To be implemented by subclasses."""
        raise NotImplementedError

    def save(self, *args, **kwargs):
        """Save data to connector synchronously. To be implemented by subclasses."""
        raise NotImplementedError

    def get_finished_requests(self):
        """Get finished loaded or saved requests"""
        raise NotImplementedError

    def shutdown(self):
        """Stop background loops and close the connector."""
        raise NotImplementedError
