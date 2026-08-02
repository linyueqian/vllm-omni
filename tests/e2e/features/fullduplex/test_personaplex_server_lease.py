# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-session lease teardown: a cancelled engine call must be drained.

Cancelling an asyncio task blocked on a worker-thread engine call returns
immediately while the thread keeps running (executor futures cannot be
interrupted). The server therefore tracks the in-flight future on the session
state and waits it out before releasing the capacity-one lease; otherwise a
fast reconnect could call ``open()`` on the shared engine while the previous
connection's final ``step()`` is still mutating it.
"""

import asyncio
import contextlib
import threading

import pytest

pytest.importorskip("sphn")  # serving-only dependency; skip where absent

from vllm_omni.experimental.fullduplex.personaplex.config import PersonaPlexConfig
from vllm_omni.experimental.fullduplex.personaplex.serving.server import DuplexServer
from vllm_omni.experimental.fullduplex.personaplex.session import PersonaPlexServingSessionState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_cancelled_engine_call_is_drained_before_lease_release():
    server = DuplexServer(PersonaPlexConfig())  # engine never loaded: stub calls only
    state = PersonaPlexServingSessionState()

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_engine_op():
        started.set()
        release.wait(timeout=5.0)
        finished.set()
        return []

    task = asyncio.create_task(server._engine_call(state, slow_engine_op))
    await asyncio.to_thread(started.wait, 5.0)

    # Cancel while the worker thread is mid-call: the await returns at once,
    # the thread keeps running, and its completion event stays tracked on the
    # state (the asyncio future itself reads as cancelled/done here, which is
    # exactly why the drain must not rely on it).
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert state.inflight is not None
    assert not state.inflight.is_set()

    # The drain must genuinely wait for the worker, not just return.
    drain = asyncio.create_task(server._drain_inflight(state))
    await asyncio.sleep(0.05)
    assert not drain.done()

    release.set()
    await asyncio.wait_for(drain, timeout=5.0)
    assert finished.is_set()
    assert state.inflight.is_set()


@pytest.mark.asyncio
async def test_completed_engine_call_clears_inflight():
    server = DuplexServer(PersonaPlexConfig())
    state = PersonaPlexServingSessionState()
    result = await server._engine_call(state, lambda: "ok")
    assert result == "ok"
    assert state.inflight is None
    await server._drain_inflight(state)  # no-op on a clean state
