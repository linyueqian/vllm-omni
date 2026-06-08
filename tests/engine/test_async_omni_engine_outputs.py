"""Tests for AsyncOmniEngine.try_get_output and try_get_output_async.

Focuses on the critical behavior: when the orchestrator thread dies,
subsequent attempts to collect output raise RuntimeError.
"""

import queue
import threading
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import DuplexControlResultMessage, ErrorMessage, OutputMessage
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine(output_queue, mocker: MockerFixture, *, thread_alive: bool = True) -> AsyncOmniEngine:
    """Create an AsyncOmniEngine bypassing __init__."""
    engine = object.__new__(AsyncOmniEngine)
    engine.output_queue = output_queue
    engine.orchestrator_thread = mocker.MagicMock(
        is_alive=mocker.MagicMock(return_value=thread_alive),
    )
    return engine


def test_try_get_output_raises_after_orchestrator_dies(mocker: MockerFixture):
    """Draining remaining results then hitting an empty queue with a dead
    orchestrator must raise RuntimeError so callers know the pipeline is gone."""
    mock_queue = mocker.MagicMock()
    # First call succeeds; second call finds the queue empty.
    mock_queue.sync_q.get.side_effect = [
        OutputMessage(
            request_id="r1",
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="r1"),
            finished=False,
        ),
        queue.Empty,
    ]

    engine = _make_engine(mock_queue, mocker, thread_alive=True)

    # Collect the one buffered result.
    assert engine.try_get_output().request_id == "r1"

    # Orchestrator thread crashes between polls.
    engine.orchestrator_thread.is_alive.return_value = False

    with pytest.raises(RuntimeError, match="Orchestrator died unexpectedly"):
        engine.try_get_output()


@pytest.mark.asyncio
async def test_try_get_output_async_raises_after_orchestrator_dies(mocker: MockerFixture):
    """Same scenario as above but for the async variant."""
    mock_queue = mocker.MagicMock()
    mock_queue.sync_q.get_nowait.side_effect = [
        OutputMessage(
            request_id="r1",
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="r1"),
            finished=False,
        ),
        queue.Empty,
    ]

    engine = _make_engine(mock_queue, mocker, thread_alive=True)

    assert (await engine.try_get_output_async()).request_id == "r1"

    engine.orchestrator_thread.is_alive.return_value = False

    with pytest.raises(RuntimeError, match="Orchestrator died unexpectedly"):
        await engine.try_get_output_async()


def test_fatal_error_message_surfaces_through_try_get_output(mocker: MockerFixture):
    """When the orchestrator thread crashes, it enqueues a fatal error message.

    ``try_get_output`` must return this message so the caller
    (``OmniBase._handle_output_message``) can detect the fatal flag.
    """
    fatal_msg = ErrorMessage(error="Orchestrator thread crashed", fatal=True)

    mock_queue = mocker.MagicMock()
    mock_queue.sync_q.get.return_value = fatal_msg

    engine = _make_engine(mock_queue, mocker, thread_alive=False)

    msg = engine.try_get_output()
    assert msg is not None
    assert msg.type == "error"
    assert msg.fatal is True
    assert "crashed" in msg.error


@pytest.mark.asyncio
async def test_fatal_error_message_surfaces_through_try_get_output_async(mocker: MockerFixture):
    """Async variant of the fatal error message test."""
    fatal_msg = ErrorMessage(error="Orchestrator thread crashed", fatal=True)

    mock_queue = mocker.MagicMock()
    mock_queue.sync_q.get_nowait.return_value = fatal_msg

    engine = _make_engine(mock_queue, mocker, thread_alive=False)

    msg = await engine.try_get_output_async()
    assert msg is not None
    assert msg.type == "error"
    assert msg.fatal is True


def test_open_duplex_session_waits_for_control_ack(mocker: MockerFixture):
    request_q = queue.Queue()
    rpc_q = queue.Queue()
    rpc_q.put_nowait(
        DuplexControlResultMessage(
            control_id="ctrl-1",
            operation="open",
            session_id="sid",
            ok=False,
            stage_results=[{"stage_id": 0, "replica_id": 0, "result": {"supported": False}}],
            unsupported_count=1,
            error_count=0,
        )
    )

    engine = object.__new__(AsyncOmniEngine)
    engine.request_queue = SimpleNamespace(sync_q=request_q)
    engine.rpc_output_queue = SimpleNamespace(sync_q=rpc_q)
    engine._rpc_lock = threading.Lock()
    mocker.patch("vllm_omni.engine.async_omni_engine.uuid.uuid4", return_value=SimpleNamespace(hex="ctrl-1"))

    result = engine.open_duplex_session("sid", timeout=1)

    msg = request_q.get_nowait()
    assert msg.type == "open_duplex_session"
    assert msg.control_id == "ctrl-1"
    assert msg.timeout == 1
    assert result["unsupported_count"] == 1
    assert result["stage_results"][0]["result"]["supported"] is False


def test_open_duplex_session_raises_on_stage_control_error(mocker: MockerFixture):
    request_q = queue.Queue()
    rpc_q = queue.Queue()
    rpc_q.put_nowait(
        DuplexControlResultMessage(
            control_id="ctrl-error",
            operation="open",
            session_id="sid",
            ok=False,
            stage_results=[{"stage_id": 0, "replica_id": 0, "result": {"supported": False, "error": "boom"}}],
            unsupported_count=1,
            error_count=1,
        )
    )

    engine = object.__new__(AsyncOmniEngine)
    engine.request_queue = SimpleNamespace(sync_q=request_q)
    engine.rpc_output_queue = SimpleNamespace(sync_q=rpc_q)
    engine._rpc_lock = threading.Lock()
    mocker.patch("vllm_omni.engine.async_omni_engine.uuid.uuid4", return_value=SimpleNamespace(hex="ctrl-error"))

    with pytest.raises(RuntimeError, match="duplex open failed"):
        engine.open_duplex_session("sid", timeout=1)
