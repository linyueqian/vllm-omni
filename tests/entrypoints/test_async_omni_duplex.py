from types import SimpleNamespace

import pytest

from vllm_omni.engine.messages import OutputMessage
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_async_omni_open_duplex_forwards_session_config_and_timeout():
    calls = []

    async def open_duplex_session_async(session_id, **kwargs):
        calls.append((session_id, kwargs))
        return {"ok": True}

    app = object.__new__(AsyncOmni)
    app.engine = SimpleNamespace(open_duplex_session_async=open_duplex_session_async)

    result = await app.open_duplex_session_async(
        "sid",
        capabilities={"implementation_level": "model_native_duplex"},
        session_config={"instructions": "Be brief.", "idle_timeout_s": 30},
        timeout=7.5,
    )

    assert result == {"ok": True}
    assert calls == [
        (
            "sid",
            {
                "session_mode": "duplex",
                "capabilities": {"implementation_level": "model_native_duplex"},
                "session_config": {"instructions": "Be brief.", "idle_timeout_s": 30},
                "timeout": 7.5,
            },
        )
    ]


@pytest.mark.asyncio
async def test_async_omni_duplex_runtime_controls_forward_timeout():
    calls = []

    async def append_duplex_input_async(session_id, **kwargs):
        calls.append(("append", session_id, kwargs))
        return {"ok": True}

    async def signal_duplex_turn_async(session_id, **kwargs):
        calls.append(("signal", session_id, kwargs))
        return {"ok": True}

    async def close_duplex_session_async(session_id, **kwargs):
        calls.append(("close", session_id, kwargs))
        return {"ok": True}

    app = object.__new__(AsyncOmni)
    app.engine = SimpleNamespace(
        append_duplex_input_async=append_duplex_input_async,
        signal_duplex_turn_async=signal_duplex_turn_async,
        close_duplex_session_async=close_duplex_session_async,
    )

    await app.append_duplex_input_async("sid", mode="append_audio_chunk", payload={}, timeout=12.5)
    await app.signal_duplex_turn_async("sid", event="barge_in", timeout=13.5)
    await app.close_duplex_session_async("sid", reason="done", timeout=14.5)

    assert calls == [
        (
            "append",
            "sid",
            {
                "mode": "append_audio_chunk",
                "payload": {},
                "final": False,
                "timeout": 12.5,
            },
        ),
        (
            "signal",
            "sid",
            {
                "event": "barge_in",
                "payload": None,
                "timeout": 13.5,
            },
        ),
        (
            "close",
            "sid",
            {
                "reason": "done",
                "timeout": 14.5,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_async_omni_duplex_append_can_defer_data_plane_output_collection():
    async def append_duplex_input_async(session_id, **kwargs):
        del session_id, kwargs
        return {
            "stage_results": [
                {
                    "result": {
                        "data_plane_append": True,
                        "request_id": "duplex-sid-e0-stage0",
                        "response_stage_id": 1,
                    }
                }
            ]
        }

    app = object.__new__(AsyncOmni)
    app.engine = SimpleNamespace(append_duplex_input_async=append_duplex_input_async)
    app.request_states = {}
    app._final_output_handler = lambda: None

    result = await app.append_duplex_input_async(
        "sid",
        mode="append_audio_chunk",
        payload={},
        collect_outputs=False,
    )

    assert "data_plane_outputs" not in result


@pytest.mark.asyncio
async def test_async_omni_duplex_collect_waits_for_response_stage():
    app = object.__new__(AsyncOmni)
    req_state = ClientRequestState("duplex-sid-e0-stage0")
    stage0_output = OmniRequestOutput(
        request_id="duplex-sid-e0-stage0",
        stage_id=0,
        finished=False,
    )
    stage1_output = OmniRequestOutput(
        request_id="duplex-sid-e0-stage0",
        stage_id=1,
        final_output_type="audio",
        finished=False,
    )
    await req_state.queue.put(
        OutputMessage(
            request_id="duplex-sid-e0-stage0",
            stage_id=0,
            engine_outputs=stage0_output,
            finished=False,
        )
    )
    await req_state.queue.put(
        OutputMessage(
            request_id="duplex-sid-e0-stage0",
            stage_id=1,
            engine_outputs=stage1_output,
            finished=False,
        )
    )

    outputs = await app._collect_duplex_data_plane_outputs(
        "duplex-sid-e0-stage0",
        req_state,
        response_stage_id=1,
        timeout=1.0,
    )

    assert outputs == [stage1_output]


@pytest.mark.asyncio
async def test_async_omni_duplex_collect_wraps_raw_response_stage_output():
    app = object.__new__(AsyncOmni)
    app.log_stats = False
    app._enable_ar_profiler = False
    app.engine = SimpleNamespace(
        num_stages=2,
        get_stage_metadata=lambda stage_id: SimpleNamespace(
            final_output=stage_id == 1,
            final_output_type="audio",
        ),
    )
    req_state = ClientRequestState("duplex-sid-e0-stage0")
    raw_stage1_output = SimpleNamespace(
        finished=False,
        outputs=[],
        stage_durations={},
        peak_memory_mb=0.0,
        final_output_type="audio",
    )
    await req_state.queue.put(
        OutputMessage(
            request_id="duplex-sid-e0-stage0",
            stage_id=1,
            engine_outputs=raw_stage1_output,
            finished=False,
        )
    )

    outputs = await app._collect_duplex_data_plane_outputs(
        "duplex-sid-e0-stage0",
        req_state,
        response_stage_id=1,
        timeout=1.0,
    )

    assert len(outputs) == 1
    assert outputs[0].request_id == "duplex-sid-e0-stage0"
    assert outputs[0].finished is False
    assert outputs[0].stage_id == 1
    assert outputs[0].final_output_type == "audio"
    assert outputs[0].request_output is raw_stage1_output


def test_async_omni_duplex_request_info_includes_response_stage():
    request_id, response_stage_id = AsyncOmni._duplex_data_plane_request_info(
        {
            "stage_results": [
                {
                    "result": {
                        "data_plane_append": True,
                        "request_id": "duplex-sid-e0-stage0",
                        "response_stage_id": 1,
                    }
                }
            ]
        }
    )

    assert request_id == "duplex-sid-e0-stage0"
    assert response_stage_id == 1
