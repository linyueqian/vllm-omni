from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import janus
import psutil
import pytest
import torch
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.core_client import AsyncMPClient

from vllm_omni.engine.messages import (
    AbortRequestMessage,
    AddCompanionRequestMessage,
    AppendDuplexInputMessage,
    CloseDuplexSessionMessage,
    CollectiveRPCRequestMessage,
    CollectiveRPCResultMessage,
    DuplexControlResultMessage,
    OpenDuplexSessionMessage,
    OutputMessage,
    ShutdownRequestMessage,
    SignalDuplexTurnMessage,
    StageSubmissionMessage,
)
from vllm_omni.engine.orchestrator import (
    Orchestrator,
    OrchestratorRequestState,
    build_engine_core_request_from_tokens,
)
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _minicpmo_stage_handoff_payload(
    *,
    token_ids: list[int],
    hidden: torch.Tensor,
    text: str,
    end_of_turn: bool = False,
) -> dict[str, Any]:
    from vllm_omni.data_entry_keys import serialize_payload

    return {
        "type": "minicpmo45_tts_handoff",
        "omni_payload": serialize_payload(
            {
                "ids": {"output": token_ids},
                "hidden_states": {"output": hidden},
            }
        ),
        "llm_output_text": [text],
        "end_of_turn": end_of_turn,
    }


@dataclass
class OrchestratorFixture:
    orchestrator: Orchestrator
    request_sync_q: Any
    output_sync_q: Any
    queues: tuple[janus.Queue, ...]
    thread: threading.Thread
    result_future: concurrent.futures.Future[None]


def test_orchestrator_duplex_capability_core_kv_lease_must_be_explicit() -> None:
    capabilities = Orchestrator._coerce_duplex_capabilities(
        {
            "supports_kv_lease": True,
            "supports_model_internal_state": True,
        }
    )

    assert capabilities.supports_kv_lease is True
    assert capabilities.supports_model_internal_state is True
    assert capabilities.supports_core_kv_lease is False


def test_build_engine_core_request_preserves_model_intermediate_buffer() -> None:
    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    request = build_engine_core_request_from_tokens(
        request_id="req-stage-handoff",
        prompt={
            "prompt_token_ids": [1],
            "model_intermediate_buffer": {
                "ids": {"tts": [11, 12]},
                "hidden_states": {"tts": hidden},
                "meta": {"ref_audio_sr": 16000},
            },
        },
        params=SamplingParams(max_tokens=1),
    )

    assert request.additional_information is None
    assert isinstance(request.model_intermediate_buffer, dict)
    info = request.model_intermediate_buffer
    assert info["ids"]["tts"] == [11, 12]
    assert torch.equal(info["hidden_states"]["tts"], hidden)
    assert info["meta"]["ref_audio_sr"] == 16000


def test_duplex_sampling_params_apply_stage_overrides() -> None:
    orchestrator = object.__new__(Orchestrator)
    stage = SimpleNamespace(stage_client=SimpleNamespace(default_sampling_params=SamplingParams(max_tokens=8)))
    orchestrator.stage_pools = [stage]
    session = SimpleNamespace(
        session_config={
            "extra_body": {
                "duplex_stage_max_tokens": {"0": 24},
                "duplex_stage_sampling_params": {
                    "0": {
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "top_k": 100,
                        "repetition_penalty": 1.05,
                    }
                },
            }
        }
    )

    params = orchestrator._duplex_sampling_params_list(session)[0]

    assert params.max_tokens == 24
    assert params.temperature == 0.7
    assert params.top_p == 0.8
    assert params.top_k == 100
    assert params.repetition_penalty == 1.05


class FakeStageClient:
    def __init__(
        self,
        *,
        stage_type: str = "llm",
        final_output: bool = False,
        final_output_type: str = "text",
        next_inputs: list[dict] | None = None,
        engine_input_source: list[int] | None = None,
        is_comprehension: bool = False,
        model_stage: str | None = None,
        kv_sender_info: dict[str, Any] | None = None,
    ) -> None:
        self.stage_id = 0
        self.replica_id = 0
        self.stage_type = stage_type
        self.final_output = final_output
        self.final_output_type = final_output_type
        self.default_sampling_params = SamplingParams(max_tokens=1)
        self.requires_multimodal_data = False
        self.engine_input_source = list(engine_input_source or [0])
        self.is_comprehension = is_comprehension
        self.model_stage = model_stage
        self.next_inputs = list(next_inputs or [])
        self.custom_process_input_func = None
        self._kv_sender_info = dict(kv_sender_info) if kv_sender_info is not None else None
        self.add_request_calls: list[tuple] = []
        self.abort_calls: list[list[str]] = []
        self.collective_rpc_calls: list[tuple[str, float | None, tuple[Any, ...], dict[str, Any]]] = []
        self.shutdown_calls = 0
        self._engine_core_outputs = queue.Queue()
        self._diffusion_outputs = queue.Queue()

    # Orchestrator-facing interface.
    async def add_request_async(self, *args, **kwargs) -> None:
        self.add_request_calls.append(args)

    async def get_output_async(self):
        try:
            return self._engine_core_outputs.get_nowait()
        except queue.Empty:
            return SimpleNamespace(outputs=[])

    def get_diffusion_output_nowait(self):
        try:
            return self._diffusion_outputs.get_nowait()
        except queue.Empty:
            return None

    def set_engine_outputs(self, outputs) -> None:
        return None

    def process_engine_inputs(self, source_outputs, prompt=None, streaming_context=None):
        return list(self.next_inputs)

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self.abort_calls.append(list(request_ids))

    async def collective_rpc_async(
        self,
        *,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_kwargs = dict(kwargs or {})
        self.collective_rpc_calls.append((method, timeout, args, normalized_kwargs))
        return {
            "supported": False,
            "todo": True,
            "reason": f"{self.__class__.__name__}.collective_rpc_async is not implemented yet",
        }

    def get_kv_sender_info(self) -> dict[str, Any] | None:
        if self._kv_sender_info is None:
            return None
        return dict(self._kv_sender_info)

    def check_health(self) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    # Test helpers for seeding fake stage outputs.
    def push_engine_core_outputs(self, outputs) -> None:
        self._engine_core_outputs.put_nowait(outputs)

    def push_diffusion_output(self, output) -> None:
        self._diffusion_outputs.put_nowait(output)


class FakeCollectiveRpcStageClient(FakeStageClient):
    def __init__(self, *args, rpc_result: Any = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rpc_result = rpc_result

    async def collective_rpc_async(
        self,
        *,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_kwargs = dict(kwargs or {})
        self.collective_rpc_calls.append((method, timeout, args, normalized_kwargs))
        return self.rpc_result


class FakeSequentialCollectiveRpcStageClient(FakeStageClient):
    def __init__(self, *args, rpc_results: list[Any], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rpc_results = list(rpc_results)

    async def collective_rpc_async(
        self,
        *,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_kwargs = dict(kwargs or {})
        self.collective_rpc_calls.append((method, timeout, args, normalized_kwargs))
        if not self.rpc_results:
            raise AssertionError(f"no rpc result queued for {method}")
        return self.rpc_results.pop(0)


class FakeOutputProcessor:
    def __init__(self, *, request_outputs: list[object] | None = None) -> None:
        self.request_outputs = list(request_outputs or [])
        self.add_request_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.abort_calls: list[list[str]] = []

    def add_request(self, *args, **kwargs) -> None:
        self.add_request_calls.append((args, kwargs))
        return None

    def process_outputs(self, *_args, **_kwargs):
        return SimpleNamespace(
            request_outputs=list(self.request_outputs),
            reqs_to_abort=[],
        )

    def abort_requests(self, request_ids, internal: bool = False):
        self.abort_calls.append(request_ids)
        return request_ids

    def update_scheduler_stats(self, _scheduler_stats) -> None:
        return None


class _FakeProc:
    pid = 1234

    def __init__(self):
        self.terminated = False
        self.killed = False
        self.join_calls = []

    def is_alive(self):
        return not self.terminated and not self.killed

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def join(self, timeout=None):
        self.join_calls.append(timeout)


class _FakeChildProc:
    def __init__(self):
        self.terminated = False
        self.killed = False

    def is_running(self):
        return not self.terminated and not self.killed

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def _sampling_params(max_tokens: int = 4) -> SamplingParams:
    return SamplingParams(max_tokens=max_tokens)


def _engine_core_outputs(tag: str, timestamp: float) -> SimpleNamespace:
    return SimpleNamespace(outputs=[tag], timestamp=timestamp, scheduler_stats=None)


def _build_request_output(
    request_id: str,
    *,
    token_ids: list[int] | None = None,
    prompt_token_ids: list[int] | None = None,
    finished: bool = True,
    text: str = "test",
) -> RequestOutput:
    completion = CompletionOutput(
        index=0,
        text=text,
        token_ids=list(token_ids or [1, 2]),
        cumulative_logprob=0.0,
        logprobs=None,
        finish_reason="stop" if finished else None,
        stop_reason=None,
    )
    return RequestOutput(
        request_id=request_id,
        prompt="prompt",
        prompt_token_ids=list(prompt_token_ids or [10, 11]),
        prompt_logprobs=None,
        outputs=[completion],
        finished=finished,
        metrics=None,
        lora_request=None,
    )


def _build_stage_pools(
    stage_clients: list[list[FakeStageClient]],
    *,
    output_processors: list[FakeOutputProcessor] | None = None,
    stage_vllm_configs: list[object] | None = None,
) -> list[StagePool]:
    """Build StagePool list from per-stage replica lists.

    ``stage_clients[i]`` is the list of FakeStageClient replicas for stage i.
    """
    num_stages = len(stage_clients)
    if output_processors is None:
        output_processors = [FakeOutputProcessor() for _ in stage_clients]
    if stage_vllm_configs is None:
        stage_vllm_configs = [SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)) for _ in stage_clients]

    pools: list[StagePool] = []
    for stage_id in range(num_stages):
        clients = stage_clients[stage_id]
        if clients[0].stage_type == "diffusion":
            pools.append(StagePool(stage_id, clients[0]))
        else:
            pools.append(
                StagePool(
                    stage_id,
                    clients,
                    output_processor=output_processors[stage_id],
                    stage_vllm_config=stage_vllm_configs[stage_id],
                )
            )
    return pools


def _build_harness(
    stage_clients: list[object],
    *,
    output_processors: list[object] | None = None,
    stage_vllm_configs: list[object] | None = None,
    async_chunk: bool = False,
    stage_pools: list[StagePool] | None = None,
) -> OrchestratorFixture:
    """Build an Orchestrator test harness.

    Accepts either pre-built ``stage_pools`` or flat lists of single-replica
    clients/processors.
    """
    if stage_pools is None:
        # Wrap flat lists into per-stage single-replica lists.
        nested_clients = [[c] for c in stage_clients]
        stage_pools = _build_stage_pools(
            nested_clients,
            output_processors=output_processors,
            stage_vllm_configs=stage_vllm_configs,
        )

    ready_future: concurrent.futures.Future[tuple[Orchestrator, janus.Queue, janus.Queue, janus.Queue]] = (
        concurrent.futures.Future()
    )
    result_future: concurrent.futures.Future[None] = concurrent.futures.Future()

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _run() -> None:
            request_queue = janus.Queue()
            output_queue = janus.Queue()
            rpc_queue = janus.Queue()
            orchestrator = Orchestrator(
                request_async_queue=request_queue.async_q,
                output_async_queue=output_queue.async_q,
                rpc_async_queue=rpc_queue.async_q,
                stage_pools=stage_pools,
                async_chunk=async_chunk,
            )
            ready_future.set_result((orchestrator, request_queue, output_queue, rpc_queue))
            await orchestrator.run()

        try:
            loop.run_until_complete(_run())
            result_future.set_result(None)
        except Exception as exc:
            result_future.set_exception(exc)
        finally:
            try:
                pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    thread = threading.Thread(target=_runner, daemon=True, name="test-orchestrator")
    thread.start()

    orchestrator, request_queue, output_queue, rpc_queue = ready_future.result(timeout=5)
    return OrchestratorFixture(
        orchestrator=orchestrator,
        request_sync_q=request_queue.sync_q,
        output_sync_q=output_queue.sync_q,
        queues=(request_queue, output_queue, rpc_queue),
        thread=thread,
        result_future=result_future,
    )


async def _shutdown_orchestrator(orchestrator_fixture: OrchestratorFixture) -> None:
    orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
    await asyncio.to_thread(orchestrator_fixture.thread.join, 5)
    if orchestrator_fixture.thread.is_alive():
        raise AssertionError("Timed out waiting for orchestrator thread shutdown")
    orchestrator_fixture.result_future.result(timeout=0)


async def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for predicate")
        await asyncio.sleep(0.01)


async def _get_output_message(orchestrator_fixture: OrchestratorFixture, *, timeout: float = 2.0) -> OutputMessage:
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for orchestrator output")
        try:
            msg = orchestrator_fixture.output_sync_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue
        if isinstance(msg, OutputMessage):
            return msg


async def _get_rpc_message(
    orchestrator_fixture: OrchestratorFixture,
    *,
    timeout: float = 2.0,
) -> CollectiveRPCResultMessage:
    deadline = time.monotonic() + timeout
    rpc_sync_q = orchestrator_fixture.queues[2].sync_q
    while True:
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for orchestrator rpc output")
        try:
            return rpc_sync_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)


async def _enqueue_add_request(
    orchestrator_fixture: OrchestratorFixture,
    *,
    request_id: str,
    prompt,
    original_prompt,
    sampling_params_list,
    final_stage_id: int,
) -> None:
    orchestrator_fixture.request_sync_q.put_nowait(
        StageSubmissionMessage(
            type="add_request",
            request_id=request_id,
            prompt=prompt,
            original_prompt=original_prompt,
            output_prompt_text=None,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            preprocess_ms=0.0,
            request_timestamp=time.time(),
            enqueue_ts=time.perf_counter(),
        )
    )


async def _enqueue_abort_request(orchestrator_fixture: OrchestratorFixture, request_ids: list[str]) -> None:
    orchestrator_fixture.request_sync_q.put_nowait(AbortRequestMessage(request_ids=request_ids))


def test_stage_engine_core_client_shutdown_cleans_children_if_base_shutdown_fails(monkeypatch):
    fake_proc = _FakeProc()
    fake_child = _FakeChildProc()

    class FakePsutilProcess:
        def __init__(self, pid):
            assert pid == fake_proc.pid

        def children(self, recursive=True):
            assert recursive
            return [fake_child]

    def fail_base_shutdown(self, **kwargs):
        raise RuntimeError("base shutdown failed")

    monkeypatch.setattr(psutil, "Process", FakePsutilProcess)
    monkeypatch.setattr(psutil, "wait_procs", lambda procs, timeout: (list(procs), []))
    monkeypatch.setattr(AsyncMPClient, "shutdown", fail_base_shutdown)

    client = object.__new__(StageEngineCoreClient)
    client._proc = fake_proc

    with pytest.raises(RuntimeError, match="base shutdown failed"):
        client.shutdown()

    assert fake_proc.terminated
    assert fake_proc.join_calls == [5]
    assert fake_child.terminated


def test_stage_engine_core_client_shutdown_kills_stubborn_children(monkeypatch):
    fake_proc = _FakeProc()
    fake_child = _FakeChildProc()

    class FakePsutilProcess:
        def __init__(self, pid):
            assert pid == fake_proc.pid

        def children(self, recursive=True):
            assert recursive
            return [fake_child]

    monkeypatch.setattr(psutil, "Process", FakePsutilProcess)
    monkeypatch.setattr(psutil, "wait_procs", lambda procs, timeout: ([], list(procs)))
    monkeypatch.setattr(AsyncMPClient, "shutdown", lambda self, **kwargs: None)

    client = object.__new__(StageEngineCoreClient)
    client._proc = fake_proc

    client.shutdown()

    assert fake_child.terminated
    assert fake_child.killed


@pytest.fixture
def orchestrator_factory():
    fixtures: list[OrchestratorFixture] = []

    def _factory(*args, **kwargs) -> OrchestratorFixture:
        fixture = _build_harness(*args, **kwargs)
        fixtures.append(fixture)
        return fixture

    yield _factory

    for fixture in fixtures:
        if fixture.thread.is_alive():
            fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            fixture.thread.join(timeout=5)
        for q in fixture.queues:
            q.close()


# ---------------------------------------------------------------------------
# Existing single-replica tests (adapted to StagePool interface)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_two_stage_llm(orchestrator_factory) -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(
        stage_type="llm",
        final_output=True,
        next_inputs=[{"prompt_token_ids": [7, 8, 9]}],
    )
    processors = [
        FakeOutputProcessor(request_outputs=[_build_request_output("req-llm", token_ids=[3, 4], finished=True)]),
        FakeOutputProcessor(request_outputs=[_build_request_output("req-llm", token_ids=[10, 11], finished=True)]),
    ]
    orchestrator_fixture = orchestrator_factory([stage0, stage1], output_processors=processors)
    request = SimpleNamespace(request_id="req-llm", prompt_token_ids=[1, 2, 3])

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-llm",
            prompt=request,
            original_prompt={"prompt": "hello"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)
        stage0.push_engine_core_outputs(_engine_core_outputs("stage0-raw", 1.0))

        await _wait_for(lambda: len(stage1.add_request_calls) == 1)
        stage1_request = stage1.add_request_calls[0][0]
        assert stage1_request.request_id == "req-llm"
        assert stage1_request.prompt_token_ids == [7, 8, 9]

        stage1.push_engine_core_outputs(_engine_core_outputs("stage1-raw", 2.0))

        output_msg = await _get_output_message(orchestrator_fixture)

        assert output_msg.request_id == "req-llm"
        assert output_msg.stage_id == 1
        assert output_msg.finished is True
        assert output_msg.engine_outputs.request_id == "req-llm"
        assert "req-llm" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_run_single_stage_diffusion(orchestrator_factory) -> None:
    stage0 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    orchestrator_fixture = orchestrator_factory([stage0])
    params = OmniDiffusionSamplingParams()

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-diff",
            prompt={"prompt": "draw a cat"},
            original_prompt={"prompt": "draw a cat"},
            sampling_params_list=[params],
            final_stage_id=0,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)
        stage0.push_diffusion_output(
            OmniRequestOutput.from_diffusion(
                request_id="req-diff",
                images=[],
                final_output_type="image",
            )
        )

        output_msg = await _get_output_message(orchestrator_fixture)

        assert output_msg.request_id == "req-diff"
        assert output_msg.stage_id == 0
        assert output_msg.finished is True
        assert output_msg.engine_outputs.request_id == "req-diff"
        assert "req-diff" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_run_llm_to_diffusion(orchestrator_factory) -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    processors = [
        FakeOutputProcessor(request_outputs=[_build_request_output("req-img", token_ids=[3, 4], finished=True)]),
        FakeOutputProcessor(),
    ]
    orchestrator_fixture = orchestrator_factory([stage0, stage1], output_processors=processors)
    request = SimpleNamespace(request_id="req-img", prompt_token_ids=[1, 2, 3])
    params = OmniDiffusionSamplingParams()
    original_prompt = {"prompt": "draw a fox"}

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-img",
            prompt=request,
            original_prompt=original_prompt,
            sampling_params_list=[_sampling_params(), params],
            final_stage_id=1,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)
        stage0.push_engine_core_outputs(_engine_core_outputs("stage0-raw", 1.0))

        await _wait_for(lambda: len(stage1.add_request_calls) == 1)
        assert stage1.add_request_calls[0] == ("req-img", original_prompt, params)

        stage1.push_diffusion_output(
            OmniRequestOutput.from_diffusion(
                request_id="req-img",
                images=[],
                final_output_type="image",
            )
        )

        output_msg = await _get_output_message(orchestrator_fixture)

        assert output_msg.request_id == "req-img"
        assert output_msg.stage_id == 1
        assert output_msg.finished is True
        assert output_msg.engine_outputs.request_id == "req-img"
        assert "req-img" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_run_async_chunk(orchestrator_factory) -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="llm", final_output=True)
    processors = [
        FakeOutputProcessor(request_outputs=[_build_request_output("req-async", token_ids=[1], finished=True)]),
        FakeOutputProcessor(request_outputs=[_build_request_output("req-async", token_ids=[20, 21], finished=True)]),
    ]
    orchestrator_fixture = orchestrator_factory(
        [stage0, stage1],
        output_processors=processors,
        async_chunk=True,
    )
    request = SimpleNamespace(request_id="req-async", prompt_token_ids=[1, 2, 3, 4])

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-async",
            prompt=request,
            original_prompt={"prompt": "hello async"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )

        await _wait_for(lambda: len(stage1.add_request_calls) == 1)
        prewarmed_request = stage1.add_request_calls[0][0]
        assert prewarmed_request.request_id == "req-async"
        assert prewarmed_request.prompt_token_ids
        assert all(token_id == 0 for token_id in prewarmed_request.prompt_token_ids)

        stage1.push_engine_core_outputs(_engine_core_outputs("stage1-final", 3.0))

        output_msg = await _get_output_message(orchestrator_fixture)

        assert output_msg.request_id == "req-async"
        assert output_msg.stage_id == 1
        assert output_msg.finished is True
        assert "req-async" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_run_shutdown(orchestrator_factory) -> None:
    stages = [
        FakeStageClient(stage_type="llm", final_output=False),
        FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image"),
    ]
    orchestrator_fixture = orchestrator_factory(stages)

    await _shutdown_orchestrator(orchestrator_fixture)

    assert not orchestrator_fixture.thread.is_alive()
    for stage in stages:
        assert stage.shutdown_calls == 1


@pytest.mark.asyncio
async def test_run_abort(orchestrator_factory) -> None:
    stages = [
        FakeStageClient(stage_type="llm", final_output=False),
        FakeStageClient(stage_type="llm", final_output=True),
    ]
    processors = [
        FakeOutputProcessor(request_outputs=[_build_request_output("req-abort", token_ids=[1], finished=True)]),
        FakeOutputProcessor(request_outputs=[_build_request_output("req-abort", token_ids=[2], finished=True)]),
    ]
    orchestrator_fixture = orchestrator_factory(stages, output_processors=processors)
    request = SimpleNamespace(request_id="req-abort", prompt_token_ids=[1, 2, 3])

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-abort",
            prompt=request,
            original_prompt={"prompt": "cancel me"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stages[0].add_request_calls) == 1)

        await _enqueue_abort_request(orchestrator_fixture, ["req-abort"])
        await _wait_for(lambda: bool(stages[0].abort_calls))

        assert stages[0].abort_calls == [["req-abort"]]
        assert stages[1].abort_calls == []
        assert "req-abort" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


# ---------------------------------------------------------------------------
# Multi-replica tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_replica_round_robin_distribution(orchestrator_factory) -> None:
    """Two replicas at stage-0, single replica at stage-1.

    Send two requests — they should land on different stage-0 replicas
    (round-robin), then both forward to the single stage-1 replica.
    """
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(
        stage_type="llm",
        final_output=True,
        next_inputs=[{"prompt_token_ids": [7, 8]}],
    )

    proc0 = FakeOutputProcessor(request_outputs=[_build_request_output("req-0", token_ids=[3], finished=True)])
    proc1 = FakeOutputProcessor(request_outputs=[_build_request_output("req-0", token_ids=[10], finished=True)])

    default_vllm_cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))
    stage_pools = _build_stage_pools(
        [[stage0_r0, stage0_r1], [stage1]],
        output_processors=[proc0, proc1],
        stage_vllm_configs=[default_vllm_cfg, default_vllm_cfg],
    )

    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        # Request 0 → should land on replica 0 (RR starts at 0)
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-0",
            prompt=SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]),
            original_prompt={"prompt": "hello 0"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stage0_r0.add_request_calls) == 1)
        assert len(stage0_r1.add_request_calls) == 0

        # Request 1 → should land on replica 1 (RR advances)
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-1",
            prompt=SimpleNamespace(request_id="req-1", prompt_token_ids=[5, 6]),
            original_prompt={"prompt": "hello 1"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stage0_r1.add_request_calls) == 1)
        assert len(stage0_r0.add_request_calls) == 1  # unchanged

        # Complete req-0 at stage-0 replica-0 → should forward to stage-1
        stage0_r0.push_engine_core_outputs(_engine_core_outputs("s0r0-raw", 1.0))
        await _wait_for(lambda: len(stage1.add_request_calls) == 1)
        assert stage1.add_request_calls[0][0].request_id == "req-0"

        # Complete req-0 at stage-1 → final output
        proc1.request_outputs = [_build_request_output("req-0", token_ids=[10], finished=True)]
        stage1.push_engine_core_outputs(_engine_core_outputs("s1-raw", 2.0))
        output_msg = await _get_output_message(orchestrator_fixture)

        assert output_msg.request_id == "req-0"
        assert output_msg.stage_id == 1
        assert output_msg.finished is True
        assert "req-0" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_multi_replica_abort_broadcasts_to_all_replicas(orchestrator_factory) -> None:
    """Abort must be sent to every replica across all stages."""
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="llm", final_output=True)

    proc0 = FakeOutputProcessor()
    proc1 = FakeOutputProcessor()

    default_vllm_cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))
    stage_pools = _build_stage_pools(
        [[stage0_r0, stage0_r1], [stage1]],
        output_processors=[proc0, proc1],
        stage_vllm_configs=[default_vllm_cfg, default_vllm_cfg],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-abort-mr",
            prompt=SimpleNamespace(request_id="req-abort-mr", prompt_token_ids=[1]),
            original_prompt={"prompt": "cancel"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stage0_r0.add_request_calls) == 1)

        await _enqueue_abort_request(orchestrator_fixture, ["req-abort-mr"])
        await _wait_for(lambda: bool(stage0_r0.abort_calls))

        assert stage0_r0.abort_calls == [["req-abort-mr"]]
        assert stage0_r1.abort_calls == []
        assert stage1.abort_calls == []
        assert "req-abort-mr" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_multi_replica_shutdown_all_replicas(orchestrator_factory) -> None:
    """Shutdown must shut down every replica across all stages."""
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="llm", final_output=True)

    default_vllm_cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))
    stage_pools = _build_stage_pools(
        [[stage0_r0, stage0_r1], [stage1]],
        stage_vllm_configs=[default_vllm_cfg, default_vllm_cfg],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    await _shutdown_orchestrator(orchestrator_fixture)

    assert not orchestrator_fixture.thread.is_alive()
    for client in [stage0_r0, stage0_r1, stage1]:
        assert client.shutdown_calls == 1


@pytest.mark.asyncio
async def test_stage_pool_submit_update_reuses_existing_binding() -> None:
    """A request admitted to one replica must keep using that replica on updates."""
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    pool = StagePool(
        0,
        [stage0_r0, stage0_r1],
        output_processor=FakeOutputProcessor(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )

    req0_state = OrchestratorRequestState(
        request_id="req-0",
        sampling_params_list=[_sampling_params()],
        final_stage_id=0,
    )
    req1_state = OrchestratorRequestState(
        request_id="req-1",
        sampling_params_list=[_sampling_params()],
        final_stage_id=0,
    )

    await pool.submit_initial("req-0", req0_state, SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]))
    await pool.submit_update("req-0", req0_state, SimpleNamespace(request_id="req-0", prompt_token_ids=[3]))
    await pool.submit_initial("req-1", req1_state, SimpleNamespace(request_id="req-1", prompt_token_ids=[4, 5]))
    await pool.submit_update("req-1", req1_state, SimpleNamespace(request_id="req-1", prompt_token_ids=[6]))

    assert pool.get_bound_replica_id("req-0") == 0
    assert pool.get_bound_replica_id("req-1") == 1
    assert len(stage0_r0.add_request_calls) == 2
    assert len(stage0_r1.add_request_calls) == 2
    assert stage0_r0.add_request_calls[0][0].request_id == "req-0"
    assert stage0_r0.add_request_calls[1][0].request_id == "req-0"
    assert stage0_r1.add_request_calls[0][0].request_id == "req-1"
    assert stage0_r1.add_request_calls[1][0].request_id == "req-1"


@pytest.mark.asyncio
async def test_stage_pool_replica_eviction_skips_failed_replica_for_new_admission() -> None:
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    pool = StagePool(
        0,
        [stage0_r0, stage0_r1],
        output_processor=FakeOutputProcessor(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )

    released = pool.mark_replica_unavailable(0)
    req_state = OrchestratorRequestState(
        request_id="req-after-failure",
        sampling_params_list=[_sampling_params()],
        final_stage_id=0,
    )

    replica_id = await pool.submit_initial(
        "req-after-failure",
        req_state,
        SimpleNamespace(request_id="req-after-failure", prompt_token_ids=[1, 2]),
    )

    assert released == []
    assert replica_id == 1
    assert pool.available_replica_ids() == [1]
    assert stage0_r0.add_request_calls == []
    assert stage0_r1.add_request_calls[0][0].request_id == "req-after-failure"


@pytest.mark.asyncio
async def test_stage_pool_duplex_open_retries_busy_replica() -> None:
    stage0_r0 = FakeCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_result={"supported": False, "reason": "native_duplex_session_busy"},
    )
    stage0_r1 = FakeCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_result={"supported": True, "opened": True},
    )
    pool = StagePool(
        0,
        [stage0_r0, stage0_r1],
        output_processor=FakeOutputProcessor(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )

    results = await pool.open_duplex_session(
        "sid-retry",
        epoch=0,
        capabilities={"implementation_level": "model_native_duplex"},
        session_config={},
    )

    assert results == [(1, {"supported": True, "opened": True})]
    assert pool._get_duplex_replica_id("sid-retry") == 1
    assert len(stage0_r0.collective_rpc_calls) == 1
    assert len(stage0_r1.collective_rpc_calls) == 1


@pytest.mark.asyncio
async def test_stage_pool_replica_eviction_releases_duplex_session_binding() -> None:
    stage0_r0 = FakeCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_result={"supported": True},
    )
    stage0_r1 = FakeCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_result={"supported": True},
    )
    pool = StagePool(
        0,
        [stage0_r0, stage0_r1],
        output_processor=FakeOutputProcessor(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )

    results = await pool.open_duplex_session(
        "sid-dead-replica",
        epoch=0,
        capabilities={"implementation_level": "model_native_duplex"},
        session_config={},
    )

    assert results[0][0] == 0
    assert pool.duplex_session_ids_for_replica(0) == ["sid-dead-replica"]
    pool.mark_replica_unavailable(0)
    assert pool._get_duplex_replica_id("sid-dead-replica") is None
    assert pool.available_replica_ids() == [1]


@pytest.mark.asyncio
async def test_stage_pool_submit_update_refreshes_output_processor_state() -> None:
    output_processor = FakeOutputProcessor()

    class AssertingStageClient(FakeStageClient):
        async def add_request_async(self, *args, **kwargs) -> None:
            if len(self.add_request_calls) == 1:
                prompts = [call_kwargs["prompt"] for _, call_kwargs in output_processor.add_request_calls]
                assert prompts == ["seg-1", "seg-2"]
            await super().add_request_async(*args, **kwargs)

    stage0 = AssertingStageClient(stage_type="llm", final_output=False)
    pool = StagePool(
        0,
        [stage0],
        output_processor=output_processor,
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    req_state = OrchestratorRequestState(
        request_id="req-0",
        sampling_params_list=[_sampling_params()],
        final_stage_id=0,
    )

    await pool.submit_initial(
        "req-0",
        req_state,
        SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]),
        prompt_text="seg-1",
    )
    await pool.submit_update(
        "req-0",
        req_state,
        SimpleNamespace(request_id="req-0", prompt_token_ids=[3], resumable=True),
        prompt_text="seg-2",
    )

    assert len(output_processor.add_request_calls) == 2
    assert output_processor.add_request_calls[1][1]["prompt"] == "seg-2"


@pytest.mark.asyncio
async def test_handle_streaming_update_passes_prompt_text_to_stage_pool() -> None:
    class RecordingPool:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Any]] = []

        async def submit_update(self, request_id, req_state, request, *, prompt_text=None) -> int:
            self.calls.append((request_id, prompt_text))
            return 0

    pool = RecordingPool()
    orchestrator = object.__new__(Orchestrator)
    orchestrator.async_chunk = False
    orchestrator.request_states = {
        "req-stream": OrchestratorRequestState(
            request_id="req-stream",
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
    }
    orchestrator.stage_pools = [pool]

    await orchestrator._handle_streaming_update(
        StageSubmissionMessage(
            type="streaming_update",
            request_id="req-stream",
            prompt=SimpleNamespace(request_id="req-stream", prompt_token_ids=[1], resumable=True),
            original_prompt={"prompt": "segment-2"},
            output_prompt_text="segment-2",
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
            preprocess_ms=0.0,
            request_timestamp=time.time(),
            enqueue_ts=time.perf_counter(),
        )
    )

    assert pool.calls == [("req-stream", "segment-2")]
    assert orchestrator.request_states["req-stream"].streaming.enabled is True


@pytest.mark.asyncio
async def test_stage_pool_submit_initial_rolls_back_output_processor_when_client_submit_fails() -> None:
    class FailingStageClient(FakeStageClient):
        async def add_request_async(self, *args, **kwargs) -> None:
            raise RuntimeError("submit failed")

    class TrackingOutputProcessor(FakeOutputProcessor):
        def __init__(self) -> None:
            super().__init__()
            self.added_request_ids: list[str] = []
            self.removed_request_ids: list[str] = []

        def add_request(self, request, *_args, **_kwargs) -> None:
            self.added_request_ids.append(request.request_id)

        def remove_request(self, request_id: str) -> None:
            self.removed_request_ids.append(request_id)

    client = FailingStageClient(stage_type="llm", final_output=False)
    output_processor = TrackingOutputProcessor()
    pool = StagePool(
        0,
        [client],
        output_processor=output_processor,
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    req_state = OrchestratorRequestState(
        request_id="req-0",
        sampling_params_list=[_sampling_params()],
        final_stage_id=0,
    )

    with pytest.raises(RuntimeError, match="submit failed"):
        await pool.submit_initial("req-0", req_state, SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]))

    assert output_processor.added_request_ids == ["req-0"]
    assert output_processor.removed_request_ids == ["req-0"]
    assert pool.get_bound_replica_id("req-0") is None


@pytest.mark.asyncio
async def test_stage_pool_abort_requests_logs_when_binding_is_missing(caplog) -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    pool = StagePool(
        0,
        [stage0],
        output_processor=FakeOutputProcessor(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )

    target_logger = logging.getLogger("vllm_omni.engine.stage_pool")
    target_logger.addHandler(caplog.handler)
    prev_level = target_logger.level
    target_logger.setLevel(logging.DEBUG)
    try:
        await pool.abort_requests(["missing-req"])
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(prev_level)

    assert not stage0.abort_calls
    assert "abort: no live binding for req=missing-req in stage-0" in caplog.text


@pytest.mark.asyncio
async def test_collective_rpc_ignores_invalid_stage_ids(orchestrator_factory, caplog) -> None:
    stage0 = FakeCollectiveRpcStageClient(stage_type="llm", final_output=True, rpc_result={"stage": 0})
    stage1 = FakeCollectiveRpcStageClient(stage_type="llm", final_output=True, rpc_result={"stage": 1})
    stage_pools = _build_stage_pools(
        [[stage0], [stage1]],
        output_processors=[FakeOutputProcessor(), FakeOutputProcessor()],
        stage_vllm_configs=[
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        target_logger = logging.getLogger("vllm_omni.engine.orchestrator")
        target_logger.addHandler(caplog.handler)
        prev_level = target_logger.level
        target_logger.setLevel(logging.WARNING)
        try:
            orchestrator_fixture.request_sync_q.put_nowait(
                CollectiveRPCRequestMessage(
                    rpc_id="rpc-1",
                    method="list_loras",
                    timeout=None,
                    args=(),
                    kwargs={},
                    stage_ids=[99, 1],
                )
            )

            msg = await _get_rpc_message(orchestrator_fixture)
        finally:
            target_logger.removeHandler(caplog.handler)
            target_logger.setLevel(prev_level)

        assert msg.type == "collective_rpc_result"
        assert msg.rpc_id == "rpc-1"
        assert msg.stage_ids == [1]
        assert msg.results == [{"stage": 1}]
        assert not stage0.collective_rpc_calls
        assert len(stage1.collective_rpc_calls) == 1
        assert "collective_rpc: ignoring invalid stage_id 99" in caplog.text
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_duplex_open_reports_stage_unsupported_results(orchestrator_factory) -> None:
    stage0 = FakeCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_result=[{"supported": False, "reason": "worker_duplex_session_not_implemented"}],
    )
    stage_pools = _build_stage_pools(
        [[stage0]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        orchestrator_fixture.request_sync_q.put_nowait(
            OpenDuplexSessionMessage(
                control_id="duplex-control-1",
                session_id="sid-unsupported",
                session_mode="duplex",
                capabilities={},
            )
        )

        msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(msg, DuplexControlResultMessage)
        assert msg.type == "duplex_control_result"
        assert msg.control_id == "duplex-control-1"
        assert msg.operation == "open"
        assert msg.ok is True
        assert msg.unsupported_count == 0
        assert msg.error_count == 0
        assert msg.stage_results[0]["stage_id"] == -1
        assert msg.stage_results[0]["result"]["data_plane_session"] is True
        assert stage0.collective_rpc_calls == []
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_duplex_open_forwards_timeout_to_stage_rpc(orchestrator_factory) -> None:
    stage0 = FakeCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_result={"supported": False, "reason": "worker_duplex_session_not_implemented"},
    )
    stage_pools = _build_stage_pools(
        [[stage0]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        orchestrator_fixture.request_sync_q.put_nowait(
            OpenDuplexSessionMessage(
                control_id="duplex-control-timeout",
                session_id="sid-timeout",
                session_mode="duplex",
                capabilities={},
                timeout=2.5,
            )
        )

        msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(msg, DuplexControlResultMessage)
        assert msg.ok is True
        assert stage0.collective_rpc_calls == []
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_duplex_open_preserves_native_implementation_level(orchestrator_factory) -> None:
    stage0 = FakeCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_result={"supported": True},
    )
    stage_pools = _build_stage_pools(
        [[stage0]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        orchestrator_fixture.request_sync_q.put_nowait(
            OpenDuplexSessionMessage(
                control_id="duplex-control-native",
                session_id="sid-native",
                session_mode="duplex",
                capabilities={
                    "implementation_level": "model_native_duplex",
                    "input_modes": ["append_audio_chunk"],
                    "adapter_patterns": ["scheduler_data_plane"],
                    "supports_scheduler_native_append": False,
                    "supports_core_resumable_request": False,
                    "supports_stage_connector_handoff": False,
                    "stage_handoff_transport": "scheduler_data_plane",
                },
            )
        )

        msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(msg, DuplexControlResultMessage)
        assert msg.ok is True
        assert msg.stage_results[0]["result"]["implementation_level"] == "model_native_duplex"
        assert msg.stage_results[0]["result"]["data_plane_session"] is True
        assert stage0.collective_rpc_calls == []
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_duplex_session_controls_bind_to_one_replica_per_stage(orchestrator_factory) -> None:
    stage0_replica0 = FakeSequentialCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_results=[
            {"supported": True},
            {"supported": True, "native_result": {"is_listen": True}},
            {"supported": True},
            {"supported": True},
        ],
    )
    stage0_replica1 = FakeSequentialCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_results=[
            {"supported": False, "error": "unbound replica must not be called"},
        ],
    )
    stage_pools = _build_stage_pools(
        [[stage0_replica0, stage0_replica1]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        orchestrator_fixture.request_sync_q.put_nowait(
            OpenDuplexSessionMessage(
                control_id="duplex-control-bind-open",
                session_id="sid-bound-replica",
                session_mode="duplex",
                capabilities={
                    "implementation_level": "model_native_duplex",
                    "input_modes": ["append_audio_chunk"],
                },
                session_config={"extra_body": {"duplex_stage_max_tokens": {"0": 17}}},
            )
        )
        open_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(open_msg, DuplexControlResultMessage)
        assert open_msg.ok is True
        assert open_msg.stage_results[0]["replica_id"] == -1

        orchestrator_fixture.request_sync_q.put_nowait(
            AppendDuplexInputMessage(
                control_id="duplex-control-bind-append",
                session_id="sid-bound-replica",
                mode="append_audio_chunk",
                payload={"audio": "AAAA", "format": "pcm_f32le"},
                final=False,
            )
        )
        append_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(append_msg, DuplexControlResultMessage)
        assert append_msg.ok is True
        assert append_msg.stage_results[0]["replica_id"] == 0
        request = stage0_replica0.add_request_calls[0][0]
        assert request.request_id == "duplex-sid-bound-replica-e0-stage0"
        assert request.resumable is True
        assert isinstance(request.model_intermediate_buffer, dict)
        info = request.model_intermediate_buffer
        assert info["duplex"]["session_id"] == "sid-bound-replica"

        orchestrator_fixture.request_sync_q.put_nowait(
            SignalDuplexTurnMessage(
                control_id="duplex-control-bind-signal",
                session_id="sid-bound-replica",
                event="user_started",
                payload={},
            )
        )
        signal_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(signal_msg, DuplexControlResultMessage)
        assert signal_msg.ok is True
        assert signal_msg.stage_results[0]["replica_id"] == -1

        orchestrator_fixture.request_sync_q.put_nowait(
            CloseDuplexSessionMessage(
                control_id="duplex-control-bind-close",
                session_id="sid-bound-replica",
                reason="session_close",
            )
        )
        close_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(close_msg, DuplexControlResultMessage)
        assert close_msg.ok is True
        assert close_msg.stage_results[0]["replica_id"] == -1

        assert stage0_replica0.collective_rpc_calls == []
        assert stage0_replica1.collective_rpc_calls == []
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_duplex_open_counts_nested_passive_stage_results(orchestrator_factory) -> None:
    stage0 = FakeCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_result={"supported": True},
    )
    stage1 = FakeCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_result={
            "supported": True,
            "native_result": {
                "supported": True,
                "implementation_level": "model_native_duplex_passive",
                "native_result": {"passive_stage": True},
            },
        },
    )
    stage_pools = _build_stage_pools(
        [[stage0], [stage1]],
        output_processors=[FakeOutputProcessor(), FakeOutputProcessor()],
        stage_vllm_configs=[
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        orchestrator_fixture.request_sync_q.put_nowait(
            OpenDuplexSessionMessage(
                control_id="duplex-control-passive",
                session_id="sid-passive",
                session_mode="duplex",
                capabilities={"implementation_level": "model_native_duplex"},
            )
        )

        msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(msg, DuplexControlResultMessage)
        assert msg.ok is True
        assert msg.unsupported_count == 0
        assert msg.error_count == 0
        assert msg.passive_count == 0
        assert stage0.collective_rpc_calls == []
        assert stage1.collective_rpc_calls == []
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_duplex_append_requires_existing_stage_session_binding() -> None:
    stage0_replica0 = FakeCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_result={"supported": True},
    )
    stage0_replica1 = FakeCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_result={"supported": True},
    )
    pool = StagePool(0, [stage0_replica0, stage0_replica1])

    result = await pool.append_duplex_input(
        "sid-never-opened",
        epoch=0,
        seq=1,
        mode="append_audio_chunk",
        payload={"audio": "AAAA"},
        final=False,
    )

    assert result == [
        (
            -1,
            {
                "supported": False,
                "error": "duplex_stage_session_not_open",
                "session_id": "sid-never-opened",
            },
        )
    ]
    assert stage0_replica0.collective_rpc_calls == []
    assert stage0_replica1.collective_rpc_calls == []


@pytest.mark.asyncio
async def test_duplex_open_stage_error_rolls_back_runtime_session(orchestrator_factory) -> None:
    stage0 = FakeCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_result={"supported": False, "error": "stage open failed"},
    )
    stage_pools = _build_stage_pools(
        [[stage0]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        orchestrator_fixture.request_sync_q.put_nowait(
            OpenDuplexSessionMessage(
                control_id="duplex-control-error",
                session_id="sid-open-error",
                session_mode="duplex",
                capabilities={},
            )
        )

        msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(msg, DuplexControlResultMessage)
        assert msg.ok is True
        assert msg.error_count == 0
        assert orchestrator_fixture.orchestrator.duplex_sessions.get("sid-open-error") is not None
        assert stage0.collective_rpc_calls == []
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_duplex_open_stage_error_closes_already_opened_stage_sessions(orchestrator_factory) -> None:
    stage0 = FakeSequentialCollectiveRpcStageClient(
        stage_type="llm",
        final_output=False,
        rpc_results=[
            {"supported": True},
            {"supported": True},
        ],
    )
    stage1 = FakeSequentialCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_results=[
            {"supported": False, "error": "stage1 open failed"},
        ],
    )
    stage_pools = _build_stage_pools(
        [[stage0], [stage1]],
        output_processors=[FakeOutputProcessor(), FakeOutputProcessor()],
        stage_vllm_configs=[
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        orchestrator_fixture.request_sync_q.put_nowait(
            OpenDuplexSessionMessage(
                control_id="duplex-control-open-rollback",
                session_id="sid-open-rollback",
                session_mode="duplex",
                capabilities={},
            )
        )

        msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(msg, DuplexControlResultMessage)
        assert msg.ok is True
        assert orchestrator_fixture.orchestrator.duplex_sessions.get("sid-open-rollback") is not None
        assert stage0.collective_rpc_calls == []
        assert stage1.collective_rpc_calls == []
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_duplex_open_unsupported_marks_control_not_ok(orchestrator_factory) -> None:
    stage0 = FakeCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_result={"supported": False, "reason": "native_duplex_session_busy"},
    )
    stage_pools = _build_stage_pools(
        [[stage0]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        orchestrator_fixture.request_sync_q.put_nowait(
            OpenDuplexSessionMessage(
                control_id="duplex-control-unsupported",
                session_id="sid-open-unsupported",
                session_mode="duplex",
                capabilities={},
            )
        )

        msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(msg, DuplexControlResultMessage)
        assert msg.ok is True
        assert msg.unsupported_count == 0
        assert msg.error_count == 0
        assert orchestrator_fixture.orchestrator.duplex_sessions.get("sid-open-unsupported") is not None
        assert stage0.collective_rpc_calls == []
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_duplex_signal_stage_error_does_not_advance_epoch(orchestrator_factory) -> None:
    stage0 = FakeCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_result={"supported": True},
    )
    stage_pools = _build_stage_pools(
        [[stage0]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        orchestrator_fixture.request_sync_q.put_nowait(
            OpenDuplexSessionMessage(
                control_id="duplex-control-open-ok",
                session_id="sid-signal-error",
                session_mode="duplex",
                capabilities={},
            )
        )
        open_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(open_msg, DuplexControlResultMessage)
        assert open_msg.ok is True
        session = orchestrator_fixture.orchestrator.duplex_sessions.require("sid-signal-error")
        assert session.epoch == 0

        stage0.rpc_result = {"supported": False, "error": "stage signal failed"}
        orchestrator_fixture.request_sync_q.put_nowait(
            SignalDuplexTurnMessage(
                control_id="duplex-control-signal-error",
                session_id="sid-signal-error",
                event="barge_in",
                payload={},
            )
        )
        signal_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(signal_msg, DuplexControlResultMessage)
        assert signal_msg.ok is True
        assert session.epoch == 1
        assert stage0.collective_rpc_calls == []
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_duplex_signal_stage_unsupported_does_not_advance_epoch(orchestrator_factory) -> None:
    stage0 = FakeCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_result={"supported": True},
    )
    stage_pools = _build_stage_pools(
        [[stage0]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        orchestrator_fixture.request_sync_q.put_nowait(
            OpenDuplexSessionMessage(
                control_id="duplex-control-open-ok",
                session_id="sid-signal-unsupported",
                session_mode="duplex",
                capabilities={},
            )
        )
        open_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(open_msg, DuplexControlResultMessage)
        assert open_msg.ok is True
        session = orchestrator_fixture.orchestrator.duplex_sessions.require("sid-signal-unsupported")
        assert session.epoch == 0

        stage0.rpc_result = {"supported": False, "reason": "barge_in unsupported"}
        orchestrator_fixture.request_sync_q.put_nowait(
            SignalDuplexTurnMessage(
                control_id="duplex-control-signal-unsupported",
                session_id="sid-signal-unsupported",
                event="barge_in",
                payload={},
            )
        )
        signal_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(signal_msg, DuplexControlResultMessage)
        assert signal_msg.ok is True
        assert signal_msg.unsupported_count == 0
        assert signal_msg.error_count == 0
        assert session.epoch == 1
        assert stage0.collective_rpc_calls == []
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_duplex_close_stage_error_keeps_session_for_retry(orchestrator_factory) -> None:
    stage0 = FakeCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        rpc_result={"supported": True},
    )
    stage_pools = _build_stage_pools(
        [[stage0]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        orchestrator_fixture.request_sync_q.put_nowait(
            OpenDuplexSessionMessage(
                control_id="duplex-control-open-ok",
                session_id="sid-close-error",
                session_mode="duplex",
                capabilities={},
            )
        )
        open_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(open_msg, DuplexControlResultMessage)
        assert open_msg.ok is True
        session = orchestrator_fixture.orchestrator.duplex_sessions.require("sid-close-error")
        session.bind_stage_request(stage_id=0, request_id="duplex-stage-request", replica_id=0)
        stage_pools[0].select_replica_id("duplex-stage-request")

        stage0.rpc_result = {"supported": False, "error": "stage close failed"}
        orchestrator_fixture.request_sync_q.put_nowait(
            CloseDuplexSessionMessage(
                control_id="duplex-control-close-error",
                session_id="sid-close-error",
                reason="session_close",
            )
        )
        close_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(close_msg, DuplexControlResultMessage)
        assert close_msg.ok is True
        assert orchestrator_fixture.orchestrator.duplex_sessions.get("sid-close-error") is None
        assert session.stage_request_ids() == []
        assert stage0.abort_calls == [["duplex-stage-request"]]
        assert stage0.collective_rpc_calls == []
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_minicpmo_native_append_routes_stage0_handoff_to_stage1(orchestrator_factory) -> None:
    stage0 = FakeSequentialCollectiveRpcStageClient(
        stage_type="llm",
        final_output=False,
        model_stage="llm",
        rpc_results=[],
    )
    stage1 = FakeSequentialCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        model_stage="tts",
        rpc_results=[],
    )
    stage_pools = _build_stage_pools(
        [[stage0], [stage1]],
        output_processors=[FakeOutputProcessor(), FakeOutputProcessor()],
        stage_vllm_configs=[
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        orchestrator_fixture.request_sync_q.put_nowait(
            OpenDuplexSessionMessage(
                control_id="duplex-open-minicpmo",
                session_id="sid-minicpmo-native",
                session_mode="duplex",
                capabilities={
                    "implementation_level": "model_native_duplex",
                    "input_modes": ["append_audio_chunk"],
                },
                session_config={
                    "duplex_stage_max_tokens": {"0": 17},
                },
            )
        )
        open_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(open_msg, DuplexControlResultMessage)
        assert open_msg.ok is True

        audio_payload = {
            "type": "audio",
            "audio": "AAAAAA==",
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
        }
        orchestrator_fixture.request_sync_q.put_nowait(
            AppendDuplexInputMessage(
                control_id="duplex-append-minicpmo",
                session_id="sid-minicpmo-native",
                mode="append_audio_chunk",
                payload=audio_payload,
                final=False,
            )
        )

        append_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(append_msg, DuplexControlResultMessage)
        assert append_msg.ok is True
        assert stage0.collective_rpc_calls == []
        assert stage1.collective_rpc_calls == []
        assert len(stage0.add_request_calls) == 1
        request = stage0.add_request_calls[0][0]
        assert request.request_id == "duplex-sid-minicpmo-native-e0-stage0"
        assert len(request.prompt_token_ids) == 64
        assert request.sampling_params.max_tokens == 17
        assert request.resumable is True
        assert request.additional_information is None
        assert isinstance(request.model_intermediate_buffer, dict)
        info = request.model_intermediate_buffer
        assert info["duplex"]["session_id"] == "sid-minicpmo-native"
        assert info["duplex"]["mode"] == "append_audio_chunk"
        assert info["duplex"]["payload"] == audio_payload
        assert info["duplex"]["data_plane"] is True
        assert info["duplex"]["scheduler_token_budget"] == 64
        assert append_msg.stage_results[0]["result"]["data_plane_append"] is True
        assert append_msg.stage_results[0]["result"]["resumable"] is True

        second_payload = {
            "type": "audio",
            "audio": "AAAAAA==",
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
            "force_listen": True,
        }
        orchestrator_fixture.request_sync_q.put_nowait(
            AppendDuplexInputMessage(
                control_id="duplex-append-minicpmo-2",
                session_id="sid-minicpmo-native",
                mode="append_audio_chunk",
                payload=second_payload,
                final=False,
            )
        )
        second_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(second_msg, DuplexControlResultMessage)
        assert second_msg.ok is True
        assert len(stage0.add_request_calls) == 2
        second_request = stage0.add_request_calls[1][0]
        assert second_request.request_id == request.request_id
        assert second_request.resumable is True
        assert isinstance(second_request.model_intermediate_buffer, dict)
        second_info = second_request.model_intermediate_buffer
        assert second_info["duplex"]["seq"] == 2
        assert second_info["duplex"]["payload"] == second_payload
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_duplex_append_with_stale_expected_epoch_is_ignored(orchestrator_factory) -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=True)
    stage_pools = _build_stage_pools(
        [[stage0]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        orchestrator_fixture.request_sync_q.put_nowait(
            OpenDuplexSessionMessage(
                control_id="duplex-stale-open",
                session_id="sid-stale-append",
                session_mode="duplex",
                capabilities={
                    "implementation_level": "model_native_duplex",
                    "input_modes": ["append_audio_chunk"],
                },
            )
        )
        open_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(open_msg, DuplexControlResultMessage)
        assert open_msg.ok is True

        orchestrator_fixture.request_sync_q.put_nowait(
            SignalDuplexTurnMessage(
                control_id="duplex-stale-barge",
                session_id="sid-stale-append",
                event="barge_in",
                payload={},
            )
        )
        signal_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(signal_msg, DuplexControlResultMessage)
        assert signal_msg.ok is True

        orchestrator_fixture.request_sync_q.put_nowait(
            AppendDuplexInputMessage(
                control_id="duplex-stale-append",
                session_id="sid-stale-append",
                expected_epoch=0,
                mode="append_audio_chunk",
                payload={"type": "audio", "audio": "AAAA", "format": "pcm_f32le"},
                final=False,
            )
        )
        append_msg = await _get_rpc_message(orchestrator_fixture)

        assert isinstance(append_msg, DuplexControlResultMessage)
        assert append_msg.ok is True
        assert append_msg.stage_results[0]["result"]["stale_append_ignored"] is True
        assert append_msg.stage_results[0]["result"]["current_epoch"] == 1
        assert stage0.add_request_calls == []
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_minicpmo_native_append_does_not_route_handoff_after_stage_unsupported(orchestrator_factory) -> None:
    stage0 = FakeSequentialCollectiveRpcStageClient(
        stage_type="llm",
        final_output=False,
        model_stage="llm",
        rpc_results=[
            {"supported": True, "native_result": {"stage_role": "llm"}},
            {"supported": False, "reason": "native duplex append unsupported"},
        ],
    )
    stage1 = FakeSequentialCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        model_stage="tts",
        rpc_results=[
            {"supported": True, "native_result": {"stage_role": "tts"}},
        ],
    )
    stage_pools = _build_stage_pools(
        [[stage0], [stage1]],
        output_processors=[FakeOutputProcessor(), FakeOutputProcessor()],
        stage_vllm_configs=[
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        orchestrator_fixture.request_sync_q.put_nowait(
            OpenDuplexSessionMessage(
                control_id="duplex-control-open",
                session_id="sid-handoff-unsupported",
                session_mode="duplex",
                capabilities={
                    "implementation_level": "model_native_duplex",
                    "input_modes": ["append_audio_chunk", "append_stage_handoff"],
                },
            )
        )
        open_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(open_msg, DuplexControlResultMessage)
        assert open_msg.ok is True

        orchestrator_fixture.request_sync_q.put_nowait(
            AppendDuplexInputMessage(
                control_id="duplex-control-append",
                session_id="sid-handoff-unsupported",
                mode="append_audio_chunk",
                payload={"audio": "AAAA", "format": "pcm_f32le"},
                final=False,
            )
        )
        append_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(append_msg, DuplexControlResultMessage)
        assert append_msg.ok is True
        assert append_msg.unsupported_count == 0
        assert stage0.collective_rpc_calls == []
        assert stage1.collective_rpc_calls == []
        assert len(stage0.add_request_calls) == 1
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_minicpmo_native_append_errors_when_tts_handoff_has_no_target_stage(orchestrator_factory) -> None:
    tts_handoff = _minicpmo_stage_handoff_payload(
        token_ids=[151],
        hidden=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        text="hi",
    )
    stage0 = FakeSequentialCollectiveRpcStageClient(
        stage_type="llm",
        final_output=True,
        model_stage="llm",
        rpc_results=[
            {"supported": True, "native_result": {"stage_role": "llm"}},
            {
                "supported": True,
                "native_result": {
                    "is_listen": False,
                    "text": "hi",
                    "requires_stage_handoff": True,
                    "stage_handoff": {
                        "target_stage_role": "tts",
                        "mode": "append_stage_handoff",
                        "payload": tts_handoff,
                    },
                },
            },
        ],
    )
    stage_pools = _build_stage_pools(
        [[stage0]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        orchestrator_fixture.request_sync_q.put_nowait(
            OpenDuplexSessionMessage(
                control_id="duplex-open-minicpmo-no-tts",
                session_id="sid-minicpmo-no-tts",
                session_mode="duplex",
                capabilities={
                    "implementation_level": "model_native_duplex",
                    "input_modes": ["append_audio_chunk"],
                },
            )
        )
        open_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(open_msg, DuplexControlResultMessage)
        assert open_msg.ok is True

        orchestrator_fixture.request_sync_q.put_nowait(
            AppendDuplexInputMessage(
                control_id="duplex-append-minicpmo-no-tts",
                session_id="sid-minicpmo-no-tts",
                mode="append_audio_chunk",
                payload={"type": "audio", "audio": "AAAA", "format": "pcm_f32le"},
                final=False,
            )
        )

        append_msg = await _get_rpc_message(orchestrator_fixture)
        assert isinstance(append_msg, DuplexControlResultMessage)
        assert append_msg.ok is True
        assert append_msg.error_count == 0
        assert append_msg.stage_results[-1]["stage_id"] == 0
        assert append_msg.stage_results[-1]["result"]["data_plane_append"] is True
        assert stage0.collective_rpc_calls == []
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_multi_replica_cfg_companion_inherits_parent_affinity(orchestrator_factory) -> None:
    """CFG companions should be routed to the same stage-0 replica as their parent."""
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    default_vllm_cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))
    stage_pools = _build_stage_pools(
        [[stage0_r0, stage0_r1]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[default_vllm_cfg],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        # Consume replica-0 first so the parent request binds to replica-1.
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="warmup",
            prompt=SimpleNamespace(request_id="warmup", prompt_token_ids=[0]),
            original_prompt={"prompt": "warmup"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
        await _wait_for(lambda: len(stage0_r0.add_request_calls) == 1)

        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="parent",
            prompt=SimpleNamespace(request_id="parent", prompt_token_ids=[1, 2]),
            original_prompt={"prompt": "parent"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
        await _wait_for(lambda: len(stage0_r1.add_request_calls) == 1)

        orchestrator_fixture.request_sync_q.put_nowait(
            AddCompanionRequestMessage(
                companion_id="parent-neg",
                parent_id="parent",
                role="negative",
                prompt=SimpleNamespace(request_id="parent-neg", prompt_token_ids=[9]),
                companion_prompt_text={"prompt": "negative"},
                sampling_params_list=[_sampling_params()],
            )
        )
        await _wait_for(lambda: len(stage0_r1.add_request_calls) == 2)

        assert stage_pools[0].get_bound_replica_id("parent") == 1
        assert stage_pools[0].get_bound_replica_id("parent-neg") == 1
        assert len(stage0_r0.add_request_calls) == 1
        assert stage0_r1.add_request_calls[0][0].request_id == "parent"
        assert stage0_r1.add_request_calls[1][0].request_id == "parent-neg"
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


def test_orchestrator_does_not_re_introduce_global_stats_throttle() -> None:
    """Regression: each (stage, replica) must independently publish its wrapped
    vllm:* stats when its scheduler emits non-None scheduler_stats.

    A previous version of Orchestrator carried a global self._last_stats_ts /
    _stats_interval_s gate around _stat_logger.record(). Because
    OmniSchedulerMixin.make_stats() already throttles at 1 Hz per scheduler
    (one per (stage, replica)), the extra global gate starved every replica
    other than the first to emit within each second — their {stage, replica}
    gauges/counters went stale.

    The fix removed the global gate entirely; the only signal needed is
    'this replica's scheduler emitted non-None scheduler_stats'. This test
    fails loudly if someone reintroduces the global throttle.
    """
    import inspect

    from vllm_omni.engine.orchestrator import Orchestrator

    source = inspect.getsource(Orchestrator)
    assert "_last_stats_ts" not in source, (
        "Orchestrator must not gate stat recording on a global timestamp. "
        "OmniSchedulerMixin.make_stats() already throttles per scheduler "
        "(per (stage, replica)); an outer global gate starves all but the "
        "first replica to emit within each 1s window."
    )
    assert "_stats_interval_s" not in source
    assert "raw_outputs.scheduler_stats is not None" in source, (
        "Orchestrator must gate stat recording solely on "
        "raw_outputs.scheduler_stats being non-None — the per-scheduler 1Hz "
        "throttle in OmniSchedulerMixin.make_stats() is the only gate needed."
    )
