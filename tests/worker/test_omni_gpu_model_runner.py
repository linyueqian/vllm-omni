from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.outputs import SamplerOutput

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class DummyBuffer:
    """A minimal buffer wrapper that exposes the `.gpu` attribute."""

    def __init__(self, t: torch.Tensor):
        self.gpu = t


class DummyInputBatch:
    """A minimal input batch that only provides `req_ids`."""

    def __init__(self, req_ids):
        self.req_ids = req_ids


class DummyReqState:
    """A minimal request state container."""

    pass


class MiMoAudioForConditionalGeneration(torch.nn.Module):
    """Dummy model whose class name must exactly match the production check."""

    def __init__(self):
        super().__init__()

    # No real forward needed for these tests.


class ReplaceSampledTokensModel(torch.nn.Module):
    """Returns a replacement sampled-token tensor from the post-sample hook."""

    def __init__(self):
        super().__init__()
        self.observed_sampled_token_ids = None

    def postprocess_sampled_tokens(self, sampled_token_ids, req_ids, req_id_to_index, model_intermediate_buffer):
        assert req_ids == ["r1", "r2"]
        assert req_id_to_index == {"r1": 0, "r2": 1}
        assert model_intermediate_buffer == {"r1": {"token": 1}, "r2": {"token": 2}}
        self.observed_sampled_token_ids = sampled_token_ids.clone()
        return sampled_token_ids + 10


class OverlaySampledTokensModel(torch.nn.Module):
    """Validates that post-sample hooks receive overlaid pending updates."""

    def __init__(self):
        super().__init__()
        self.observed_buffer = None
        self.pooler_output_buffer_keys = ("audio_token_ids",)

    def postprocess_sampled_tokens(self, sampled_token_ids, req_ids, req_id_to_index, model_intermediate_buffer):
        del sampled_token_ids, req_ids, req_id_to_index
        self.observed_buffer = model_intermediate_buffer
        return None


class RawTokenPreprocessModel(torch.nn.Module):
    """Tracks whether preprocess receives raw input ids without an embed slice."""

    has_preprocess = True
    requires_raw_input_tokens = True

    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.observed_input_embeds = []

    def preprocess(self, input_ids, input_embeds, **info_dict):
        self.observed_input_embeds.append(input_embeds)
        req_embeds = input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, self.hidden_size)
        return input_ids + 100, req_embeds, {"marker_seen": info_dict.get("marker")}


class MultimodalPreprocessModel(torch.nn.Module):
    """Tracks fallback raw-token slices when multimodal preprocess runs from embeds."""

    has_preprocess = True
    requires_raw_input_tokens = False

    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.observed_input_ids = []
        self.observed_input_embeds = []

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, is_multimodal=None):
        del multimodal_embeddings, is_multimodal
        return input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, self.hidden_size)

    def preprocess(self, input_ids, input_embeds, **info_dict):
        self.observed_input_ids.append(input_ids.clone())
        self.observed_input_embeds.append(input_embeds.clone() if isinstance(input_embeds, torch.Tensor) else None)
        req_embeds = input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, self.hidden_size)
        return input_ids, req_embeds, {"marker_seen": info_dict.get("marker")}


class DummyTalkerMTP(torch.nn.Module):
    """A fake talker_mtp module for deterministic CPU testing."""

    def forward(self, req_input_ids, req_embeds, last_talker_hidden, text_step):
        # Deterministic behavior:
        # - output embeds = input embeds + 1
        # - output codes = [[0], [1], ...]
        bsz = req_embeds.shape[0]
        new_embeds = req_embeds + 1.0
        codes = torch.arange(bsz, dtype=torch.int64).view(bsz, 1)
        return new_embeds, codes


@contextmanager
def _noop_forward_context(*args, **kwargs):
    """A no-op context manager to replace vLLM forward context in CPU tests."""
    yield


def _make_runner(req_ids=("r1", "r2"), hidden_size=4):
    # Create an instance without calling OmniGPUModelRunner.__init__
    runner = object.__new__(OmniGPUModelRunner)

    # Minimal attributes used by OmniGPUModelRunner._talker_mtp_forward
    runner.input_batch = DummyInputBatch(list(req_ids))
    runner.requests = {rid: DummyReqState() for rid in req_ids}
    runner.model_intermediate_buffer = {}

    # query_start_loc.cpu[req_index] is used to locate the token position
    # in the flattened `inputs_embeds`.
    runner.query_start_loc = type("QSL", (), {})()
    # Map: r1 -> offset 0, r2 -> offset 3
    runner.query_start_loc.cpu = torch.tensor([0, 3], dtype=torch.int32)

    bsz = len(req_ids)
    runner.talker_mtp_input_ids = DummyBuffer(torch.zeros((bsz,), dtype=torch.int64))
    runner.talker_mtp_inputs_embeds = DummyBuffer(torch.zeros((bsz, hidden_size), dtype=torch.float32))
    runner.last_talker_hidden = DummyBuffer(torch.zeros((bsz, hidden_size), dtype=torch.float32))
    runner.text_step = DummyBuffer(torch.zeros((bsz, hidden_size), dtype=torch.float32))

    runner.talker_mtp = DummyTalkerMTP()
    runner.model = SimpleNamespace(talker_mtp_output_key="code_predictor_codes")
    runner.vllm_config = object()

    # Provide a minimal implementation that returns the expected 4-tuple.
    def _determine_batch_execution_and_padding(**kwargs):
        return None, object(), None, None, None

    runner._determine_batch_execution_and_padding = _determine_batch_execution_and_padding

    # Use the real merge method from OmniGPUModelRunner.
    return runner


def _make_runner_for_mimo(req_id="r_mimo"):
    """Create a minimal runner with MiMoAudio-like model and request state."""
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = MiMoAudioForConditionalGeneration()

    # Minimal vllm_config / model_config used by helper.
    class _DummyModelConfig:
        async_chunk = False

    class _DummyVllmConfig:
        model_config = _DummyModelConfig()

    runner.vllm_config = _DummyVllmConfig()

    # Attach a single request state with mm_features and additional_information_cpu.
    req_state = DummyReqState()
    req_state.mm_features = ["mm_feature_obj"]
    req_state.additional_information_cpu = {"some_key": "some_value"}

    runner.requests = {req_id: req_state}

    return runner


def _make_preprocess_runner(model, hidden_size=4):
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = model
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.supports_mm_inputs = False
    runner.enable_prompt_embeds = False
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.positions = DummyBuffer(torch.arange(8, dtype=torch.int64))
    runner.input_ids = DummyBuffer(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    runner.inputs_embeds = DummyBuffer(torch.full((4, hidden_size), -1.0, dtype=torch.float32))
    runner.input_batch = SimpleNamespace(
        req_ids=["r1"],
        num_computed_tokens_cpu=np.array([0], dtype=np.int32),
    )
    runner.requests = {"r1": SimpleNamespace(prompt_token_ids=[], mm_features=[])}
    runner.model_intermediate_buffer = {"r1": {"marker": "r1"}}
    runner.query_start_loc = SimpleNamespace(cpu=torch.tensor([0], dtype=torch.int32))
    runner.dtype = torch.float32
    runner.device = torch.device("cpu")
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=False))
    runner._init_model_kwargs = lambda: {}
    return runner


def _make_mm_preprocess_runner(model, hidden_size=4):
    runner = _make_preprocess_runner(model, hidden_size=hidden_size)
    runner.supports_mm_inputs = True
    runner.encoder_cache = None
    runner._execute_mm_encoder = lambda scheduler_output: None
    runner._gather_mm_embeddings = lambda scheduler_output: (None, None)
    runner._prepare_mm_inputs = lambda num_input_tokens: (
        None,
        runner.inputs_embeds.gpu[:num_input_tokens],
    )
    runner._extract_mm_kwargs = lambda scheduler_output: {}
    runner.maybe_get_ec_connector_output = _noop_forward_context
    return runner


class StopAfterBookkeepingError(Exception):
    pass


def _make_sample_tokens_runner(model):
    runner = object.__new__(GPUARModelRunner)
    runner.model = model
    runner.speculative_config = None
    runner.use_async_scheduling = False
    runner.input_batch = SimpleNamespace(
        req_ids=["r1", "r2"],
        req_id_to_index={"r1": 0, "r2": 1},
        sampling_metadata=SimpleNamespace(no_penalties=True),
        prev_sampled_token_ids=None,
        num_tokens_no_spec=np.array([1, 1], dtype=np.int32),
        token_ids_cpu=np.array([[1, 0, 0, 0], [2, 0, 0, 0]], dtype=np.int32),
        vocab_size=32000,
    )
    runner.model_intermediate_buffer = {"r1": {"token": 1}, "r2": {"token": 2}}
    runner.requests = {
        "r1": SimpleNamespace(output_token_ids=[1]),
        "r2": SimpleNamespace(output_token_ids=[2]),
    }
    runner.execute_model_state = (
        SimpleNamespace(total_num_scheduled_tokens=2, num_scheduled_tokens={"r1": 1, "r2": 1}),
        None,
        None,
        None,
        torch.zeros((2, 4), dtype=torch.float32),
        torch.zeros((2, 4), dtype=torch.float32),
        None,
        None,
        None,
        None,
        None,
    )
    runner._sample = lambda logits, spec_decode_metadata: SamplerOutput(
        sampled_token_ids=torch.tensor([[1], [2]], dtype=torch.int32),
        logprobs_tensors=None,
    )
    runner.max_model_len = 4
    runner.query_start_loc = SimpleNamespace(cpu=torch.tensor([0, 1], dtype=torch.int32))
    runner._omni_num_scheduled_tokens_np = np.array([1, 1], dtype=np.int32)
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(engine_output_type="omni"))
    runner.model_config = SimpleNamespace(enable_return_routed_experts=False)
    runner.supports_mm_inputs = False
    runner.kv_connector_output = None
    runner.eplb_step = lambda: None
    runner.finalize_kv_connector = lambda: None
    return runner


def test_talker_mtp_forward_cpu_updates_inputs_and_info(monkeypatch):
    # Patch the module-level `set_forward_context` symbol used inside
    # OmniGPUModelRunner._talker_mtp_forward.
    import vllm_omni.worker.gpu_model_runner as mod  # Must be the same module that defines OmniGPUModelRunner

    monkeypatch.setattr(mod, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1", "r2"), hidden_size=4)

    def fake_determine(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens, use_cascade_attn):
        batch_desc = SimpleNamespace(num_tokens=int(num_tokens))
        return (False, batch_desc, None, None, None)

    monkeypatch.setattr(runner, "_determine_batch_execution_and_padding", fake_determine.__get__(runner, type(runner)))

    # Initialize per-request embeds (batch-major inside talker_mtp_inputs_embeds)
    runner.talker_mtp_inputs_embeds.gpu[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    runner.talker_mtp_inputs_embeds.gpu[1] = torch.tensor([10.0, 20.0, 30.0, 40.0])

    # Flattened `inputs_embeds`: offsets 0 and 3 will be overwritten
    inputs_embeds = torch.zeros((6, 4), dtype=torch.float32)

    # Call the original implementation from OmniGPUModelRunner (no re-implementation)
    OmniGPUModelRunner._talker_mtp_forward(runner, ["r1", "r2"], inputs_embeds)

    # Validate embeds were written back (+1)
    assert torch.allclose(inputs_embeds[0], torch.tensor([2.0, 3.0, 4.0, 5.0]))
    assert torch.allclose(inputs_embeds[3], torch.tensor([11.0, 21.0, 31.0, 41.0]))

    # Validate per-request additional_information_cpu was updated
    info_r1 = runner.requests["r1"].additional_information_cpu
    info_r2 = runner.requests["r2"].additional_information_cpu
    assert int(info_r1["code_predictor_codes"][0, 0]) == 0
    assert int(info_r2["code_predictor_codes"][0, 0]) == 1


def test_talker_mtp_forward_cpu_empty_batch_noop(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    inputs_embeds = torch.randn((2, 4))
    before = inputs_embeds.clone()

    OmniGPUModelRunner._talker_mtp_forward(runner, [], inputs_embeds)

    # Ensure no changes were made
    assert torch.allclose(inputs_embeds, before)


def test_update_intermediate_buffer_writes_to_buffer_and_setattr(monkeypatch):
    """Validate that _update_intermediate_buffer writes to model_intermediate_buffer
    (forward path) and mirrors to additional_information_cpu setattr (backward compat)."""
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    update = {"my_tensor": torch.tensor([1.0, 2.0]), "my_list": [3, 4]}
    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", update)

    # Forward: buffer is populated
    assert "r1" in runner.model_intermediate_buffer
    buf = runner.model_intermediate_buffer["r1"]
    assert torch.allclose(buf["my_tensor"], torch.tensor([1.0, 2.0]))
    assert buf["my_list"] == [3, 4]

    # Backward compat: setattr is also populated
    info_cpu = runner.requests["r1"].additional_information_cpu
    assert torch.allclose(info_cpu["my_tensor"], torch.tensor([1.0, 2.0]))
    assert info_cpu["my_list"] == [3, 4]


def test_update_intermediate_buffer_accumulates():
    """Validate that successive merges accumulate keys in the buffer."""
    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", {"a": torch.tensor([1.0])})
    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", {"b": torch.tensor([2.0])})

    buf = runner.model_intermediate_buffer["r1"]
    assert "a" in buf and "b" in buf
    assert torch.allclose(buf["a"], torch.tensor([1.0]))
    assert torch.allclose(buf["b"], torch.tensor([2.0]))


def test_update_intermediate_buffer_skips_empty_update():
    """Validate that an empty update dict is a no-op."""
    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", {})

    assert "r1" not in runner.model_intermediate_buffer


def test_update_intermediate_buffer_skips_unknown_req_id():
    """Validate that merge is a no-op when req_id is not in self.requests."""
    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    OmniGPUModelRunner._update_intermediate_buffer(runner, "unknown_req", {"key": torch.tensor([1.0])})

    assert "unknown_req" not in runner.model_intermediate_buffer


def test_maybe_attach_mimo_audio_req_infos_enriches_dict():
    runner = _make_runner_for_mimo()
    req_id = "r_mimo"
    req_state = runner.requests[req_id]

    # Existing req_infos should be copied and enriched, not mutated in place.
    original_req_infos = {"existing": 1}
    enriched = OmniGPUModelRunner._maybe_attach_mimo_audio_req_infos(runner, req_state, original_req_infos, req_id)

    assert enriched is not original_req_infos
    assert enriched["existing"] == 1
    # mm_features should be filled from req_state when missing
    assert enriched["mm_features"] == req_state.mm_features
    # req_id should always be attached
    assert enriched["req_id"] == req_id


def test_maybe_attach_mimo_audio_req_infos_no_req_state_returns_input():
    runner = _make_runner_for_mimo()
    req_id = "missing"
    req_state = None
    req_infos = {"k": "v"}

    result = OmniGPUModelRunner._maybe_attach_mimo_audio_req_infos(runner, req_state, req_infos, req_id)

    # When no req_state, helper should be a no-op.
    assert result is req_infos


def test_sample_tokens_applies_postprocessed_tokens_before_bookkeeping():
    runner = _make_sample_tokens_runner(ReplaceSampledTokensModel())
    captured = {}

    def fake_bookkeeping(
        self,
        scheduler_output,
        sampler_output,
        logits,
        hidden_states,
        num_scheduled_tokens,
        spec_decode_metadata,
    ):
        captured["sampled_token_ids"] = sampler_output.sampled_token_ids.clone()
        raise StopAfterBookkeepingError

    runner._bookkeeping_sync = fake_bookkeeping.__get__(runner, type(runner))

    with pytest.raises(StopAfterBookkeepingError):
        GPUARModelRunner.sample_tokens(runner, grammar_output=None)

    assert torch.equal(runner.model.observed_sampled_token_ids, torch.tensor([[1], [2]], dtype=torch.int32))
    assert torch.equal(captured["sampled_token_ids"], torch.tensor([[11], [12]], dtype=torch.int32))


def test_sample_tokens_passes_pending_updates_to_postprocess_without_committing_before_bookkeeping():
    runner = _make_sample_tokens_runner(OverlaySampledTokensModel())

    def fake_collect(*args, **kwargs):
        del args, kwargs
        return {"r1": {"pending": 11}, "r2": {"pending": 22}}

    captured = {}

    def fake_bookkeeping(
        self,
        scheduler_output,
        sampler_output,
        logits,
        hidden_states,
        num_scheduled_tokens,
        spec_decode_metadata,
    ):
        del scheduler_output, sampler_output, logits, hidden_states, num_scheduled_tokens, spec_decode_metadata
        captured["buffer_before_bookkeeping"] = {
            req_id: dict(info) for req_id, info in self.model_intermediate_buffer.items()
        }
        raise StopAfterBookkeepingError

    runner._collect_additional_information_updates = fake_collect
    runner._bookkeeping_sync = fake_bookkeeping.__get__(runner, type(runner))

    with pytest.raises(StopAfterBookkeepingError):
        GPUARModelRunner.sample_tokens(runner, grammar_output=None)

    assert runner.model.observed_buffer == {
        "r1": {"token": 1, "pending": 11},
        "r2": {"token": 2, "pending": 22},
    }
    assert captured["buffer_before_bookkeeping"] == {"r1": {"token": 1}, "r2": {"token": 2}}


def test_preprocess_passes_none_input_embeds_for_raw_token_models(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))

    runner = _make_preprocess_runner(RawTokenPreprocessModel(hidden_size=4), hidden_size=4)
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"r1": 2},
        scheduled_encoder_inputs=None,
    )

    input_ids, inputs_embeds, *_ = OmniGPUModelRunner._preprocess(
        runner,
        scheduler_output,
        num_input_tokens=2,
    )

    assert runner.model.observed_input_embeds == [None]
    assert torch.equal(input_ids, torch.tensor([101, 102], dtype=torch.int32))
    assert inputs_embeds.data_ptr() == runner.inputs_embeds.gpu[:2].data_ptr()
    assert torch.equal(
        inputs_embeds,
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
        ),
    )
    assert runner.model_intermediate_buffer["r1"]["marker_seen"] == "r1"


def test_preprocess_uses_buffered_input_ids_when_multimodal_path_returns_none(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))

    runner = _make_mm_preprocess_runner(MultimodalPreprocessModel(hidden_size=4), hidden_size=4)
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"r1": 2},
        scheduled_encoder_inputs=None,
    )

    input_ids, inputs_embeds, *_ = OmniGPUModelRunner._preprocess(
        runner,
        scheduler_output,
        num_input_tokens=2,
    )

    assert input_ids is None
    assert len(runner.model.observed_input_ids) == 1
    assert torch.equal(runner.model.observed_input_ids[0], torch.tensor([1, 2], dtype=torch.int32))
    assert torch.equal(
        runner.model.observed_input_embeds[0],
        runner.inputs_embeds.gpu[:2],
    )
    assert torch.equal(
        inputs_embeds,
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
        ),
    )
    assert runner.model_intermediate_buffer["r1"]["marker_seen"] == "r1"
