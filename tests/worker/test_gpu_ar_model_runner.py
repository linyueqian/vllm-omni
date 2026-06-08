from types import SimpleNamespace

import pytest

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_runner(engine_output_type: str | None, downstream_req_ids: set[str]) -> GPUARModelRunner:
    runner = object.__new__(GPUARModelRunner)
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(engine_output_type=engine_output_type),
    )
    runner._request_needs_downstream_stage_payload = lambda rid: rid in downstream_req_ids
    return runner


def test_resolve_pooler_payload_req_ids_audio_terminal_stage_keeps_payload():
    runner = _make_runner(engine_output_type="audio", downstream_req_ids=set())

    engine_output_type, payload_req_ids = GPUARModelRunner._resolve_pooler_payload_req_ids(runner, ["r1", "r2"])

    assert engine_output_type == "audio"
    assert payload_req_ids == ["r1", "r2"]


def test_resolve_pooler_payload_req_ids_text_terminal_stage_drops_payload():
    runner = _make_runner(engine_output_type="text", downstream_req_ids=set())

    engine_output_type, payload_req_ids = GPUARModelRunner._resolve_pooler_payload_req_ids(runner, ["r1", "r2"])

    assert engine_output_type == "text"
    assert payload_req_ids == []


def test_resolve_pooler_payload_req_ids_downstream_stage_uses_filtered_requests():
    runner = _make_runner(engine_output_type="latent", downstream_req_ids={"r2"})

    engine_output_type, payload_req_ids = GPUARModelRunner._resolve_pooler_payload_req_ids(runner, ["r1", "r2", "r3"])

    assert engine_output_type == "latent"
    assert payload_req_ids == ["r2"]


def test_duplex_forward_with_runner_context_exposes_scheduler_kv_contract():
    runner = object.__new__(GPUARModelRunner)

    hook = runner.duplex_forward_with_runner_context

    assert getattr(hook, "uses_scheduler_metadata") is True
    assert getattr(hook, "uses_runner_kv_cache") is True
    assert getattr(hook, "vllm_omni_runner_context_contract") is True
    assert getattr(runner, "supports_native_duplex_runner_context") is True


def test_duplex_forward_with_runner_context_delegates_to_runner_impl():
    runner = object.__new__(GPUARModelRunner)
    calls = []

    def impl(**kwargs):
        calls.append(kwargs)
        return {
            "logits": "logits",
            "hidden_states": "hidden",
            "sampled_token_id": 7,
            "kv_cache_length": kwargs["context_len"],
        }

    runner._duplex_forward_with_runner_context_impl = impl

    result = runner.duplex_forward_with_runner_context(
        session_id="sid",
        inputs_embeds="embeds",
        context_len=3,
        previous_context_len=2,
        reset_kv=False,
    )

    assert calls == [
        {
            "session_id": "sid",
            "inputs_embeds": "embeds",
            "context_len": 3,
            "previous_context_len": 2,
            "reset_kv": False,
        }
    ]
    assert result == {
        "logits": "logits",
        "hidden_states": "hidden",
        "sampled_token_id": 7,
        "kv_cache_length": 3,
        "uses_model_runner_scheduler": True,
        "runner_kv_backed": True,
    }


def test_model_sampler_identifies_duplex_rows_without_mutating_metadata():
    runner = object.__new__(GPUARModelRunner)
    runner.input_batch = SimpleNamespace(
        req_ids=["duplex-sid-e0-stage0-s1", "plain-request"],
        req_output_token_ids=[[], []],
        sampled_token_ids_cpu=None,
        prev_req_id_to_index=None,
    )
    runner.model_intermediate_buffer = {
        "duplex-sid-e0-stage0-s1": {"duplex": {"data_plane": True}},
        "plain-request": {},
    }
    runner.requests = {}
    metadata = SimpleNamespace(output_token_ids=[[], []])

    result = runner._sampling_metadata_for_model_sampler(metadata)

    assert result is metadata
    assert not hasattr(result, "_vllm_omni_duplex_rows")
    assert runner._model_sampler_duplex_rows() == [0]


def test_model_sampler_does_not_treat_request_id_prefix_as_duplex_row():
    runner = object.__new__(GPUARModelRunner)
    runner.input_batch = SimpleNamespace(
        req_ids=["duplex-looking-plain-request"],
    )
    runner.model_intermediate_buffer = {
        "duplex-looking-plain-request": {},
    }
    runner.requests = {}

    assert runner._model_sampler_duplex_rows() == []


def test_call_model_sampler_passes_duplex_rows_when_supported():
    calls = []

    def model_sample(logits, sampling_metadata, *, duplex_rows=None):
        calls.append((logits, sampling_metadata, duplex_rows))
        return "sampled"

    assert (
        GPUARModelRunner._call_model_sampler(
            model_sample,
            "logits",
            "metadata",
            duplex_rows=[0],
        )
        == "sampled"
    )
    assert calls == [("logits", "metadata", [0])]


def test_call_model_sampler_falls_back_for_legacy_signature():
    calls = []

    def model_sample(logits, sampling_metadata):
        calls.append((logits, sampling_metadata))
        return "sampled"

    assert (
        GPUARModelRunner._call_model_sampler(
            model_sample,
            "logits",
            "metadata",
            duplex_rows=[0],
        )
        == "sampled"
    )
    assert calls == [("logits", "metadata")]
