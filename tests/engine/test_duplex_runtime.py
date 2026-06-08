import pytest

from vllm_omni.engine.duplex import (
    DuplexAdapterPattern,
    DuplexInputMode,
    DuplexRuntimeCapabilities,
    DuplexSessionRuntimeManager,
    SessionMode,
    duplex_data_plane_request_info,
    duplex_scheduler_token_budget,
)


def test_duplex_runtime_tracks_stage_bindings_and_barge_in_epoch():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        "sid-1",
        session_mode=SessionMode.DUPLEX,
        capabilities=DuplexRuntimeCapabilities(
            adapter_patterns={DuplexAdapterPattern.CHUNK_GROUP_APPEND},
            input_modes={DuplexInputMode.APPEND_TOKENS},
            supports_kv_lease=True,
            supports_stage_resumption=True,
        ),
    )
    session.bind_stage_request(stage_id=0, request_id="req-stage0", replica_id=1)
    session.bind_stage_request(stage_id=1, request_id="req-stage1", replica_id=0)

    update = session.append_input({"tokens": [1, 2, 3]}, mode=DuplexInputMode.APPEND_TOKENS)
    new_epoch, stale_request_ids = session.barge_in()

    assert update.seq == 1
    assert new_epoch == 1
    assert stale_request_ids == ["req-stage0", "req-stage1"]
    assert session.stage_bindings == {}
    assert session.pending_inputs == []


def test_duplex_runtime_core_kv_lease_is_not_model_internal_state():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        "sid-model-internal",
        session_mode=SessionMode.DUPLEX,
        capabilities=DuplexRuntimeCapabilities(
            supports_kv_lease=True,
            supports_core_kv_lease=False,
            supports_model_internal_state=True,
        ),
    )

    session.bind_stage_request(stage_id=0, request_id="req-stage0")

    assert session.stage_bindings[0].lease_active is False


def test_duplex_runtime_core_kv_lease_marks_stage_binding_active():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        "sid-core-lease",
        session_mode=SessionMode.DUPLEX,
        capabilities=DuplexRuntimeCapabilities(
            supports_kv_lease=True,
            supports_core_kv_lease=True,
        ),
    )

    session.bind_stage_request(stage_id=0, request_id="req-stage0")

    assert session.stage_bindings[0].lease_active is True


def test_duplex_runtime_pending_inputs_store_metadata_not_raw_payload():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        "sid-audio",
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )
    audio_payload = {
        "type": "audio",
        "audio": "A" * 4096,
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
    }

    update = session.append_input(audio_payload, mode=DuplexInputMode.APPEND_AUDIO_CHUNK)

    assert not hasattr(update, "payload")
    assert update.payload_meta == {
        "type": "dict",
        "keys": ["audio", "format", "sample_rate_hz", "type"],
        "audio_bytes": 3072,
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
    }
    assert session.pending_inputs == [update]


def test_duplex_runtime_rejects_unsupported_append_mode():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        "sid-2",
        capabilities=DuplexRuntimeCapabilities(input_modes={DuplexInputMode.TURN_COMMIT_ONLY}),
    )

    with pytest.raises(ValueError, match="not supported"):
        session.append_input({"tokens": [1]}, mode=DuplexInputMode.APPEND_TOKENS)


def test_duplex_runtime_serializes_capability_patterns():
    caps = DuplexRuntimeCapabilities(
        adapter_patterns={
            DuplexAdapterPattern.CHUNK_GROUP_APPEND,
            DuplexAdapterPattern.EXPERIMENTAL_WORKER_CONTROL_RPC,
            DuplexAdapterPattern.PER_STEP_TENSOR_HANDOFF,
        },
        input_modes={
            DuplexInputMode.APPEND_TOKENS,
            DuplexInputMode.ROLLBACK_TO_CHECKPOINT,
        },
        supports_audio_truncate=True,
        chunk_period_ms=1000,
        target_barge_in_latency_ms=300,
    )

    data = caps.as_dict()

    assert data["adapter_patterns"] == [
        "chunk_group_append",
        "experimental_worker_control_rpc",
        "per_step_tensor_handoff",
    ]
    assert data["input_modes"] == ["append_tokens", "rollback_to_checkpoint"]
    assert data["supports_audio_truncate"] is True
    assert data["chunk_period_ms"] == 1000
    assert data["target_barge_in_latency_ms"] == 300


def test_duplex_data_plane_request_info_extracts_structured_stage_result():
    request_id, response_stage_id = duplex_data_plane_request_info(
        {
            "stage_results": [
                {"result": {"supported": True}},
                {
                    "result": {
                        "data_plane_append": True,
                        "request_id": "duplex-sid-e0-stage0-s1",
                        "response_stage_id": 1,
                    }
                },
            ]
        }
    )

    assert request_id == "duplex-sid-e0-stage0-s1"
    assert response_stage_id == 1


def test_duplex_data_plane_request_info_rejects_missing_request_id():
    assert duplex_data_plane_request_info(
        {
            "stage_results": [
                {
                    "result": {
                        "data_plane_append": True,
                        "request_id": "",
                        "response_stage_id": 1,
                    }
                }
            ]
        }
    ) == (None, None)


def test_duplex_scheduler_token_budget_estimates_pcm_slots():
    assert (
        duplex_scheduler_token_budget(
            {
                "audio": "AAAAAA==",
                "format": "pcm_f32le",
                "sample_rate_hz": 16000,
            }
        )
        == 64
    )


def test_duplex_scheduler_token_budget_ignores_client_budget_fields():
    assert (
        duplex_scheduler_token_budget(
            {
                "audio": "AAAAAA==",
                "format": "pcm_f32le",
                "duplex_num_input_tokens": 999,
                "num_input_tokens": 999,
            }
        )
        == 64
    )
