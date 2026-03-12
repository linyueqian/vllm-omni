from types import SimpleNamespace

import torch
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)
from transformers.modeling_outputs import BaseModelOutput

import vllm_omni.model_executor.models.funaudiochat.funaudiochat as fac_mod
from vllm_omni.model_executor.models.funaudiochat.funaudiochat import (
    DEFAULT_SP_GEN_KWARGS,
    FunAudioChatForConditionalGeneration,
)


def _make_model_stub(
    *,
    audio_bos_id: int = 42,
    audio_eos_id: int = 99,
    group_size: int = 5,
    hidden_size: int = 4,
):
    model = object.__new__(FunAudioChatForConditionalGeneration)
    model.config = SimpleNamespace(
        audio_config=SimpleNamespace(group_size=group_size, eos_token_id=audio_eos_id),
        text_config=SimpleNamespace(audio_bos_index=audio_bos_id, audio_eos_index=audio_eos_id),
    )
    model.sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
    model._batch_preprocess_in_progress = False
    model._batch_req_infos = []
    model._batch_sidecar_results = []
    model._postprocess_cursor = 0
    model._logged_stage0_backend = True
    model.get_language_model = lambda: SimpleNamespace(
        embed_input_ids=lambda input_ids: torch.zeros(
            (input_ids.reshape(-1).numel(), hidden_size),
            dtype=torch.float32,
            device=input_ids.device,
        )
    )
    model.audio_tower = lambda audio_ids: BaseModelOutput(
        last_hidden_state=torch.full(
            (audio_ids.shape[0], 1, hidden_size),
            2.0,
            dtype=torch.float32,
            device=audio_ids.device,
        )
    )
    model._get_stage0_backend = lambda: "TEST"
    return model


def test_default_sp_gen_kwargs_match_official_defaults():
    assert DEFAULT_SP_GEN_KWARGS == {
        "text_greedy": True,
        "only_crq_sampling": True,
        "disable_speech": False,
        "force_text_abos": True,
    }


def test_pooler_output_buffer_only_snapshots_incremental_audio_groups():
    assert FunAudioChatForConditionalGeneration.pooler_output_buffer_keys == ("audio_token_ids",)


def test_build_crq_sampling_config_matches_official_sampling_defaults():
    model = _make_model_stub()
    sampling_metadata = type(
        "SamplingMetadataStub",
        (),
        {
            "repetition_penalties": torch.tensor([1.2]),
            "temperature": torch.tensor([0.8]),
            "top_p": torch.tensor([0.9]),
            "top_k": torch.tensor([0]),
        },
    )()

    processors, do_sample = model._build_crq_sampling_config(
        sampling_metadata=sampling_metadata,
        req_index=0,
    )

    assert do_sample is True
    assert any(isinstance(p, RepetitionPenaltyLogitsProcessor) for p in processors)
    assert any(isinstance(p, TemperatureLogitsWarper) for p in processors)
    assert any(isinstance(p, TopPLogitsWarper) for p in processors)


def test_build_crq_sampling_config_is_empty_for_greedy_without_penalties():
    model = _make_model_stub()
    model.sp_gen_kwargs["text_greedy"] = False
    sampling_metadata = type(
        "SamplingMetadataStub",
        (),
        {
            "repetition_penalties": torch.tensor([1.0]),
            "temperature": None,
            "top_p": None,
            "top_k": None,
        },
    )()

    processors, do_sample = model._build_crq_sampling_config(
        sampling_metadata=sampling_metadata,
        req_index=0,
    )

    assert do_sample is False
    assert len(processors) == 0


def test_build_crq_sampling_config_restores_official_audio_sampling_when_text_path_is_greedy():
    model = _make_model_stub()
    model.sp_gen_kwargs["text_greedy"] = True
    sampling_metadata = type(
        "SamplingMetadataStub",
        (),
        {
            "repetition_penalties": torch.tensor([1.2]),
            "temperature": torch.tensor([0.0]),
            "top_p": torch.tensor([1.0]),
            "top_k": torch.tensor([-1]),
        },
    )()

    processors, do_sample = model._build_crq_sampling_config(
        sampling_metadata=sampling_metadata,
        req_index=0,
    )

    assert do_sample is True
    assert len(processors) == 3
    assert any(isinstance(p, RepetitionPenaltyLogitsProcessor) for p in processors)
    assert any(isinstance(p, TemperatureLogitsWarper) for p in processors)
    assert any(isinstance(p, TopPLogitsWarper) for p in processors)


def test_resolve_text_seq_len_prefill_accumulates_prompt_tokens():
    assert FunAudioChatForConditionalGeneration._resolve_text_seq_len(None, 5) == (5, 5)
    assert FunAudioChatForConditionalGeneration._resolve_text_seq_len(5, 3) == (8, 8)


def test_resolve_text_seq_len_decode_advances_for_next_step():
    assert FunAudioChatForConditionalGeneration._resolve_text_seq_len(8, 1) == (8, 9)
    assert FunAudioChatForConditionalGeneration._resolve_text_seq_len(None, 1) == (1, 2)


def test_resolve_next_speech_state_stays_text_only_until_audio_bos_is_sampled():
    final_token, next_speech_active, next_force_pending = (
        FunAudioChatForConditionalGeneration._resolve_next_speech_state(
            sampled_token_id=7,
            generate_speech=False,
            finish_speech=False,
            force_audio_bos_pending=False,
            audio_bos_id=42,
            audio_eos_id=99,
        )
    )

    assert final_token == 7
    assert next_speech_active is False
    assert next_force_pending is False


def test_resolve_next_speech_state_arms_speech_after_audio_bos_is_sampled():
    final_token, next_speech_active, next_force_pending = (
        FunAudioChatForConditionalGeneration._resolve_next_speech_state(
            sampled_token_id=42,
            generate_speech=False,
            finish_speech=False,
            force_audio_bos_pending=False,
            audio_bos_id=42,
            audio_eos_id=99,
        )
    )

    assert final_token == 42
    assert next_speech_active is True
    assert next_force_pending is False


def test_resolve_next_speech_state_force_text_abos_overrides_sampled_token():
    final_token, next_speech_active, next_force_pending = (
        FunAudioChatForConditionalGeneration._resolve_next_speech_state(
            sampled_token_id=7,
            generate_speech=False,
            finish_speech=False,
            force_audio_bos_pending=True,
            audio_bos_id=42,
            audio_eos_id=99,
        )
    )

    assert final_token == 42
    assert next_speech_active is True
    assert next_force_pending is False


def test_resolve_next_speech_state_finish_speech_overrides_final_token_to_audio_eos():
    final_token, next_speech_active, next_force_pending = (
        FunAudioChatForConditionalGeneration._resolve_next_speech_state(
            sampled_token_id=7,
            generate_speech=True,
            finish_speech=True,
            force_audio_bos_pending=False,
            audio_bos_id=42,
            audio_eos_id=99,
        )
    )

    assert final_token == 99
    assert next_speech_active is False
    assert next_force_pending is False


def test_postprocess_sampled_tokens_updates_buffer_from_final_sampled_token():
    model = _make_model_stub()
    sampled_token_ids = torch.tensor([42], dtype=torch.long)
    model_intermediate_buffer = {
        "req0": {
            fac_mod._GENERATE_SPEECH_KEY: False,
            fac_mod._FORCE_AUDIO_BOS_KEY: False,
            fac_mod._FINISH_SPEECH_KEY: False,
        }
    }

    updated = model.postprocess_sampled_tokens(
        sampled_token_ids=sampled_token_ids,
        req_ids=["req0"],
        req_id_to_index={"req0": 0},
        model_intermediate_buffer=model_intermediate_buffer,
    )

    assert updated.tolist() == [42]
    assert model_intermediate_buffer["req0"][fac_mod._GENERATE_SPEECH_KEY] is True
    assert model_intermediate_buffer["req0"][fac_mod._FORCE_AUDIO_BOS_KEY] is False
    assert fac_mod._FINISH_SPEECH_KEY not in model_intermediate_buffer["req0"]


def test_postprocess_sampled_tokens_force_text_abos_overrides_sampled_token():
    model = _make_model_stub()
    sampled_token_ids = torch.tensor([7], dtype=torch.long)
    model_intermediate_buffer = {
        "req0": {
            fac_mod._GENERATE_SPEECH_KEY: False,
            fac_mod._FORCE_AUDIO_BOS_KEY: True,
            fac_mod._FINISH_SPEECH_KEY: False,
        }
    }

    updated = model.postprocess_sampled_tokens(
        sampled_token_ids=sampled_token_ids,
        req_ids=["req0"],
        req_id_to_index={"req0": 0},
        model_intermediate_buffer=model_intermediate_buffer,
    )

    assert updated.tolist() == [42]
    assert model_intermediate_buffer["req0"][fac_mod._GENERATE_SPEECH_KEY] is True
    assert model_intermediate_buffer["req0"][fac_mod._FORCE_AUDIO_BOS_KEY] is False


def test_postprocess_sampled_tokens_overwrites_emitted_token_to_audio_eos_on_finish():
    model = _make_model_stub()
    sampled_token_ids = torch.tensor([7], dtype=torch.long)
    model_intermediate_buffer = {
        "req0": {
            fac_mod._GENERATE_SPEECH_KEY: True,
            fac_mod._FORCE_AUDIO_BOS_KEY: False,
            fac_mod._FINISH_SPEECH_KEY: True,
        }
    }

    updated = model.postprocess_sampled_tokens(
        sampled_token_ids=sampled_token_ids,
        req_ids=["req0"],
        req_id_to_index={"req0": 0},
        model_intermediate_buffer=model_intermediate_buffer,
    )

    assert updated.tolist() == [99]
    assert model_intermediate_buffer["req0"][fac_mod._GENERATE_SPEECH_KEY] is False
    assert model_intermediate_buffer["req0"][fac_mod._FORCE_AUDIO_BOS_KEY] is False
    assert fac_mod._FINISH_SPEECH_KEY not in model_intermediate_buffer["req0"]


def test_postprocess_sampled_tokens_noops_for_spec_decode_shapes():
    model = _make_model_stub()
    sampled_token_ids = torch.tensor([[7, 8]], dtype=torch.long)
    model_intermediate_buffer = {
        "req0": {
            fac_mod._GENERATE_SPEECH_KEY: False,
            fac_mod._FORCE_AUDIO_BOS_KEY: True,
            fac_mod._FINISH_SPEECH_KEY: False,
        }
    }

    updated = model.postprocess_sampled_tokens(
        sampled_token_ids=sampled_token_ids,
        req_ids=["req0"],
        req_id_to_index={"req0": 0},
        model_intermediate_buffer=model_intermediate_buffer,
    )

    assert torch.equal(updated, sampled_token_ids)
    assert model_intermediate_buffer["req0"][fac_mod._FORCE_AUDIO_BOS_KEY] is True


def test_chunked_prefill_preprocess_keeps_speech_inactive():
    model = _make_model_stub()

    _, _, first_update = model.preprocess(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        input_embeds=None,
    )
    _, _, second_update = model.preprocess(
        input_ids=torch.tensor([4, 5], dtype=torch.long),
        input_embeds=None,
        **first_update,
    )

    assert first_update[fac_mod._GENERATE_SPEECH_KEY] is False
    assert first_update[fac_mod._FORCE_AUDIO_BOS_KEY] is True
    assert second_update[fac_mod._GENERATE_SPEECH_KEY] is False
    assert second_update[fac_mod._FORCE_AUDIO_BOS_KEY] is True
    assert torch.equal(first_update["audio_token_ids"], torch.full((1, 5), -1, dtype=torch.long))
    assert torch.equal(second_update["audio_token_ids"], torch.full((1, 5), -1, dtype=torch.long))


def test_preprocess_single_token_text_decode_returns_text_embeddings():
    model = _make_model_stub()

    _, req_embeds, _ = model.preprocess(
        input_ids=torch.tensor([7], dtype=torch.long),
        input_embeds=None,
    )

    assert torch.equal(req_embeds, torch.zeros((1, 4), dtype=torch.float32))


def test_preprocess_first_speech_step_without_codec_history_returns_text_embeddings():
    model = _make_model_stub()

    _, req_embeds, _ = model.preprocess(
        input_ids=torch.tensor([42], dtype=torch.long),
        input_embeds=None,
        **{
            fac_mod._GENERATE_SPEECH_KEY: True,
            fac_mod._SPEECH_IDS_KEY: torch.empty((1, 0), dtype=torch.long),
        },
    )

    assert torch.equal(req_embeds, torch.zeros((1, 4), dtype=torch.float32))


def test_preprocess_active_speech_with_codec_history_blends_audio_features():
    model = _make_model_stub(hidden_size=4)

    _, req_embeds, _ = model.preprocess(
        input_ids=torch.tensor([42], dtype=torch.long),
        input_embeds=None,
        **{
            fac_mod._GENERATE_SPEECH_KEY: True,
            fac_mod._SPEECH_IDS_KEY: torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
        },
    )

    assert torch.equal(req_embeds, torch.full((1, 4), 1.0))


def test_run_audio_sidecar_decode_warmup_updates_cache_only():
    model = _make_model_stub(hidden_size=4)

    class AudioInvertTowerStub:
        def __init__(self):
            self.crq_audio_embeds = None
            self.crq_past_key_values = None
            self.crq_do_sample = None
            self.crq_logits_processor = None
            self.crq_speech_ids = None

        def crq_generate_forward(self, *, inputs_embeds, return_dict=True):
            del return_dict
            self.last_inputs_embeds = inputs_embeds
            self.crq_audio_embeds = torch.full((1, 4), 5.0, dtype=torch.float32, device=inputs_embeds.device)
            self.crq_past_key_values = (torch.full((1, 1), 7.0, dtype=torch.float32, device=inputs_embeds.device),)

    model.audio_invert_tower = AudioInvertTowerStub()

    warmup_state = model._run_audio_sidecar_decode_warmup(
        hidden_state=torch.zeros(4, dtype=torch.float32),
        current_input_token_id=7,
        speech_ids=torch.empty((1, 0), dtype=torch.long),
        cached_audio_embeds=None,
        cached_past_key_values=None,
        logits_processor=[],
        do_sample=True,
    )

    assert list(model.audio_invert_tower.last_inputs_embeds.shape) == [1, 1, 4]
    assert torch.equal(warmup_state[fac_mod._CRQ_AUDIO_EMBEDS_KEY], torch.full((1, 4), 5.0))
    assert torch.equal(warmup_state[fac_mod._CRQ_PAST_KEY_VALUES_KEY][0], torch.full((1, 1), 7.0))


def test_postprocess_prefill_warmup_updates_cache_without_emitting_audio():
    model = _make_model_stub(hidden_size=4)
    model._batch_sidecar_results = [
        {
            fac_mod._AUDIO_TOKEN_IDS_KEY: torch.full((1, 5), -1, dtype=torch.long),
            fac_mod._CRQ_AUDIO_EMBEDS_KEY: None,
            fac_mod._CRQ_PAST_KEY_VALUES_KEY: None,
            fac_mod._FORCE_AUDIO_BOS_KEY: True,
            fac_mod._FINISH_SPEECH_KEY: False,
            fac_mod._GENERATE_SPEECH_KEY: False,
            fac_mod._SPEECH_IDS_KEY: torch.empty((1, 0), dtype=torch.long),
            "_run_prefill_crq_warmup": True,
            "_prefill_input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "_prefill_crq_logits_processor": [],
            "_prefill_crq_do_sample": False,
            "audio_token_ids": torch.full((1, 5), -1, dtype=torch.long),
        }
    ]
    model._postprocess_cursor = 0

    def _prefill_warmup(**kwargs):
        del kwargs
        return {
            fac_mod._CRQ_AUDIO_EMBEDS_KEY: torch.full((1, 4), 9.0),
            fac_mod._CRQ_PAST_KEY_VALUES_KEY: (torch.full((1, 1), 3.0),),
        }

    model._run_audio_sidecar_prefill_warmup = _prefill_warmup

    output = model.postprocess(torch.zeros((3, 4), dtype=torch.float32))

    assert torch.equal(output["audio_token_ids"], torch.full((1, 5), -1, dtype=torch.long))
    assert torch.equal(output[fac_mod._CRQ_AUDIO_EMBEDS_KEY], torch.full((1, 4), 9.0))
    assert torch.equal(output[fac_mod._CRQ_PAST_KEY_VALUES_KEY][0], torch.full((1, 1), 3.0))
