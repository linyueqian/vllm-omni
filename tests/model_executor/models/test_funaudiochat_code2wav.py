from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.funaudiochat.funaudiochat_code2wav import (
    FunAudioChatCosyVoice3Code2Wav,
)


def test_split_tokens_like_official_keeps_short_inputs_as_single_segment():
    token = torch.arange(100, dtype=torch.long)

    segments = FunAudioChatCosyVoice3Code2Wav._split_tokens_like_official(token)

    assert len(segments) == 1
    assert torch.equal(segments[0], token)


def test_split_tokens_like_official_rebalances_tiny_tail_segment():
    token = torch.arange(760, dtype=torch.long)

    segments = FunAudioChatCosyVoice3Code2Wav._split_tokens_like_official(token)

    assert [segment.numel() for segment in segments] == [380, 380]
    assert torch.equal(torch.cat(segments, dim=0), token)


def _build_code2wav_stub() -> FunAudioChatCosyVoice3Code2Wav:
    model = object.__new__(FunAudioChatCosyVoice3Code2Wav)
    model.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cpu")))
    model._max_codec_token_id = 6560
    model._dummy_profile_token_len = 32
    model._logged_dummy_profile_cap = False
    return model


def test_build_decode_tokens_keeps_real_input_ids_without_sampling_metadata():
    model = _build_code2wav_stub()
    input_ids = torch.tensor([12, 34, 56], dtype=torch.long)

    token_batches, is_dummy_profile = model._build_decode_tokens(input_ids, sampling_metadata=None)

    assert len(token_batches) == 1
    assert token_batches[0].tolist() == [[12, 34, 56]]
    assert is_dummy_profile is False


def test_build_decode_tokens_uses_prompt_token_ids_when_input_ids_are_empty():
    model = _build_code2wav_stub()
    sampling_metadata = SimpleNamespace(prompt_token_ids=[1, 2, 3, 4])

    token_batches, is_dummy_profile = model._build_decode_tokens(
        torch.empty((0,), dtype=torch.long),
        sampling_metadata,
    )

    assert len(token_batches) == 1
    assert token_batches[0].tolist() == [[1, 2, 3, 4]]
    assert is_dummy_profile is False


def test_build_decode_tokens_treats_all_zero_missing_metadata_as_dummy_profile():
    model = _build_code2wav_stub()
    input_ids = torch.zeros((64,), dtype=torch.long)

    token_batches, is_dummy_profile = model._build_decode_tokens(input_ids, sampling_metadata=None)

    assert len(token_batches) == 1
    assert token_batches[0].shape == (1, 32)
    assert is_dummy_profile is True


def test_build_decode_tokens_no_longer_rejects_long_sequences_before_segmentation():
    model = _build_code2wav_stub()
    input_ids = torch.arange(10235, dtype=torch.long) % 6000

    token_batches, is_dummy_profile = model._build_decode_tokens(input_ids, sampling_metadata=None)

    assert len(token_batches) == 1
    assert token_batches[0].shape == (1, 10235)
    assert is_dummy_profile is False


def test_build_decode_tokens_preserves_batched_prompt_token_ids_per_request():
    model = _build_code2wav_stub()
    sampling_metadata = SimpleNamespace(prompt_token_ids=[[1, 2, 3], [4, 5]])

    token_batches, is_dummy_profile = model._build_decode_tokens(
        torch.empty((0,), dtype=torch.long),
        sampling_metadata,
    )

    assert [token.tolist() for token in token_batches] == [[[1, 2, 3]], [[4, 5]]]
    assert is_dummy_profile is False
