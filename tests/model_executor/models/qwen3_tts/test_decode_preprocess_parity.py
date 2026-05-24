"""Parity test for the scalar / batched decode-preprocess paths.

The talker exposes a batched ``preprocess_decode_batch`` plus a scalar
fast-path that loops to the existing single-request ``preprocess()`` when
the decode batch is small or has no ``task_type=Base`` requests. This test
asserts the two paths produce identical outputs so the fast-path is a true
byte-equivalent shortcut, not an approximation.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    _DEFAULT_SCALAR_DECODE_PREPROCESS_THRESHOLD,
    Qwen3TTSTalkerForConditionalGeneration,
)


def _make_minimal_talker(*, threshold: int | None = None, compact_min: int = 256):
    model = Qwen3TTSTalkerForConditionalGeneration.__new__(Qwen3TTSTalkerForConditionalGeneration)
    model.talker_config = SimpleNamespace(codec_pad_id=7, num_code_groups=16)
    model._scalar_decode_preprocess_threshold = (
        threshold if threshold is not None else _DEFAULT_SCALAR_DECODE_PREPROCESS_THRESHOLD
    )
    model._trailing_text_compact_min_frames = compact_min

    def fake_embed_input_ids(input_ids):
        return input_ids.to(torch.float32).reshape(-1, 1, 1).expand(-1, 1, 4)

    model.embed_input_ids = fake_embed_input_ids
    return model


def _build_req_info(*, task_type: str, text_offset: int, seed: int):
    """Build one request payload with a predictable trailing-text tensor."""
    trailing = torch.arange(seed, seed + 12, dtype=torch.float32).reshape(3, 4)
    last_hidden = torch.full((4,), float(seed % 7), dtype=torch.float32)
    tts_pad = torch.full((1, 4), float(-seed), dtype=torch.float32)
    return {
        "text": ["hello"],
        "task_type": [task_type],
        "hidden_states": {"trailing_text": trailing, "last": last_hidden},
        "embed": {"tts_pad": tts_pad},
        "meta": {"talker_text_offset": text_offset},
    }


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("task_type", ["Base", "CustomVoice"])
def test_scalar_and_batched_paths_agree(batch_size: int, task_type: str) -> None:
    """Same inputs → identical (out_ids, out_embeds, past_hidden, text_step, updates)."""
    req_infos = [_build_req_info(task_type=task_type, text_offset=i % 3, seed=10 + i) for i in range(batch_size)]
    input_ids = torch.arange(100, 100 + batch_size, dtype=torch.long)

    scalar_model = _make_minimal_talker(threshold=batch_size + 1)
    batched_model = _make_minimal_talker(threshold=0)

    scalar_out = scalar_model.preprocess_decode_batch(
        input_ids=input_ids,
        req_infos=[dict(info) for info in req_infos],
    )
    batched_out = batched_model.preprocess_decode_batch(
        input_ids=input_ids,
        req_infos=[dict(info) for info in req_infos],
    )

    s_ids, s_embeds, s_past, s_step, s_updates = scalar_out
    b_ids, b_embeds, b_past, b_step, b_updates = batched_out

    assert s_ids.tolist() == b_ids.tolist()
    assert torch.equal(s_embeds, b_embeds)
    assert torch.equal(s_past, b_past)
    assert torch.equal(s_step, b_step)
    assert len(s_updates) == len(b_updates)
    for s_u, b_u in zip(s_updates, b_updates):
        assert s_u["meta"]["talker_text_offset"] == b_u["meta"]["talker_text_offset"]
        assert s_u["meta"]["codec_streaming"] == b_u["meta"]["codec_streaming"]
        s_has_hs = "hidden_states" in s_u
        b_has_hs = "hidden_states" in b_u
        assert s_has_hs == b_has_hs
        if s_has_hs:
            assert torch.equal(
                s_u["hidden_states"]["trailing_text"],
                b_u["hidden_states"]["trailing_text"],
            )


def test_routing_uses_scalar_for_small_batch() -> None:
    model = _make_minimal_talker(threshold=4)
    req_infos = [_build_req_info(task_type="Base", text_offset=0, seed=1) for _ in range(4)]
    assert model._should_use_scalar_decode_preprocess(req_infos) is True


def test_routing_uses_batched_for_large_base_batch() -> None:
    model = _make_minimal_talker(threshold=4)
    req_infos = [_build_req_info(task_type="Base", text_offset=0, seed=1) for _ in range(8)]
    assert model._should_use_scalar_decode_preprocess(req_infos) is False


def test_routing_uses_scalar_when_no_base_request() -> None:
    model = _make_minimal_talker(threshold=4)
    req_infos = [_build_req_info(task_type="CustomVoice", text_offset=0, seed=i) for i in range(8)]
    assert model._should_use_scalar_decode_preprocess(req_infos) is True


def test_routing_threshold_zero_means_size_check_disabled() -> None:
    model = _make_minimal_talker(threshold=0)
    base_batch = [_build_req_info(task_type="Base", text_offset=0, seed=i) for i in range(2)]
    custom_batch = [_build_req_info(task_type="CustomVoice", text_offset=0, seed=i) for i in range(2)]
    assert model._should_use_scalar_decode_preprocess(base_batch) is False
    assert model._should_use_scalar_decode_preprocess(custom_batch) is True
