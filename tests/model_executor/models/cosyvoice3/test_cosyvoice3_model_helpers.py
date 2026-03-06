# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 as cosy_mod
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyCode2Wav:
    def __init__(self, vocab_size: int, num_samples: int = 32):
        self.input_embedding = SimpleNamespace(num_embeddings=vocab_size)
        self.num_samples = num_samples

    def __call__(self, **kwargs):
        return torch.linspace(-1.0, 1.0, self.num_samples, dtype=torch.float32)


def _make_code2wav_model(*, with_stride_cfg: bool = False, num_samples: int = 32) -> CosyVoice3Model:
    model = object.__new__(CosyVoice3Model)
    nn.Module.__init__(model)
    model.model_stage = "code2wav"
    hift_cfg = {} if not with_stride_cfg else {"upsample_rates": [8, 5, 3], "istft_params": {"hop_len": 4}}
    model.config = SimpleNamespace(
        sample_rate=24000,
        hift=hift_cfg,
        token_frame_rate=25 if with_stride_cfg else 0,
        token_mel_ratio=2 if with_stride_cfg else 0,
    )
    model.code2wav = _DummyCode2Wav(vocab_size=4, num_samples=num_samples)
    return model


def test_split_request_ids_uses_seq_token_counts():
    ids = torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)
    chunks = CosyVoice3Model._split_request_ids(ids, [2, 2, 2])
    assert [c.tolist() for c in chunks] == [[10, 11], [12, 13], [14]]


def test_split_request_ids_honors_single_request_seq_token_counts():
    ids = torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)
    chunks = CosyVoice3Model._split_request_ids(ids, [3])
    assert [c.tolist() for c in chunks] == [[10, 11, 12]]


def test_sanitize_codec_tokens_filters_out_of_range():
    model = _make_code2wav_model()
    raw = torch.tensor([-1, 0, 3, 4, 99], dtype=torch.long)
    clean = model._sanitize_codec_tokens(raw)
    assert clean.tolist() == [0, 3]


def test_forward_warns_when_left_context_trim_is_unavailable(monkeypatch):
    model = _make_code2wav_model()
    warning_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(cosy_mod.logger, "warning_once", lambda *a, **kw: warning_calls.append((a, kw)))

    runtime_info = [
        {
            "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
            "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            "left_context_size": 2,
        }
    ]

    out = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    assert len(out.multimodal_outputs["audio"]) == 1
    assert out.multimodal_outputs["audio"][0].numel() > 0
    assert any("cannot trim left context" in str(args[0]) for args, _ in warning_calls)


def test_forward_caps_chunk_audio_length_to_token_span():
    model = _make_code2wav_model(with_stride_cfg=True, num_samples=10000)
    runtime_info = [
        {
            "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
            "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            "left_context_size": 2,
        }
    ]

    out = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    # With stride cfg: samples_per_token = 8*5*3*4*2 = 960.
    # total tokens=3, left_context=2 -> expected non-overlap span = 1 token.
    assert out.multimodal_outputs["audio"][0].numel() == 960


def test_forward_ignores_single_request_padded_tail_tokens():
    model = _make_code2wav_model(with_stride_cfg=True, num_samples=10000)
    runtime_info = [
        {
            "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
            "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            "left_context_size": 0,
        }
    ]

    out = model.forward(
        input_ids=torch.tensor([0, 1, 2, 3, 3], dtype=torch.long),
        positions=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    # The padded tail must not contribute to code2wav length.
    assert out.multimodal_outputs["audio"][0].numel() == 2880
