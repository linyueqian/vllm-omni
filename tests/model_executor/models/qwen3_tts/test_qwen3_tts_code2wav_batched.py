# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract test: batched Code2Wav forward must match sequential forward."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import Qwen3TTSCode2Wav

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _BatchAwareFakeDecoder(nn.Module):
    """Fake decoder that respects the batch dimension deterministically."""

    def __init__(self, total_upsample: int = 4):
        super().__init__()
        self.total_upsample = total_upsample

    def chunked_decode(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: [B, Q, F]; produce [B, 1, wav_len] deterministically per sample.
        assert codes.dim() == 3, f"expected [B, Q, F], got {tuple(codes.shape)}"
        bs, _, frames = codes.shape
        wav_len = frames * self.total_upsample + 6
        wavs = []
        for b in range(bs):
            # Make each per-sample output depend on the sample content so any
            # cross-sample leakage shows up as a mismatch.
            seed = int(codes[b].sum().item()) & 0xFFFF
            wav = torch.arange(wav_len, dtype=torch.float32) + float(seed)
            wavs.append(wav.view(1, 1, -1))
        return torch.cat(wavs, dim=0)


def _make_model(num_quantizers: int = 2, total_upsample: int = 4) -> Qwen3TTSCode2Wav:
    model = Qwen3TTSCode2Wav(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model="unused"),
            device_config=SimpleNamespace(device=torch.device("cpu")),
        )
    )
    model._decoder = _BatchAwareFakeDecoder(total_upsample=total_upsample)
    model._num_quantizers = num_quantizers
    model._output_sample_rate = 24000
    model._total_upsample = total_upsample
    model._ensure_speech_tokenizer_loaded = lambda: None
    return model


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_batched_forward_matches_sequential(batch_size: int):
    q, frames = 2, 25
    torch.manual_seed(42)
    model = _make_model(num_quantizers=q)
    per_req_ids = [torch.randint(0, 1024, (q * frames,), dtype=torch.long) for _ in range(batch_size)]
    concat_ids = torch.cat(per_req_ids) if per_req_ids else torch.empty(0, dtype=torch.long)
    counts = [q * frames] * batch_size

    batched_out = model.forward(
        input_ids=concat_ids,
        seq_token_counts=counts,
        runtime_additional_information=[{"left_context_size": 0}] * batch_size,
    )
    batched_audios = batched_out.multimodal_outputs["model_outputs"]

    seq_audios = []
    for pids in per_req_ids:
        seq_out = model.forward(
            input_ids=pids,
            seq_token_counts=[q * frames],
            runtime_additional_information=[{"left_context_size": 0}],
        )
        seq_audios.append(seq_out.multimodal_outputs["model_outputs"][0])

    assert len(batched_audios) == batch_size
    for i in range(batch_size):
        torch.testing.assert_close(batched_audios[i], seq_audios[i], atol=1e-5, rtol=1e-5)


def test_batched_forward_handles_heterogeneous_lengths():
    """Heterogeneous frame counts must fall back to per-request decode, not crash."""
    q = 2
    model = _make_model(num_quantizers=q)
    a = torch.randint(0, 1024, (q * 25,), dtype=torch.long)
    b = torch.randint(0, 1024, (q * 40,), dtype=torch.long)

    out = model.forward(
        input_ids=torch.cat([a, b]),
        seq_token_counts=[q * 25, q * 40],
        runtime_additional_information=[{"left_context_size": 0}, {"left_context_size": 0}],
    )
    audios = out.multimodal_outputs["model_outputs"]
    assert len(audios) == 2
    assert audios[0].numel() > 0
    assert audios[1].numel() > audios[0].numel()
