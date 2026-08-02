# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.personaplex import (
    personaplex_code2wav,
    personaplex_mimi,
)
from vllm_omni.model_executor.models.personaplex.personaplex_code2wav import (
    PersonaPlexCode2Wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeStreamingMimi(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decode_frame_calls = 0
        self.reset_calls = 0

    def decode(self, codes: torch.Tensor, return_dict: bool = True):
        assert return_dict
        frames = int(codes.shape[-1])
        return SimpleNamespace(audio_values=torch.arange(frames * 4, dtype=torch.float32).reshape(1, 1, -1))

    def streaming_init(self, batch_size: int) -> None:
        assert batch_size == 1

    def decode_frame(self, codes: torch.Tensor) -> torch.Tensor:
        assert codes.shape == (1, 2)
        self.decode_frame_calls += 1
        return torch.full((1, 4), float(self.decode_frame_calls), dtype=torch.float32)

    def reset_streaming(self) -> None:
        self.reset_calls += 1


def test_moshi_mimi_checkpoint_maps_local_codec_weights() -> None:
    mapper = getattr(personaplex_mimi, "_map_moshi_codec_weights", None)
    assert callable(mapper), "PersonaPlex Mimi must load its bundled codec weights without Hugging Face"

    source = {
        "encoder.model.3.conv.conv.weight": torch.ones(1),
        "decoder.model.2.convtr.convtr.bias": torch.ones(1),
        "downsample.conv.conv.conv.weight": torch.ones(1),
        "upsample.convtr.convtr.convtr.weight": torch.ones(1),
        "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum": torch.ones(1),
        "quantizer.rvq_rest.vq.layers.3._codebook._initialized": torch.ones(1),
        "encoder_transformer.transformer.layers.0.norm1.weight": torch.ones(1),
    }

    mapped = mapper(source)

    assert set(mapped) == {
        "encoder.layers.3.conv.weight",
        "decoder.layers.2.conv.bias",
        "downsample.conv.weight",
        "upsample.conv.weight",
        "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum",
        "quantizer.acoustic_residual_vector_quantizer.layers.3.codebook.initialized",
    }
    assert mapped["encoder.layers.3.conv.weight"] is source["encoder.model.3.conv.conv.weight"]


def _model() -> tuple[PersonaPlexCode2Wav, _FakeStreamingMimi]:
    mimi_config = SimpleNamespace(
        num_codebooks=2,
        sample_rate=24000,
        samples_per_frame=4,
        mimi_name=None,
    )
    config = SimpleNamespace(mimi_config=mimi_config, mimi_name=None)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model="/unused", hf_config=config),
        device_config=SimpleNamespace(device="cpu"),
    )
    model = PersonaPlexCode2Wav(vllm_config=vllm_config)
    mimi = _FakeStreamingMimi()
    model.mimi = mimi
    model._mimi_device = torch.device("cpu")
    return model, mimi


def _codes(frames: int, *, start: int = 0) -> torch.Tensor:
    return torch.stack(
        [
            torch.arange(start, start + frames, dtype=torch.long),
            torch.arange(start + 100, start + 100 + frames, dtype=torch.long),
        ]
    ).reshape(-1)


def _audio(output) -> torch.Tensor:
    return output.multimodal_outputs["model_outputs"][0]


def test_resumable_cumulative_codes_emit_only_new_pcm() -> None:
    model, mimi = _model()

    first = model(input_ids=_codes(2), request_ids=["req"])
    second = model(input_ids=_codes(3), request_ids=["req"])

    assert _audio(first).numel() == 8
    assert _audio(second).numel() == 4
    assert mimi.decode_frame_calls == 3


def test_resumable_delta_codes_append_to_decoder_state() -> None:
    model, mimi = _model()

    first = model(input_ids=_codes(2), request_ids=["req"])
    second = model(input_ids=_codes(1, start=100), request_ids=["req"])

    assert _audio(first).numel() == 8
    assert _audio(second).numel() == 4
    assert mimi.decode_frame_calls == 3


def test_connector_codec_payload_replaces_scheduler_placeholder() -> None:
    model, mimi = _model()
    connector_codes = _codes(2)

    output = model(
        input_ids=torch.tensor([0]),
        request_ids=["req"],
        runtime_additional_information=[{"codes": {"audio": connector_codes}}],
    )

    assert _audio(output).numel() == 8
    assert mimi.decode_frame_calls == 2


def test_dummy_profile_input_skips_codec_decode_without_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    model, mimi = _model()
    runtime_info = [{"meta": {"personaplex_dummy_profile": True}}]
    warnings: list[tuple] = []
    monkeypatch.setattr(personaplex_code2wav.logger, "warning", lambda *args: warnings.append(args))

    output = model(
        input_ids=torch.arange(3),
        runtime_additional_information=runtime_info,
    )

    assert _audio(output).numel() == 0
    assert mimi.decode_frame_calls == 0
    assert warnings == []


def test_malformed_online_input_still_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    model, mimi = _model()
    warnings: list[tuple] = []
    monkeypatch.setattr(personaplex_code2wav.logger, "warning", lambda *args: warnings.append(args))

    output = model(input_ids=torch.arange(3))

    assert _audio(output).numel() == 0
    assert mimi.decode_frame_calls == 0
    assert len(warnings) == 1
    assert "not divisible by" in warnings[0][0]


def test_dummy_profile_runtime_information_marks_each_request() -> None:
    model, _ = _model()

    assert model.get_dummy_runtime_additional_information(2) == [
        {"meta": {"personaplex_dummy_profile": True}},
        {"meta": {"personaplex_dummy_profile": True}},
    ]


def test_request_id_falls_back_to_runtime_information() -> None:
    model, _ = _model()
    info = [{"request_id": "runtime-req"}]

    model(input_ids=_codes(2), runtime_additional_information=info)
    second = model(input_ids=_codes(3), runtime_additional_information=info)

    assert _audio(second).numel() == 4


def test_finished_request_resets_streaming_decoder() -> None:
    model, mimi = _model()
    model(input_ids=_codes(2), request_ids=["req"])

    model.on_requests_finished(["req"])

    assert mimi.reset_calls == 1
    replacement = model(input_ids=_codes(1), request_ids=["replacement"])
    assert _audio(replacement).numel() == 4
