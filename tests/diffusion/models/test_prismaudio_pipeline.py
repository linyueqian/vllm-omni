# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_request(
    additional_information: dict[str, torch.Tensor] | None = None,
    extra_args: dict[str, object] | None = None,
) -> OmniDiffusionRequest:
    features = additional_information or {
        "video_features": torch.randn(1, 10, 1024),
        "text_features": torch.randn(1, 32, 1024),
        "sync_features": torch.randn(1, 16, 768),
    }
    return OmniDiffusionRequest(
        prompts=[
            {
                "prompt": "semantic and temporal cot text",
                "additional_information": features,
            }
        ],
        sampling_params=OmniDiffusionSamplingParams(
            extra_args=extra_args or {},
        ),
        request_ids=["req-1"],
    )


def test_prismaudio_pipeline_is_registered():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    assert model_cls is not None
    assert model_cls.__name__ == "PrismAudioPipeline"


def test_prismaudio_pipeline_lazy_import():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    assert model_cls is not None
    assert model_cls.__module__ == "vllm_omni.diffusion.models.prismaudio.pipeline_prismaudio"


def test_prismaudio_pipeline_can_be_constructed_with_fake_components():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    pipeline = model_cls(transformer=nn.Identity(), vae=nn.Identity())

    assert pipeline is not None
    assert pipeline.transformer is not None
    assert pipeline.vae is not None


def test_prismaudio_pipeline_requires_video_features():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(transformer=nn.Identity(), vae=nn.Identity())
    req = _make_request(
        additional_information={
            "text_features": torch.randn(1, 32, 1024),
            "sync_features": torch.randn(1, 16, 768),
        }
    )

    with pytest.raises(ValueError, match="video_features"):
        pipeline.forward(req)


def test_prismaudio_pipeline_requires_text_features():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(transformer=nn.Identity(), vae=nn.Identity())
    req = _make_request(
        additional_information={
            "video_features": torch.randn(1, 10, 1024),
            "sync_features": torch.randn(1, 16, 768),
        }
    )

    with pytest.raises(ValueError, match="text_features"):
        pipeline.forward(req)


def test_prismaudio_pipeline_requires_sync_features():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(transformer=nn.Identity(), vae=nn.Identity())
    req = _make_request(
        additional_information={
            "video_features": torch.randn(1, 10, 1024),
            "text_features": torch.randn(1, 32, 1024),
        }
    )

    with pytest.raises(ValueError, match="sync_features"):
        pipeline.forward(req)


def test_prismaudio_pipeline_reads_sampling_extra_args():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(transformer=nn.Identity(), vae=nn.Identity())
    req = _make_request(
        extra_args={
            "num_inference_steps": 24,
            "cfg_scale": 5.0,
        }
    )

    sampling_args = pipeline._parse_sampling_args(req)

    assert sampling_args == {
        "num_inference_steps": 24,
        "cfg_scale": 5.0,
    }


def test_prismaudio_checkpoint_loader_supports_raw_state_dict(tmp_path):
    from vllm_omni.diffusion.models.prismaudio.pipeline_prismaudio import load_prismaudio_state_dict

    ckpt_path = tmp_path / "prismaudio.ckpt"
    expected = {"weight": torch.randn(2, 2)}
    torch.save(expected, ckpt_path)

    actual = load_prismaudio_state_dict(ckpt_path)

    assert list(actual) == ["weight"]
    assert torch.equal(actual["weight"], expected["weight"])


def test_prismaudio_checkpoint_loader_supports_nested_state_dict(tmp_path):
    from vllm_omni.diffusion.models.prismaudio.pipeline_prismaudio import load_prismaudio_state_dict

    ckpt_path = tmp_path / "vae.ckpt"
    expected = {"autoencoder.weight": torch.randn(2, 2)}
    torch.save({"state_dict": expected}, ckpt_path)

    actual = load_prismaudio_state_dict(ckpt_path)

    assert list(actual) == ["autoencoder.weight"]
    assert torch.equal(actual["autoencoder.weight"], expected["autoencoder.weight"])


def test_prismaudio_checkpoint_loader_reports_key_mismatches(tmp_path):
    from vllm_omni.diffusion.models.prismaudio.pipeline_prismaudio import load_module_from_prismaudio_checkpoint

    ckpt_path = tmp_path / "prismaudio.ckpt"
    module = nn.Linear(2, 2, bias=False)
    expected_weight = torch.randn_like(module.weight)
    torch.save(
        {
            "weight": expected_weight,
            "unexpected": torch.randn(1),
        },
        ckpt_path,
    )

    report = load_module_from_prismaudio_checkpoint(module, ckpt_path)

    assert torch.equal(module.weight, expected_weight)
    assert report.missing_keys == []
    assert report.unexpected_keys == ["unexpected"]


def test_prismaudio_checkpoint_loader_strips_autoencoder_prefix(tmp_path):
    from vllm_omni.diffusion.models.prismaudio.pipeline_prismaudio import load_module_from_prismaudio_checkpoint

    ckpt_path = tmp_path / "vae.ckpt"
    module = nn.Linear(2, 2, bias=False)
    expected_weight = torch.randn_like(module.weight)
    torch.save(
        {
            "state_dict": {
                "autoencoder.weight": expected_weight,
            }
        },
        ckpt_path,
    )

    report = load_module_from_prismaudio_checkpoint(module, ckpt_path, prefix="autoencoder.")

    assert torch.equal(module.weight, expected_weight)
    assert report.missing_keys == []
    assert report.unexpected_keys == []


def test_prismaudio_sampling_uses_requested_euler_steps():
    from vllm_omni.diffusion.models.prismaudio.pipeline_prismaudio import sample_discrete_euler

    class _RecordingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls: list[dict[str, object]] = []

        def forward(self, x, t, **kwargs):
            self.calls.append(
                {
                    "shape": tuple(x.shape),
                    "t": t.detach().cpu().tolist(),
                    "cfg_scale": kwargs.get("cfg_scale"),
                    "batch_cfg": kwargs.get("batch_cfg"),
                }
            )
            return t.view(-1, 1, 1).expand_as(x)

    model = _RecordingModel()
    noise = torch.zeros(2, 3, 4)

    output = sample_discrete_euler(
        model,
        noise,
        steps=4,
        cfg_scale=5.0,
        batch_cfg=True,
    )

    assert output.shape == noise.shape
    assert torch.allclose(output, torch.full_like(noise, -0.625))
    assert len(model.calls) == 4
    assert [call["t"] for call in model.calls] == [
        [1.0, 1.0],
        [0.75, 0.75],
        [0.5, 0.5],
        [0.25, 0.25],
    ]
    assert all(call["cfg_scale"] == 5.0 for call in model.calls)
    assert all(call["batch_cfg"] is True for call in model.calls)
