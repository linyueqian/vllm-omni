# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
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


def test_prismaudio_pipeline_load_weights_uses_standard_loader_contract():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    transformer = nn.Linear(2, 2, bias=False)
    vae = nn.Linear(2, 2, bias=False)
    pipeline = model_cls(transformer=transformer, vae=vae)

    expected_transformer_weight = torch.randn_like(transformer.weight)
    expected_vae_weight = torch.randn_like(vae.weight)

    loaded = pipeline.load_weights(
        [
            ("transformer.weight", expected_transformer_weight),
            ("vae.weight", expected_vae_weight),
        ]
    )

    assert torch.equal(transformer.weight, expected_transformer_weight)
    assert torch.equal(vae.weight, expected_vae_weight)
    assert loaded == {"transformer.weight", "vae.weight"}


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


def test_prismaudio_pipeline_prefers_request_guidance_scale():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(transformer=nn.Identity(), vae=nn.Identity())
    req = _make_request()
    req.sampling_params.guidance_scale = 7.25
    req.sampling_params.guidance_scale_provided = True

    sampling_args = pipeline._parse_sampling_args(req)

    assert sampling_args == {
        "num_inference_steps": req.sampling_params.num_inference_steps or 24,
        "cfg_scale": 7.25,
    }


def test_prismaudio_checkpoint_loader_supports_raw_state_dict(tmp_path):
    from vllm_omni.diffusion.models.prismaudio.pipeline_prismaudio import load_prismaudio_state_dict

    ckpt_path = tmp_path / "prismaudio.ckpt"
    expected = {"weight": torch.randn(2, 2)}
    torch.save(expected, ckpt_path)

    actual = load_prismaudio_state_dict(ckpt_path)

    assert list(actual) == ["weight"]
    assert torch.equal(actual["weight"], expected["weight"])


@pytest.mark.parametrize("prefix", ["module.", "model."])
def test_prismaudio_checkpoint_loader_strips_common_wrapper_prefixes(prefix, tmp_path):
    from vllm_omni.diffusion.models.prismaudio.pipeline_prismaudio import load_module_from_prismaudio_checkpoint

    ckpt_path = tmp_path / "wrapped.ckpt"
    module = nn.Linear(2, 2, bias=False)
    expected_weight = torch.randn_like(module.weight)
    torch.save({f"{prefix}weight": expected_weight}, ckpt_path)

    report = load_module_from_prismaudio_checkpoint(module, ckpt_path)

    assert torch.equal(module.weight, expected_weight)
    assert report.missing_keys == []
    assert report.unexpected_keys == []


def test_prismaudio_checkpoint_loader_uses_weights_only_for_torch_load(monkeypatch, tmp_path):
    from vllm_omni.diffusion.models.prismaudio.pipeline_prismaudio import load_prismaudio_state_dict

    ckpt_path = tmp_path / "prismaudio.ckpt"
    ckpt_path.write_bytes(b"placeholder")
    captured: dict[str, object] = {}

    def _fake_torch_load(path, *args, **kwargs):
        captured["path"] = path
        captured["kwargs"] = dict(kwargs)
        return {"weight": torch.randn(2, 2)}

    monkeypatch.setattr(torch, "load", _fake_torch_load)

    actual = load_prismaudio_state_dict(ckpt_path)

    assert list(actual) == ["weight"]
    assert captured["path"] == str(ckpt_path)
    assert captured["kwargs"] == {
        "map_location": "cpu",
        "weights_only": True,
    }


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


def test_prismaudio_pipeline_returns_audio_output():
    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls: list[dict[str, object]] = []

        def forward(self, x, t, **kwargs):
            self.calls.append(
                {
                    "video_features_shape": tuple(kwargs["video_features"].shape),
                    "text_features_shape": tuple(kwargs["text_features"].shape),
                    "sync_features_shape": tuple(kwargs["sync_features"].shape),
                    "cfg_scale": kwargs["cfg_scale"],
                    "batch_cfg": kwargs["batch_cfg"],
                    "t": t.detach().cpu().tolist(),
                }
            )
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :] * 3.0)

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    transformer = _FakeTransformer()
    pipeline = model_cls(transformer=transformer, vae=_FakeVAE())
    initial_latents = torch.zeros(1, 64, 4)
    req = _make_request(
        extra_args={
            "num_inference_steps": 4,
            "cfg_scale": 5.0,
        }
    )
    req.sampling_params.latents = initial_latents

    output = pipeline.forward(req)

    assert output.output is not None
    assert output.error is None
    assert output.output.shape == (1, 2, 4)
    assert torch.allclose(output.output, torch.full((1, 2, 4), -3.0))
    assert len(transformer.calls) == 4
    assert transformer.calls[0]["video_features_shape"] == (1, 10, 1024)
    assert transformer.calls[0]["text_features_shape"] == (1, 32, 1024)
    assert transformer.calls[0]["sync_features_shape"] == (1, 16, 768)
    assert transformer.calls[0]["cfg_scale"] == 5.0
    assert transformer.calls[0]["batch_cfg"] is True


def test_prismaudio_pipeline_supports_official_wrapper_conditioning_path():
    class _InnerTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls: list[dict[str, object]] = []

        def forward(self, x, t, **kwargs):
            self.calls.append(
                {
                    "cross_attn_cond_shape": tuple(kwargs["cross_attn_cond"].shape),
                    "cross_attn_mask_shape": tuple(kwargs["cross_attn_mask"].shape),
                    "cfg_scale": kwargs["cfg_scale"],
                    "batch_cfg": kwargs["batch_cfg"],
                }
            )
            return torch.ones_like(x)

    class _OfficialStyleWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _InnerTransformer()
            self.conditioner_calls: list[dict[str, object]] = []
            self.get_conditioning_inputs_calls: list[dict[str, object]] = []

        def conditioner(self, metadata, device):
            self.conditioner_calls.append(
                {
                    "metadata_len": len(metadata),
                    "device": str(device),
                    "keys": sorted(metadata[0].keys()),
                }
            )
            return {
                "video_features": (
                    torch.ones(1, 10, 1024, device=device),
                    torch.ones(1, 10, device=device, dtype=torch.bool),
                ),
                "text_features": (
                    torch.ones(1, 32, 1024, device=device),
                    torch.ones(1, 32, device=device, dtype=torch.bool),
                ),
            }

        def get_conditioning_inputs(self, conditioning_tensors):
            self.get_conditioning_inputs_calls.append({"keys": sorted(conditioning_tensors.keys())})
            return {
                "cross_attn_cond": torch.cat(
                    [
                        conditioning_tensors["video_features"][0],
                        conditioning_tensors["text_features"][0],
                    ],
                    dim=1,
                ),
                "cross_attn_mask": torch.cat(
                    [
                        conditioning_tensors["video_features"][1],
                        conditioning_tensors["text_features"][1],
                    ],
                    dim=1,
                ),
            }

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :] * 2.0)

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    transformer = _OfficialStyleWrapper()
    pipeline = model_cls(transformer=transformer, vae=_FakeVAE())
    req = _make_request(extra_args={"num_inference_steps": 2, "cfg_scale": 4.0})
    req.sampling_params.latents = torch.zeros(1, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (1, 2, 4)
    assert torch.allclose(output.output, torch.full((1, 2, 4), -2.0))
    assert transformer.conditioner_calls == [
        {
            "metadata_len": 1,
            "device": str(req.sampling_params.latents.device),
            "keys": ["sync_features", "text_features", "video_features"],
        }
    ]
    assert transformer.get_conditioning_inputs_calls == [{"keys": ["text_features", "video_features"]}]
    assert len(transformer.model.calls) == 2
    assert transformer.model.calls[0]["cross_attn_cond_shape"] == (1, 42, 1024)
    assert transformer.model.calls[0]["cross_attn_mask_shape"] == (1, 42)
    assert transformer.model.calls[0]["cfg_scale"] == 4.0
    assert transformer.model.calls[0]["batch_cfg"] is True


def test_prismaudio_pipeline_rejects_multiple_prompts():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(transformer=nn.Identity(), vae=nn.Identity())
    req = OmniDiffusionRequest(
        prompts=[
            {
                "prompt": "first",
                "additional_information": {
                    "video_features": torch.randn(1, 10, 1024),
                    "text_features": torch.randn(1, 32, 1024),
                    "sync_features": torch.randn(1, 16, 768),
                },
            },
            {
                "prompt": "second",
                "additional_information": {
                    "video_features": torch.randn(1, 10, 1024),
                    "text_features": torch.randn(1, 32, 1024),
                    "sync_features": torch.randn(1, 16, 768),
                },
            },
        ],
        sampling_params=OmniDiffusionSamplingParams(extra_args={}),
        request_ids=["req-1", "req-2"],
    )

    with pytest.raises(ValueError, match="exactly one prompt"):
        pipeline.forward(req)


def test_prismaudio_pipeline_builds_runtime_config_and_components_from_od_config(tmp_path):
    model_config_path = tmp_path / "prismaudio.json"
    model_config_path.write_text(
        """
{
  "sample_rate": 44100,
  "audio_channels": 2,
  "model": {
    "io_channels": 64,
    "pretransform": {
      "config": {
        "latent_dim": 64
      }
    }
  }
}
        """.strip()
    )

    class _FactoryTransformer(nn.Module):
        pass

    class _FactoryVAE(nn.Module):
        pass

    od_config = OmniDiffusionConfig(
        model=str(model_config_path),
        model_class_name="PrismAudioPipeline",
        model_paths={
            "transformer": "/tmp/prismaudio.ckpt",
            "vae": "/tmp/vae.ckpt",
        },
        model_config={
            "transformer_factory": lambda runtime_config: _FactoryTransformer(),
            "vae_factory": lambda runtime_config: _FactoryVAE(),
        },
    )

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(od_config=od_config)

    assert pipeline.runtime_config.sample_rate == 44100
    assert pipeline.runtime_config.audio_channels == 2
    assert pipeline.runtime_config.latent_channels == 64
    assert pipeline.runtime_config.transformer_checkpoint_path == "/tmp/prismaudio.ckpt"
    assert pipeline.runtime_config.vae_checkpoint_path == "/tmp/vae.ckpt"
    assert isinstance(pipeline.transformer, _FactoryTransformer)
    assert isinstance(pipeline.vae, _FactoryVAE)


def test_prismaudio_pipeline_builds_official_wrapper_from_model_factory_import_path():
    import sys
    import types

    class _InnerTransformer(nn.Module):
        def forward(self, x, t, **kwargs):
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _OfficialStylePretransform(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :] * 2.0)

    class _OfficialStyleWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _InnerTransformer()
            self.pretransform = _OfficialStylePretransform()

        def conditioner(self, metadata, device):
            return {
                "video_features": (
                    torch.ones(1, 10, 1024, device=device),
                    torch.ones(1, 10, device=device, dtype=torch.bool),
                ),
                "text_features": (
                    torch.ones(1, 32, 1024, device=device),
                    torch.ones(1, 32, device=device, dtype=torch.bool),
                ),
            }

        def get_conditioning_inputs(self, conditioning_tensors):
            return {
                "cross_attn_cond": torch.cat(
                    [
                        conditioning_tensors["video_features"][0],
                        conditioning_tensors["text_features"][0],
                    ],
                    dim=1,
                ),
                "cross_attn_mask": torch.cat(
                    [
                        conditioning_tensors["video_features"][1],
                        conditioning_tensors["text_features"][1],
                    ],
                    dim=1,
                ),
            }

    module_name = "prismaudio_test_builders"
    builder_module = types.ModuleType(module_name)

    def _build_wrapper(runtime_config):
        assert runtime_config.sample_rate == 44100
        return _OfficialStyleWrapper()

    builder_module.build_wrapper = _build_wrapper
    sys.modules[module_name] = builder_module

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": {
                "sample_rate": 44100,
                "audio_channels": 2,
                "model": {
                    "io_channels": 64,
                },
            },
            "model_factory": f"{module_name}.build_wrapper",
        },
    )

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(od_config=od_config)

    assert isinstance(pipeline.transformer, _OfficialStyleWrapper)
    assert pipeline.vae is pipeline.transformer.pretransform

    req = _make_request(extra_args={"num_inference_steps": 2, "cfg_scale": 4.0})
    req.sampling_params.latents = torch.zeros(1, 64, 4)
    output = pipeline.forward(req)

    assert output.output.shape == (1, 2, 4)
    assert torch.allclose(output.output, torch.full((1, 2, 4), -2.0))


def test_prismaudio_pipeline_builds_official_wrapper_from_raw_model_config_factory_spec():
    import sys
    import types

    class _InnerTransformer(nn.Module):
        def forward(self, x, t, **kwargs):
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _OfficialStylePretransform(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :] * 2.0)

    class _OfficialStyleWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _InnerTransformer()
            self.pretransform = _OfficialStylePretransform()

        def conditioner(self, metadata, device):
            return {
                "video_features": (
                    torch.ones(1, 10, 1024, device=device),
                    torch.ones(1, 10, device=device, dtype=torch.bool),
                ),
                "text_features": (
                    torch.ones(1, 32, 1024, device=device),
                    torch.ones(1, 32, device=device, dtype=torch.bool),
                ),
            }

        def get_conditioning_inputs(self, conditioning_tensors):
            return {
                "cross_attn_cond": torch.cat(
                    [
                        conditioning_tensors["video_features"][0],
                        conditioning_tensors["text_features"][0],
                    ],
                    dim=1,
                ),
                "cross_attn_mask": torch.cat(
                    [
                        conditioning_tensors["video_features"][1],
                        conditioning_tensors["text_features"][1],
                    ],
                    dim=1,
                ),
            }

    module_name = "prismaudio_test_raw_builders"
    builder_module = types.ModuleType(module_name)

    def _build_wrapper_from_raw_model_config(raw_model_config):
        assert raw_model_config["model_type"] == "diffusion_cond"
        assert raw_model_config["sample_rate"] == 44100
        assert raw_model_config["model"]["io_channels"] == 64
        return _OfficialStyleWrapper()

    builder_module.build_wrapper_from_raw_model_config = _build_wrapper_from_raw_model_config
    sys.modules[module_name] = builder_module

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": {
                "model_type": "diffusion_cond",
                "sample_rate": 44100,
                "audio_channels": 2,
                "model": {
                    "io_channels": 64,
                },
            },
            "model_factory": {
                "path": f"{module_name}.build_wrapper_from_raw_model_config",
                "input": "raw_model_config",
            },
        },
    )

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(od_config=od_config)

    assert isinstance(pipeline.transformer, _OfficialStyleWrapper)
    assert pipeline.vae is pipeline.transformer.pretransform

    req = _make_request(extra_args={"num_inference_steps": 2, "cfg_scale": 4.0})
    req.sampling_params.latents = torch.zeros(1, 64, 4)
    output = pipeline.forward(req)

    assert output.output.shape == (1, 2, 4)
    assert torch.allclose(output.output, torch.full((1, 2, 4), -2.0))


def test_prismaudio_pipeline_load_weights_reads_runtime_checkpoint_paths(tmp_path):
    transformer_ckpt_path = tmp_path / "prismaudio.ckpt"
    vae_ckpt_path = tmp_path / "vae.ckpt"

    inline_model_config = {
        "sample_rate": 44100,
        "audio_channels": 2,
        "model": {
            "io_channels": 64,
            "pretransform": {
                "config": {
                    "latent_dim": 64,
                }
            },
        },
    }

    transformer_weight = torch.randn(2, 2)
    vae_weight = torch.randn(2, 2)
    torch.save({"weight": transformer_weight}, transformer_ckpt_path)
    torch.save({"state_dict": {"autoencoder.weight": vae_weight}}, vae_ckpt_path)

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_paths={
            "transformer": str(transformer_ckpt_path),
            "vae": str(vae_ckpt_path),
        },
        model_config={
            "prismaudio_model_config": inline_model_config,
            "transformer_factory": lambda runtime_config: nn.Linear(2, 2, bias=False),
            "vae_factory": lambda runtime_config: nn.Linear(2, 2, bias=False),
        },
    )

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(od_config=od_config)

    loaded = pipeline.load_weights([])

    assert torch.equal(pipeline.transformer.weight, transformer_weight)
    assert torch.equal(pipeline.vae.weight, vae_weight)
    assert loaded == {"transformer.weight", "vae.weight"}


def test_prismaudio_pipeline_load_weights_requires_initialized_components_for_runtime_checkpoints():
    inline_model_config = {
        "sample_rate": 44100,
        "audio_channels": 2,
        "model": {
            "io_channels": 64,
        },
    }

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_paths={
            "transformer": "/tmp/prismaudio.ckpt",
        },
        model_config={
            "prismaudio_model_config": inline_model_config,
        },
    )

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(od_config=od_config)

    with pytest.raises(NotImplementedError, match="initialized transformer module"):
        pipeline.load_weights([])


def test_prismaudio_pipeline_load_weights_rejects_runtime_checkpoint_without_matching_keys(tmp_path):
    transformer_ckpt_path = tmp_path / "prismaudio.ckpt"
    torch.save({"bad.weight": torch.randn(2, 2)}, transformer_ckpt_path)

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_paths={
            "transformer": str(transformer_ckpt_path),
        },
        model_config={
            "prismaudio_model_config": {
                "sample_rate": 44100,
                "audio_channels": 2,
                "model": {
                    "io_channels": 64,
                },
            },
            "transformer_factory": lambda runtime_config: nn.Linear(2, 2, bias=False),
        },
    )

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(od_config=od_config)

    with pytest.raises(ValueError, match="loaded no matching weights"):
        pipeline.load_weights([])
