# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.utils import DiffusionRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_request(
    additional_information: dict[str, torch.Tensor] | None = None,
    extra_args: dict[str, object] | None = None,
    prompts: list[dict[str, object]] | None = None,
) -> OmniDiffusionRequest:
    features = additional_information or {
        "video_features": torch.randn(1, 10, 1024),
        "text_features": torch.randn(1, 32, 1024),
        "sync_features": torch.randn(1, 16, 768),
    }
    request_prompts = prompts or [
        {
            "prompt": "semantic and temporal cot text",
            "additional_information": features,
        }
    ]
    return OmniDiffusionRequest(
        prompts=request_prompts,
        sampling_params=OmniDiffusionSamplingParams(
            extra_args=extra_args or {},
        ),
        request_ids=[f"req-{idx + 1}" for idx in range(len(request_prompts))],
    )


def _make_minimal_official_prismaudio_model_config(*, include_pretransform: bool = False) -> dict[str, object]:
    model_config: dict[str, object] = {
        "model_type": "diffusion_cond",
        "sample_rate": 44100,
        "audio_channels": 2,
        "sample_size": 397312,
        "model": {
            "io_channels": 64,
            "conditioning": {
                "cond_dim": 128,
                "configs": [
                    {
                        "id": "video_features",
                        "type": "cond_mlp",
                        "config": {
                            "dim": 128,
                            "output_dim": 128,
                        },
                    },
                    {
                        "id": "text_features",
                        "type": "cond_mlp",
                        "config": {
                            "dim": 128,
                            "output_dim": 128,
                        },
                    },
                    {
                        "id": "sync_features",
                        "type": "sync_mlp",
                        "config": {
                            "dim": 128,
                            "output_dim": 128,
                        },
                    },
                ],
            },
            "diffusion": {
                "type": "dit",
                "cross_attention_cond_ids": ["video_features", "text_features"],
                "add_cond_ids": ["video_features"],
                "sync_cond_ids": ["sync_features"],
                "config": {
                    "embed_dim": 128,
                    "depth": 2,
                    "num_heads": 2,
                    "io_channels": 64,
                    "cond_token_dim": 128,
                    "add_token_dim": 128,
                    "sync_token_dim": 128,
                    "project_cond_tokens": False,
                    "transformer_type": "continuous_transformer",
                    "use_gated": True,
                    "use_sync_gated": True,
                    "attn_kwargs": {"qk_norm": "rns"},
                },
            },
        },
    }

    if include_pretransform:
        model_config["model"]["pretransform"] = {
            "type": "autoencoder",
            "iterate_batch": True,
            "config": {
                "encoder": {
                    "type": "oobleck",
                    "config": {
                        "in_channels": 2,
                        "channels": 128,
                        "c_mults": [1, 2, 4, 8, 16],
                        "strides": [2, 4, 4, 8, 8],
                        "latent_dim": 128,
                        "use_snake": True,
                    },
                },
                "decoder": {
                    "type": "oobleck",
                    "config": {
                        "out_channels": 2,
                        "channels": 128,
                        "c_mults": [1, 2, 4, 8, 16],
                        "strides": [2, 4, 4, 8, 8],
                        "latent_dim": 64,
                        "use_snake": True,
                        "final_tanh": False,
                    },
                },
                "bottleneck": {"type": "vae"},
                "latent_dim": 64,
                "downsampling_ratio": 2048,
                "io_channels": 2,
            },
        }

    return model_config


def _make_minimal_official_mm_prismaudio_model_config() -> dict[str, object]:
    return {
        "model_type": "mm_diffusion_cond",
        "sample_rate": 44100,
        "audio_channels": 2,
        "sample_size": 397312,
        "model": {
            "io_channels": 64,
            "conditioning": {
                "cond_dim": 128,
                "configs": [
                    {
                        "id": "metaclip_features",
                        "type": "mm_unchang",
                        "config": {"dim": 1024, "output_dim": 1024},
                    },
                    {
                        "id": "metaclip_text_features",
                        "type": "mm_unchang",
                        "config": {"dim": 1024, "output_dim": 1024},
                    },
                    {
                        "id": "sync_features",
                        "type": "mm_unchang",
                        "config": {"dim": 768, "output_dim": 768},
                    },
                    {
                        "id": "t5_features",
                        "type": "mm_unchang",
                        "config": {"dim": 2048, "output_dim": 2048},
                    },
                ],
            },
            "diffusion": {
                "type": "mmdit",
                "mm_cond_ids": [
                    "metaclip_features",
                    "sync_features",
                    "metaclip_text_features",
                    "t5_features",
                ],
                "config": {
                    "latent_dim": 64,
                    "clip_dim": 1024,
                    "sync_dim": 768,
                    "text_dim": 2048,
                    "hidden_dim": 128,
                    "depth": 2,
                    "fused_depth": 1,
                    "num_heads": 2,
                    "latent_seq_len": 194,
                    "clip_seq_len": 72,
                    "sync_seq_len": 216,
                    "v2": True,
                    "kernel_size": 3,
                },
            },
        },
    }


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


def test_prismaudio_pipeline_requires_dynamic_official_conditioning_ids():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_mm_prismaudio_model_config(),
        },
    )
    pipeline = model_cls(od_config=od_config, transformer=nn.Identity(), vae=nn.Identity())
    req = _make_request(
        additional_information={
            "metaclip_features": torch.randn(1, 72, 1024),
            "metaclip_text_features": torch.randn(1, 72, 1024),
            "sync_features": torch.randn(1, 216, 768),
        }
    )

    with pytest.raises(ValueError, match="t5_features"):
        pipeline.forward(req)


def test_prismaudio_pipeline_rejects_non_tensor_features():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
        },
    )
    pipeline = model_cls(od_config=od_config, transformer=nn.Identity(), vae=nn.Identity())
    req = _make_request(
        additional_information={
            "video_features": [[1.0, 2.0]],
            "text_features": torch.randn(1, 32, 128),
            "sync_features": torch.randn(1, 216, 128),
        }
    )

    with pytest.raises(TypeError, match="video_features"):
        pipeline.forward(req)


def test_prismaudio_pipeline_rejects_bad_dynamic_conditioning_width():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_mm_prismaudio_model_config(),
        },
    )
    pipeline = model_cls(od_config=od_config, transformer=nn.Identity(), vae=nn.Identity())
    req = _make_request(
        additional_information={
            "metaclip_features": torch.randn(1, 72, 1024),
            "metaclip_text_features": torch.randn(1, 72, 1024),
            "sync_features": torch.randn(1, 216, 768),
            "t5_features": torch.randn(1, 32, 1024),
        }
    )

    with pytest.raises(ValueError, match="t5_features"):
        pipeline.forward(req)


def test_prismaudio_pipeline_rejects_invalid_feature_rank():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
        },
    )
    pipeline = model_cls(od_config=od_config, transformer=nn.Identity(), vae=nn.Identity())
    req = _make_request(
        additional_information={
            "video_features": torch.randn(128),
            "text_features": torch.randn(1, 32, 128),
            "sync_features": torch.randn(1, 216, 128),
        }
    )

    with pytest.raises(ValueError, match="video_features"):
        pipeline.forward(req)


def test_prismaudio_pipeline_rejects_invalid_feature_width():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
        },
    )
    pipeline = model_cls(od_config=od_config, transformer=nn.Identity(), vae=nn.Identity())
    req = _make_request(
        additional_information={
            "video_features": torch.randn(1, 10, 64),
            "text_features": torch.randn(1, 32, 128),
            "sync_features": torch.randn(1, 216, 128),
        }
    )

    with pytest.raises(ValueError, match="video_features"):
        pipeline.forward(req)


def test_prismaudio_pipeline_rejects_non_floating_features():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
        },
    )
    pipeline = model_cls(od_config=od_config, transformer=nn.Identity(), vae=nn.Identity())
    req = _make_request(
        additional_information={
            "video_features": torch.ones(1, 10, 128, dtype=torch.int64),
            "text_features": torch.randn(1, 32, 128),
            "sync_features": torch.randn(1, 216, 128),
        }
    )

    with pytest.raises(TypeError, match="video_features"):
        pipeline.forward(req)


def test_prismaudio_pipeline_loads_conditioning_from_npz_path(tmp_path):
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    class _FakeTransformer(nn.Module):
        def forward(self, x, t, **kwargs):
            assert tuple(kwargs["video_features"].shape) == (1, 10, 1024)
            assert tuple(kwargs["text_features"].shape) == (1, 32, 1024)
            assert tuple(kwargs["sync_features"].shape) == (1, 16, 768)
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    feature_path = tmp_path / "conditioning.npz"
    import numpy as np

    np.savez(
        feature_path,
        video_features=np.ones((10, 1024), dtype=np.float32),
        text_features=np.ones((32, 1024), dtype=np.float32),
        sync_features=np.ones((16, 768), dtype=np.float32),
    )

    pipeline = model_cls(transformer=_FakeTransformer(), vae=_FakeVAE())
    req = _make_request(
        prompts=[
            {
                "prompt": "from file",
                "additional_information": {
                    "conditioning_path": str(feature_path),
                },
            }
        ],
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.zeros(1, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (1, 2, 4)


def test_prismaudio_pipeline_loads_conditioning_from_pt_path(tmp_path):
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    class _FakeTransformer(nn.Module):
        def forward(self, x, t, **kwargs):
            assert tuple(kwargs["video_features"].shape) == (1, 10, 1024)
            assert tuple(kwargs["text_features"].shape) == (1, 32, 1024)
            assert tuple(kwargs["sync_features"].shape) == (1, 16, 768)
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    feature_path = tmp_path / "conditioning.pt"
    torch.save(
        {
            "video_features": torch.ones(10, 1024),
            "text_features": torch.ones(32, 1024),
            "sync_features": torch.ones(16, 768),
        },
        feature_path,
    )

    pipeline = model_cls(transformer=_FakeTransformer(), vae=_FakeVAE())
    req = _make_request(
        prompts=[
            {
                "prompt": "from file",
                "additional_information": {
                    "conditioning_path": str(feature_path),
                },
            }
        ],
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.zeros(1, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (1, 2, 4)


def test_prismaudio_pipeline_batches_multiple_conditioning_paths(tmp_path):
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_means: list[float] | None = None

        def forward(self, x, t, **kwargs):
            self.batch_means = kwargs["video_features"].mean(dim=(1, 2)).tolist()
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    import numpy as np

    feature_path_a = tmp_path / "conditioning_a.npz"
    np.savez(
        feature_path_a,
        video_features=np.ones((10, 1024), dtype=np.float32),
        text_features=np.ones((32, 1024), dtype=np.float32),
        sync_features=np.ones((16, 768), dtype=np.float32),
    )
    feature_path_b = tmp_path / "conditioning_b.npz"
    np.savez(
        feature_path_b,
        video_features=np.ones((10, 1024), dtype=np.float32) * 2,
        text_features=np.ones((32, 1024), dtype=np.float32) * 2,
        sync_features=np.ones((16, 768), dtype=np.float32) * 2,
    )

    transformer = _FakeTransformer()
    pipeline = model_cls(transformer=transformer, vae=_FakeVAE())
    req = _make_request(
        prompts=[
            {
                "prompt": "from file a",
                "additional_information": {"conditioning_path": str(feature_path_a)},
            },
            {
                "prompt": "from file b",
                "additional_information": {"conditioning_path": str(feature_path_b)},
            },
        ],
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.zeros(2, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (2, 2, 4)
    assert transformer.batch_means == [1.0, 2.0]


def test_prismaudio_pipeline_rejects_missing_conditioning_path(tmp_path):
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(transformer=nn.Identity(), vae=nn.Identity())
    missing_path = tmp_path / "missing.npz"
    req = _make_request(
        prompts=[
            {
                "prompt": "from missing file",
                "additional_information": {
                    "conditioning_path": str(missing_path),
                },
            }
        ]
    )

    with pytest.raises(ValueError, match="conditioning_path"):
        pipeline.forward(req)


def test_prismaudio_pipeline_uses_conditioning_factory_when_features_are_absent():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_means: list[float] | None = None

        def forward(self, x, t, **kwargs):
            self.batch_means = kwargs["video_features"].mean(dim=(1, 2)).tolist()
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    calls: list[str] = []

    def _conditioning_factory(prompt, runtime_config):
        calls.append(prompt["prompt"])
        assert runtime_config.sample_rate == 44100
        scale = 1.0 if prompt["prompt"] == "first" else 2.0
        return {
            "video_features": torch.ones(10, 128) * scale,
            "text_features": torch.ones(32, 128) * scale,
            "sync_features": torch.ones(16, 128) * scale,
        }

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
            "conditioning_factory": _conditioning_factory,
        },
    )

    transformer = _FakeTransformer()
    pipeline = model_cls(od_config=od_config, transformer=transformer, vae=_FakeVAE())
    req = _make_request(
        prompts=[
            {"prompt": "first", "additional_information": {"caption_cot": "first caption"}},
            {"prompt": "second", "additional_information": {"caption_cot": "second caption"}},
        ],
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.zeros(2, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (2, 2, 4)
    assert calls == ["first", "second"]
    assert transformer.batch_means == [1.0, 2.0]


def test_prismaudio_pipeline_surfaces_bad_conditioning_factory_output():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    def _conditioning_factory(prompt, runtime_config):
        return {"video_features": torch.ones(10, 128)}

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
            "conditioning_factory": _conditioning_factory,
        },
    )

    pipeline = model_cls(od_config=od_config, transformer=nn.Identity(), vae=nn.Identity())
    req = _make_request(
        prompts=[
            {"prompt": "first", "additional_information": {"caption_cot": "first caption"}},
        ]
    )

    with pytest.raises(ValueError, match="text_features"):
        pipeline.forward(req)


def test_prismaudio_pipeline_builds_conditioning_from_import_path_factory_spec():
    import sys
    import types

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_means: list[float] | None = None

        def forward(self, x, t, **kwargs):
            self.batch_means = kwargs["video_features"].mean(dim=(1, 2)).tolist()
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    module_name = "prismaudio_conditioning_factory_test_module"
    builder_module = types.ModuleType(module_name)

    def _build_conditioning(additional_information, runtime_config):
        scale = float(additional_information["scale"])
        assert runtime_config.sample_rate == 44100
        return {
            "video_features": torch.ones(10, 128) * scale,
            "text_features": torch.ones(32, 128) * scale,
            "sync_features": torch.ones(16, 128) * scale,
        }

    builder_module.build_conditioning = _build_conditioning
    sys.modules[module_name] = builder_module

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
            "conditioning_factory": {
                "path": f"{module_name}.build_conditioning",
                "input": "additional_information_and_runtime_config",
            },
        },
    )

    transformer = _FakeTransformer()
    pipeline = model_cls(od_config=od_config, transformer=transformer, vae=_FakeVAE())
    req = _make_request(
        prompts=[
            {"prompt": "first", "additional_information": {"scale": 1.0}},
            {"prompt": "second", "additional_information": {"scale": 2.0}},
        ],
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.zeros(2, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (2, 2, 4)
    assert transformer.batch_means == [1.0, 2.0]


def test_prismaudio_pipeline_builds_conditioning_from_feature_extractor_factory():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_means: list[float] | None = None

        def forward(self, x, t, **kwargs):
            self.batch_means = kwargs["video_features"].mean(dim=(1, 2)).tolist()
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    class _FakeFeatureExtractor:
        def encode_t5_text(self, captions):
            assert captions == ["first caption", "second caption"]
            return torch.stack([torch.ones(32, 128), torch.ones(32, 128) * 2], dim=0)

        def encode_video_and_text_with_videoprism(self, clip_input, captions):
            assert tuple(clip_input.shape) == (2, 3, 4, 4, 3)
            assert captions == ["first caption", "second caption"]
            video_feat = torch.zeros(2, 1)
            frame_embed = torch.stack([torch.ones(10, 128), torch.ones(10, 128) * 2], dim=0)
            ignored = None
            text_feat = torch.zeros(2, 1)
            return video_feat, frame_embed, ignored, text_feat

        def encode_video_with_sync(self, sync_input):
            assert tuple(sync_input.shape) == (2, 5, 3, 4, 4)
            return torch.stack([torch.ones(16, 128), torch.ones(16, 128) * 2], dim=0)

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
            "feature_extractor_factory": lambda runtime_config: _FakeFeatureExtractor(),
        },
    )

    transformer = _FakeTransformer()
    pipeline = model_cls(od_config=od_config, transformer=transformer, vae=_FakeVAE())
    req = _make_request(
        prompts=[
            {
                "prompt": "first prompt",
                "additional_information": {
                    "caption_cot": "first caption",
                    "clip_chunk": torch.ones(3, 4, 4, 3),
                    "sync_chunk": torch.ones(5, 3, 4, 4),
                },
            },
            {
                "prompt": "second prompt",
                "additional_information": {
                    "caption_cot": "second caption",
                    "clip_chunk": torch.ones(3, 4, 4, 3) * 2,
                    "sync_chunk": torch.ones(5, 3, 4, 4) * 2,
                },
            },
        ],
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.zeros(2, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (2, 2, 4)
    assert transformer.batch_means == [1.0, 2.0]


def test_prismaudio_pipeline_builds_mm_conditioning_from_feature_extractor_factory():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_means: list[float] | None = None
            self.t5_batch_means: list[float] | None = None

        def forward(self, x, t, **kwargs):
            self.batch_means = kwargs["metaclip_features"].mean(dim=(1, 2)).tolist()
            self.t5_batch_means = kwargs["t5_features"].mean(dim=(1, 2)).tolist()
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    class _FakeMMFeatureExtractor:
        def encode_video_with_clip(self, clip_input):
            assert tuple(clip_input.shape) == (2, 3, 224, 224, 3)
            return torch.stack([torch.ones(72, 1024), torch.ones(72, 1024) * 2], dim=0)

        def encode_text(self, captions):
            assert captions == ["first prompt", "second prompt"]
            global_text = torch.stack([torch.ones(1024), torch.ones(1024) * 2], dim=0)
            token_text = torch.stack([torch.ones(77, 1024), torch.ones(77, 1024) * 2], dim=0)
            return global_text, token_text

        def encode_t5_text(self, captions):
            assert captions == ["first caption", "second caption"]
            return torch.stack([torch.ones(32, 2048), torch.ones(32, 2048) * 2], dim=0)

        def encode_video_with_sync(self, sync_input):
            assert tuple(sync_input.shape) == (2, 25, 3, 224, 224)
            return torch.stack([torch.ones(216, 768), torch.ones(216, 768) * 2], dim=0)

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_mm_prismaudio_model_config(),
            "feature_extractor_factory": lambda runtime_config: _FakeMMFeatureExtractor(),
        },
    )

    transformer = _FakeTransformer()
    pipeline = model_cls(od_config=od_config, transformer=transformer, vae=_FakeVAE())
    req = _make_request(
        prompts=[
            {
                "prompt": "first prompt",
                "additional_information": {
                    "caption_cot": "first caption",
                    "clip_chunk": torch.ones(3, 224, 224, 3),
                    "sync_chunk": torch.ones(25, 3, 224, 224),
                },
            },
            {
                "prompt": "second prompt",
                "additional_information": {
                    "caption_cot": "second caption",
                    "clip_chunk": torch.ones(3, 224, 224, 3) * 2,
                    "sync_chunk": torch.ones(25, 3, 224, 224) * 2,
                },
            },
        ],
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.zeros(2, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (2, 2, 4)
    assert transformer.batch_means == [1.0, 2.0]
    assert transformer.t5_batch_means == [1.0, 2.0]


def test_prismaudio_pipeline_rejects_missing_feature_extractor_inputs():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    class _FakeFeatureExtractor:
        def encode_t5_text(self, captions):
            return torch.ones(1, 32, 128)

        def encode_video_and_text_with_videoprism(self, clip_input, captions):
            return torch.zeros(1, 1), torch.ones(1, 10, 128), None, torch.zeros(1, 1)

        def encode_video_with_sync(self, sync_input):
            return torch.ones(1, 16, 128)

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
            "feature_extractor_factory": lambda runtime_config: _FakeFeatureExtractor(),
        },
    )

    pipeline = model_cls(od_config=od_config, transformer=nn.Identity(), vae=nn.Identity())
    req = _make_request(
        prompts=[
            {
                "prompt": "missing sync input",
                "additional_information": {
                    "caption_cot": "caption only",
                    "clip_chunk": torch.ones(3, 4, 4, 3),
                },
            },
        ]
    )

    with pytest.raises(ValueError, match="sync_chunk"):
        pipeline.forward(req)


def test_prismaudio_pipeline_builds_feature_extractor_from_import_path_factory():
    import sys
    import types

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_means: list[float] | None = None

        def forward(self, x, t, **kwargs):
            self.batch_means = kwargs["video_features"].mean(dim=(1, 2)).tolist()
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    class _FakeFeatureExtractor:
        def encode_t5_text(self, captions):
            return torch.stack([torch.ones(32, 128), torch.ones(32, 128) * 2], dim=0)

        def encode_video_and_text_with_videoprism(self, clip_input, captions):
            return torch.zeros(2, 1), torch.stack([torch.ones(10, 128), torch.ones(10, 128) * 2], dim=0), None, None

        def encode_video_with_sync(self, sync_input):
            return torch.stack([torch.ones(16, 128), torch.ones(16, 128) * 2], dim=0)

    module_name = "prismaudio_feature_extractor_factory_test_module"
    builder_module = types.ModuleType(module_name)

    def _build_feature_extractor(runtime_config):
        assert runtime_config.audio_channels == 2
        return _FakeFeatureExtractor()

    builder_module.build_feature_extractor = _build_feature_extractor
    sys.modules[module_name] = builder_module

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
            "feature_extractor_factory": f"{module_name}.build_feature_extractor",
        },
    )

    transformer = _FakeTransformer()
    pipeline = model_cls(od_config=od_config, transformer=transformer, vae=_FakeVAE())
    req = _make_request(
        prompts=[
            {
                "prompt": "first prompt",
                "additional_information": {
                    "caption_cot": "first caption",
                    "clip_chunk": torch.ones(3, 4, 4, 3),
                    "sync_chunk": torch.ones(5, 3, 4, 4),
                },
            },
            {
                "prompt": "second prompt",
                "additional_information": {
                    "caption_cot": "second caption",
                    "clip_chunk": torch.ones(3, 4, 4, 3),
                    "sync_chunk": torch.ones(5, 3, 4, 4),
                },
            },
        ],
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.zeros(2, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (2, 2, 4)
    assert transformer.batch_means == [1.0, 2.0]


def test_prismaudio_pipeline_uses_default_official_feature_extractor_builder(monkeypatch):
    import sys
    import types

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_means: list[float] | None = None

        def forward(self, x, t, **kwargs):
            self.batch_means = kwargs["video_features"].mean(dim=(1, 2)).tolist()
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    class _OfficialFeaturesUtils:
        def __init__(self, *, vae_ckpt=None, vae_config=None, synchformer_ckpt=None, enable_conditions=True, need_vae_encoder=True):
            assert vae_ckpt is None
            assert vae_config == "/tmp/vae_config.json"
            assert synchformer_ckpt == "/tmp/synchformer.ckpt"
            assert enable_conditions is True
            assert need_vae_encoder is False

        def encode_t5_text(self, captions):
            return torch.stack([torch.ones(32, 128), torch.ones(32, 128) * 2], dim=0)

        def encode_video_and_text_with_videoprism(self, clip_input, captions):
            return torch.zeros(2, 1), torch.stack([torch.ones(10, 128), torch.ones(10, 128) * 2], dim=0), None, None

        def encode_video_with_sync(self, sync_input):
            return torch.stack([torch.ones(16, 128), torch.ones(16, 128) * 2], dim=0)

    data_utils_module = types.ModuleType("data_utils")
    v2a_utils_module = types.ModuleType("data_utils.v2a_utils")
    feature_utils_module = types.ModuleType("data_utils.v2a_utils.feature_utils_288")
    feature_utils_module.FeaturesUtils = _OfficialFeaturesUtils
    monkeypatch.setitem(sys.modules, "data_utils", data_utils_module)
    monkeypatch.setitem(sys.modules, "data_utils.v2a_utils", v2a_utils_module)
    monkeypatch.setitem(sys.modules, "data_utils.v2a_utils.feature_utils_288", feature_utils_module)

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
            "feature_extractor_config": {
                "vae_config": "/tmp/vae_config.json",
                "synchformer_ckpt": "/tmp/synchformer.ckpt",
                "need_vae_encoder": False,
            },
        },
    )

    transformer = _FakeTransformer()
    pipeline = model_cls(od_config=od_config, transformer=transformer, vae=_FakeVAE())
    req = _make_request(
        prompts=[
            {
                "prompt": "first prompt",
                "additional_information": {
                    "caption_cot": "first caption",
                    "clip_chunk": torch.ones(3, 4, 4, 3),
                    "sync_chunk": torch.ones(5, 3, 4, 4),
                },
            },
            {
                "prompt": "second prompt",
                "additional_information": {
                    "caption_cot": "second caption",
                    "clip_chunk": torch.ones(3, 4, 4, 3),
                    "sync_chunk": torch.ones(5, 3, 4, 4),
                },
            },
        ],
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.zeros(2, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (2, 2, 4)
    assert transformer.batch_means == [1.0, 2.0]


def test_prismaudio_pipeline_uses_mm_default_official_feature_extractor_builder(monkeypatch):
    import sys
    import types

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_means: list[float] | None = None
            self.t5_batch_means: list[float] | None = None

        def forward(self, x, t, **kwargs):
            self.batch_means = kwargs["metaclip_features"].mean(dim=(1, 2)).tolist()
            self.t5_batch_means = kwargs["t5_features"].mean(dim=(1, 2)).tolist()
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    class _OfficialMMFeaturesUtils:
        def __init__(self, *, vae_ckpt=None, vae_config=None, synchformer_ckpt=None, enable_conditions=True, need_vae_encoder=True):
            assert vae_ckpt is None
            assert vae_config == "/tmp/mm_vae_config.json"
            assert synchformer_ckpt == "/tmp/mm_synchformer.ckpt"
            assert enable_conditions is True
            assert need_vae_encoder is False

        def encode_video_with_clip(self, clip_input):
            return torch.stack([torch.ones(72, 1024), torch.ones(72, 1024) * 2], dim=0)

        def encode_text(self, captions):
            return (
                torch.stack([torch.ones(1024), torch.ones(1024) * 2], dim=0),
                torch.stack([torch.ones(77, 1024), torch.ones(77, 1024) * 2], dim=0),
            )

        def encode_t5_text(self, captions):
            return torch.stack([torch.ones(32, 2048), torch.ones(32, 2048) * 2], dim=0)

        def encode_video_with_sync(self, sync_input):
            return torch.stack([torch.ones(216, 768), torch.ones(216, 768) * 2], dim=0)

    data_utils_module = types.ModuleType("data_utils")
    v2a_utils_module = types.ModuleType("data_utils.v2a_utils")
    feature_utils_module = types.ModuleType("data_utils.v2a_utils.feature_utils_224")
    feature_utils_module.FeaturesUtils = _OfficialMMFeaturesUtils
    monkeypatch.setitem(sys.modules, "data_utils", data_utils_module)
    monkeypatch.setitem(sys.modules, "data_utils.v2a_utils", v2a_utils_module)
    monkeypatch.setitem(sys.modules, "data_utils.v2a_utils.feature_utils_224", feature_utils_module)

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_mm_prismaudio_model_config(),
            "feature_extractor_config": {
                "vae_config": "/tmp/mm_vae_config.json",
                "synchformer_ckpt": "/tmp/mm_synchformer.ckpt",
                "need_vae_encoder": False,
            },
        },
    )

    transformer = _FakeTransformer()
    pipeline = model_cls(od_config=od_config, transformer=transformer, vae=_FakeVAE())
    req = _make_request(
        prompts=[
            {
                "prompt": "first prompt",
                "additional_information": {
                    "caption_cot": "first caption",
                    "clip_chunk": torch.ones(3, 224, 224, 3),
                    "sync_chunk": torch.ones(25, 3, 224, 224),
                },
            },
            {
                "prompt": "second prompt",
                "additional_information": {
                    "caption_cot": "second caption",
                    "clip_chunk": torch.ones(3, 224, 224, 3),
                    "sync_chunk": torch.ones(25, 3, 224, 224),
                },
            },
        ],
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.zeros(2, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (2, 2, 4)
    assert transformer.batch_means == [1.0, 2.0]
    assert transformer.t5_batch_means == [1.0, 2.0]


def test_prismaudio_pipeline_surfaces_missing_dependency_from_default_feature_extractor_builder(monkeypatch):
    import importlib

    original_import_module = importlib.import_module

    def _fake_import_module(name, package=None):
        if name == "data_utils.v2a_utils.feature_utils_288":
            raise ModuleNotFoundError("No module named 'videoprism'")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
            "feature_extractor_config": {
                "vae_config": "/tmp/vae_config.json",
                "synchformer_ckpt": "/tmp/synchformer.ckpt",
            },
        },
    )

    pipeline = model_cls(od_config=od_config, transformer=nn.Identity(), vae=nn.Identity())
    req = _make_request(
        prompts=[
            {
                "prompt": "first prompt",
                "additional_information": {
                    "caption_cot": "first caption",
                    "clip_chunk": torch.ones(3, 4, 4, 3),
                    "sync_chunk": torch.ones(5, 3, 4, 4),
                },
            },
        ]
    )

    with pytest.raises(ModuleNotFoundError, match="feature extractor"):
        pipeline.forward(req)


def test_prismaudio_pipeline_uses_video_preprocessor_factory_before_feature_extractor():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_means: list[float] | None = None

        def forward(self, x, t, **kwargs):
            self.batch_means = kwargs["video_features"].mean(dim=(1, 2)).tolist()
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    class _FakeFeatureExtractor:
        def encode_t5_text(self, captions):
            return torch.stack([torch.ones(32, 128), torch.ones(32, 128) * 2], dim=0)

        def encode_video_and_text_with_videoprism(self, clip_input, captions):
            scales = clip_input.mean(dim=(1, 2, 3, 4))
            return (
                torch.zeros(len(captions), 1),
                torch.stack([torch.ones(10, 128) * float(scale) for scale in scales], dim=0),
                None,
                None,
            )

        def encode_video_with_sync(self, sync_input):
            scales = sync_input.mean(dim=(1, 2, 3, 4))
            return torch.stack([torch.ones(16, 128) * float(scale) for scale in scales], dim=0)

    def _video_preprocessor_factory(prompt, runtime_config):
        scale = 1.0 if prompt["additional_information"]["video_path"] == "video-a.mp4" else 2.0
        return {
            "clip_chunk": torch.ones(3, 4, 4, 3) * scale,
            "sync_chunk": torch.ones(5, 3, 4, 4) * scale,
        }

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
            "video_preprocessor_factory": _video_preprocessor_factory,
            "feature_extractor_factory": lambda runtime_config: _FakeFeatureExtractor(),
        },
    )

    transformer = _FakeTransformer()
    pipeline = model_cls(od_config=od_config, transformer=transformer, vae=_FakeVAE())
    req = _make_request(
        prompts=[
            {
                "prompt": "first prompt",
                "additional_information": {
                    "caption_cot": "first caption",
                    "video_path": "video-a.mp4",
                },
            },
            {
                "prompt": "second prompt",
                "additional_information": {
                    "caption_cot": "second caption",
                    "video_path": "video-b.mp4",
                },
            },
        ],
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.zeros(2, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (2, 2, 4)
    assert transformer.batch_means == [1.0, 2.0]


def test_prismaudio_pipeline_rejects_missing_video_preprocessor_inputs():
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    class _FakeFeatureExtractor:
        def encode_t5_text(self, captions):
            return torch.ones(1, 32, 128)

        def encode_video_and_text_with_videoprism(self, clip_input, captions):
            return torch.zeros(1, 1), torch.ones(1, 10, 128), None, None

        def encode_video_with_sync(self, sync_input):
            return torch.ones(1, 16, 128)

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
            "video_preprocessor_factory": lambda prompt, runtime_config: {"clip_chunk": torch.ones(3, 4, 4, 3)},
            "feature_extractor_factory": lambda runtime_config: _FakeFeatureExtractor(),
        },
    )

    pipeline = model_cls(od_config=od_config, transformer=nn.Identity(), vae=nn.Identity())
    req = _make_request(
        prompts=[
            {
                "prompt": "broken preprocessing",
                "additional_information": {
                    "caption_cot": "caption only",
                    "video_path": "video-a.mp4",
                },
            },
        ]
    )

    with pytest.raises(ValueError, match="sync_chunk"):
        pipeline.forward(req)


def test_prismaudio_pipeline_uses_default_official_video_preprocessor_builder(monkeypatch):
    import sys
    import types

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_means: list[float] | None = None

        def forward(self, x, t, **kwargs):
            self.batch_means = kwargs["video_features"].mean(dim=(1, 2)).tolist()
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    class _FakeFeatureExtractor:
        def encode_t5_text(self, captions):
            return torch.stack([torch.ones(32, 128), torch.ones(32, 128) * 2], dim=0)

        def encode_video_and_text_with_videoprism(self, clip_input, captions):
            scales = clip_input.mean(dim=(1, 2, 3, 4))
            return (
                torch.zeros(len(captions), 1),
                torch.stack([torch.ones(10, 128) * float(scale) for scale in scales], dim=0),
                None,
                None,
            )

        def encode_video_with_sync(self, sync_input):
            scales = sync_input.mean(dim=(1, 2, 3, 4))
            return torch.stack([torch.ones(16, 128) * float(scale) for scale in scales], dim=0)

    official_app_module = types.ModuleType("app")

    def _extract_video_frames(video_path):
        scale = 1.0 if video_path == "video-a.mp4" else 2.0
        return torch.ones(3, 4, 4, 3) * scale, torch.ones(5, 3, 4, 4) * scale, 1.0

    official_app_module.extract_video_frames = _extract_video_frames
    monkeypatch.setitem(sys.modules, "app", official_app_module)

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
            "video_preprocessor_config": {},
            "feature_extractor_factory": lambda runtime_config: _FakeFeatureExtractor(),
        },
    )

    transformer = _FakeTransformer()
    pipeline = model_cls(od_config=od_config, transformer=transformer, vae=_FakeVAE())
    req = _make_request(
        prompts=[
            {
                "prompt": "first prompt",
                "additional_information": {
                    "caption_cot": "first caption",
                    "video_path": "video-a.mp4",
                },
            },
            {
                "prompt": "second prompt",
                "additional_information": {
                    "caption_cot": "second caption",
                    "video_path": "video-b.mp4",
                },
            },
        ],
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.zeros(2, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (2, 2, 4)
    assert transformer.batch_means == [1.0, 2.0]


def test_prismaudio_pipeline_surfaces_missing_dependency_from_default_video_preprocessor_builder(monkeypatch):
    import importlib

    original_import_module = importlib.import_module

    def _fake_import_module(name, package=None):
        if name == "app":
            raise ModuleNotFoundError("No module named 'torio'")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
            "video_preprocessor_config": {},
            "feature_extractor_factory": lambda runtime_config: object(),
        },
    )

    pipeline = model_cls(od_config=od_config, transformer=nn.Identity(), vae=nn.Identity())
    req = _make_request(
        prompts=[
            {
                "prompt": "first prompt",
                "additional_information": {
                    "caption_cot": "first caption",
                    "video_path": "video-a.mp4",
                },
            },
        ]
    )

    with pytest.raises(ModuleNotFoundError, match="video preprocessor"):
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


def test_prismaudio_pipeline_default_official_builder_does_not_patch_sys_modules(monkeypatch):
    import sys
    import types

    observed_modules = {}

    def _create_model_from_config(raw_model_config):
        observed_modules["wandb"] = "wandb" in sys.modules
        observed_modules["lightning"] = "lightning" in sys.modules
        raise ModuleNotFoundError("No module named 'lightning'")

    prismaudio_module = types.ModuleType("PrismAudio")
    prismaudio_models_module = types.ModuleType("PrismAudio.models")
    prismaudio_models_module.create_model_from_config = _create_model_from_config
    prismaudio_module.models = prismaudio_models_module
    monkeypatch.setitem(sys.modules, "PrismAudio", prismaudio_module)
    monkeypatch.setitem(sys.modules, "PrismAudio.models", prismaudio_models_module)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    monkeypatch.delitem(sys.modules, "lightning", raising=False)

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(),
        },
    )

    with pytest.raises(ModuleNotFoundError, match="required dependency is missing"):
        model_cls(od_config=od_config)

    assert observed_modules == {
        "wandb": False,
        "lightning": False,
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


def test_prismaudio_checkpoint_loader_preserves_nested_model_prefix_for_wrappers(tmp_path):
    from vllm_omni.diffusion.models.prismaudio.pipeline_prismaudio import load_module_from_prismaudio_checkpoint

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(2, 2, bias=False)

    ckpt_path = tmp_path / "prismaudio.ckpt"
    wrapper = _Wrapper()
    expected_weight = torch.randn_like(wrapper.model.weight)
    torch.save(
        {
            "model.weight": expected_weight,
        },
        ckpt_path,
    )

    report = load_module_from_prismaudio_checkpoint(wrapper, ckpt_path)

    assert torch.equal(wrapper.model.weight, expected_weight)
    assert report.missing_keys == []
    assert report.unexpected_keys == []


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


def test_prismaudio_checkpoint_loader_accepts_none_return_from_load_state_dict(tmp_path):
    from vllm_omni.diffusion.models.prismaudio.pipeline_prismaudio import load_module_from_prismaudio_checkpoint

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(2, 2, bias=False)

        def load_state_dict(self, state_dict, strict=True):
            self.model.load_state_dict(state_dict, strict=strict)
            return None

    ckpt_path = tmp_path / "vae.ckpt"
    wrapper = _Wrapper()
    expected_weight = torch.randn_like(wrapper.model.weight)
    torch.save(
        {
            "state_dict": {
                "autoencoder.weight": expected_weight,
            }
        },
        ckpt_path,
    )

    report = load_module_from_prismaudio_checkpoint(wrapper, ckpt_path, prefix="autoencoder.")

    assert torch.equal(wrapper.model.weight, expected_weight)
    assert report.loaded_keys == ["weight"]
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


def test_prismaudio_pipeline_batches_dynamic_official_conditioning_ids():
    class _FakeMMTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls: list[dict[str, object]] = []

        def forward(self, x, t, **kwargs):
            self.calls.append(
                {
                    "keys": sorted(kwargs.keys()),
                    "metaclip_shape": tuple(kwargs["metaclip_features"].shape),
                    "text_shape": tuple(kwargs["metaclip_text_features"].shape),
                    "sync_shape": tuple(kwargs["sync_features"].shape),
                    "t5_shape": tuple(kwargs["t5_features"].shape),
                    "cfg_scale": kwargs["cfg_scale"],
                    "batch_cfg": kwargs["batch_cfg"],
                }
            )
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_mm_prismaudio_model_config(),
        },
    )

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    transformer = _FakeMMTransformer()
    pipeline = model_cls(od_config=od_config, transformer=transformer, vae=_FakeVAE())
    req = _make_request(
        prompts=[
            {
                "prompt": "first",
                "additional_information": {
                    "metaclip_features": torch.ones(1, 72, 1024),
                    "metaclip_text_features": torch.ones(1, 72, 1024),
                    "sync_features": torch.ones(1, 216, 768),
                    "t5_features": torch.ones(1, 32, 2048),
                },
            },
            {
                "prompt": "second",
                "additional_information": {
                    "metaclip_features": torch.ones(1, 72, 1024) * 2,
                    "metaclip_text_features": torch.ones(1, 72, 1024) * 2,
                    "sync_features": torch.ones(1, 216, 768) * 2,
                    "t5_features": torch.ones(1, 32, 2048) * 2,
                },
            },
        ],
        extra_args={
            "num_inference_steps": 2,
            "cfg_scale": 4.0,
        },
    )
    req.sampling_params.latents = torch.zeros(2, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (2, 2, 4)
    assert len(transformer.calls) == 2
    assert transformer.calls[0]["keys"] == [
        "batch_cfg",
        "cfg_scale",
        "metaclip_features",
        "metaclip_text_features",
        "sync_features",
        "t5_features",
    ]
    assert transformer.calls[0]["metaclip_shape"] == (2, 72, 1024)
    assert transformer.calls[0]["text_shape"] == (2, 72, 1024)
    assert transformer.calls[0]["sync_shape"] == (2, 216, 768)
    assert transformer.calls[0]["t5_shape"] == (2, 32, 2048)
    assert transformer.calls[0]["cfg_scale"] == 4.0
    assert transformer.calls[0]["batch_cfg"] is True


def test_prismaudio_pipeline_initializes_latents_from_runtime_config_when_missing():
    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls: list[dict[str, object]] = []

        def forward(self, x, t, **kwargs):
            self.calls.append(
                {
                    "latent_shape": tuple(x.shape),
                    "cfg_scale": kwargs["cfg_scale"],
                    "batch_cfg": kwargs["batch_cfg"],
                }
            )
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": {
                "sample_rate": 44100,
                "sample_size": 397312,
                "audio_channels": 2,
                "model": {
                    "io_channels": 64,
                    "pretransform": {
                        "config": {
                            "latent_dim": 64,
                            "downsampling_ratio": 2048,
                        }
                    },
                },
            },
        },
    )

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    transformer = _FakeTransformer()
    pipeline = model_cls(od_config=od_config, transformer=transformer, vae=_FakeVAE())
    req = _make_request(extra_args={"num_inference_steps": 2, "cfg_scale": 5.0})

    output = pipeline.forward(req)

    assert output.output.shape == (1, 2, 194)
    assert len(transformer.calls) == 2
    assert transformer.calls[0]["latent_shape"] == (1, 64, 194)
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


def test_prismaudio_pipeline_strips_single_prompt_batch_dim_for_official_wrapper_conditioner():
    class _InnerTransformer(nn.Module):
        def forward(self, x, t, **kwargs):
            return torch.zeros_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    class _OfficialStyleWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _InnerTransformer()
            self.conditioner_shapes: dict[str, tuple[int, ...]] | None = None

        def conditioner(self, metadata, device):
            sample = metadata[0]
            self.conditioner_shapes = {
                key: tuple(value.shape) for key, value in sample.items() if isinstance(value, torch.Tensor)
            }
            return {}

        def get_conditioning_inputs(self, conditioning_tensors):
            assert conditioning_tensors == {}
            return {}

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    transformer = _OfficialStyleWrapper()
    pipeline = model_cls(transformer=transformer, vae=_FakeVAE())
    req = _make_request(
        additional_information={
            "video_features": torch.randn(1, 10, 1024),
            "text_features": torch.randn(1, 32, 1024),
            "sync_features": torch.randn(1, 216, 768),
        },
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.randn(1, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (1, 2, 4)
    assert transformer.conditioner_shapes == {
        "video_features": (10, 1024),
        "text_features": (32, 1024),
        "sync_features": (216, 768),
    }


def test_prismaudio_pipeline_casts_official_wrapper_conditioning_to_runtime_dtype():
    class _InnerTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1024, 1024, bias=False, dtype=torch.bfloat16)

        def forward(self, x, t, **kwargs):
            return torch.zeros_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    class _OfficialStyleWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _InnerTransformer()
            self.seen_dtypes: dict[str, torch.dtype] | None = None

        def conditioner(self, metadata, device):
            sample = metadata[0]
            self.seen_dtypes = {
                key: value.dtype for key, value in sample.items() if isinstance(value, torch.Tensor)
            }
            return {
                "video_features": (
                    self.model.proj(sample["video_features"].to(device=device)),
                    torch.ones(sample["video_features"].shape[0], device=device, dtype=torch.bool),
                ),
            }

        def get_conditioning_inputs(self, conditioning_tensors):
            return {
                "cross_attn_cond": conditioning_tensors["video_features"][0],
                "cross_attn_mask": conditioning_tensors["video_features"][1],
            }

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    transformer = _OfficialStyleWrapper()
    pipeline = model_cls(transformer=transformer, vae=_FakeVAE())
    req = _make_request(
        additional_information={
            "video_features": torch.randn(1, 10, 1024, dtype=torch.float32),
            "text_features": torch.randn(1, 32, 1024, dtype=torch.float32),
            "sync_features": torch.randn(1, 216, 768, dtype=torch.float32),
        },
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.randn(1, 64, 4, dtype=torch.bfloat16)

    output = pipeline.forward(req)

    assert output.output.shape == (1, 2, 4)
    assert transformer.seen_dtypes == {
        "video_features": torch.bfloat16,
        "text_features": torch.bfloat16,
        "sync_features": torch.bfloat16,
    }


def test_prismaudio_post_process_returns_numpy_audio():
    from vllm_omni.diffusion.models.prismaudio.pipeline_prismaudio import get_prismaudio_post_process_func

    post_process = get_prismaudio_post_process_func(OmniDiffusionConfig(model="prismaudio"))

    audio = torch.randn(1, 2, 32, dtype=torch.bfloat16)
    output = post_process(audio)

    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 2, 32)


def test_prismaudio_pipeline_uses_sampling_seed_for_auto_initialized_latents():
    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

        def forward(self, x, t, **kwargs):
            return torch.zeros_like(x)

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return latents[:, :2, :]

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": {
                "sample_rate": 44100,
                "sample_size": 397312,
                "audio_channels": 2,
                "model": {
                    "io_channels": 64,
                    "pretransform": {
                        "config": {
                            "latent_dim": 64,
                            "downsampling_ratio": 2048,
                        }
                    },
                },
            },
        },
    )

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(od_config=od_config, transformer=_FakeTransformer(), vae=_FakeVAE())

    sampling_a = OmniDiffusionSamplingParams(seed=1234, extra_args={})
    sampling_b = OmniDiffusionSamplingParams(seed=1234, extra_args={})
    prompts = _make_request().prompts

    latents_a = pipeline._prepare_latents_from_sampling(prompts, sampling_a)
    latents_b = pipeline._prepare_latents_from_sampling(prompts, sampling_b)

    assert torch.equal(latents_a, latents_b)


def test_prismaudio_pipeline_uses_sampling_generator_for_auto_initialized_latents():
    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

        def forward(self, x, t, **kwargs):
            return torch.zeros_like(x)

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return latents[:, :2, :]

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": {
                "sample_rate": 44100,
                "sample_size": 397312,
                "audio_channels": 2,
                "model": {
                    "io_channels": 64,
                    "pretransform": {
                        "config": {
                            "latent_dim": 64,
                            "downsampling_ratio": 2048,
                        }
                    },
                },
            },
        },
    )

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(od_config=od_config, transformer=_FakeTransformer(), vae=_FakeVAE())

    sampling_a = OmniDiffusionSamplingParams(generator=torch.Generator(device="cpu").manual_seed(4321), extra_args={})
    sampling_b = OmniDiffusionSamplingParams(generator=torch.Generator(device="cpu").manual_seed(4321), extra_args={})
    prompts = _make_request().prompts

    latents_a = pipeline._prepare_latents_from_sampling(prompts, sampling_a)
    latents_b = pipeline._prepare_latents_from_sampling(prompts, sampling_b)

    assert torch.equal(latents_a, latents_b)


def test_prismaudio_pipeline_moves_fallback_conditioning_to_sampling_device():
    class _FallbackTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1, device="meta"))

        def forward(self, x, t, **kwargs):
            return torch.zeros_like(x)

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(transformer=_FallbackTransformer(), vae=nn.Identity())
    conditioning = {
        "video_features": torch.randn(1, 10, 1024),
        "text_features": torch.randn(1, 32, 1024),
        "sync_features": torch.randn(1, 16, 768),
    }

    _, sampling_conditioning = pipeline._get_sampling_model_and_conditioning(
        conditioning,
        device=torch.device("meta"),
    )

    assert sampling_conditioning["video_features"].device.type == "meta"
    assert sampling_conditioning["text_features"].device.type == "meta"
    assert sampling_conditioning["sync_features"].device.type == "meta"


def test_prismaudio_pipeline_batches_multiple_prompt_conditioning_payloads():
    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls: list[dict[str, object]] = []

        def forward(self, x, t, **kwargs):
            self.calls.append(
                {
                    "video_features_shape": tuple(kwargs["video_features"].shape),
                    "video_features_batch_means": kwargs["video_features"].mean(dim=(1, 2)).tolist(),
                    "text_features_shape": tuple(kwargs["text_features"].shape),
                    "sync_features_shape": tuple(kwargs["sync_features"].shape),
                    "cfg_scale": kwargs["cfg_scale"],
                    "batch_cfg": kwargs["batch_cfg"],
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
    req = _make_request(
        prompts=[
            {
                "prompt": "first",
                "additional_information": {
                    "video_features": torch.ones(1, 10, 1024),
                    "text_features": torch.ones(1, 32, 1024),
                    "sync_features": torch.ones(1, 16, 768),
                },
            },
            {
                "prompt": "second",
                "additional_information": {
                    "video_features": torch.ones(1, 10, 1024) * 2,
                    "text_features": torch.ones(1, 32, 1024) * 2,
                    "sync_features": torch.ones(1, 16, 768) * 2,
                },
            },
        ],
        extra_args={"num_inference_steps": 2, "cfg_scale": 5.0},
    )
    req.sampling_params.latents = torch.zeros(2, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (2, 2, 4)
    assert torch.allclose(output.output, torch.full((2, 2, 4), -3.0))
    assert len(transformer.calls) == 2
    assert transformer.calls[0]["video_features_shape"] == (2, 10, 1024)
    assert transformer.calls[0]["text_features_shape"] == (2, 32, 1024)
    assert transformer.calls[0]["sync_features_shape"] == (2, 16, 768)
    assert transformer.calls[0]["video_features_batch_means"] == [1.0, 2.0]
    assert transformer.calls[0]["cfg_scale"] == 5.0
    assert transformer.calls[0]["batch_cfg"] is True


def test_prismaudio_pipeline_batches_multiple_prompts_for_official_wrapper_conditioner():
    class _InnerTransformer(nn.Module):
        def forward(self, x, t, **kwargs):
            return torch.ones_like(x)

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    class _OfficialStyleWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _InnerTransformer()
            self.conditioner_payload: list[dict[str, tuple[int, ...]]] | None = None

        def conditioner(self, metadata, device):
            self.conditioner_payload = [
                {key: tuple(value.shape) for key, value in sample.items() if isinstance(value, torch.Tensor)}
                for sample in metadata
            ]
            batch = len(metadata)
            return {
                "video_features": (
                    torch.ones(batch, 10, 1024, device=device),
                    torch.ones(batch, 10, device=device, dtype=torch.bool),
                ),
                "text_features": (
                    torch.ones(batch, 32, 1024, device=device),
                    torch.ones(batch, 32, device=device, dtype=torch.bool),
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

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    transformer = _OfficialStyleWrapper()
    pipeline = model_cls(transformer=transformer, vae=_FakeVAE())
    req = _make_request(
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
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.zeros(2, 64, 4)

    output = pipeline.forward(req)

    assert output.output.shape == (2, 2, 4)
    assert transformer.conditioner_payload == [
        {
            "video_features": (10, 1024),
            "text_features": (32, 1024),
            "sync_features": (16, 768),
        },
        {
            "video_features": (10, 1024),
            "text_features": (32, 1024),
            "sync_features": (16, 768),
        },
    ]


def test_prismaudio_pipeline_supports_step_execution_with_multiple_prompts():
    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls: list[dict[str, object]] = []

        def forward(self, x, t, **kwargs):
            self.calls.append(
                {
                    "latent_shape": tuple(x.shape),
                    "video_features_shape": tuple(kwargs["video_features"].shape),
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
    req = _make_request(
        prompts=[
            {
                "prompt": "first",
                "additional_information": {
                    "video_features": torch.ones(1, 10, 1024),
                    "text_features": torch.ones(1, 32, 1024),
                    "sync_features": torch.ones(1, 16, 768),
                },
            },
            {
                "prompt": "second",
                "additional_information": {
                    "video_features": torch.ones(1, 10, 1024) * 2,
                    "text_features": torch.ones(1, 32, 1024) * 2,
                    "sync_features": torch.ones(1, 16, 768) * 2,
                },
            },
        ],
        extra_args={"num_inference_steps": 2, "cfg_scale": 5.0},
    )
    req.sampling_params.latents = torch.zeros(2, 64, 4)
    state = DiffusionRequestState(
        req_id="req-1",
        sampling=req.sampling_params,
        prompts=req.prompts,
    )

    pipeline.prepare_encode(state)
    while not state.denoise_completed:
        noise_pred = pipeline.denoise_step(state)
        assert noise_pred is not None
        pipeline.step_scheduler(state, noise_pred)
    output = pipeline.post_decode(state)

    assert output.output.shape == (2, 2, 4)
    assert torch.allclose(output.output, torch.full((2, 2, 4), -3.0))
    assert len(transformer.calls) == 2
    assert transformer.calls[0]["latent_shape"] == (2, 64, 4)
    assert transformer.calls[0]["video_features_shape"] == (2, 10, 1024)


def test_prismaudio_pipeline_supports_step_execution_with_fake_components():
    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls: list[dict[str, object]] = []

        def forward(self, x, t, **kwargs):
            self.calls.append(
                {
                    "latent_shape": tuple(x.shape),
                    "t": t.detach().cpu().tolist(),
                    "cfg_scale": kwargs["cfg_scale"],
                    "batch_cfg": kwargs["batch_cfg"],
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
    req.sampling_params.latents = initial_latents.clone()
    state = DiffusionRequestState(
        req_id="req-1",
        sampling=req.sampling_params,
        prompts=req.prompts,
    )

    pipeline.prepare_encode(state)
    while not state.denoise_completed:
        noise_pred = pipeline.denoise_step(state)
        assert noise_pred is not None
        pipeline.step_scheduler(state, noise_pred)
    output = pipeline.post_decode(state)

    assert output.output is not None
    assert output.output.shape == (1, 2, 4)
    assert torch.allclose(output.output, torch.full((1, 2, 4), -3.0))
    assert len(transformer.calls) == 4
    assert transformer.calls[0]["latent_shape"] == (1, 64, 4)
    assert transformer.calls[0]["cfg_scale"] == 5.0
    assert transformer.calls[0]["batch_cfg"] is True


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


def test_prismaudio_pipeline_uses_default_official_builder_when_available(monkeypatch):
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

    prismaudio_module = types.ModuleType("PrismAudio")
    prismaudio_models_module = types.ModuleType("PrismAudio.models")

    def _create_model_from_config(raw_model_config):
        assert raw_model_config["model_type"] == "diffusion_cond"
        return _OfficialStyleWrapper()

    prismaudio_models_module.create_model_from_config = _create_model_from_config
    prismaudio_module.models = prismaudio_models_module
    monkeypatch.setitem(sys.modules, "PrismAudio", prismaudio_module)
    monkeypatch.setitem(sys.modules, "PrismAudio.models", prismaudio_models_module)

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


def test_prismaudio_pipeline_surfaces_missing_dependency_from_default_official_builder(monkeypatch):
    import PrismAudio.models as official_models

    def _missing_dependency(_raw_model_config):
        raise ModuleNotFoundError("No module named 'dac'")

    monkeypatch.setattr(official_models, "create_model_from_config", _missing_dependency)

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
        },
    )

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")

    with pytest.raises(ModuleNotFoundError, match="official PrismAudio builder"):
        model_cls(od_config=od_config)


def test_prismaudio_pipeline_surfaces_numpy_compatibility_error_from_default_official_builder(monkeypatch):
    import numpy as np
    import PrismAudio.models as official_models

    monkeypatch.delattr(np, "float_", raising=False)

    def _builder(_raw_model_config):
        _ = np.float_
        raise AssertionError("unreachable")

    monkeypatch.setattr(official_models, "create_model_from_config", _builder)

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
        },
    )

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    with pytest.raises(RuntimeError, match="NumPy 2.x compatibility issue"):
        model_cls(od_config=od_config)


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


def test_prismaudio_pipeline_load_weights_supports_wrapper_transformer_and_split_vae_checkpoint(tmp_path):
    transformer_ckpt_path = tmp_path / "prismaudio.ckpt"
    vae_ckpt_path = tmp_path / "vae.ckpt"

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(2, 2, bias=False)
            self.pretransform = nn.Linear(2, 2, bias=False)

    wrapper = _Wrapper()
    expected_transformer_weight = torch.randn_like(wrapper.model.weight)
    expected_vae_weight = torch.randn_like(wrapper.pretransform.weight)
    torch.save({"model.weight": expected_transformer_weight}, transformer_ckpt_path)
    torch.save({"state_dict": {"autoencoder.weight": expected_vae_weight}}, vae_ckpt_path)

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_paths={
            "transformer": str(transformer_ckpt_path),
            "vae": str(vae_ckpt_path),
        },
        model_config={
            "prismaudio_model_config": {
                "sample_rate": 44100,
                "audio_channels": 2,
                "model": {
                    "io_channels": 64,
                },
            },
        },
    )

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(od_config=od_config, transformer=wrapper, vae=wrapper.pretransform)

    loaded = pipeline.load_weights([])

    assert torch.equal(pipeline.transformer.model.weight, expected_transformer_weight)
    assert torch.equal(pipeline.vae.weight, expected_vae_weight)
    assert "transformer.model.weight" in loaded
    assert "vae.weight" in loaded


def test_prismaudio_pipeline_od_config_owned_wrapper_smoke_path(tmp_path):
    import sys
    import types

    transformer_ckpt_path = tmp_path / "prismaudio.ckpt"
    vae_ckpt_path = tmp_path / "vae.ckpt"

    class _InnerTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.0))
            self.calls: list[dict[str, object]] = []

        def forward(self, x, t, **kwargs):
            self.calls.append(
                {
                    "latent_shape": tuple(x.shape),
                    "cross_attn_cond_shape": tuple(kwargs["cross_attn_cond"].shape),
                    "cross_attn_mask_shape": tuple(kwargs["cross_attn_mask"].shape),
                    "cfg_scale": kwargs["cfg_scale"],
                }
            )
            return torch.ones_like(x) * self.scale

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _OfficialStylePretransform(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def decode(self, latents):
            return _Decoded(latents[:, :2, :] * self.scale)

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

    module_name = "prismaudio_test_smoke_builders"
    builder_module = types.ModuleType(module_name)

    def _build_wrapper_from_raw_model_config(raw_model_config):
        assert raw_model_config["model_type"] == "diffusion_cond"
        return _OfficialStyleWrapper()

    builder_module.build_wrapper_from_raw_model_config = _build_wrapper_from_raw_model_config
    sys.modules[module_name] = builder_module

    torch.save({"model.scale": torch.tensor(2.0)}, transformer_ckpt_path)
    torch.save({"state_dict": {"autoencoder.scale": torch.tensor(3.0)}}, vae_ckpt_path)

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_paths={
            "transformer": str(transformer_ckpt_path),
            "vae": str(vae_ckpt_path),
        },
        model_config={
            "prismaudio_model_config": {
                "model_type": "diffusion_cond",
                "sample_rate": 44100,
                "sample_size": 397312,
                "audio_channels": 2,
                "model": {
                    "io_channels": 64,
                    "pretransform": {
                        "config": {
                            "latent_dim": 64,
                            "downsampling_ratio": 2048,
                        }
                    },
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

    loaded = pipeline.load_weights([])
    assert "transformer.model.scale" in loaded
    assert "vae.scale" in loaded
    assert pipeline.transformer.model.scale.item() == pytest.approx(2.0)
    assert pipeline.vae.scale.item() == pytest.approx(3.0)

    torch.manual_seed(0)
    req = _make_request(extra_args={"num_inference_steps": 2, "cfg_scale": 4.0})
    output = pipeline.forward(req)

    assert output.output.shape == (1, 2, 194)
    assert len(pipeline.transformer.model.calls) == 2
    assert pipeline.transformer.model.calls[0]["latent_shape"] == (1, 64, 194)
    assert pipeline.transformer.model.calls[0]["cross_attn_cond_shape"] == (1, 42, 1024)
    assert pipeline.transformer.model.calls[0]["cross_attn_mask_shape"] == (1, 42)
    assert pipeline.transformer.model.calls[0]["cfg_scale"] == 4.0


def test_prismaudio_pipeline_default_official_builder_smoke_path(tmp_path, monkeypatch):
    import sys
    import types

    transformer_ckpt_path = tmp_path / "prismaudio.ckpt"
    vae_ckpt_path = tmp_path / "vae.ckpt"

    class _InnerTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.0))
            self.calls: list[dict[str, object]] = []

        def forward(self, x, t, **kwargs):
            self.calls.append(
                {
                    "latent_shape": tuple(x.shape),
                    "cross_attn_cond_shape": tuple(kwargs["cross_attn_cond"].shape),
                    "cross_attn_mask_shape": tuple(kwargs["cross_attn_mask"].shape),
                    "cfg_scale": kwargs["cfg_scale"],
                }
            )
            return torch.ones_like(x) * self.scale

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _OfficialStylePretransform(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def decode(self, latents):
            return _Decoded(latents[:, :2, :] * self.scale)

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

    prismaudio_module = types.ModuleType("PrismAudio")
    prismaudio_models_module = types.ModuleType("PrismAudio.models")

    def _create_model_from_config(raw_model_config):
        assert raw_model_config["model_type"] == "diffusion_cond"
        return _OfficialStyleWrapper()

    prismaudio_models_module.create_model_from_config = _create_model_from_config
    prismaudio_module.models = prismaudio_models_module
    monkeypatch.setitem(sys.modules, "PrismAudio", prismaudio_module)
    monkeypatch.setitem(sys.modules, "PrismAudio.models", prismaudio_models_module)

    torch.save({"model.scale": torch.tensor(2.0)}, transformer_ckpt_path)
    torch.save({"state_dict": {"autoencoder.scale": torch.tensor(3.0)}}, vae_ckpt_path)

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_paths={
            "transformer": str(transformer_ckpt_path),
            "vae": str(vae_ckpt_path),
        },
        model_config={
            "prismaudio_model_config": {
                "model_type": "diffusion_cond",
                "sample_rate": 44100,
                "sample_size": 397312,
                "audio_channels": 2,
                "model": {
                    "io_channels": 64,
                    "pretransform": {
                        "config": {
                            "latent_dim": 64,
                            "downsampling_ratio": 2048,
                        }
                    },
                },
            },
        },
    )

    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    pipeline = model_cls(od_config=od_config)

    loaded = pipeline.load_weights([])
    assert "transformer.model.scale" in loaded
    assert "vae.scale" in loaded
    assert pipeline.transformer.model.scale.item() == pytest.approx(2.0)
    assert pipeline.vae.scale.item() == pytest.approx(3.0)

    torch.manual_seed(0)
    req = _make_request(extra_args={"num_inference_steps": 2, "cfg_scale": 4.0})
    output = pipeline.forward(req)

    assert output.output.shape == (1, 2, 194)
    assert len(pipeline.transformer.model.calls) == 2
    assert pipeline.transformer.model.calls[0]["latent_shape"] == (1, 64, 194)
    assert pipeline.transformer.model.calls[0]["cross_attn_cond_shape"] == (1, 42, 1024)
    assert pipeline.transformer.model.calls[0]["cross_attn_mask_shape"] == (1, 42)
    assert pipeline.transformer.model.calls[0]["cfg_scale"] == 4.0


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


def _build_real_official_pipeline_or_skip(od_config: OmniDiffusionConfig):
    model_cls = DiffusionModelRegistry._try_load_model_cls("PrismAudioPipeline")
    try:
        return model_cls(od_config=od_config)
    except ModuleNotFoundError as exc:
        pytest.skip(f"official PrismAudio builder dependency missing in test env: {exc}")
    except RuntimeError as exc:
        if "NumPy 2.x compatibility issue" in str(exc):
            pytest.skip(str(exc))
        raise


def test_prismaudio_pipeline_can_construct_real_official_builder_when_available():
    pytest.importorskip("PrismAudio.models")

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
                    "conditioning": {
                        "cond_dim": 128,
                        "configs": [
                            {
                                "id": "video_features",
                                "type": "cond_mlp",
                                "config": {
                                    "dim": 128,
                                    "output_dim": 128,
                                },
                            },
                            {
                                "id": "text_features",
                                "type": "cond_mlp",
                                "config": {
                                    "dim": 128,
                                    "output_dim": 128,
                                },
                            },
                            {
                                "id": "sync_features",
                                "type": "sync_mlp",
                                "config": {
                                    "dim": 128,
                                    "output_dim": 128,
                                },
                            },
                        ],
                    },
                    "diffusion": {
                        "type": "dit",
                        "cross_attention_cond_ids": ["video_features", "text_features"],
                        "add_cond_ids": ["video_features"],
                        "sync_cond_ids": ["sync_features"],
                        "config": {
                            "embed_dim": 128,
                            "depth": 2,
                            "num_heads": 2,
                            "io_channels": 64,
                            "cond_token_dim": 128,
                            "add_token_dim": 128,
                            "sync_token_dim": 128,
                            "project_cond_tokens": False,
                            "transformer_type": "continuous_transformer",
                            "use_gated": True,
                            "use_sync_gated": True,
                            "attn_kwargs": {"qk_norm": "rns"},
                        },
                    },
                },
            },
        },
    )

    pipeline = _build_real_official_pipeline_or_skip(od_config)

    assert type(pipeline.transformer).__module__.startswith("PrismAudio.models.")
    assert type(pipeline.transformer).__name__ == "ConditionedDiffusionModelWrapper"
    assert hasattr(pipeline.transformer, "conditioner")
    assert hasattr(pipeline.transformer, "model")
    assert pipeline.vae is None


def test_prismaudio_pipeline_can_run_real_official_builder_with_fake_vae_when_available():
    pytest.importorskip("PrismAudio.models")

    class _Decoded:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE(nn.Module):
        def decode(self, latents):
            return _Decoded(latents[:, :2, :])

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": {
                "model_type": "diffusion_cond",
                "sample_rate": 44100,
                "audio_channels": 2,
                "sample_size": 397312,
                "model": {
                    "io_channels": 64,
                    "conditioning": {
                        "cond_dim": 128,
                        "configs": [
                            {
                                "id": "video_features",
                                "type": "cond_mlp",
                                "config": {
                                    "dim": 128,
                                    "output_dim": 128,
                                },
                            },
                            {
                                "id": "text_features",
                                "type": "cond_mlp",
                                "config": {
                                    "dim": 128,
                                    "output_dim": 128,
                                },
                            },
                            {
                                "id": "sync_features",
                                "type": "sync_mlp",
                                "config": {
                                    "dim": 128,
                                    "output_dim": 128,
                                },
                            },
                        ],
                    },
                    "diffusion": {
                        "type": "dit",
                        "cross_attention_cond_ids": ["video_features", "text_features"],
                        "add_cond_ids": ["video_features"],
                        "sync_cond_ids": ["sync_features"],
                        "config": {
                            "embed_dim": 128,
                            "depth": 2,
                            "num_heads": 2,
                            "io_channels": 64,
                            "cond_token_dim": 128,
                            "add_token_dim": 128,
                            "sync_token_dim": 128,
                            "project_cond_tokens": False,
                            "transformer_type": "continuous_transformer",
                            "use_gated": True,
                            "use_sync_gated": True,
                            "attn_kwargs": {"qk_norm": "rns"},
                        },
                    },
                },
            },
        },
    )

    pipeline = _build_real_official_pipeline_or_skip(od_config)
    pipeline.vae = _FakeVAE()
    req = _make_request(
        additional_information={
            "video_features": torch.randn(1, 10, 128),
            "text_features": torch.randn(1, 32, 128),
            "sync_features": torch.randn(1, 216, 128),
        },
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.randn(1, 64, 194)

    output = pipeline.forward(req)

    assert output.error is None
    assert output.output is not None
    assert output.output.shape == (1, 2, 194)


def test_prismaudio_pipeline_can_construct_real_official_pretransform_when_available():
    pytest.importorskip("PrismAudio.models")

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(include_pretransform=True),
        },
    )

    pipeline = _build_real_official_pipeline_or_skip(od_config)

    assert type(pipeline.transformer).__module__.startswith("PrismAudio.models.")
    assert type(pipeline.vae).__module__.startswith("PrismAudio.models.")
    assert type(pipeline.vae).__name__ == "AutoencoderPretransform"
    assert pipeline.vae is getattr(pipeline.transformer, "pretransform", None)
    assert getattr(pipeline.vae, "downsampling_ratio", None) == 2048


def test_prismaudio_pipeline_can_run_real_official_pretransform_decode_when_available():
    pytest.importorskip("PrismAudio.models")

    od_config = OmniDiffusionConfig(
        model="prismaudio",
        model_class_name="PrismAudioPipeline",
        model_config={
            "prismaudio_model_config": _make_minimal_official_prismaudio_model_config(include_pretransform=True),
        },
    )

    pipeline = _build_real_official_pipeline_or_skip(od_config)
    req = _make_request(
        additional_information={
            "video_features": torch.randn(1, 10, 128),
            "text_features": torch.randn(1, 32, 128),
            "sync_features": torch.randn(1, 216, 128),
        },
        extra_args={"num_inference_steps": 1, "cfg_scale": 2.0},
    )
    req.sampling_params.latents = torch.randn(1, 64, 194)

    output = pipeline.forward(req)

    assert output.error is None
    assert output.output is not None
    assert output.output.shape == (1, 2, 397312)
