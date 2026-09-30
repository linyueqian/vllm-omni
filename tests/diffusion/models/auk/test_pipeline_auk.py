# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK stage-1 pipeline: request parsing on CPU, generation parity on a GPU.

The CPU tests stub the transformer, the codec and the ODE so they exercise only
what the pipeline itself owns: how a request turns into a target length, a
sampling schedule and a noise generator.

The parity test replays the saved upstream reference for the zero-shot TTS case
and reports latent, per-frame and waveform distances rather than asserting bit
equality, because two differences live in the conditioning rather than in the
port: the saved text condition is an fp32 thinker forward while the reference
generated under bf16 autocast, and the reference's stochastic VAE draw came from
the process-global RNG seeded at 1234.

It replays with ``vae_sample=False``. That is the deliberate choice: the
reference re-seeds the global RNG immediately before drawing the ODE noise, so a
per-request generator reproduces that noise exactly (verified bit-identical) as
long as nothing else has drawn from it. Sampling the VAE posterior consumes the
generator first, which moves the noise and produces a different, equally valid
realization; comparing latents then measures the noise, not the port. The
posterior mean costs a reference latent that is 0.045 MSE (0.969 cosine) away
from the draw the reference used, which is the smaller of the two errors.
Generate the reference artifacts once with the upstream package installed
(``tools/auk_parity_reference.py --auk-repo ... --ckpt-dir ... --qwen-dir ...
--out /path/to/auk-parity``; it writes ``parity_ref/<variant>`` and
``fusion_ref``), then run::

    AUK_OMNI_CKPT_DIR=/path/to/auk-omni-base \
    AUK_PARITY_REF=/path/to/auk-parity/parity_ref/base \
    python -m pytest -s tests/diffusion/models/auk/test_pipeline_auk.py

``AUK_OMNI_CKPT_DIR`` is the assembled directory this pipeline loads, which is
not the same thing as the released AuK snapshot that the transformer and codec
parity tests in this directory read from ``AUK_CKPT_DIR``. ``AUK_CKPT_DIR`` is
accepted as a fallback, and the test skips rather than fails when the directory
it resolves to is not an assembled one.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.auk import pipeline_auk
from vllm_omni.diffusion.models.auk.batching import AuKBatchRequest
from vllm_omni.diffusion.models.auk.pipeline_auk import AuKPipeline
from vllm_omni.diffusion.models.interface import supports_step_execution
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

LATENT_DIM = 64
HOP = 480
SAMPLE_RATE = 24000
TEXT_HIDDEN_DIM = 2048

# The distilled student's grid, exact values from the released checkpoint.
FLASH_T_GRID = [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]

CKPT_DIR = os.environ.get("AUK_OMNI_CKPT_DIR") or os.environ.get("AUK_CKPT_DIR")
PARITY_REF = os.environ.get("AUK_PARITY_REF")
# An assembled directory is the only thing this pipeline can load.
_stub_uids = itertools.count()
IS_ASSEMBLED = bool(CKPT_DIR) and (Path(CKPT_DIR).expanduser() / "config.json").is_file()

# Measured on the released base checkpoint: mean frame cosine 0.961 to 0.968,
# relative latent MSE 0.051 to 0.063, log-mel distance 0.34. That spread is
# bf16 against fp32 DiT weights plus where the codec folds its weight norm,
# neither of which is worth gating on. A port that has lost the conditioning
# sits at 0.110 cosine, 1.82 relative MSE and 2.88 log-mel, so these gates
# still separate the two by an order of magnitude.
PARITY_MIN_FRAME_COSINE = 0.93
PARITY_MAX_RELATIVE_MSE = 0.15
PARITY_MAX_MEL_DISTANCE = 0.6


class _StubTransformer(nn.Module):
    """Records its construction arguments; owns one weight so strict loading runs."""

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: float,
        latent_dim: int,
        text_hidden_dim: int,
        num_layers: int,
        num_single_layers: int,
        attn_mask_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.init_kwargs = {
            "dim": dim,
            "heads": heads,
            "dim_head": dim_head,
            "ff_mult": ff_mult,
            "latent_dim": latent_dim,
            "text_hidden_dim": text_hidden_dim,
            "num_layers": num_layers,
            "num_single_layers": num_single_layers,
            "attn_mask_enabled": attn_mask_enabled,
        }
        self.latent_dim = latent_dim
        self.proj = nn.Linear(latent_dim, latent_dim, bias=False)


class _StubVAE(nn.Module):
    """Deterministic stand-in for the codec, one latent frame per hop."""

    hop_size = HOP
    sample_rate = SAMPLE_RATE
    latent_dim = LATENT_DIM

    def __init__(self) -> None:
        super().__init__()
        self.encode_calls: list[dict[str, Any]] = []
        self.decode_calls = 0
        self.weights_path: str | None = None
        self.config: dict[str, Any] = {}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> _StubVAE:
        vae = cls()
        vae.config = dict(cfg)
        return vae

    def load_weights(self, path: str, **_: Any) -> tuple[list[str], list[str]]:
        self.weights_path = str(path)
        return [], []

    def encode(
        self,
        wav: torch.Tensor,
        *,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        frames = wav.shape[-1] // HOP
        self.encode_calls.append({"samples": int(wav.shape[-1]), "sample": sample, "generator": generator})
        if sample:
            return torch.randn(1, frames, LATENT_DIM, generator=generator, device=wav.device)
        return torch.zeros(1, frames, LATENT_DIM, device=wav.device)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.decode_calls += 1
        return torch.zeros(1, latents.shape[1] * HOP, device=latents.device)

    def decode_context_frames(self) -> tuple[int, int]:
        # A stateless stub: no context, so any tile size is valid.
        return 0, 0


def _stub_sampler(calls: list[dict[str, Any]]):
    """Record the ODE arguments and return noise drawn from the request generator."""

    def sample_latents(dit: nn.Module, **kwargs: Any) -> torch.Tensor:
        calls.append(kwargs)
        shape = (1, kwargs["gen_frames"], kwargs["latent_dim"])
        generator = kwargs.get("generator")
        if generator is None:
            return torch.zeros(*shape, device=kwargs["device"], dtype=kwargs["dtype"])
        return torch.randn(*shape, generator=generator, device=kwargs["device"], dtype=kwargs["dtype"])

    return sample_latents


def _write_checkpoint(root: Path, variant: str) -> Path:
    """Write a tiny assembled checkpoint the stubs can load."""

    from safetensors.torch import save_file

    model_dir = root / f"auk-{variant}"
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "auk",
        "architectures": ["AuKForConditionalGeneration"],
        "dit": {
            "dim": 16,
            "heads": 2,
            "dim_head": 8,
            "ff_mult": 2,
            "text_hidden_dim": TEXT_HIDDEN_DIM,
            "num_layers": 1,
            "num_single_layers": 1,
            "attn_mask_enabled": True,
        },
        "vae": {
            "latent_dim": LATENT_DIM,
            "downsample_rate": HOP,
            "target_sample_rate": SAMPLE_RATE,
            "model_init_kwargs": {"latent_dim": LATENT_DIM},
        },
        "variant": variant,
        "flash_t_grid": FLASH_T_GRID,
        "defaults": {"nfe": 32, "cfg": 2.0, "sway": -1.0},
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    save_file(
        {
            "transformer.proj.weight": torch.zeros(LATENT_DIM, LATENT_DIM),
            "layer_weights": torch.zeros(36),
            "layer_scale": torch.ones(1),
        },
        str(model_dir / "auk.safetensors"),
    )
    save_file({"global_mean": torch.zeros(LATENT_DIM)}, str(model_dir / "vae.safetensors"))
    return model_dir


@pytest.fixture
def build_pipeline(tmp_path, monkeypatch):
    """Build an AuKPipeline whose transformer, codec and ODE are stubs."""

    monkeypatch.setattr(pipeline_auk, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline_auk, "AuKTransformer", _StubTransformer)
    monkeypatch.setattr(pipeline_auk, "AuKVAE", _StubVAE)

    def _build(
        variant: str = "base", *, model_config: dict[str, Any] | None = None
    ) -> tuple[AuKPipeline, list[dict[str, Any]]]:
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(pipeline_auk, "sample_latents", _stub_sampler(calls))
        od_config = OmniDiffusionConfig(
            model=str(_write_checkpoint(tmp_path, variant)),
            dtype=torch.float32,
            model_class_name="AuKPipeline",
            model_config=model_config or {},
        )
        return AuKPipeline(od_config=od_config), calls

    return _build


def _prompt(
    *,
    tokens: int = 8,
    audio: Any = None,
    knobs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the diffusion prompt the stage input processor would emit."""

    prompt: dict[str, Any] = {
        "prompt": "",
        "prompt_embeds": torch.zeros(tokens, TEXT_HIDDEN_DIM),
        "additional_information": {"auk": dict(knobs or {})},
    }
    if audio is not None:
        prompt["multi_modal_data"] = {"audio": audio}
    return prompt


def _batch(prompt: dict[str, Any], **sampling: Any) -> DiffusionRequestBatch:
    request = OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(**sampling),
        request_id="auk-test-0",
    )
    return DiffusionRequestBatch(requests=[request])


def _silence(seconds: float, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32), sample_rate


@pytest.mark.core_model
@pytest.mark.cpu
class TestRequestParsing:
    @pytest.mark.parametrize(
        "model_config, expected_slots",
        [({}, 32), ({"max_dit_graphs": 1}, 1), ({"max_dit_graphs": 3}, 3), ({"max_dit_graphs": 64}, 64)],
    )
    def test_max_dit_graphs_reaches_lazy_cache(self, build_pipeline, model_config, expected_slots):
        pipeline, _ = build_pipeline(model_config=model_config)
        assert pipeline.cudagraph_wrapper.max_graphs == expected_slots
        assert not pipeline.cudagraph_wrapper._cache

    @pytest.mark.parametrize("max_dit_graphs", [0, -1, True, False, 1.5, "3", None])
    def test_invalid_max_dit_graphs_is_rejected(self, build_pipeline, max_dit_graphs):
        with pytest.raises(ValueError, match="AuK max_dit_graphs must be a positive integer"):
            build_pipeline(model_config={"max_dit_graphs": max_dit_graphs})

    def test_pipeline_declares_audio_output(self, build_pipeline):
        pipeline, _ = build_pipeline()

        assert pipeline.support_audio_output is True
        assert pipeline.audio_sample_rate == SAMPLE_RATE
        assert pipeline.supports_request_batch is True
        # The warmup request cannot carry an encoder-stage text condition.
        assert pipeline.dummy_run_num_frames == 0

    def test_config_reaches_the_transformer(self, build_pipeline):
        pipeline, _ = build_pipeline()

        assert pipeline.dit.init_kwargs["latent_dim"] == LATENT_DIM
        assert pipeline.dit.init_kwargs["text_hidden_dim"] == TEXT_HIDDEN_DIM
        assert pipeline.dit.init_kwargs["attn_mask_enabled"] is True

    def test_gen_seconds_sets_the_target_length(self, build_pipeline):
        pipeline, calls = build_pipeline()

        outputs = pipeline.forward(_batch(_prompt(audio=_silence(2.0), knobs={"gen_seconds": 6.0}), seed=1))

        assert calls[0]["gen_frames"] == math.ceil(6.0 * SAMPLE_RATE / HOP)
        assert calls[0]["ref"].shape == (1, 100, LATENT_DIM)
        assert calls[0]["ref_mask"].shape == (1, 100)
        assert calls[0]["c_mask"].shape == (1, 8)
        assert calls[0]["sampler"] is pipeline.cudagraph_wrapper
        assert outputs[0].output.shape == (300 * HOP,)
        assert outputs[0].output.dtype is torch.float32

    def test_target_length_falls_back_to_the_source_clip(self, build_pipeline):
        pipeline, calls = build_pipeline()

        pipeline.forward(_batch(_prompt(audio=_silence(2.0), knobs={"gen_seconds": None}), seed=1))

        assert calls[0]["gen_frames"] == 100

    def test_text_only_request_without_a_duration_is_rejected(self, build_pipeline):
        pipeline, calls = build_pipeline()

        with pytest.raises(ValueError, match="gen_seconds"):
            pipeline.forward(_batch(_prompt(knobs={"gen_seconds": None}), seed=1))
        assert calls == []

    def test_text_only_request_gets_an_empty_reference(self, build_pipeline):
        pipeline, calls = build_pipeline()

        pipeline.forward(_batch(_prompt(knobs={"gen_seconds": 3.5}), seed=1))

        assert calls[0]["ref"].shape == (1, 0, LATENT_DIM)
        assert calls[0]["gen_frames"] == 175

    def test_source_clip_is_resampled_to_the_codec_rate(self, build_pipeline):
        pipeline, _ = build_pipeline()

        pipeline.forward(_batch(_prompt(audio=_silence(1.0, 16000), knobs={"gen_seconds": 1.0}), seed=1))

        assert pipeline.vae.encode_calls[0]["samples"] == SAMPLE_RATE

    def test_stereo_source_is_mixed_to_mono(self, build_pipeline):
        pipeline, _ = build_pipeline()
        stereo = np.zeros((2, SAMPLE_RATE), dtype=np.float32)

        pipeline.forward(_batch(_prompt(audio=(stereo, SAMPLE_RATE), knobs={"gen_seconds": 1.0}), seed=1))

        assert pipeline.vae.encode_calls[0]["samples"] == SAMPLE_RATE

    def test_missing_text_condition_is_rejected(self, build_pipeline):
        pipeline, _ = build_pipeline()
        prompt = _prompt(knobs={"gen_seconds": 1.0})
        del prompt["prompt_embeds"]

        with pytest.raises(ValueError, match="prompt_embeds"):
            pipeline.forward(_batch(prompt, seed=1))

    def test_base_variant_uses_the_requested_schedule(self, build_pipeline):
        pipeline, calls = build_pipeline()

        pipeline.forward(
            _batch(
                _prompt(audio=_silence(1.0), knobs={"gen_seconds": 1.0, "sway": -1.0}),
                seed=1,
                num_inference_steps=16,
                guidance_scale=3.0,
            )
        )

        assert calls[0]["nfe"] == 16
        assert calls[0]["cfg_strength"] == 3.0
        assert calls[0]["sway_sampling_coef"] == -1.0
        assert calls[0]["t_grid"] is None

    def test_schedule_defaults_come_from_the_checkpoint(self, build_pipeline):
        pipeline, calls = build_pipeline()

        pipeline.forward(_batch(_prompt(audio=_silence(1.0), knobs={"gen_seconds": 1.0, "sway": None}), seed=1))

        assert calls[0]["nfe"] == 32
        assert calls[0]["cfg_strength"] == 2.0
        assert calls[0]["sway_sampling_coef"] == -1.0

    def test_flash_variant_locks_the_distilled_recipe(self, build_pipeline):
        pipeline, calls = build_pipeline("flash")

        pipeline.forward(
            _batch(
                _prompt(audio=_silence(1.0), knobs={"gen_seconds": 1.0, "sway": -1.0}),
                seed=1,
                num_inference_steps=32,
                guidance_scale=2.0,
            )
        )

        assert calls[0]["nfe"] == 4
        assert calls[0]["cfg_strength"] == 0.0
        assert calls[0]["sway_sampling_coef"] is None
        assert calls[0]["t_grid"] == FLASH_T_GRID

    def test_noise_comes_from_a_seeded_request_generator(self, build_pipeline):
        pipeline, calls = build_pipeline()

        outputs = pipeline.forward(
            _batch(_prompt(knobs={"gen_seconds": 1.0}), seed=0, output_type="latent"),
        )

        expected = torch.randn(1, 50, LATENT_DIM, generator=torch.Generator().manual_seed(0))
        assert torch.equal(outputs[0].output, expected)
        # The global RNG is never seeded on the pipeline's behalf.
        assert calls[0]["seed"] is None
        assert isinstance(calls[0]["generator"], torch.Generator)

    def test_setup_compile_warms_the_decode_buckets_on_the_pipeline_device(self, build_pipeline, mocker):
        pipeline, _ = build_pipeline()
        warmup = mocker.patch.object(pipeline.vae_decode, "warmup")
        regional = mocker.patch.object(pipeline_auk, "regionally_compile")
        mocker.patch.object(pipeline, "_warmup_dit")

        pipeline.setup_compile()

        warmup.assert_called_once_with(pipeline.device)
        # Regional is the default: the DiT's repeated blocks are compiled.
        regional.assert_called_once_with(pipeline.dit, dynamic=pipeline.od_config.diffusion_compile_dynamic)

    def test_setup_compile_honours_full_dit_granularity(self, build_pipeline, mocker):
        pipeline, _ = build_pipeline()
        pipeline.od_config.diffusion_compile_granularity = "full"
        pipeline.od_config.diffusion_compile_dynamic = False
        mocker.patch.object(pipeline.vae_decode, "warmup")
        mocker.patch.object(pipeline, "_warmup_dit")
        step = mocker.Mock()
        pipeline.dit.step = step
        compile_fn = mocker.patch.object(pipeline_auk.torch, "compile", side_effect=lambda fn, **_: fn)

        pipeline.setup_compile()

        # The samplers call step() directly, so that is what gets compiled.
        compile_fn.assert_called_once_with(step, dynamic=False)

    @pytest.mark.parametrize("variant, cfg", [("base", 2.0), ("flash", 0.0)])
    def test_setup_compile_warms_the_dit_graph_buckets(self, build_pipeline, mocker, variant, cfg):
        pipeline, _ = build_pipeline(variant)
        mocker.patch.object(pipeline.vae_decode, "warmup")
        mocker.patch.object(pipeline_auk, "regionally_compile")
        wrapper = mocker.Mock(enabled=True)
        pipeline.cudagraph_wrapper = wrapper

        pipeline.setup_compile()

        shapes = [(call.kwargs["x"].shape[1], call.kwargs["ref"].shape[1]) for call in wrapper.call_args_list]
        assert shapes == [(150, 0), (300, 0), (600, 0), (150, 150), (300, 150), (600, 150)]
        for call in wrapper.call_args_list:
            assert call.kwargs["cfg_strength"] == cfg
            assert call.kwargs["new_request"] is True
            assert call.kwargs["text"].shape == (1, 96, TEXT_HIDDEN_DIM)

    @pytest.mark.parametrize("model_config, enabled", [({"auk_dit_warmup_frames": []}, True), ({}, False)])
    def test_dit_warmup_can_be_skipped(self, build_pipeline, mocker, model_config, enabled):
        pipeline, _ = build_pipeline(model_config=model_config)
        wrapper = mocker.Mock(enabled=enabled)
        pipeline.cudagraph_wrapper = wrapper

        pipeline._warmup_dit()

        wrapper.assert_not_called()

    def test_latent_output_type_skips_the_decoder(self, build_pipeline):
        pipeline, _ = build_pipeline()

        outputs = pipeline.forward(_batch(_prompt(knobs={"gen_seconds": 1.0}), seed=1, output_type="latent"))

        assert outputs[0].output.shape == (1, 50, LATENT_DIM)
        assert pipeline.vae.decode_calls == 0

    def test_vae_sample_draws_from_the_request_generator(self, build_pipeline):
        pipeline, _ = build_pipeline()

        pipeline.forward(_batch(_prompt(audio=_silence(1.0), knobs={"gen_seconds": 1.0, "vae_sample": True}), seed=1))

        call = pipeline.vae.encode_calls[0]
        assert call["sample"] is True
        assert isinstance(call["generator"], torch.Generator)

    def test_reference_latents_are_cached_by_content(self, build_pipeline):
        pipeline, calls = build_pipeline()
        clip = _silence(1.0)

        for _ in range(3):
            pipeline.forward(_batch(_prompt(audio=clip, knobs={"gen_seconds": 1.0}), seed=1))
        # A different clip is a new entry, the first one still hits.
        other = (np.full_like(clip[0], 0.25), clip[1])
        pipeline.forward(_batch(_prompt(audio=other, knobs={"gen_seconds": 1.0}), seed=1))
        pipeline.forward(_batch(_prompt(audio=clip, knobs={"gen_seconds": 1.0}), seed=1))

        assert len(pipeline.vae.encode_calls) == 2
        assert calls[0]["ref"] is calls[1]["ref"] is calls[4]["ref"]

    def test_posterior_draws_and_disabled_cache_always_encode(self, build_pipeline):
        pipeline, _ = build_pipeline()
        for _ in range(2):
            pipeline.forward(
                _batch(_prompt(audio=_silence(1.0), knobs={"gen_seconds": 1.0, "vae_sample": True}), seed=1)
            )
        assert len(pipeline.vae.encode_calls) == 2

        pipeline, _ = build_pipeline(model_config={"auk_ref_cache_size": 0})
        for _ in range(2):
            pipeline.forward(_batch(_prompt(audio=_silence(1.0), knobs={"gen_seconds": 1.0}), seed=1))
        assert len(pipeline.vae.encode_calls) == 2

    def test_reference_cache_is_bounded(self, build_pipeline):
        pipeline, _ = build_pipeline(model_config={"auk_ref_cache_size": 2})
        for level in (0.1, 0.2, 0.3):
            clip = (np.full(SAMPLE_RATE, level, dtype=np.float32), SAMPLE_RATE)
            pipeline.forward(_batch(_prompt(audio=clip, knobs={"gen_seconds": 1.0}), seed=1))
        assert len(pipeline._ref_cache) == 2


class _StubBatchRunner:
    """Batched DiT stand-in: every velocity is ``value``; records each call's membership."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self.calls: list[list[int]] = []

    def make_request(self, *, text, ref, cfg, grid, latents) -> AuKBatchRequest:
        return AuKBatchRequest(
            uid=next(_stub_uids),
            text=text,
            ref=ref,
            c=text,
            prompt=None,
            prompt_uncond=None,
            cfg=cfg,
            grid=grid,
            modulation=torch.zeros(grid.numel() - 1, 1),
            latents=latents,
        )

    def velocities(self, requests) -> list[torch.Tensor]:
        self.calls.append([request.uid for request in requests])
        return [torch.full_like(request.latents, self.value) for request in requests]


class _StubSingleStep:
    """Single-request graph stand-in recording whether each call reloaded its context."""

    enabled = True

    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self.new_request: list[bool] = []

    def __call__(self, *, x, new_request, **_: Any) -> torch.Tensor:
        self.new_request.append(new_request)
        return torch.full_like(x, self.value)


def _multi_batch(*requests: tuple[dict[str, Any], dict[str, Any]]) -> DiffusionRequestBatch:
    return DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt=prompt,
                sampling_params=OmniDiffusionSamplingParams(**sampling),
                request_id=f"auk-test-{i}",
            )
            for i, (prompt, sampling) in enumerate(requests)
        ]
    )


def _step_state(request_id: str, prompt: dict[str, Any], **sampling: Any) -> StepRequestState:
    return StepRequestState(request_id=request_id, sampling=OmniDiffusionSamplingParams(**sampling), prompt=prompt)


def _run_step_waves(pipeline: AuKPipeline, states: list[StepRequestState]) -> None:
    """Drive the step protocol the way the model runner does, one wave per step."""
    running = list(states)
    while running:
        input_batch = InputBatch.make_batch(running)
        noise_pred = pipeline.denoise_step(input_batch, states=running)
        offset = 0
        for state in running:
            rows = state.latents.shape[0]
            pipeline.step_scheduler(state, noise_pred[offset : offset + rows])
            offset += rows
        assert offset == noise_pred.shape[0]
        running = [state for state in running if not state.denoise_completed]


@pytest.mark.core_model
@pytest.mark.cpu
class TestRequestBatching:
    def test_pipeline_supports_both_batching_modes(self, build_pipeline):
        pipeline, _ = build_pipeline()

        assert pipeline.supports_request_batch is True
        assert supports_step_execution(pipeline)

    def test_batched_forward_runs_each_request_on_its_own_grid(self, build_pipeline):
        pipeline, calls = build_pipeline()
        pipeline.batch_runner = runner = _StubBatchRunner(value=1.0)

        outputs = pipeline.forward(
            _multi_batch(
                (_prompt(knobs={"gen_seconds": 1.0}), {"seed": 1, "num_inference_steps": 3, "output_type": "latent"}),
                (_prompt(knobs={"gen_seconds": 2.0}), {"seed": 2, "num_inference_steps": 5, "output_type": "latent"}),
            )
        )

        # Both requests share three steps; the longer grid then runs alone.
        assert [len(members) for members in runner.calls] == [2, 2, 2, 1, 1]
        assert not calls, "the single-request sampler must not run for a batch"
        for output, (seed, frames) in zip(outputs, [(1, 50), (2, 100)]):
            noise = torch.randn(1, frames, LATENT_DIM, generator=torch.Generator().manual_seed(seed))
            # A constant unit velocity integrates to noise + 1 over [0, 1].
            torch.testing.assert_close(output.output, noise + 1.0)

    def test_batched_forward_isolates_a_failing_request(self, build_pipeline):
        pipeline, _ = build_pipeline()
        pipeline.batch_runner = _StubBatchRunner()
        broken = _prompt(knobs={"gen_seconds": 1.0})
        broken.pop("prompt_embeds")

        outputs = pipeline.forward(
            _multi_batch(
                (broken, {"seed": 1, "output_type": "latent"}),
                (_prompt(knobs={"gen_seconds": 1.0}), {"seed": 2, "output_type": "latent"}),
            )
        )

        assert len(outputs) == 2
        assert "prompt_embeds" in outputs[0].error
        assert outputs[1].error is None
        assert outputs[1].output.shape == (1, 50, LATENT_DIM)

    def test_step_execution_batches_requests_and_matches_the_integral(self, build_pipeline):
        pipeline, _ = build_pipeline()
        pipeline.batch_runner = runner = _StubBatchRunner(value=1.0)
        states = [
            _step_state("a", _prompt(knobs={"gen_seconds": 1.0}), seed=1, num_inference_steps=2, output_type="latent"),
            _step_state("b", _prompt(knobs={"gen_seconds": 2.0}), seed=2, num_inference_steps=4, output_type="latent"),
        ]
        for state in states:
            pipeline.prepare_encode(state)
        # The runner stacks latents by rows: a request's rows are its frames.
        assert [tuple(state.latents.shape) for state in states] == [(50, LATENT_DIM), (100, LATENT_DIM)]
        assert [state.total_steps for state in states] == [2, 4]

        pipeline.cudagraph_wrapper = single = _StubSingleStep(value=1.0)
        _run_step_waves(pipeline, states)

        # Two batched waves, then the survivor steps alone through the single-request graph.
        assert [len(members) for members in runner.calls] == [2, 2]
        assert single.new_request == [True, False]
        for state, (seed, frames) in zip(states, [(1, 50), (2, 100)]):
            output = pipeline.post_decode(state)
            noise = torch.randn(1, frames, LATENT_DIM, generator=torch.Generator().manual_seed(seed))
            torch.testing.assert_close(output.output, noise + 1.0)
            assert "auk" not in state.extra

    def test_single_step_reloads_its_graph_context_when_the_request_changes(self, build_pipeline):
        pipeline, _ = build_pipeline()
        pipeline.batch_runner = _StubBatchRunner()
        pipeline.cudagraph_wrapper = single = _StubSingleStep()
        first, second = (
            _step_state(name, _prompt(knobs={"gen_seconds": 1.0}), seed=i, num_inference_steps=3)
            for i, name in enumerate("ab")
        )
        for state in (first, second):
            pipeline.prepare_encode(state)

        for state in (first, first, second, first):
            noise_pred = pipeline.denoise_step(InputBatch.make_batch([state]), states=[state])
            pipeline.step_scheduler(state, noise_pred)

        assert single.new_request == [True, False, True, True]


def _reference_audio_path(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        for item in message.get("content") or ():
            if isinstance(item, dict) and item.get("type") == "audio":
                path = item.get("audio") or item.get("audio_url")
                if path:
                    return str(path)
    raise AssertionError("the saved reference case carries no source clip")


def _log_mel_distance(left: torch.Tensor, right: torch.Tensor, sample_rate: int) -> float:
    """Mean absolute log-mel difference between two mono waveforms."""

    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=80)
    length = min(left.shape[-1], right.shape[-1])
    spectra = [torch.log(mel(wav[..., :length].float()) + 1e-5) for wav in (left, right)]
    return float((spectra[0] - spectra[1]).abs().mean())


@pytest.mark.local_model
@pytest.mark.diffusion
@pytest.mark.skipif(
    not (IS_ASSEMBLED and PARITY_REF),
    reason="needs AUK_PARITY_REF and AUK_OMNI_CKPT_DIR pointing at an assembled checkpoint directory",
)
def test_zero_shot_tts_tracks_the_upstream_reference():
    """Replay the saved zero-shot TTS case and report the distances."""

    parity_dir = Path(PARITY_REF)
    fusion_dir = Path(os.environ.get("AUK_FUSION_REF") or parity_dir.parent.parent / "fusion_ref")
    reference = torch.load(parity_dir / "zs_tts_en.pt", map_location="cpu", weights_only=False)
    fusion = torch.load(fusion_dir / "zs_tts_audio.pt", map_location="cpu", weights_only=False)

    # soundfile rather than torchaudio.load: the repo's test convention, and
    # torchaudio's decoder needs an ffmpeg the port environment does not have.
    samples, source_rate = sf.read(_reference_audio_path(reference["messages"]), dtype="float32", always_2d=True)
    source = torch.from_numpy(samples).transpose(0, 1)
    knobs = {"gen_seconds": reference["gen_seconds"], "sway": -1.0, "t_grid": None, "vae_sample": False}
    prompt = {
        "prompt": "",
        # The encoder stage emits bf16; the saved fusion is an fp32 forward.
        "prompt_embeds": fusion["fused"].to(torch.bfloat16),
        "multi_modal_data": {"audio": (source, int(source_rate))},
        "additional_information": {"auk": knobs},
    }
    od_config = OmniDiffusionConfig(model=CKPT_DIR, dtype=torch.bfloat16, model_class_name="AuKPipeline")
    pipeline = AuKPipeline(od_config=od_config)

    def run(output_type: str) -> torch.Tensor:
        batch = _batch(
            dict(prompt),
            seed=reference["seed"],
            num_inference_steps=32,
            guidance_scale=2.0,
            output_type=output_type,
        )
        return pipeline.forward(batch)[0].output

    # Two deterministic runs: the ODE is seeded, so the latents belong to the
    # waveform even though the decode happens in the second call.
    latents = run("latent")
    waveform = run("pt")

    ref_frames = int(reference["ref_latent_lens"][0])
    expected = reference["generated"][:, ref_frames:, :].float()
    assert latents.shape == expected.shape
    assert torch.isfinite(latents).all()
    assert torch.isfinite(waveform).all()
    assert waveform.shape[-1] == expected.shape[1] * HOP

    cosine = F.cosine_similarity(latents[0], expected[0], dim=-1)
    metrics = {
        "gen_frames": int(expected.shape[1]),
        "ref_frames": ref_frames,
        "latent_mse": float(torch.mean((latents - expected) ** 2)),
        "latent_relative_mse": float(torch.mean((latents - expected) ** 2) / expected.var()),
        "frame_cosine_mean": float(cosine.mean()),
        "frame_cosine_min": float(cosine.min()),
        "log_mel_distance": _log_mel_distance(waveform, reference["audio"].reshape(-1), int(reference["sr"])),
        "rms_port": float(waveform.pow(2).mean().sqrt()),
        "rms_reference": float(reference["audio"].reshape(-1).pow(2).mean().sqrt()),
    }
    print("\nAuK zero-shot TTS parity:", json.dumps(metrics, indent=1))

    assert metrics["frame_cosine_mean"] > PARITY_MIN_FRAME_COSINE
    assert metrics["latent_relative_mse"] < PARITY_MAX_RELATIVE_MSE
    assert metrics["log_mel_distance"] < PARITY_MAX_MEL_DISTANCE
