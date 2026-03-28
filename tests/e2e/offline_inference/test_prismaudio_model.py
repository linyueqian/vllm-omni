# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tests.utils import hardware_test
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


_DEFAULT_VENDOR_CONFIG = Path("/home/chen/vendor/ThinkSound-prismaudio/PrismAudio/configs/model_configs/prismaudio.json")
_ENV_TRANSFORMER_CKPT = "PRISMAUDIO_E2E_TRANSFORMER_CKPT"
_ENV_VAE_CKPT = "PRISMAUDIO_E2E_VAE_CKPT"
_ENV_FEATURES = "PRISMAUDIO_E2E_FEATURES"
_ENV_CONFIG = "PRISMAUDIO_E2E_CONFIG"
_ENV_STEPS = "PRISMAUDIO_E2E_NUM_STEPS"
_ENV_CFG_SCALE = "PRISMAUDIO_E2E_CFG_SCALE"
_ENV_SEED = "PRISMAUDIO_E2E_SEED"
_ENV_DTYPE = "PRISMAUDIO_E2E_DTYPE"


def _require_existing_path_from_env(env_name: str) -> Path:
    raw_value = os.environ.get(env_name)
    if not raw_value:
        pytest.skip(f"{env_name} is not set; skipping PrismAudio real-model e2e smoke test.")

    path = Path(raw_value).expanduser()
    if not path.exists():
        pytest.skip(f"{env_name} points to a missing path: {path}")
    return path


def _resolve_prismaudio_config_path() -> Path:
    configured_path = os.environ.get(_ENV_CONFIG)
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.exists():
            pytest.skip(f"{_ENV_CONFIG} points to a missing path: {path}")
        return path

    if not _DEFAULT_VENDOR_CONFIG.exists():
        pytest.skip(
            f"Default vendored PrismAudio config is missing at {_DEFAULT_VENDOR_CONFIG}; "
            f"set {_ENV_CONFIG} to a valid official config path."
        )
    return _DEFAULT_VENDOR_CONFIG


def _load_conditioning_fixture(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".npz":
        npz_data = np.load(path, allow_pickle=True)
        raw_data = {key: npz_data[key] for key in npz_data.files}
    else:
        raw_data = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(raw_data, dict):
        raise TypeError(f"PrismAudio conditioning fixture must decode to a dict, got {type(raw_data)!r}.")

    conditioning: dict[str, torch.Tensor] = {}
    for feature_name in ("video_features", "text_features", "sync_features"):
        value = raw_data.get(feature_name)
        if value is None:
            raise KeyError(
                f"PrismAudio conditioning fixture {path} is missing required key {feature_name!r}. "
                "Provide a fixture containing `video_features`, `text_features`, and `sync_features`."
            )
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"PrismAudio conditioning fixture key {feature_name!r} must be a tensor-like value, "
                f"got {type(value)!r}."
            )
        conditioning[feature_name] = value

    return conditioning


def _make_local_prismaudio_model_dir(tmp_path: Path, official_config_path: Path) -> Path:
    model_dir = tmp_path / "local-prismaudio-e2e"
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)

    model_index = {
        "_class_name": "PrismAudioPipeline",
        "_diffusers_version": "0.0.0",
    }
    (model_dir / "model_index.json").write_text(json.dumps(model_index), encoding="utf-8")

    official_config = json.loads(official_config_path.read_text())
    transformer_config = {
        "model_type": official_config.get("model_type", "diffusion_cond"),
        "sample_rate": official_config.get("sample_rate", 44100),
        "audio_channels": official_config.get("audio_channels", 2),
    }
    (transformer_dir / "config.json").write_text(json.dumps(transformer_config), encoding="utf-8")
    return model_dir


def _resolve_dtype() -> str:
    return os.environ.get(_ENV_DTYPE, "bfloat16")


@pytest.mark.asyncio
@hardware_test(res={"cuda": "L4"})
async def test_prismaudio_real_model_e2e_smoke(tmp_path: Path) -> None:
    transformer_ckpt = _require_existing_path_from_env(_ENV_TRANSFORMER_CKPT)
    vae_ckpt = _require_existing_path_from_env(_ENV_VAE_CKPT)
    features_path = _require_existing_path_from_env(_ENV_FEATURES)
    official_config_path = _resolve_prismaudio_config_path()

    if current_omni_platform.device_type != "cuda":
        pytest.skip("PrismAudio real-model smoke test currently requires CUDA.")

    conditioning = _load_conditioning_fixture(features_path)
    local_model_dir = _make_local_prismaudio_model_dir(tmp_path, official_config_path)
    model_config = json.loads(official_config_path.read_text())
    expected_sample_rate = int(model_config.get("sample_rate", 44100))
    expected_audio_channels = int(model_config.get("audio_channels", 2))
    expected_num_samples = int(model_config.get("sample_size", 0))

    init_start = time.perf_counter()
    diffusion = AsyncOmniDiffusion(
        model=str(local_model_dir),
        model_config={"prismaudio_model_config_path": str(official_config_path)},
        model_paths={
            "transformer": str(transformer_ckpt),
            "vae": str(vae_ckpt),
        },
        dtype=_resolve_dtype(),
        num_gpus=1,
    )
    init_elapsed_s = time.perf_counter() - init_start

    try:
        num_inference_steps = int(os.environ.get(_ENV_STEPS, "4"))
        cfg_scale = float(os.environ.get(_ENV_CFG_SCALE, "5.0"))
        seed = int(os.environ.get(_ENV_SEED, "42"))
        sampling_params = OmniDiffusionSamplingParams(
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
            num_outputs_per_prompt=1,
            extra_args={
                "cfg_scale": cfg_scale,
            },
        )

        prompt = {
            "prompt": "PrismAudio e2e smoke request",
            "additional_information": conditioning,
        }

        inference_start = time.perf_counter()
        result = await diffusion.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id="prismaudio-e2e-smoke",
        )
        inference_elapsed_s = time.perf_counter() - inference_start
    finally:
        diffusion.close()

    assert isinstance(result, OmniRequestOutput)
    assert result.final_output_type == "audio"

    audio = result.multimodal_output.get("audio")
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 3
    assert audio.shape[0] == 1
    assert audio.shape[1] == expected_audio_channels
    assert audio.shape[2] > 0
    if expected_num_samples > 0:
        assert audio.shape[2] == expected_num_samples

    print(
        "\n[PrismAudio E2E]"
        f" init_s={init_elapsed_s:.2f}"
        f" inference_s={inference_elapsed_s:.2f}"
        f" steps={num_inference_steps}"
        f" cfg_scale={cfg_scale:.2f}"
        f" sample_rate={expected_sample_rate}"
        f" audio_shape={tuple(audio.shape)}"
        f" peak_memory_mb={result.peak_memory_mb:.2f}"
    )
