# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from os import PathLike
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors_file
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest


@dataclass
class PrismAudioCheckpointLoadReport:
    missing_keys: list[str]
    unexpected_keys: list[str]
    loaded_keys: list[str]


def load_prismaudio_state_dict(checkpoint_path: str | PathLike[str]) -> dict[str, torch.Tensor]:
    checkpoint_path = str(checkpoint_path)
    if checkpoint_path.endswith(".safetensors"):
        raw_state = load_safetensors_file(checkpoint_path)
    else:
        raw_state = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(raw_state, dict):
        raise TypeError(
            f"PrismAudio checkpoint at {checkpoint_path!r} must deserialize to a dict, "
            f"but received {type(raw_state)!r}."
        )

    state_dict = raw_state.get("state_dict", raw_state)
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"PrismAudio checkpoint at {checkpoint_path!r} must contain a dict-like `state_dict`, "
            f"but received {type(state_dict)!r}."
        )

    return state_dict


def _strip_prefix_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefix: str | None = None,
) -> dict[str, torch.Tensor]:
    if not prefix:
        return dict(state_dict)

    prefix_len = len(prefix)
    return {name[prefix_len:]: tensor for name, tensor in state_dict.items() if name.startswith(prefix)}


def load_module_from_prismaudio_checkpoint(
    module: nn.Module,
    checkpoint_path: str | PathLike[str],
    *,
    prefix: str | None = None,
    strict: bool = False,
) -> PrismAudioCheckpointLoadReport:
    state_dict = load_prismaudio_state_dict(checkpoint_path)
    filtered_state_dict = _strip_prefix_from_state_dict(state_dict, prefix)

    incompatible = module.load_state_dict(filtered_state_dict, strict=strict)
    loaded_keys = sorted(set(filtered_state_dict) - set(incompatible.unexpected_keys))
    return PrismAudioCheckpointLoadReport(
        missing_keys=sorted(list(incompatible.missing_keys)),
        unexpected_keys=sorted(list(incompatible.unexpected_keys)),
        loaded_keys=loaded_keys,
    )


@torch.no_grad()
def sample_discrete_euler(
    model: nn.Module,
    x: torch.Tensor,
    steps: int,
    sigma_max: float = 1.0,
    **extra_args: Any,
) -> torch.Tensor:
    timesteps = torch.linspace(sigma_max, 0, steps + 1, device=x.device, dtype=x.dtype)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)
        dt = t_prev - t_curr
        x = x + dt * model(x, t_curr_tensor, **extra_args)

    return x


class PrismAudioPipeline(nn.Module, SupportAudioOutput):
    """Minimal PrismAudio request-contract shell.

    This first integration stage matches the official PrismAudio inference
    contract where precomputed conditioning features are loaded before the
    diffusion model runs. Sampling and waveform decode are intentionally left
    for follow-up changes once the request plumbing is reviewed.
    """

    support_audio_output = True
    required_feature_names = ("video_features", "text_features", "sync_features")
    default_num_inference_steps = 24
    default_cfg_scale = 5.0

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig | None = None,
        transformer: nn.Module | None = None,
        vae: nn.Module | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.od_config = od_config
        self.transformer = transformer
        self.vae = vae
        self.prefix = prefix

    def _get_additional_information(self, prompt: Any) -> dict[str, Any]:
        if isinstance(prompt, str):
            return {}
        if not isinstance(prompt, Mapping):
            raise TypeError(
                "PrismAudioPipeline expects each prompt to be a string or mapping "
                f"but received {type(prompt)!r}."
            )

        additional_information = prompt.get("additional_information", {})
        if additional_information is None:
            return {}
        if not isinstance(additional_information, Mapping):
            raise TypeError(
                "PrismAudioPipeline expects prompt.additional_information to be a mapping "
                f"but received {type(additional_information)!r}."
            )
        return dict(additional_information)

    def _validate_required_features(self, additional_information: Mapping[str, Any]) -> None:
        for feature_name in self.required_feature_names:
            if additional_information.get(feature_name) is None:
                raise ValueError(
                    f"PrismAudioPipeline requires precomputed `{feature_name}` in "
                    "`prompt.additional_information`. End-to-end video preprocessing "
                    "is not implemented in this integration stage."
                )

    def _parse_sampling_args(self, req: OmniDiffusionRequest) -> dict[str, float | int]:
        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
        num_inference_steps = extra_args.get("num_inference_steps", req.sampling_params.num_inference_steps)
        if num_inference_steps is None:
            num_inference_steps = self.default_num_inference_steps

        cfg_scale = extra_args.get("cfg_scale", self.default_cfg_scale)
        return {
            "num_inference_steps": int(num_inference_steps),
            "cfg_scale": float(cfg_scale),
        }

    def forward(self, req: OmniDiffusionRequest, *args: Any, **kwargs: Any) -> DiffusionOutput:
        if not req.prompts:
            raise ValueError("PrismAudioPipeline requires at least one prompt.")

        for prompt in req.prompts:
            additional_information = self._get_additional_information(prompt)
            self._validate_required_features(additional_information)

        sampling_args = self._parse_sampling_args(req)

        raise NotImplementedError(
            "PrismAudioPipeline currently validates the request contract only. "
            "Checkpoint loading, rectified-flow sampling, and waveform decode "
            f"are not implemented yet. Parsed sampling args: {sampling_args!r}."
        )
