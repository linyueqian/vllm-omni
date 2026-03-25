# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors_file
from torch import nn
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest


@dataclass
class PrismAudioCheckpointLoadReport:
    missing_keys: list[str]
    unexpected_keys: list[str]
    loaded_keys: list[str]


@dataclass
class PrismAudioRuntimeConfig:
    sample_rate: int
    audio_channels: int
    latent_channels: int
    transformer_checkpoint_path: str | None = None
    vae_checkpoint_path: str | None = None
    raw_model_config: dict[str, Any] | None = None


def load_prismaudio_state_dict(checkpoint_path: str | PathLike[str]) -> dict[str, torch.Tensor]:
    checkpoint_path = str(checkpoint_path)
    if checkpoint_path.endswith(".safetensors"):
        raw_state = load_safetensors_file(checkpoint_path)
    else:
        try:
            raw_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
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


def _strip_common_checkpoint_prefixes(
    state_dict: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    stripped_state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        stripped_name = name
        prefix_stripped = True
        while prefix_stripped:
            prefix_stripped = False
            for prefix in prefixes:
                if stripped_name.startswith(prefix):
                    stripped_name = stripped_name[len(prefix) :]
                    prefix_stripped = True
                    break
        stripped_state_dict[stripped_name] = tensor
    return stripped_state_dict


def load_module_from_prismaudio_checkpoint(
    module: nn.Module,
    checkpoint_path: str | PathLike[str],
    *,
    prefix: str | None = None,
    strict: bool = False,
) -> PrismAudioCheckpointLoadReport:
    state_dict = load_prismaudio_state_dict(checkpoint_path)
    prefix_chain = tuple(p for p in (prefix, "module.", "model.") if p)
    filtered_state_dict = _strip_common_checkpoint_prefixes(state_dict, prefix_chain)

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
    """PrismAudio pipeline with request validation, sampling, and audio decode.

    This integration stage matches the official PrismAudio inference contract
    where precomputed conditioning features are loaded before diffusion runs.
    The pipeline can drive a minimal injected-component sampling and waveform
    decode flow, but it does not yet assemble the full checkpoint-backed model
    graph on its own.
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
        self.runtime_config: PrismAudioRuntimeConfig | None = None

        if self.od_config is not None:
            self.runtime_config = self._build_runtime_config(self.od_config)
            self._maybe_initialize_components_from_config()

    def _load_prismaudio_model_config(self, od_config: OmniDiffusionConfig) -> dict[str, Any]:
        inline_model_config = dict(getattr(od_config, "model_config", {}) or {})
        raw_model_config = inline_model_config.get("prismaudio_model_config")
        if isinstance(raw_model_config, Mapping):
            return dict(raw_model_config)

        config_path = inline_model_config.get("prismaudio_model_config_path")
        if config_path is None and isinstance(od_config.model, str) and od_config.model.endswith(".json"):
            config_path = od_config.model

        if config_path is None:
            return {}

        config_data = json.loads(Path(config_path).read_text())
        if not isinstance(config_data, dict):
            raise TypeError(f"PrismAudio model config at {config_path!r} must decode to a dict.")
        return config_data

    def _build_runtime_config(self, od_config: OmniDiffusionConfig) -> PrismAudioRuntimeConfig:
        raw_model_config = self._load_prismaudio_model_config(od_config)
        model_value = raw_model_config.get("model", {})
        model_section = model_value if isinstance(model_value, Mapping) else {}
        pretransform_value = model_section.get("pretransform", {})
        pretransform = pretransform_value if isinstance(pretransform_value, Mapping) else {}
        pretransform_config_value = pretransform.get("config", {})
        pretransform_config = pretransform_config_value if isinstance(pretransform_config_value, Mapping) else {}
        sample_rate = int(raw_model_config.get("sample_rate", 44100))
        audio_channels = int(raw_model_config.get("audio_channels", 2))
        latent_channels = int(model_section.get("io_channels", pretransform_config.get("latent_dim", 64)))
        return PrismAudioRuntimeConfig(
            sample_rate=sample_rate,
            audio_channels=audio_channels,
            latent_channels=latent_channels,
            transformer_checkpoint_path=od_config.model_paths.get("transformer"),
            vae_checkpoint_path=od_config.model_paths.get("vae"),
            raw_model_config=raw_model_config,
        )

    def _maybe_initialize_components_from_config(self) -> None:
        if self.od_config is None:
            return

        factories = getattr(self.od_config, "model_config", {}) or {}
        runtime_config = self.runtime_config
        if runtime_config is None:
            return

        if self.transformer is None:
            model_factory = self._resolve_factory_spec(factories.get("model_factory"))
            if model_factory is not None:
                wrapper = self._call_factory_spec(model_factory, runtime_config)
                if isinstance(wrapper, nn.Module):
                    self.transformer = wrapper
                    pretransform = getattr(wrapper, "pretransform", None)
                    if self.vae is None and isinstance(pretransform, nn.Module):
                        self.vae = pretransform

        if self.transformer is None:
            transformer_factory = self._resolve_factory_spec(factories.get("transformer_factory"))
            if transformer_factory is not None:
                self.transformer = self._call_factory_spec(transformer_factory, runtime_config)

        if self.vae is None:
            vae_factory = self._resolve_factory_spec(factories.get("vae_factory"))
            if vae_factory is not None:
                self.vae = self._call_factory_spec(vae_factory, runtime_config)

    def _resolve_factory_spec(self, spec: Any) -> Any:
        if isinstance(spec, Mapping):
            path = spec.get("path")
            if not isinstance(path, str):
                raise TypeError("PrismAudio factory spec mappings require a string `path`.")
            return {
                "callable": self._resolve_factory_spec(path),
                "input": spec.get("input", "runtime_config"),
            }
        if callable(spec):
            return spec
        if not isinstance(spec, str):
            return None
        if ":" in spec:
            module_name, attr_name = spec.split(":", 1)
            return getattr(importlib.import_module(module_name), attr_name)
        return resolve_obj_by_qualname(spec)

    def _call_factory_spec(self, spec: Any, runtime_config: PrismAudioRuntimeConfig) -> Any:
        if isinstance(spec, Mapping):
            factory_callable = spec["callable"]
            factory_input = spec.get("input", "runtime_config")
            if factory_input == "runtime_config":
                return factory_callable(runtime_config)
            if factory_input == "raw_model_config":
                return factory_callable(runtime_config.raw_model_config or {})
            raise ValueError(f"Unsupported PrismAudio factory input mode: {factory_input!r}.")
        return spec(runtime_config)

    def _get_additional_information(self, prompt: Any) -> dict[str, Any]:
        if isinstance(prompt, str):
            return {}
        if not isinstance(prompt, Mapping):
            raise TypeError(
                f"PrismAudioPipeline expects each prompt to be a string or mapping but received {type(prompt)!r}."
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

        if req.sampling_params.guidance_scale_provided:
            cfg_scale = req.sampling_params.guidance_scale
        else:
            cfg_scale = extra_args.get("cfg_scale", self.default_cfg_scale)
        return {
            "num_inference_steps": int(num_inference_steps),
            "cfg_scale": float(cfg_scale),
        }

    def _prepare_latents(self, req: OmniDiffusionRequest) -> torch.Tensor:
        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
        latents = extra_args.get("latents")
        if latents is None:
            latents = req.sampling_params.latents
        if latents is None:
            latents = req.sampling_params.audio_latents
        if latents is None:
            raise ValueError(
                "PrismAudioPipeline requires initial `latents` for the current injected-component "
                "execution path. Real checkpoint-backed latent initialization is not implemented yet."
            )
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"PrismAudioPipeline expected `latents` to be a torch.Tensor, got {type(latents)!r}.")
        return latents

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise NotImplementedError("PrismAudioPipeline requires a VAE module before decoding latents.")

        if hasattr(self.vae, "decode"):
            decoded = self.vae.decode(latents)
        else:
            decoded = self.vae(latents)

        if isinstance(decoded, torch.Tensor):
            return decoded
        sample = getattr(decoded, "sample", None)
        if isinstance(sample, torch.Tensor):
            return sample
        raise TypeError(
            "PrismAudioPipeline expected VAE decode output to be a tensor or an object with `.sample` tensor, "
            f"got {type(decoded)!r}."
        )

    def _get_sampling_model_and_conditioning(
        self,
        conditioning: Mapping[str, Any],
        *,
        device: torch.device,
    ) -> tuple[nn.Module, dict[str, Any]]:
        if self.transformer is None:
            raise NotImplementedError("PrismAudioPipeline requires a transformer module before sampling latents.")

        conditioner = getattr(self.transformer, "conditioner", None)
        get_conditioning_inputs = getattr(self.transformer, "get_conditioning_inputs", None)
        inner_model = getattr(self.transformer, "model", None)
        if callable(conditioner) and callable(get_conditioning_inputs) and isinstance(inner_model, nn.Module):
            conditioning_tensors = conditioner((dict(conditioning),), device)
            sampling_conditioning = dict(get_conditioning_inputs(conditioning_tensors))
            return inner_model, sampling_conditioning

        return self.transformer, {
            "video_features": conditioning["video_features"],
            "text_features": conditioning["text_features"],
            "sync_features": conditioning["sync_features"],
        }

    def _load_runtime_checkpoint_weights(self) -> set[str]:
        runtime_config = self.runtime_config
        if runtime_config is None:
            return set()

        loaded_keys: set[str] = set()
        if runtime_config.transformer_checkpoint_path and self.transformer is None:
            raise NotImplementedError(
                "PrismAudioPipeline cannot load the runtime transformer checkpoint without an initialized "
                "transformer module. Provide a `transformer_factory` or inject a constructed transformer."
            )
        if runtime_config.vae_checkpoint_path and self.vae is None:
            raise NotImplementedError(
                "PrismAudioPipeline cannot load the runtime VAE checkpoint without an initialized VAE module. "
                "Provide a `vae_factory` or inject a constructed VAE."
            )

        if self.transformer is not None and runtime_config.transformer_checkpoint_path:
            report = load_module_from_prismaudio_checkpoint(
                self.transformer,
                runtime_config.transformer_checkpoint_path,
            )
            if not report.loaded_keys:
                raise ValueError(
                    "PrismAudioPipeline loaded no matching weights for the runtime transformer checkpoint "
                    f"{runtime_config.transformer_checkpoint_path!r}."
                )
            if report.missing_keys:
                raise ValueError(
                    "PrismAudioPipeline runtime transformer checkpoint is missing required weights: "
                    f"{report.missing_keys!r}."
                )
            loaded_keys.update(f"transformer.{key}" for key in report.loaded_keys)

        if self.vae is not None and runtime_config.vae_checkpoint_path:
            report = load_module_from_prismaudio_checkpoint(
                self.vae,
                runtime_config.vae_checkpoint_path,
                prefix="autoencoder.",
            )
            if not report.loaded_keys:
                raise ValueError(
                    "PrismAudioPipeline loaded no matching weights for the runtime VAE checkpoint "
                    f"{runtime_config.vae_checkpoint_path!r}."
                )
            if report.missing_keys:
                raise ValueError(
                    f"PrismAudioPipeline runtime VAE checkpoint is missing required weights: {report.missing_keys!r}."
                )
            loaded_keys.update(f"vae.{key}" for key in report.loaded_keys)

        return loaded_keys

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded_keys = loader.load_weights(weights)
        loaded_keys.update(self._load_runtime_checkpoint_weights())
        return loaded_keys

    def forward(self, req: OmniDiffusionRequest, *args: Any, **kwargs: Any) -> DiffusionOutput:
        if not req.prompts:
            raise ValueError("PrismAudioPipeline requires at least one prompt.")
        if len(req.prompts) != 1:
            raise ValueError(
                "PrismAudioPipeline currently supports exactly one prompt per request. "
                "Batching multiple prompt-specific conditioning payloads is not implemented yet."
            )

        conditioning: dict[str, Any] | None = None
        for prompt in req.prompts:
            additional_information = self._get_additional_information(prompt)
            self._validate_required_features(additional_information)
            if conditioning is None:
                conditioning = additional_information

        sampling_args = self._parse_sampling_args(req)
        assert conditioning is not None
        latents = self._prepare_latents(req)
        sampling_model, sampling_conditioning = self._get_sampling_model_and_conditioning(
            conditioning,
            device=latents.device,
        )
        sampled_latents = sample_discrete_euler(
            sampling_model,
            latents,
            steps=int(sampling_args["num_inference_steps"]),
            **sampling_conditioning,
            cfg_scale=float(sampling_args["cfg_scale"]),
            batch_cfg=True,
        )
        audio = self._decode_latents(sampled_latents)

        return DiffusionOutput(output=audio)
