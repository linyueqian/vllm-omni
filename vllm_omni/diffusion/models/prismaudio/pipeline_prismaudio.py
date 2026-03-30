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

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file as load_safetensors_file
from torch import nn
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.utils import DiffusionRequestState


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
    module_state_keys = tuple(module.state_dict().keys())
    preserve_model_prefix = any(name.startswith("model.") for name in module_state_keys)
    prefix_chain = tuple(p for p in (prefix, "module.") if p)
    if not preserve_model_prefix:
        prefix_chain = (*prefix_chain, "model.")
    filtered_state_dict = _strip_common_checkpoint_prefixes(state_dict, prefix_chain)

    incompatible = module.load_state_dict(filtered_state_dict, strict=strict)
    if incompatible is None:
        return PrismAudioCheckpointLoadReport(
            missing_keys=[],
            unexpected_keys=[],
            loaded_keys=sorted(filtered_state_dict.keys()),
        )
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
    supports_step_execution = True
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
        """Load the official PrismAudio model config.

        Supported inputs, in precedence order:
        1. ``od_config.model_config["prismaudio_model_config"]`` inline dict
        2. ``od_config.model_config["prismaudio_model_config_path"]`` JSON file path
        3. ``od_config.model`` when it directly points to a ``.json`` file

        Example:
            od_config.model_config = {
                "prismaudio_model_config_path": "/path/to/prismaudio.json",
            }

        The JSON/dict must preserve the upstream official builder schema, e.g.
        ``{"model_type": "diffusion_cond", "model": {...}}``.
        """
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
            if (
                model_factory is None
                and factories.get("transformer_factory") is None
                and factories.get("vae_factory") is None
                and isinstance(runtime_config.raw_model_config, Mapping)
                and runtime_config.raw_model_config.get("model_type") is not None
            ):
                model_factory = self._get_default_official_model_factory()
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

    def _get_default_official_model_factory(self) -> Mapping[str, Any] | None:
        try:
            official_models_module = importlib.import_module("PrismAudio.models")
        except ModuleNotFoundError:
            return None

        official_builder = getattr(official_models_module, "create_model_from_config", None)
        if not callable(official_builder):
            return None
        return {
            "callable": official_builder,
            "input": "raw_model_config",
            "source": "official_default_builder",
        }

    def _call_factory_spec(self, spec: Any, runtime_config: PrismAudioRuntimeConfig) -> Any:
        if isinstance(spec, Mapping):
            factory_callable = spec["callable"]
            factory_input = spec.get("input", "runtime_config")
            try:
                if factory_input == "runtime_config":
                    return factory_callable(runtime_config)
                if factory_input == "raw_model_config":
                    return factory_callable(runtime_config.raw_model_config or {})
            except ModuleNotFoundError as exc:
                if spec.get("source") == "official_default_builder":
                    raise ModuleNotFoundError(
                        "The official PrismAudio builder is importable but cannot construct the model because "
                        f"a required dependency is missing: {exc}. Install the upstream PrismAudio runtime "
                        "dependencies before using the default builder path."
                    ) from exc
            except AttributeError as exc:
                if spec.get("source") == "official_default_builder" and "np.float_" in str(exc):
                    raise RuntimeError(
                        "The official PrismAudio builder hit a NumPy 2.x compatibility issue "
                        f"while constructing the model: {exc}. Use a NumPy version compatible with the "
                        "upstream PrismAudio stack, or patch the upstream dependency to avoid `np.float_`."
                    ) from exc
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
        expected_feature_dims = self._get_expected_feature_dims()
        for feature_name in self.required_feature_names:
            feature_value = additional_information.get(feature_name)
            if feature_value is None:
                raise ValueError(
                    f"PrismAudioPipeline requires precomputed `{feature_name}` in "
                    "`prompt.additional_information`. End-to-end video preprocessing "
                    "is not implemented in this integration stage."
                )
            self._validate_feature_tensor(
                feature_name,
                feature_value,
                expected_feature_dims.get(feature_name),
            )

    def _get_expected_feature_dims(self) -> dict[str, int]:
        runtime_config = self.runtime_config
        raw_model_config = runtime_config.raw_model_config if runtime_config is not None else None
        if not isinstance(raw_model_config, Mapping):
            return {}

        model_section = raw_model_config.get("model", {})
        if not isinstance(model_section, Mapping):
            return {}

        conditioning_section = model_section.get("conditioning", {})
        if not isinstance(conditioning_section, Mapping):
            return {}

        feature_dims: dict[str, int] = {}
        for item in conditioning_section.get("configs", []):
            if not isinstance(item, Mapping):
                continue
            feature_name = item.get("id")
            feature_config = item.get("config", {})
            if not isinstance(feature_name, str) or not isinstance(feature_config, Mapping):
                continue
            feature_dim = feature_config.get("dim")
            if isinstance(feature_dim, int):
                feature_dims[feature_name] = feature_dim
        return feature_dims

    def _validate_feature_tensor(
        self,
        feature_name: str,
        feature_value: Any,
        expected_width: int | None,
    ) -> None:
        if not isinstance(feature_value, torch.Tensor):
            raise TypeError(
                f"PrismAudioPipeline expects `{feature_name}` to be a torch.Tensor, "
                f"but received {type(feature_value)!r}."
            )
        if not torch.is_floating_point(feature_value):
            raise TypeError(
                f"PrismAudioPipeline expects `{feature_name}` to use a floating-point dtype, "
                f"but received {feature_value.dtype}."
            )
        if feature_value.ndim not in (2, 3):
            raise ValueError(
                f"PrismAudioPipeline expects `{feature_name}` to have rank 2 or 3, "
                f"but received shape {tuple(feature_value.shape)}."
            )
        if feature_value.ndim == 3 and feature_value.shape[0] != 1:
            raise ValueError(
                f"PrismAudioPipeline currently supports exactly one prompt, so `{feature_name}` "
                f"may only use a leading batch dimension of 1; received shape {tuple(feature_value.shape)}."
            )
        if feature_value.shape[-1] <= 0:
            raise ValueError(
                f"PrismAudioPipeline expects `{feature_name}` to have a positive trailing feature dimension, "
                f"but received shape {tuple(feature_value.shape)}."
            )
        if expected_width is not None and feature_value.shape[-1] != expected_width:
            raise ValueError(
                f"PrismAudioPipeline expects `{feature_name}` to have trailing dimension {expected_width}, "
                f"but received shape {tuple(feature_value.shape)}."
            )

    def _parse_sampling_args(self, req: OmniDiffusionRequest) -> dict[str, float | int]:
        return self._parse_sampling_args_from_sampling(req.sampling_params)

    def _parse_sampling_args_from_sampling(self, sampling_params: Any) -> dict[str, float | int]:
        extra_args = getattr(sampling_params, "extra_args", {}) or {}
        num_inference_steps = extra_args.get("num_inference_steps", sampling_params.num_inference_steps)
        if num_inference_steps is None:
            num_inference_steps = self.default_num_inference_steps

        if sampling_params.guidance_scale_provided:
            cfg_scale = sampling_params.guidance_scale
        else:
            cfg_scale = extra_args.get("cfg_scale", self.default_cfg_scale)
        return {
            "num_inference_steps": int(num_inference_steps),
            "cfg_scale": float(cfg_scale),
        }

    def _infer_runtime_latent_shape(self, batch_size: int) -> tuple[int, int, int] | None:
        runtime_config = self.runtime_config
        if runtime_config is None:
            return None

        latent_channels = runtime_config.latent_channels
        raw_model_config = runtime_config.raw_model_config or {}
        model_value = raw_model_config.get("model", {})
        model_section = model_value if isinstance(model_value, Mapping) else {}
        pretransform_value = model_section.get("pretransform", {})
        pretransform = pretransform_value if isinstance(pretransform_value, Mapping) else {}
        pretransform_config_value = pretransform.get("config", {})
        pretransform_config = pretransform_config_value if isinstance(pretransform_config_value, Mapping) else {}
        diffusion_value = model_section.get("diffusion", {})
        diffusion_section = diffusion_value if isinstance(diffusion_value, Mapping) else {}
        diffusion_config_value = diffusion_section.get("config", {})
        diffusion_config = diffusion_config_value if isinstance(diffusion_config_value, Mapping) else {}

        latent_seq_len = diffusion_config.get("latent_seq_len")
        if latent_seq_len is None:
            sample_size = raw_model_config.get("sample_size")
            downsampling_ratio = pretransform_config.get("downsampling_ratio")
            if sample_size is not None and downsampling_ratio:
                latent_seq_len = round(float(sample_size) / float(downsampling_ratio))

        if latent_seq_len is None:
            return None

        return (batch_size, latent_channels, int(latent_seq_len))

    def _get_runtime_latent_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        for module in (self.transformer, self.vae):
            if not isinstance(module, nn.Module):
                continue
            parameter = next(module.parameters(), None)
            if parameter is not None:
                return parameter.device, parameter.dtype
            buffer = next(module.buffers(), None)
            if buffer is not None:
                return buffer.device, buffer.dtype
        return torch.device("cpu"), torch.float32

    def _resolve_sampling_generator(
        self,
        sampling_params: Any,
        *,
        device: torch.device,
    ) -> torch.Generator | list[torch.Generator] | None:
        generator = getattr(sampling_params, "generator", None)
        if generator is not None:
            return generator

        seed = getattr(sampling_params, "seed", None)
        if seed is None:
            return None

        generator_device = getattr(sampling_params, "generator_device", None) or device.type
        return torch.Generator(device=generator_device).manual_seed(seed)

    def _prepare_latents_from_sampling(self, prompts: list[Any], sampling_params: Any) -> torch.Tensor:
        extra_args = getattr(sampling_params, "extra_args", {}) or {}
        latents = extra_args.get("latents")
        if latents is None:
            latents = sampling_params.latents
        if latents is None:
            latents = sampling_params.audio_latents
        if latents is None:
            inferred_latent_shape = self._infer_runtime_latent_shape(len(prompts))
            if inferred_latent_shape is None:
                raise ValueError(
                    "PrismAudioPipeline requires initial `latents` for the current execution path, or a runtime "
                    "config that can infer latent shape from PrismAudio model metadata."
                )
            device, dtype = self._get_runtime_latent_device_dtype()
            generator = self._resolve_sampling_generator(sampling_params, device=device)
            latents = randn_tensor(inferred_latent_shape, generator=generator, device=device, dtype=dtype)
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"PrismAudioPipeline expected `latents` to be a torch.Tensor, got {type(latents)!r}.")
        return latents

    def _prepare_latents(self, req: OmniDiffusionRequest) -> torch.Tensor:
        return self._prepare_latents_from_sampling(req.prompts, req.sampling_params)

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
            conditioning_tensors = conditioner(
                (self._normalize_conditioning_for_official_wrapper(conditioning),), device
            )
            sampling_conditioning = dict(get_conditioning_inputs(conditioning_tensors))
            return inner_model, sampling_conditioning

        def _move_to_device(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(device=device)
            if isinstance(value, tuple):
                return tuple(_move_to_device(item) for item in value)
            if isinstance(value, list):
                return [_move_to_device(item) for item in value]
            if isinstance(value, Mapping):
                return {key: _move_to_device(item) for key, item in value.items()}
            return value

        return self.transformer, {
            "video_features": _move_to_device(conditioning["video_features"]),
            "text_features": _move_to_device(conditioning["text_features"]),
            "sync_features": _move_to_device(conditioning["sync_features"]),
        }

    def _normalize_conditioning_for_official_wrapper(self, conditioning: Mapping[str, Any]) -> dict[str, Any]:
        def _normalize(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                if value.ndim > 1 and value.shape[0] == 1:
                    return value[0]
                return value
            if isinstance(value, tuple):
                return tuple(_normalize(item) for item in value)
            if isinstance(value, list):
                return [_normalize(item) for item in value]
            if isinstance(value, Mapping):
                return {key: _normalize(item) for key, item in value.items()}
            return value

        return {key: _normalize(value) for key, value in conditioning.items()}

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
            transformer_missing_keys = list(report.missing_keys)
            transformer_pretransform = getattr(self.transformer, "pretransform", None)
            if isinstance(transformer_pretransform, nn.Module) and self.vae is transformer_pretransform:
                transformer_missing_keys = [
                    key for key in transformer_missing_keys if not key.startswith("pretransform.")
                ]
            if not report.loaded_keys:
                raise ValueError(
                    "PrismAudioPipeline loaded no matching weights for the runtime transformer checkpoint "
                    f"{runtime_config.transformer_checkpoint_path!r}."
                )
            if transformer_missing_keys:
                raise ValueError(
                    "PrismAudioPipeline runtime transformer checkpoint is missing required weights: "
                    f"{transformer_missing_keys!r}."
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

    def prepare_encode(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionRequestState:
        prompts = state.prompts or []
        if not prompts:
            raise ValueError("PrismAudioPipeline requires at least one prompt.")
        if len(prompts) != 1:
            raise ValueError(
                "PrismAudioPipeline currently supports exactly one prompt per request. "
                "Batching multiple prompt-specific conditioning payloads is not implemented yet."
            )

        conditioning = self._get_additional_information(prompts[0])
        self._validate_required_features(conditioning)
        sampling_args = self._parse_sampling_args_from_sampling(state.sampling)
        latents = self._prepare_latents_from_sampling(prompts, state.sampling)
        sampling_model, sampling_conditioning = self._get_sampling_model_and_conditioning(
            conditioning,
            device=latents.device,
        )

        full_timesteps = torch.linspace(
            1.0,
            0.0,
            int(sampling_args["num_inference_steps"]) + 1,
            device=latents.device,
            dtype=latents.dtype,
        )
        state.latents = latents
        state.timesteps = full_timesteps[:-1]
        state.step_index = 0
        state.extra["next_timesteps"] = full_timesteps[1:]
        state.extra["sampling_model"] = sampling_model
        state.extra["sampling_conditioning"] = sampling_conditioning
        state.extra["cfg_scale"] = float(sampling_args["cfg_scale"])
        return state

    def denoise_step(self, state: DiffusionRequestState, **kwargs: Any) -> torch.Tensor | None:
        timestep = state.current_timestep
        if timestep is None:
            return None
        assert state.latents is not None
        sampling_model = state.extra["sampling_model"]
        sampling_conditioning = state.extra["sampling_conditioning"]
        t_for_model = timestep * torch.ones(
            (state.latents.shape[0],), dtype=state.latents.dtype, device=state.latents.device
        )
        return sampling_model(
            state.latents,
            t_for_model,
            **sampling_conditioning,
            cfg_scale=state.extra["cfg_scale"],
            batch_cfg=True,
        )

    def step_scheduler(self, state: DiffusionRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        assert state.latents is not None
        timestep = state.current_timestep
        if timestep is None:
            return
        next_timesteps = state.extra["next_timesteps"]
        next_timestep = next_timesteps[state.step_index]
        dt = next_timestep - timestep
        state.latents = state.latents + dt * noise_pred
        state.step_index += 1

    def post_decode(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionOutput:
        assert state.latents is not None
        audio = self._decode_latents(state.latents)
        return DiffusionOutput(output=audio)

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
