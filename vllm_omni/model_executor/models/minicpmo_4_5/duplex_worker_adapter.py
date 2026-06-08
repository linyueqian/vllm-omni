from __future__ import annotations

from typing import Any


class PassiveNativeDuplexStage:
    is_passive_native_duplex_stage = True

    def open_duplex_session(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "supported": True,
            "implementation_level": "model_native_duplex_passive",
            "session_id": kwargs.get("session_id"),
            "epoch": kwargs.get("epoch"),
            "native_result": {"passive_stage": True},
        }

    def append_duplex_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "supported": True,
            "implementation_level": "model_native_duplex_passive",
            "session_id": kwargs.get("session_id"),
            "epoch": kwargs.get("epoch"),
            "seq": kwargs.get("seq"),
            "mode": kwargs.get("mode"),
            "final": kwargs.get("final"),
            "native_result": {"passive_stage": True},
        }

    def signal_duplex_turn(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "supported": True,
            "implementation_level": "model_native_duplex_passive",
            "session_id": kwargs.get("session_id"),
            "epoch": kwargs.get("epoch"),
            "event": kwargs.get("event"),
            "native_result": {"passive_stage": True},
        }

    def close_duplex_session(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "supported": True,
            "implementation_level": "model_native_duplex_passive",
            "session_id": kwargs.get("session_id"),
            "epoch": kwargs.get("epoch"),
            "reason": kwargs.get("reason"),
        }


def is_passive_native_duplex_stage(target: Any) -> bool:
    return bool(getattr(target, "is_passive_native_duplex_stage", False))


def maybe_load_minicpmo_native_duplex_target(
    worker: Any,
    capabilities: dict[str, Any] | None = None,
) -> Any | None:
    if capabilities is not None and capabilities.get("implementation_level") != "model_native_duplex":
        return None

    existing = getattr(worker, "_minicpmo_stage_duplex_target", None)
    if existing is not None:
        return existing

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex_runtime import (
        MiniCPMO45Stage0DuplexRuntime,
        MiniCPMO45Stage1DuplexRuntime,
    )

    patch_minicpmo_transformers_compat()
    loaded_model = _minicpmo_loaded_model(worker)
    model_path = _minicpmo_native_model_path(worker)
    model_stage = getattr(loaded_model, "model_stage", None)
    if model_stage in {"llm", "thinker"} and MiniCPMO45Stage0DuplexRuntime.can_wrap(loaded_model):
        _attach_runner_context_forward(worker, loaded_model)
        target = MiniCPMO45Stage0DuplexRuntime(
            loaded_model,
            model_path=model_path,
            device=_minicpmo_native_device(worker),
        )
        worker._minicpmo_stage_duplex_target = target
        return target
    if model_stage in {"tts", "talker"} and MiniCPMO45Stage1DuplexRuntime.can_wrap(loaded_model):
        target = MiniCPMO45Stage1DuplexRuntime(
            loaded_model,
            model_path=model_path,
            device=_minicpmo_native_device(worker),
        )
        model_runner = getattr(worker, "model_runner", None)
        set_payload_cache = getattr(target, "set_duplex_stage_payload_cache", None)
        if callable(set_payload_cache) and model_runner is not None:
            set_payload_cache(model_runner)
        worker._minicpmo_stage_duplex_target = target
        return target

    return None


def _attach_runner_context_forward(worker: Any, loaded_model: Any) -> None:
    """Expose a runner-owned duplex forward hook on the loaded stage model.

    The Stage0 duplex runtime must not rebuild attention/KV semantics locally.
    If a model runner provides a scheduled duplex forward entry point, the
    loaded model forwards calls to that runner boundary.  Otherwise no hook is
    installed and Stage0 open fails explicitly instead of falling back to an
    unscheduled model.forward path.
    """

    existing_forward = getattr(loaded_model, "duplex_forward_with_runner_context", None)
    if callable(existing_forward) and getattr(existing_forward, "vllm_omni_runner_context_contract", False) is True:
        return
    model_runner = getattr(worker, "model_runner", None)
    runner_forward = getattr(model_runner, "duplex_forward_with_runner_context", None)
    if not callable(runner_forward):
        return
    if not _runner_forward_has_scheduler_contract(model_runner, runner_forward):
        return

    def duplex_forward_with_runner_context(**kwargs: Any) -> Any:
        return runner_forward(**kwargs)

    duplex_forward_with_runner_context.uses_scheduler_metadata = True  # type: ignore[attr-defined]
    duplex_forward_with_runner_context.uses_runner_kv_cache = True  # type: ignore[attr-defined]
    duplex_forward_with_runner_context.vllm_omni_runner_context_contract = True  # type: ignore[attr-defined]
    setattr(loaded_model, "duplex_forward_with_runner_context", duplex_forward_with_runner_context)


def _runner_forward_has_scheduler_contract(model_runner: Any, runner_forward: Any) -> bool:
    """Return True only for a runner-owned scheduled/KV duplex forward path.

    A same-named method is not enough: MiniCPM-o Stage0 must not fall back to a
    private model.forward/eager path that bypasses vLLM attention metadata and
    KV block assignment. Runner implementations must opt in explicitly.
    """

    return bool(
        getattr(runner_forward, "uses_scheduler_metadata", False)
        and getattr(runner_forward, "uses_runner_kv_cache", False)
        and getattr(runner_forward, "vllm_omni_runner_context_contract", False)
        and getattr(model_runner, "supports_native_duplex_runner_context", False)
    )


def patch_minicpmo_remote_config(config: Any) -> None:
    tts_config = getattr(config, "tts_config", None)
    if tts_config is None:
        return
    defaults = {
        "top_p": 0.8,
        "top_k": 100,
        "temperature": 0.8,
        "repetition_penalty": 1.05,
    }
    for name, value in defaults.items():
        if not hasattr(tts_config, name):
            setattr(tts_config, name, value)


def patch_minicpmo_transformers_compat() -> None:
    try:
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache
        from transformers.modeling_utils import PreTrainedModel
        from transformers.models.whisper.modeling_whisper import WhisperAttention
    except Exception:
        return

    if not getattr(WhisperAttention.forward, "_minicpmo45_compat", False):
        original_whisper_forward = WhisperAttention.forward

        def whisper_forward_compat(self: Any, *args: Any, **kwargs: Any):
            legacy_cache = kwargs.pop("past_key_value", None)
            if legacy_cache is not None and kwargs.get("past_key_values") is None:
                kwargs["past_key_values"] = legacy_cache
            result = original_whisper_forward(self, *args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                return result[0], result[1], kwargs.get("past_key_values")
            return result

        whisper_forward_compat._minicpmo45_compat = True  # type: ignore[attr-defined]
        WhisperAttention.forward = whisper_forward_compat  # type: ignore[method-assign]

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):

        def all_tied_weights_keys(self: Any) -> dict[str, Any]:
            cached = getattr(self, "_all_tied_weights_keys_compat", None)
            if isinstance(cached, dict):
                return cached
            keys: dict[str, Any] = {}
            for attr_name in ("_tied_weights_keys", "_dynamic_tied_weights_keys"):
                raw_keys = getattr(self, attr_name, None)
                if raw_keys is None:
                    continue
                if isinstance(raw_keys, dict):
                    keys.update({str(k): v for k, v in raw_keys.items()})
                    continue
                if isinstance(raw_keys, str):
                    keys[raw_keys] = None
                    continue
                for key in raw_keys:
                    keys[str(key)] = None
            return keys

        def set_all_tied_weights_keys(self: Any, value: Any) -> None:
            if isinstance(value, dict):
                normalized = {str(k): v for k, v in value.items()}
            elif value is None:
                normalized = {}
            else:
                normalized = {str(key): None for key in value}
            self._all_tied_weights_keys_compat = normalized

        PreTrainedModel.all_tied_weights_keys = property(  # type: ignore[attr-defined]
            all_tied_weights_keys,
            set_all_tied_weights_keys,
        )

    class _DynamicCacheTensorList:
        def __init__(self, cache: Any, name: str) -> None:
            self._cache = cache
            self._name = name

        def __bool__(self) -> bool:
            return bool(len(self))

        def __len__(self) -> int:
            return sum(1 for layer in self._cache.layers if getattr(layer, self._name, None) is not None)

        def __iter__(self):
            for layer in self._cache.layers:
                value = getattr(layer, self._name, None)
                if value is not None:
                    yield value

        def __getitem__(self, index):
            if isinstance(index, slice):
                return list(self)[index]
            return list(self)[index]

        def __setitem__(self, index: int, value: Any) -> None:
            self._ensure_layer(index)
            setattr(self._cache.layers[index], self._name, value)
            self._cache.layers[index].is_initialized = True

        def append(self, value: Any) -> None:
            self._ensure_layer(len(self))
            self[len(self) - 1] = value

        def _ensure_layer(self, index: int) -> None:
            while len(self._cache.layers) <= index:
                self._cache.layers.append(self._cache.layer_class_to_replicate())

    def key_cache(self: Any) -> _DynamicCacheTensorList:
        return _DynamicCacheTensorList(self, "keys")

    def set_key_cache(self: Any, values: Any) -> None:
        for index, value in enumerate(values):
            key_cache(self)[index] = value

    def value_cache(self: Any) -> _DynamicCacheTensorList:
        return _DynamicCacheTensorList(self, "values")

    def set_value_cache(self: Any, values: Any) -> None:
        for index, value in enumerate(values):
            value_cache(self)[index] = value

    if not hasattr(DynamicCache(), "key_cache"):
        DynamicCache.key_cache = property(key_cache, set_key_cache)  # type: ignore[attr-defined]
        DynamicCache.value_cache = property(value_cache, set_value_cache)  # type: ignore[attr-defined]

    if not hasattr(DynamicCache, "__getitem__"):

        def dynamic_cache_getitem(self: Any, layer_idx: int) -> tuple[Any, Any]:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        DynamicCache.__getitem__ = dynamic_cache_getitem  # type: ignore[attr-defined]

    if not hasattr(DynamicCache, "get_usable_length"):

        def get_usable_length(self: Any, new_seq_length: int | None = None, layer_idx: int = 0) -> int:
            del new_seq_length
            return int(self.get_seq_length(layer_idx=layer_idx))

        DynamicCache.get_usable_length = get_usable_length  # type: ignore[attr-defined]

    if not hasattr(EncoderDecoderCache, "__getitem__"):

        def encoder_decoder_cache_getitem(self: Any, layer_idx: int) -> tuple[Any, Any]:
            return self.self_attention_cache[layer_idx]

        EncoderDecoderCache.__getitem__ = encoder_decoder_cache_getitem  # type: ignore[attr-defined]

    if not hasattr(EncoderDecoderCache, "key_cache"):

        def encoder_decoder_key_cache(self: Any) -> Any:
            return self.self_attention_cache.key_cache

        def set_encoder_decoder_key_cache(self: Any, values: Any) -> None:
            self.self_attention_cache.key_cache = values

        def encoder_decoder_value_cache(self: Any) -> Any:
            return self.self_attention_cache.value_cache

        def set_encoder_decoder_value_cache(self: Any, values: Any) -> None:
            self.self_attention_cache.value_cache = values

        EncoderDecoderCache.key_cache = property(  # type: ignore[attr-defined]
            encoder_decoder_key_cache,
            set_encoder_decoder_key_cache,
        )
        EncoderDecoderCache.value_cache = property(  # type: ignore[attr-defined]
            encoder_decoder_value_cache,
            set_encoder_decoder_value_cache,
        )


def _minicpmo_native_model_path(worker: Any) -> str | None:
    model_runner = getattr(worker, "model_runner", None)
    candidates = [
        getattr(getattr(model_runner, "model", None), "name_or_path", None),
        getattr(getattr(getattr(model_runner, "model", None), "config", None), "_name_or_path", None),
        getattr(getattr(getattr(model_runner, "model", None), "config", None), "name_or_path", None),
        getattr(getattr(getattr(model_runner, "vllm_config", None), "model_config", None), "model", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _minicpmo_loaded_model(worker: Any) -> Any | None:
    model_runner = getattr(worker, "model_runner", None)
    model = getattr(model_runner, "model", None)
    if model is None:
        return getattr(worker, "model", None)
    return model


def _minicpmo_native_device(worker: Any) -> str:
    loaded_model = _minicpmo_loaded_model(worker)
    if loaded_model is not None:
        try:
            param = next(loaded_model.parameters())
            return str(param.device)
        except Exception:
            pass
    return "cuda"
