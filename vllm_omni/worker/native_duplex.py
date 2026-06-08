from __future__ import annotations

import base64
import inspect
from collections.abc import Callable
from typing import Any

NativeDuplexProvider = Callable[[Any, dict[str, Any] | None], Any | None]

NATIVE_DUPLEX_GENERATE_PARAMS = (
    "max_new_speak_tokens_per_chunk",
    "temperature",
    "top_k",
    "top_p",
    "listen_prob_scale",
    "listen_top_k",
    "text_repetition_penalty",
    "text_repetition_window_size",
)

_NATIVE_DUPLEX_PROVIDERS: list[NativeDuplexProvider] = []
_DEFAULT_PROVIDERS_BOOTSTRAPPED = False


def register_native_duplex_provider(provider: NativeDuplexProvider) -> None:
    if provider not in _NATIVE_DUPLEX_PROVIDERS:
        _NATIVE_DUPLEX_PROVIDERS.append(provider)


def get_native_duplex_target(worker: Any, capabilities: dict[str, Any] | None = None) -> Any | None:
    if capabilities is not None and capabilities.get("implementation_level") != "model_native_duplex":
        return None
    _bootstrap_default_providers()
    for provider in tuple(_NATIVE_DUPLEX_PROVIDERS):
        target = provider(worker, capabilities)
        if target is not None:
            return target
    return None


def is_passive_native_duplex_stage(target: Any) -> bool:
    return bool(getattr(target, "is_passive_native_duplex_stage", False))


def open_native_duplex_session(
    target: Any,
    *,
    session_id: str,
    epoch: int,
    capabilities: dict[str, Any],
    session_config: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    validate_native_ref_audio_config(session_config)
    if not callable(getattr(target, "open_duplex_session", None)):
        if NativeDuplexMethodAdapter.can_wrap(target):
            target = NativeDuplexMethodAdapter(target)
        else:
            raise RuntimeError("native duplex target requires open_duplex_session or explicit native duplex methods")
    apply_native_duplex_session_config(target, session_config)

    open_fn = getattr(target, "open_duplex_session", None)
    if callable(open_fn):
        native = open_fn(
            session_id=session_id,
            epoch=epoch,
            capabilities=capabilities,
            session_config=session_config,
        )
    else:
        instructions = session_config.get("instructions")
        system_prompt = (
            instructions if isinstance(instructions, str) and instructions else "You are a helpful assistant."
        )
        ref_audio = decode_native_ref_audio_from_config(session_config)
        legacy_prepare_fn = getattr(target, "duplex_prepare", None)
        prepare_fn = legacy_prepare_fn or getattr(target, "prepare")
        native = call_native_duplex_prepare(
            prepare_fn,
            system_prompt=system_prompt,
            ref_audio=ref_audio,
            use_legacy_prompt=callable(legacy_prepare_fn),
            target=target,
        )
    return (
        {
            "supported": True,
            "session_id": session_id,
            "epoch": epoch,
            "implementation_level": "model_native_duplex",
            "native_result": as_native_result_dict(native, target=target),
        },
        target,
    )


def append_native_duplex_input(
    target: Any,
    *,
    session_id: str,
    epoch: int,
    seq: int,
    mode: str,
    payload: Any,
    final: bool,
) -> dict[str, Any]:
    append_fn = getattr(target, "append_duplex_input", None)
    if callable(append_fn):
        native = append_fn(
            session_id=session_id,
            epoch=epoch,
            seq=seq,
            mode=mode,
            payload=payload,
            final=final,
        )
    else:
        if mode != "append_audio_chunk":
            raise ValueError(f"native duplex method adapter only supports append_audio_chunk, got {mode!r}")
        if not isinstance(payload, dict):
            raise TypeError("append_audio_chunk payload must be a dict")
        audio_waveform = decode_native_audio_payload(payload)
        prefill_fn = (
            getattr(target, "duplex_prefill", None)
            or getattr(target, "streaming_prefill", None)
            or getattr(target, "prefill")
        )
        prefill_result = prefill_fn(
            audio_waveform=audio_waveform,
            frame_list=payload.get("frame_list"),
            max_slice_nums=int(payload.get("max_slice_nums", 1) or 1),
        )
        if isinstance(prefill_result, dict) and prefill_result.get("success") is False:
            native = {
                "prefill_success": False,
                "is_buffering": True,
                "reason": prefill_result.get("reason", ""),
                "cost_all": prefill_result.get("cost_all"),
            }
            return {
                "supported": True,
                "session_id": session_id,
                "epoch": epoch,
                "seq": seq,
                "mode": mode,
                "final": final,
                "implementation_level": "model_native_duplex",
                "native_result": as_native_result_dict(native, target=target),
            }
        generate_fn = (
            getattr(target, "duplex_generate", None)
            or getattr(target, "streaming_generate", None)
            or getattr(target, "generate")
        )
        force_listen = bool(payload.get("force_listen", False))
        if force_listen:
            break_fn = (
                getattr(target, "duplex_set_break", None)
                or getattr(target, "set_break_event", None)
                or getattr(target, "set_break", None)
            )
            if callable(break_fn):
                break_fn()
            generate_count = getattr(target, "_streaming_generate_count", None)
            if isinstance(generate_count, int) and hasattr(target, "force_listen_count"):
                target.force_listen_count = max(int(getattr(target, "force_listen_count", 0)), generate_count + 1)
        try:
            generate_kwargs = native_duplex_generate_kwargs(target)
            native = generate_fn(force_listen=force_listen, **generate_kwargs)
        except TypeError:
            try:
                native = generate_fn(force_listen_override=force_listen, **generate_kwargs)
            except TypeError:
                try:
                    native = generate_fn(**generate_kwargs)
                except TypeError:
                    native = generate_fn()
        finalize_fn = getattr(target, "duplex_finalize", None) or getattr(target, "finalize", None)
        if callable(finalize_fn):
            finalize_fn()
    return {
        "supported": True,
        "session_id": session_id,
        "epoch": epoch,
        "seq": seq,
        "mode": mode,
        "final": final,
        "implementation_level": "model_native_duplex",
        "native_result": as_native_result_dict(native, target=target),
    }


def decode_native_ref_audio_from_config(session_config: dict[str, Any]) -> Any | None:
    validate_native_ref_audio_config(session_config)
    extra_body = session_config.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}
    audio_data = extra_body.get("ref_audio_data")
    if audio_data is None:
        return None
    if not isinstance(audio_data, str):
        raise TypeError("native duplex ref_audio_data must be base64 pcm_f32le")
    fmt = extra_body.get("ref_audio_format") or "pcm_f32le"
    if fmt != "pcm_f32le":
        raise ValueError(f"unsupported native duplex ref_audio_format: {fmt!r}")
    import numpy as np

    raw = base64.b64decode(audio_data)
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)


def validate_native_ref_audio_config(session_config: dict[str, Any]) -> None:
    extra_body = session_config.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}
    if any(key in session_config for key in ("ref_audio_path", "tts_ref_audio_path")) or any(
        key in extra_body for key in ("ref_audio_path", "tts_ref_audio_path")
    ):
        raise ValueError("native duplex ref_audio_path is not accepted; resolve ref_audio in serving first")


def call_native_duplex_prepare(
    prepare_fn: Any,
    *,
    system_prompt: str,
    ref_audio: Any | None,
    use_legacy_prompt: bool,
    target: Any | None = None,
) -> Any:
    if use_legacy_prompt:
        prefix_system_prompt, suffix_system_prompt = format_native_duplex_system_prompt(target, system_prompt)
        return prepare_fn(
            prefix_system_prompt=prefix_system_prompt,
            suffix_system_prompt=suffix_system_prompt,
            ref_audio=ref_audio,
            prompt_wav_path=None,
        )

    try:
        return prepare_fn(
            prefix_system_prompt=system_prompt,
            ref_audio=ref_audio,
            prompt_wav_path=None,
        )
    except TypeError as prefix_exc:
        try:
            return prepare_fn(
                system_prompt_text=system_prompt,
                ref_audio_path=None,
                prompt_wav_path=None,
            )
        except TypeError:
            raise prefix_exc


def format_native_duplex_system_prompt(target: Any | None, system_prompt: str) -> tuple[str, str | None]:
    """Return model-owned prompt pieces for legacy duplex_prepare hooks."""

    formatter = getattr(target, "format_duplex_system_prompt", None) if target is not None else None
    if callable(formatter):
        formatted = formatter(system_prompt)
        if isinstance(formatted, tuple) and len(formatted) == 2:
            return str(formatted[0]), str(formatted[1]) if formatted[1] is not None else None
        if isinstance(formatted, dict):
            prefix = formatted.get("prefix_system_prompt", system_prompt)
            suffix = formatted.get("suffix_system_prompt")
            return str(prefix), str(suffix) if suffix is not None else None
        if isinstance(formatted, str):
            return formatted, None
        raise TypeError("format_duplex_system_prompt must return str, tuple(prefix, suffix), or dict")

    if target is not None and getattr(target, "native_duplex_uses_minicpmo_legacy_prompt_template", False):
        return (
            f"<|im_start|>system\n{system_prompt}\n<|audio_start|>",
            "<|audio_end|><|im_end|>",
        )

    return system_prompt, None


def decode_native_audio_payload(payload: dict[str, Any]) -> Any:
    audio = payload.get("audio") or payload.get("data")
    if not isinstance(audio, str):
        raise ValueError("audio append payload requires base64 audio")
    fmt = payload.get("format") or "pcm_f32le"
    if fmt != "pcm_f32le":
        raise ValueError(f"native duplex method adapter expects pcm_f32le audio, got {fmt!r}")
    import numpy as np

    return np.frombuffer(base64.b64decode(audio), dtype=np.float32)


def apply_native_duplex_session_config(target: Any, session_config: dict[str, Any]) -> None:
    extra_body = session_config.get("extra_body")
    if not isinstance(extra_body, dict):
        return
    for name in NATIVE_DUPLEX_GENERATE_PARAMS:
        if name in extra_body:
            setattr(target, name, extra_body[name])


def native_duplex_generate_kwargs(target: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for name in NATIVE_DUPLEX_GENERATE_PARAMS:
        value = getattr(target, name, None)
        if value is not None:
            kwargs[name] = value
    return kwargs


def as_native_result_dict(native: Any, *, target: Any | None = None) -> dict[str, Any]:
    if native is None:
        result: dict[str, Any] = {}
    elif isinstance(native, dict):
        result = dict(native)
    else:
        model_dump = getattr(native, "model_dump", None)
        if callable(model_dump):
            result = dict(model_dump())
        elif hasattr(native, "__dict__"):
            result = dict(native.__dict__)
        else:
            result = {"value": native}

    normalize_native_audio_result(result)
    normalize_native_costs(result)
    if target is not None:
        runtime_impl = getattr(target, "runtime_impl", None)
        if isinstance(runtime_impl, str) and runtime_impl:
            result.setdefault("runtime_impl", runtime_impl)
        owned_runtime = getattr(target, "owned_runtime", None)
        if isinstance(owned_runtime, bool):
            result.setdefault("owned_runtime", owned_runtime)
    if "kv_cache_length" not in result and target is not None:
        kv_cache_length = native_kv_cache_length(target)
        if kv_cache_length is not None:
            result["kv_cache_length"] = kv_cache_length
    return result


def normalize_native_audio_result(result: dict[str, Any]) -> None:
    if result.get("audio_data") is not None:
        result.pop("audio_waveform", None)
        return
    waveform = result.pop("audio_waveform", None)
    if waveform is None:
        return
    try:
        import numpy as np

        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        audio_bytes = np.asarray(waveform, dtype=np.float32).reshape(-1).tobytes()
        result["audio_data"] = base64.b64encode(audio_bytes).decode("ascii")
    except Exception as exc:
        result["audio_data_error"] = str(exc)


def normalize_native_costs(result: dict[str, Any]) -> None:
    cost_fields = {
        "cost_llm": "cost_llm_ms",
        "cost_tts_prep": "cost_tts_prep_ms",
        "cost_tts": "cost_tts_ms",
        "cost_token2wav": "cost_token2wav_ms",
        "cost_all": "cost_all_ms",
    }
    for src, dst in cost_fields.items():
        if dst in result or src not in result:
            continue
        value = result.get(src)
        if isinstance(value, int | float):
            result[dst] = float(value) * 1000.0


def native_kv_cache_length(target: Any) -> int | None:
    candidates = [
        target,
        getattr(target, "processor", None),
        getattr(target, "_model", None),
        getattr(getattr(target, "_model", None), "processor", None),
        getattr(target, "model", None),
        getattr(getattr(target, "model", None), "processor", None),
        getattr(target, "duplex", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        value = getattr(candidate, "kv_cache_length", None)
        if isinstance(value, int):
            return value
        if callable(value):
            try:
                computed = value()
            except Exception:
                computed = None
            if isinstance(computed, int):
                return computed
        get_cache_length = getattr(candidate, "get_cache_length", None)
        if callable(get_cache_length):
            try:
                computed = get_cache_length()
            except Exception:
                computed = None
            if isinstance(computed, int):
                return computed
        get_kv_cache_length = getattr(candidate, "_get_kv_cache_length", None)
        if callable(get_kv_cache_length):
            try:
                computed = get_kv_cache_length()
            except Exception:
                computed = None
            if isinstance(computed, int):
                return computed
    return None


class NativeDuplexMethodAdapter:
    """Adapt model-local prepare/prefill/generate methods to worker hooks."""

    _PREPARE_METHODS = ("duplex_prepare", "prepare")
    _PREFILL_METHODS = ("duplex_prefill", "streaming_prefill", "prefill")
    _GENERATE_METHODS = ("duplex_generate", "streaming_generate", "generate")
    runtime_impl = "vllm_omni_native_method_adapter"
    owned_runtime = False

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    @classmethod
    def can_wrap(cls, backend: Any) -> bool:
        has_explicit_duplex_methods = (
            callable(getattr(backend, "duplex_prepare", None))
            and callable(getattr(backend, "duplex_prefill", None))
            and callable(getattr(backend, "duplex_generate", None))
        )
        if has_explicit_duplex_methods:
            return True
        if not getattr(backend, "supports_native_duplex_method_adapter", False):
            return False
        return (
            cls._first_callable(backend, cls._PREPARE_METHODS) is not None
            and cls._first_callable(backend, cls._PREFILL_METHODS) is not None
            and cls._first_callable(backend, cls._GENERATE_METHODS) is not None
        )

    def prepare(
        self,
        *,
        prefix_system_prompt: str,
        ref_audio: Any | None = None,
        prompt_wav_path: str | None = None,
    ) -> Any:
        prepare_fn = self._first_callable(self.backend, self._PREPARE_METHODS)
        if prepare_fn is None:
            raise RuntimeError("native duplex method adapter requires prepare/duplex_prepare")
        if getattr(prepare_fn, "__name__", "") == "duplex_prepare":
            prefix_system_prompt, suffix_system_prompt = format_native_duplex_system_prompt(
                self.backend,
                prefix_system_prompt,
            )
            return prepare_fn(
                prefix_system_prompt=prefix_system_prompt,
                suffix_system_prompt=suffix_system_prompt,
                ref_audio=ref_audio,
                prompt_wav_path=prompt_wav_path,
            )
        try:
            return prepare_fn(
                prefix_system_prompt=prefix_system_prompt,
                ref_audio=ref_audio,
                prompt_wav_path=prompt_wav_path,
            )
        except TypeError as prefix_exc:
            try:
                return prepare_fn(
                    system_prompt_text=prefix_system_prompt,
                    ref_audio_path=None,
                    prompt_wav_path=prompt_wav_path,
                )
            except TypeError:
                raise prefix_exc

    def prefill(
        self,
        *,
        audio_waveform: Any | None,
        frame_list: Any | None = None,
        max_slice_nums: int = 1,
    ) -> Any:
        self.clear_break()
        prefill_fn = self._first_callable(self.backend, self._PREFILL_METHODS)
        if prefill_fn is None:
            raise RuntimeError("native duplex method adapter requires prefill/duplex_prefill")
        return prefill_fn(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            max_slice_nums=max_slice_nums,
        )

    def generate(self, *, force_listen: bool = False, **kwargs: Any) -> Any:
        generate_fn = self._first_callable(self.backend, self._GENERATE_METHODS)
        if generate_fn is None:
            raise RuntimeError("native duplex method adapter requires generate/duplex_generate")

        filtered_kwargs = self._filter_kwargs(generate_fn, kwargs)
        if self._accepts_kwarg(generate_fn, "force_listen"):
            return generate_fn(force_listen=force_listen, **filtered_kwargs)
        if self._accepts_kwarg(generate_fn, "force_listen_override"):
            return generate_fn(force_listen_override=force_listen, **filtered_kwargs)
        return generate_fn(**filtered_kwargs)

    def finalize(self) -> None:
        finalize_fn = self._first_callable(self.backend, ("duplex_finalize", "finalize"))
        if finalize_fn is not None:
            finalize_fn()

    def set_break(self) -> None:
        break_fn = self._first_callable(
            self.backend,
            ("duplex_set_break", "set_break_event", "set_break"),
        )
        if break_fn is not None:
            break_fn()

    def clear_break(self) -> None:
        clear_fn = self._first_callable(
            self.backend,
            ("duplex_clear_break", "clear_break_event", "clear_break"),
        )
        if clear_fn is not None:
            clear_fn()

    def stop(self) -> None:
        stop_fn = self._first_callable(
            self.backend,
            ("duplex_stop", "stop", "set_session_stop"),
        )
        if stop_fn is not None:
            stop_fn()
        reset_fn = getattr(self.backend, "reset_session", None)
        if callable(reset_fn):
            reset_fn()

    def cleanup(self) -> None:
        cleanup_fn = self._first_callable(self.backend, ("duplex_cleanup", "cleanup"))
        if cleanup_fn is not None:
            cleanup_fn()

    @staticmethod
    def _first_callable(target: Any, names: tuple[str, ...]) -> Any | None:
        for name in names:
            value = getattr(target, name, None)
            if callable(value):
                return value
        return None

    @staticmethod
    def _accepts_kwarg(fn: Any, name: str) -> bool:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return True
        if name in signature.parameters:
            return True
        return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())

    @classmethod
    def _filter_kwargs(cls, fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return dict(kwargs)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return dict(kwargs)
        return {name: value for name, value in kwargs.items() if name in signature.parameters}


def _bootstrap_default_providers() -> None:
    global _DEFAULT_PROVIDERS_BOOTSTRAPPED
    if _DEFAULT_PROVIDERS_BOOTSTRAPPED:
        return
    _DEFAULT_PROVIDERS_BOOTSTRAPPED = True
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex_worker_adapter import (
        maybe_load_minicpmo_native_duplex_target,
    )

    register_native_duplex_provider(maybe_load_minicpmo_native_duplex_target)
