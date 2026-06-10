# SPDX-License-Identifier: Apache-2.0
"""Registry of TTS serving adapters.

Adapters register themselves by stage key via ``@register_tts_adapter``.
``resolve_adapter`` picks the single adapter whose ``matches()`` predicate
accepts the deployment's present stage keys + ``model_arch``. A stage key may
map to more than one adapter (e.g. VoxCPM / VoxCPM2 both expose
``latent_generator``); ``matches()`` disambiguates.
"""

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    DiffusionTTSAdapter,
    OutputPolicy,
    PreparedRequest,
    SpeechServingContext,
    TTSModelAdapter,
)

logger = init_logger(__name__)

TTS_ADAPTER_REGISTRY: dict[str, list[type[TTSModelAdapter]]] = {}


def register_tts_adapter(cls: type[TTSModelAdapter]) -> type[TTSModelAdapter]:
    """Class decorator: index ``cls`` under each of its ``stage_keys``."""
    for key in cls.stage_keys:
        TTS_ADAPTER_REGISTRY.setdefault(key, []).append(cls)
    return cls


def all_tts_stage_keys() -> frozenset[str]:
    """All registered stage keys (replaces the hand-maintained union)."""
    return frozenset(TTS_ADAPTER_REGISTRY)


def resolve_adapter(stage_keys: set[str], model_arch: str | None) -> type[TTSModelAdapter] | None:
    """Return the single adapter matching the given stage keys + arch.

    Returns ``None`` if nothing matches. Raises ``TypeError`` if more than one
    adapter matches (an ambiguous registration that must be fixed).
    """
    candidates: list[type[TTSModelAdapter]] = []
    seen: set[type[TTSModelAdapter]] = set()
    for key in stage_keys:
        for cls in TTS_ADAPTER_REGISTRY.get(key, []):
            if cls not in seen:
                seen.add(cls)
                candidates.append(cls)
    matched = [cls for cls in candidates if cls.matches(set(stage_keys), model_arch)]
    if len(matched) > 1:
        raise TypeError(
            f"Ambiguous TTS adapter match for stage_keys={sorted(stage_keys)} "
            f"model_arch={model_arch!r}: {[c.__name__ for c in matched]}"
        )
    return matched[0] if matched else None


# Import adapter modules for their registration side effects. Keep this at the
# bottom so the registry helpers above are defined first.
from vllm_omni.entrypoints.openai.tts_adapters import qwen3_tts  # noqa: E402,F401

__all__ = [
    "ARTTSAdapter",
    "DiffusionTTSAdapter",
    "OutputPolicy",
    "PreparedRequest",
    "SpeechServingContext",
    "TTSModelAdapter",
    "TTS_ADAPTER_REGISTRY",
    "all_tts_stage_keys",
    "register_tts_adapter",
    "resolve_adapter",
]
