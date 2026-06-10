# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TTS serving adapter registry (RFC #4327).

These exercise pure-Python registry/resolution logic and do not load any model
or GPU resources.
"""

import pytest

from vllm_omni.entrypoints.openai.tts_adapters import (
    TTS_ADAPTER_REGISTRY,
    ARTTSAdapter,
    DiffusionTTSAdapter,
    all_tts_stage_keys,
    resolve_adapter,
)
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import Qwen3TTSAdapter


def test_qwen3_tts_registered_by_stage_key():
    assert "qwen3_tts" in TTS_ADAPTER_REGISTRY
    assert Qwen3TTSAdapter in TTS_ADAPTER_REGISTRY["qwen3_tts"]


def test_all_stage_keys_includes_qwen3_tts():
    assert "qwen3_tts" in all_tts_stage_keys()


def test_resolve_qwen3_tts():
    assert resolve_adapter({"qwen3_tts"}, None) is Qwen3TTSAdapter


def test_resolve_unknown_returns_none():
    assert resolve_adapter({"not_a_real_stage"}, None) is None
    assert resolve_adapter(set(), None) is None


def test_qwen3_tts_adapter_metadata():
    assert Qwen3TTSAdapter.name == "qwen3_tts"
    assert Qwen3TTSAdapter.backend == "ar"
    assert issubclass(Qwen3TTSAdapter, ARTTSAdapter)


def test_shared_stage_key_disambiguation_via_matches():
    """Two adapters may share a stage key; matches() disambiguates by arch.

    Mirrors the VoxCPM / VoxCPM2 ``latent_generator`` case without touching the
    module-level registry shared by other tests.
    """
    local: dict[str, list] = {}

    def _register(cls):
        for key in cls.stage_keys:
            local.setdefault(key, []).append(cls)
        return cls

    @_register
    class _AdapterA(ARTTSAdapter):
        stage_keys = frozenset({"shared_stage"})
        name = "a"

        @classmethod
        def matches(cls, stage_keys, model_arch):
            return model_arch == "ArchA"

        async def build(self, request):  # pragma: no cover - not invoked
            raise NotImplementedError

    @_register
    class _AdapterB(ARTTSAdapter):
        stage_keys = frozenset({"shared_stage"})
        name = "b"

        @classmethod
        def matches(cls, stage_keys, model_arch):
            return model_arch == "ArchB"

        async def build(self, request):  # pragma: no cover - not invoked
            raise NotImplementedError

    def _resolve(stage_keys, arch):
        matched = [c for cands in local.values() for c in cands if c.matches(set(stage_keys), arch)]
        return matched

    assert _resolve({"shared_stage"}, "ArchA") == [_AdapterA]
    assert _resolve({"shared_stage"}, "ArchB") == [_AdapterB]
    assert _resolve({"shared_stage"}, "ArchC") == []


def test_diffusion_adapter_extra_body_params_fallback():
    """Without a backing pipeline declaring EXTRA_BODY_PARAMS, the diffusion
    adapter degrades to an empty frozenset (implementable before #3572)."""

    class _DiffAdapter(DiffusionTTSAdapter):
        stage_keys = frozenset({"diff_stage"})
        name = "diff"

        async def build(self, request):  # pragma: no cover - not invoked
            raise NotImplementedError

    assert _DiffAdapter.extra_body_params() == frozenset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
