# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.engine.arg_utils import _resolve_bundled_hf_config_path
from vllm_omni.entrypoints.omni import OmniBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_resolve_bundled_hf_config_path_uses_cosyvoice3_bundle_by_default():
    resolved = _resolve_bundled_hf_config_path("FunAudioChatCosyVoice3Code2Wav", None)

    assert resolved is not None
    assert resolved.endswith("vllm_omni/model_executor/models/cosyvoice3/hf_config")
    assert (Path(resolved) / "config.json").is_file()


def test_resolve_bundled_hf_config_path_preserves_explicit_override():
    resolved = _resolve_bundled_hf_config_path("FunAudioChatCosyVoice3Code2Wav", "/tmp/custom-hf-config")

    assert resolved == "/tmp/custom-hf-config"


def test_get_stage_model_prefers_stage_override():
    stage = SimpleNamespace(engine_args=SimpleNamespace(model="stage-specific-model"))

    assert OmniBase._get_stage_model(stage, "fallback-model") == "stage-specific-model"


def test_get_stage_model_falls_back_when_stage_override_missing():
    stage = SimpleNamespace(engine_args=SimpleNamespace())

    assert OmniBase._get_stage_model(stage, "fallback-model") == "fallback-model"
