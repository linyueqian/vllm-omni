# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E offline tests for OmniVoice TTS model with text input and audio output.

Uses GPUGenerationWorker for both stages (iterative unmasking + DAC decoder).
CUDA graph is disabled via enforce_eager=true (required for dynamic shapes).
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

from pathlib import Path

import pytest

from tests.conftest import modify_stage_config
from tests.utils import hardware_test

MODEL = "k2-fsa/OmniVoice"


def get_stage_config(name: str = "omnivoice.yaml"):
    """Get the OmniVoice stage config path."""
    return str(Path(__file__).parent.parent.parent.parent / "vllm_omni" / "model_executor" / "stage_configs" / name)


def get_eager_config():
    """Get stage config with enforce_eager=true for both stages."""
    path = modify_stage_config(
        get_stage_config(),
        updates={
            "stage_args": {
                0: {"engine_args.enforce_eager": "true"},
                1: {"engine_args.enforce_eager": "true"},
            },
        },
    )
    return path


tts_server_params = [
    pytest.param(
        (MODEL, get_eager_config()),
        id="omnivoice_eager",
    )
]


def get_prompt():
    """Text prompt for text-to-audio."""
    return "Hello, this is a test for text to audio."


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", tts_server_params, indirect=True)
def test_omnivoice_text_to_audio(omni_runner, omni_runner_handler) -> None:
    """
    Test text input processing and audio output via offline Omni runner.
    Deploy Setting: omnivoice.yaml + enforce_eager=true
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False
    Datasets: few requests
    """
    request_config = {"input": get_prompt()}
    omni_runner_handler.send_audio_speech_request(request_config)
