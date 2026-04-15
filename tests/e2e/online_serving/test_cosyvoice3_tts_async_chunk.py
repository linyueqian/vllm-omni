# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for CosyVoice3 TTS with the `cosyvoice3_async_chunk.yaml`
streaming deployment config.

Kept in its own file because the `omni_server` fixture is module-scoped
(`tests/conftest.py`). Mixing the default (`cosyvoice3.yaml`) and async-chunk
deploy configs in one file forces mid-module server teardown/restarts that
have been observed to race with the readiness probe and cause spurious
`openai.APIConnectionError: Connection refused` on the first request after
the restart. One deploy variant per file = one server lifecycle per file.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

from pathlib import Path

import pytest

from tests.conftest import OmniServerParams
from tests.utils import hardware_test

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

# Official CosyVoice zero-shot prompt audio and its transcript
REF_AUDIO_URL = "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav"
REF_TEXT = "希望你以后能够做的比我还好呦。"


def get_stage_config(name: str = "cosyvoice3_async_chunk.yaml"):
    """Get the stage config path from vllm_omni model_executor stage_configs."""
    return str(Path(__file__).parent.parent.parent.parent / "vllm_omni" / "model_executor" / "stage_configs" / name)


def get_prompt(prompt_type="zh"):
    prompts = {
        "zh": "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的感动让我热泪盈眶。",
    }
    return prompts.get(prompt_type, prompts["zh"])


tts_async_chunk_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_stage_config(),
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="cosyvoice3_async_chunk",
    )
]


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_async_chunk_server_params, indirect=True)
def test_voice_clone_zh_002(omni_server, openai_client) -> None:
    """
    Test voice cloning TTS with Chinese text via async_chunk streaming.
    Deploy Setting: cosyvoice3_async_chunk.yaml
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio (streamed)
    Input Setting: stream=True
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("zh"),
        "stream": True,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config)
