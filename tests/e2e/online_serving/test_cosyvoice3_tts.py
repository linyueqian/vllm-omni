# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for CosyVoice3 TTS model with voice cloning.

These tests verify the /v1/audio/speech endpoint works correctly with
the CosyVoice3 model, which requires reference audio for voice cloning.

The official CosyVoice zero-shot prompt audio is fetched from GitHub
and encoded as a base64 data URI for the API requests.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import base64
import urllib.request
from pathlib import Path

import httpx
import pytest

from tests.conftest import OmniServerParams
from tests.utils import hardware_test

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

STAGE_CONFIG = str(
    Path(__file__).parent.parent.parent.parent / "vllm_omni" / "model_executor" / "stage_configs" / "cosyvoice3.yaml"
)
EXTRA_ARGS = [
    "--trust-remote-code",
    "--disable-log-stats",
]
TEST_PARAMS = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=STAGE_CONFIG,
        server_args=EXTRA_ARGS,
    )
]

# Official CosyVoice zero-shot prompt audio and its transcript
_REF_AUDIO_URL = "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav"
_REF_TEXT = "希望你以后能够做的比我还好呦。"
_ref_audio_cache: str | None = None


def _get_ref_audio_data_uri() -> str:
    """Fetch official CosyVoice zero-shot prompt audio and return as data URI.

    The result is cached so the download only happens once per test session.
    """
    global _ref_audio_cache
    if _ref_audio_cache is not None:
        return _ref_audio_cache

    with urllib.request.urlopen(_REF_AUDIO_URL, timeout=30) as resp:
        wav_bytes = resp.read()
    b64 = base64.b64encode(wav_bytes).decode()
    _ref_audio_cache = f"data:audio/wav;base64,{b64}"
    return _ref_audio_cache


def make_speech_request(
    host: str,
    port: int,
    text: str,
    ref_audio: str,
    ref_text: str,
    timeout: float = 180.0,
) -> httpx.Response:
    """Make a request to the /v1/audio/speech endpoint for CosyVoice3."""
    url = f"http://{host}:{port}/v1/audio/speech"
    payload = {
        "input": text,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
    }

    with httpx.Client(timeout=timeout) as client:
        return client.post(url, json=payload)


def verify_wav_audio(content: bytes) -> bool:
    """Verify that content is valid WAV audio data."""
    if len(content) < 44:
        return False
    return content[:4] == b"RIFF" and content[8:12] == b"WAVE"


MIN_AUDIO_BYTES = 5000


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestCosyVoice3TTS:
    """E2E tests for CosyVoice3 TTS model."""

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_speech_voice_clone_basic(self, omni_server) -> None:
        """Test basic voice cloning TTS generation with official reference audio."""
        ref_audio = _get_ref_audio_data_uri()
        response = make_speech_request(
            host=omni_server.host,
            port=omni_server.port,
            text="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的感动让我热泪盈眶。",
            ref_audio=ref_audio,
            ref_text=_REF_TEXT,
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert response.headers.get("content-type") == "audio/wav"
        assert verify_wav_audio(response.content), "Response is not valid WAV audio"
        assert len(response.content) > MIN_AUDIO_BYTES, (
            f"Audio content too small ({len(response.content)} bytes), expected at least {MIN_AUDIO_BYTES} bytes"
        )

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_speech_missing_ref_audio_rejected(self, omni_server) -> None:
        """Request without ref_audio should return an error."""
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
        payload = {
            "input": "This should fail without reference audio.",
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)

        data = response.json()
        assert "error" in data or "message" in data, f"Expected error response for missing ref_audio, got: {data}"

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_speech_missing_ref_text_rejected(self, omni_server) -> None:
        """Request with ref_audio but no ref_text should return an error."""
        ref_audio = _get_ref_audio_data_uri()
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
        payload = {
            "input": "This should fail without reference text.",
            "ref_audio": ref_audio,
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)

        data = response.json()
        assert "error" in data or "message" in data, f"Expected error response for missing ref_text, got: {data}"

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_speech_english_text(self, omni_server) -> None:
        """Test voice cloning with English synthesis text."""
        ref_audio = _get_ref_audio_data_uri()
        response = make_speech_request(
            host=omni_server.host,
            port=omni_server.port,
            text="Hello, this is a voice cloning test with English text.",
            ref_audio=ref_audio,
            ref_text=_REF_TEXT,
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert verify_wav_audio(response.content), "Response is not valid WAV audio"
        assert len(response.content) > MIN_AUDIO_BYTES
