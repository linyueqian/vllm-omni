"""Tests for SeedTTSTextDataset and SeedTTSTextSampleRequest."""
from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Minimal vllm stubs — installed before any vllm_omni import so the module
# can be loaded without a real vLLM installation.
# ---------------------------------------------------------------------------

def _ensure_vllm_stubs() -> None:
    if "vllm.benchmarks.datasets" in sys.modules:
        return

    @dataclass
    class SampleRequest:
        prompt: str = ""
        prompt_len: int = 0
        expected_output_len: int = 0
        multi_modal_data: Any = None
        request_id: str = ""

    class BenchmarkDataset:
        def __init__(self, dataset_path="", random_seed=0, disable_shuffle=False, **kw):
            self.dataset_path = dataset_path
            self.random_seed = random_seed
            self.disable_shuffle = disable_shuffle

        def maybe_oversample_requests(self, out, num_requests, prefix, no_oversample):
            pass

    class TokenizerLike:
        pass

    def get_cached_tokenizer(t):
        return t

    vllm_mod = types.ModuleType("vllm")
    vllm_benchmarks = types.ModuleType("vllm.benchmarks")
    vllm_benchmarks_datasets = types.ModuleType("vllm.benchmarks.datasets")
    vllm_tokenizers = types.ModuleType("vllm.tokenizers")
    vllm_tokenizers_hf = types.ModuleType("vllm.tokenizers.hf")

    vllm_benchmarks_datasets.BenchmarkDataset = BenchmarkDataset
    vllm_benchmarks_datasets.SampleRequest = SampleRequest
    vllm_tokenizers.TokenizerLike = TokenizerLike
    vllm_tokenizers_hf.get_cached_tokenizer = get_cached_tokenizer

    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.benchmarks"] = vllm_benchmarks
    sys.modules["vllm.benchmarks.datasets"] = vllm_benchmarks_datasets
    sys.modules["vllm.tokenizers"] = vllm_tokenizers
    sys.modules["vllm.tokenizers.hf"] = vllm_tokenizers_hf


_ensure_vllm_stubs()

# Load the data module directly (bypasses vllm_omni.__init__ heavy imports).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "vllm_omni" / "benchmarks" / "data_modules" / "seed_tts_dataset.py"
_MODULE_NAME = "vllm_omni.benchmarks.data_modules.seed_tts_dataset"

if _MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _mod
    _spec.loader.exec_module(_mod)

from vllm_omni.benchmarks.data_modules.seed_tts_dataset import (  # noqa: E402
    SeedTTSTextDataset,
    SeedTTSTextSampleRequest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def seed_tts_root(tmp_path: Path) -> Path:
    """Minimal seed-tts-style directory with 5 entries."""
    locale_dir = tmp_path / "en"
    locale_dir.mkdir()
    wav_dir = locale_dir / "prompt-wavs"
    wav_dir.mkdir()
    for i in range(5):
        (wav_dir / f"utt{i:03d}.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    meta = "\n".join(
        f"utt{i:03d}|ref text {i}|prompt-wavs/utt{i:03d}.wav|target text {i}"
        for i in range(5)
    )
    (locale_dir / "meta.lst").write_text(meta, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_seed_tts_text_dataset_omits_ref_audio(seed_tts_root):
    ds = SeedTTSTextDataset(
        dataset_path=str(seed_tts_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )
    tokenizer = MagicMock()
    tokenizer.encode = lambda text, **kw: [0] * len(text.split())
    requests = ds.sample(tokenizer, num_requests=3)
    assert len(requests) == 3
    for req in requests:
        assert isinstance(req, SeedTTSTextSampleRequest)
        assert req.seed_tts_speech_extra is None or "ref_audio" not in (req.seed_tts_speech_extra or {})
        assert req.seed_tts_ref_wav_path == ""
        assert "target text" in req.prompt
