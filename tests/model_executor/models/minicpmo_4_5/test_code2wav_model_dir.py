# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``minicpmo_4_5_code2wav._resolve_model_dir``.

Covers the hub/CI deployment path where ``model_config.model`` is a repo id
rather than a local directory (issue #5442): asset lookups must resolve to
the downloaded snapshot instead of treating the repo id as a relative path.
"""

from __future__ import annotations

import huggingface_hub
import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    _resolve_model_dir,
)


def test_local_directory_is_returned_unchanged(tmp_path, monkeypatch):
    def _fail(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("snapshot_download must not be called for local dirs")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fail)
    assert _resolve_model_dir(str(tmp_path)) == str(tmp_path)


def test_repo_id_resolves_via_snapshot_download(tmp_path, monkeypatch):
    calls = {}

    def _fake_snapshot_download(model_ref, revision=None, allow_patterns=None):
        calls["model_ref"] = model_ref
        calls["revision"] = revision
        calls["allow_patterns"] = allow_patterns
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)
    resolved = _resolve_model_dir("openbmb/MiniCPM-o-4_5", revision="abc123")
    assert resolved == str(tmp_path / "snapshot")
    assert calls["model_ref"] == "openbmb/MiniCPM-o-4_5"
    assert calls["revision"] == "abc123"
    assert calls["allow_patterns"] == ["assets/*"]


def test_snapshot_download_failure_propagates(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("offline and not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _raise)
    with pytest.raises(FileNotFoundError):
        _resolve_model_dir("openbmb/MiniCPM-o-4_5")
