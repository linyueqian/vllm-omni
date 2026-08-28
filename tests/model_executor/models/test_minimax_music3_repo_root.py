# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repository-root resolution for MiniMax Music 3's multi-component checkpoint.

The acoustic stage declares no ``model_subdir`` because it reads the repo root
itself, so on a hub-id deployment its model path is ``MiniMaxAI/MiniMax-Music3``
rather than a directory. Resolving that as a relative path silently points at
the server's working directory (issue #6638).
"""

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.minimax_music3.weights import (
    _ROOT_MARKERS,
    resolve_repo_root,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_root(path):
    for marker in _ROOT_MARKERS:
        (path / marker).mkdir(parents=True, exist_ok=True)
    return path


def test_resolve_repo_root_accepts_the_root_itself(tmp_path):
    root = _make_root(tmp_path / "snapshot")
    assert resolve_repo_root(str(root)) == root


def test_resolve_repo_root_walks_up_from_a_model_subdir(tmp_path):
    root = _make_root(tmp_path / "snapshot")
    (root / "language_model").mkdir()
    assert resolve_repo_root(str(root / "language_model")) == root


def test_resolve_repo_root_resolves_a_hub_id_to_a_local_snapshot(monkeypatch, tmp_path):
    """A hub id must resolve through the cache, not against the working directory."""
    from huggingface_hub import HfApi

    root = _make_root(tmp_path / "snapshots" / "deadbeef")
    seen = []

    def fake_snapshot_download(self, repo_id, **kwargs):
        seen.append(kwargs)
        return str(root)

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    assert resolve_repo_root("MiniMaxAI/MiniMax-Music3") == root
    # Cache-first, and only the component folders.
    assert seen[0]["local_files_only"] is True
    assert "transformer/*" in seen[0]["allow_patterns"]
    assert "qwen_7B/*" not in seen[0]["allow_patterns"]


def test_resolve_repo_root_falls_back_to_the_hub_when_the_cache_is_incomplete(monkeypatch, tmp_path):
    from huggingface_hub import HfApi

    root = _make_root(tmp_path / "snapshots" / "deadbeef")
    calls = []

    def fake_snapshot_download(self, repo_id, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("incomplete snapshot")
        return str(root)

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    assert resolve_repo_root("MiniMaxAI/MiniMax-Music3") == root
    assert len(calls) == 2


def test_resolve_repo_root_reports_the_original_reference_when_unresolvable(monkeypatch):
    from huggingface_hub import HfApi

    def fake_snapshot_download(self, repo_id, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    with pytest.raises(FileNotFoundError, match="MiniMaxAI/MiniMax-Music3"):
        resolve_repo_root("MiniMaxAI/MiniMax-Music3")


def test_resolve_repo_root_downloads_components_for_a_cold_cache_model_subdir(monkeypatch, tmp_path):
    """Cold-cache stage 0: the talker gets an EXISTING ``language_model`` dir.

    Stage init pre-downloads only ``language_model/`` and ``tokenizer/``, so
    the marker walk fails although the repo id is recoverable from the cache
    layout and one snapshot_download away from working.
    """
    from huggingface_hub import HfApi

    snapshot = tmp_path / "models--MiniMaxAI--MiniMax-Music3" / "snapshots" / "deadbeef"
    (snapshot / "language_model").mkdir(parents=True)
    (snapshot / "tokenizer").mkdir()
    seen = []

    def fake_snapshot_download(self, repo_id, **kwargs):
        seen.append(repo_id)
        _make_root(snapshot)
        return str(snapshot)

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    assert resolve_repo_root(str(snapshot / "language_model")) == snapshot
    assert seen == ["MiniMaxAI/MiniMax-Music3"]


def test_resolve_repo_root_does_not_guess_a_repo_id_for_plain_directories(monkeypatch, tmp_path):
    """A directory outside the HF cache has no recoverable repo id."""
    from huggingface_hub import HfApi

    def fake_snapshot_download(self, repo_id, **kwargs):
        raise AssertionError("must not reach the Hub for a plain directory")

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    plain = tmp_path / "language_model"
    plain.mkdir()
    with pytest.raises(FileNotFoundError, match="expected sibling folders"):
        resolve_repo_root(str(plain))
