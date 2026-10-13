# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Drama 3 main-token sampler with repetition-aware re-draw."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.fish_omni import fish_omni_model
from vllm_omni.model_executor.models.fish_omni.fish_omni_model import FishOmniForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

EOS = 5
SEMANTIC_BEGIN = 100
CODEBOOK_SIZE = 8
# Candidate columns: end of speech, then the semantic tokens.
CANDIDATES = torch.tensor([EOS, *range(SEMANTIC_BEGIN, SEMANTIC_BEGIN + CODEBOOK_SIZE)])


def _column(token: int) -> int:
    return int((CANDIDATES == token).nonzero())


def _model() -> SimpleNamespace:
    def is_semantic(tokens: torch.Tensor) -> torch.Tensor:
        return (tokens >= SEMANTIC_BEGIN) & (tokens < SEMANTIC_BEGIN + CODEBOOK_SIZE)

    return SimpleNamespace(_main_token_candidates=CANDIDATES, tower=SimpleNamespace(is_semantic=is_semantic))


def _sample(monkeypatch, first: list[int], redraw: list[int], history: list[list[int]]) -> list[int]:
    """Run ``sample`` with scripted draws; returns the sampled main tokens."""
    draws = iter([first, redraw])
    calls = []

    def scripted_sample_rows(logits, temperature, top_k, top_p):
        calls.append((temperature.tolist(), top_k.tolist(), top_p.tolist()))
        return torch.tensor([_column(t) for t in next(draws)])

    monkeypatch.setattr(fish_omni_model, "sample_rows", scripted_sample_rows)
    rows = len(first)
    metadata = SimpleNamespace(
        temperature=torch.full((rows,), 0.8),
        top_k=torch.full((rows,), 5, dtype=torch.long),
        top_p=torch.full((rows,), 0.7),
        output_token_ids=history,
    )
    logits = torch.zeros((rows, 256))
    output = FishOmniForConditionalGeneration.sample(_model(), logits, metadata)
    # The first draw uses the request parameters; the re-draw uses fixed ones.
    assert calls[0] == ([pytest.approx(0.8)] * rows, [5] * rows, [pytest.approx(0.7)] * rows)
    assert calls[1] == ([1.0] * rows, [30] * rows, [pytest.approx(0.95)] * rows)
    assert output.sampled_token_ids.shape == (rows, 1)
    return output.sampled_token_ids.squeeze(-1).tolist()


def test_repeated_semantic_token_is_redrawn(monkeypatch):
    assert _sample(monkeypatch, first=[101], redraw=[103], history=[[100, 101, 102]]) == [103]


def test_non_repeated_token_keeps_first_draw(monkeypatch):
    assert _sample(monkeypatch, first=[104], redraw=[103], history=[[100, 101, 102]]) == [104]


def test_redraw_may_end_speech(monkeypatch):
    assert _sample(monkeypatch, first=[101], redraw=[EOS], history=[[101]]) == [EOS]


def test_end_of_speech_is_never_redrawn(monkeypatch):
    assert _sample(monkeypatch, first=[EOS], redraw=[101], history=[[EOS, EOS]]) == [EOS]


def test_repeat_outside_window_keeps_first_draw(monkeypatch):
    history = [[101] + [102] * 16]
    assert _sample(monkeypatch, first=[101], redraw=[103], history=history) == [101]


def test_rows_are_independent(monkeypatch):
    history = [[101], [100], [], [103]]
    tokens = _sample(monkeypatch, first=[101, 101, 101, 103], redraw=[104, 105, 106, EOS], history=history)
    assert tokens == [104, 101, 101, EOS]
