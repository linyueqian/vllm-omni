# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Drama 3 main-token sampler with repetition-aware re-draw.

The request parameters (top_k=1) make the first draw the arg-max token. The
re-draw uses fixed parameters (temperature 1.0, top_k 30, top_p 0.95), so over
many identical rows it spreads across the near-tied candidates it keeps. A row whose
first draw repeats its recent history must therefore often return another
token; a row without a repeat always keeps the arg-max token.
"""

import pytest
import torch
from torch import nn
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.fish_omni.fish_omni_model import FishOmniForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

EOS = 5
SEMANTIC_BEGIN = 100
CODEBOOK_SIZE = 8
VOCAB = 128
ROWS = 2000
# Near-tied candidates: the arg-max semantic token, end of speech, and two more
# semantic tokens; top_p 0.95 drops the last of them.
ARGMAX, OTHER_A, OTHER_B = SEMANTIC_BEGIN + 1, SEMANTIC_BEGIN + 2, SEMANTIC_BEGIN + 3


class SemanticRange(nn.Module):
    def is_semantic(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens >= SEMANTIC_BEGIN) & (tokens < SEMANTIC_BEGIN + CODEBOOK_SIZE)


def _sampler_model() -> FishOmniForConditionalGeneration:
    """A model instance with only the state ``sample`` reads (no backbone)."""
    model = FishOmniForConditionalGeneration.__new__(FishOmniForConditionalGeneration)
    nn.Module.__init__(model)
    candidates = torch.tensor([EOS, *range(SEMANTIC_BEGIN, SEMANTIC_BEGIN + CODEBOOK_SIZE)])
    model.register_buffer("_main_token_candidates", candidates, persistent=False)
    model.tower = SemanticRange()
    return model


def _logits(top: int, second: int) -> torch.Tensor:
    logits = torch.full((ROWS, VOCAB), -30.0)
    logits[:, OTHER_A] = 0.0
    logits[:, OTHER_B] = 0.0
    logits[:, second] = 0.01
    logits[:, top] = 0.02
    return logits


def _metadata(history: list[int]) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.full((ROWS,), 1.0),
        all_greedy=False,
        all_random=True,
        top_p=torch.full((ROWS,), 1.0),
        top_k=torch.full((ROWS,), 1, dtype=torch.long),
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(ROWS),
        presence_penalties=torch.zeros(ROWS),
        repetition_penalties=torch.ones(ROWS),
        output_token_ids=[list(history) for _ in range(ROWS)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def _sample(logits: torch.Tensor, history: list[int]) -> torch.Tensor:
    torch.manual_seed(0)
    output = _sampler_model().sample(logits, _metadata(history))
    assert output is not None
    assert output.sampled_token_ids.shape == (ROWS, 1)
    return output.sampled_token_ids.squeeze(-1)


def test_non_repeated_token_keeps_first_draw():
    tokens = _sample(_logits(ARGMAX, OTHER_A), history=[OTHER_A, OTHER_B])
    assert bool((tokens == ARGMAX).all())


def test_repeated_semantic_token_is_redrawn():
    tokens = _sample(_logits(ARGMAX, OTHER_A), history=[OTHER_B, ARGMAX])
    kept = (tokens == ARGMAX).float().mean().item()
    assert 0.3 < kept < 0.7
    assert set(tokens.tolist()) <= {ARGMAX, OTHER_A, OTHER_B}


def test_redraw_may_end_speech():
    tokens = _sample(_logits(ARGMAX, EOS), history=[ARGMAX])
    assert bool((tokens == EOS).any())


def test_end_of_speech_is_never_redrawn():
    tokens = _sample(_logits(EOS, OTHER_A), history=[EOS, EOS])
    assert bool((tokens == EOS).all())


def test_repeat_outside_window_keeps_first_draw():
    tokens = _sample(_logits(ARGMAX, OTHER_A), history=[ARGMAX] + [OTHER_B] * 16)
    assert bool((tokens == ARGMAX).all())
