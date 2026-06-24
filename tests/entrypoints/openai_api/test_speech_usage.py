# SPDX-License-Identifier: Apache-2.0
"""Unit tests for speech token-usage accounting (issue #4646).

These exercise the pure usage logic without a model/server: input-token
breakdown (text + reference-audio gating) and output-token accumulation.
"""

from dataclasses import dataclass

import pytest

from vllm_omni.entrypoints.openai.speech_usage import (
    SpeechOutputTokenCounter,
    build_speech_usage,
    final_output_token_count,
    gate_audio_tokens,
    qwen3_tts_input_token_details,
    usage_headers,
)


# A deterministic stand-in for a real tokenizer: 1 token per whitespace word.
def word_count(text: str) -> int:
    return len(text.split())


# --- input breakdown: gating ------------------------------------------------


def test_gate_audio_tokens_base_in_context_counts_ref_frames():
    assert (
        gate_audio_tokens(
            task_type="Base", x_vector_only_mode=False, icl_mode_override=None, ref_code_length=120
        )
        == 120
    )


def test_gate_audio_tokens_customvoice_is_zero():
    assert (
        gate_audio_tokens(
            task_type="CustomVoice", x_vector_only_mode=False, icl_mode_override=None, ref_code_length=120
        )
        == 0
    )


def test_gate_audio_tokens_x_vector_only_is_zero_even_with_ref():
    # x-vector cloning inserts NO codec frames, so audio_tokens must be 0
    # even though ref_code_length is populated.
    assert (
        gate_audio_tokens(
            task_type="Base", x_vector_only_mode=True, icl_mode_override=None, ref_code_length=120
        )
        == 0
    )


def test_gate_audio_tokens_icl_override_wins():
    assert (
        gate_audio_tokens(
            task_type="Base", x_vector_only_mode=False, icl_mode_override=False, ref_code_length=120
        )
        == 0
    )
    assert (
        gate_audio_tokens(
            task_type="Base", x_vector_only_mode=True, icl_mode_override=True, ref_code_length=80
        )
        == 80
    )


def test_gate_audio_tokens_bad_ref_length_is_zero():
    assert (
        gate_audio_tokens(
            task_type="Base", x_vector_only_mode=False, icl_mode_override=None, ref_code_length="oops"
        )
        == 0
    )


# --- input breakdown: Qwen3-TTS reproduces issue #4646 observation ----------


def test_customvoice_text_scales_with_input_no_audio():
    short = qwen3_tts_input_token_details(
        input_text="hi there",
        instructions=None,
        tts_params={"task_type": ["CustomVoice"]},
        count_text_tokens=word_count,
    )
    long = qwen3_tts_input_token_details(
        input_text="hi there this is a much longer sentence",
        instructions=None,
        tts_params={"task_type": ["CustomVoice"]},
        count_text_tokens=word_count,
    )
    assert short.audio_tokens == 0 and long.audio_tokens == 0
    assert long.text_tokens > short.text_tokens  # scales with input


def test_base_icl_text_tracks_input_audio_tracks_ref():
    # The core #4646 fix: text_tokens reflect the input (not dropped), and
    # audio_tokens track the reference audio independently.
    params = {"task_type": ["Base"], "x_vector_only_mode": [False], "ref_code_length": [100]}
    short = qwen3_tts_input_token_details(
        input_text="hi", instructions=None, tts_params=params, count_text_tokens=word_count
    )
    long = qwen3_tts_input_token_details(
        input_text="hi there friend", instructions=None, tts_params=params, count_text_tokens=word_count
    )
    # text now varies with input (the bug was that it was dropped for Base)
    assert long.text_tokens > short.text_tokens
    # audio fixed by the reference, independent of input text
    assert short.audio_tokens == long.audio_tokens == 100


def test_instructions_count_toward_text():
    without = qwen3_tts_input_token_details(
        input_text="hello world",
        instructions=None,
        tts_params={"task_type": ["CustomVoice"]},
        count_text_tokens=word_count,
    )
    with_instr = qwen3_tts_input_token_details(
        input_text="hello world",
        instructions="speak slowly and warmly",
        tts_params={"task_type": ["CustomVoice"]},
        count_text_tokens=word_count,
    )
    assert with_instr.text_tokens == without.text_tokens + word_count("speak slowly and warmly")


# --- build usage ------------------------------------------------------------


def test_build_speech_usage_aggregates_and_totals():
    details = qwen3_tts_input_token_details(
        input_text="hi there",
        instructions=None,
        tts_params={"task_type": ["Base"], "x_vector_only_mode": [False], "ref_code_length": [100]},
        count_text_tokens=word_count,
    )
    usage = build_speech_usage(details, output_tokens=250)
    assert usage.input_tokens == details.text_tokens + 100
    assert usage.output_tokens == 250
    assert usage.total_tokens == usage.input_tokens + 250
    assert usage.input_token_details.text_tokens == 2
    assert usage.input_token_details.audio_tokens == 100


def test_usage_headers_are_strings():
    details = qwen3_tts_input_token_details(
        input_text="hi",
        instructions=None,
        tts_params={"task_type": ["CustomVoice"]},
        count_text_tokens=word_count,
    )
    headers = usage_headers(build_speech_usage(details, output_tokens=10))
    assert headers["x-usage-output-tokens"] == "10"
    assert all(isinstance(v, str) for v in headers.values())


# --- output token accumulation ---------------------------------------------


@dataclass
class _FakeCompletion:
    token_ids: list


@dataclass
class _FakeRes:
    stage_id: int | None
    outputs: list


def test_streaming_total_sums_deltas():
    acc = SpeechOutputTokenCounter()
    for delta in ([1, 2, 3], [4, 5], [6]):
        acc.observe(_FakeRes(stage_id=0, outputs=[_FakeCompletion(token_ids=delta)]))
    assert acc.streaming_total() == 6  # 3 + 2 + 1 deltas


def test_non_stage0_outputs_ignored():
    acc = SpeechOutputTokenCounter()
    acc.observe(_FakeRes(stage_id=1, outputs=[_FakeCompletion(token_ids=[1, 2, 3, 4])]))
    assert acc.streaming_total() == 0


def test_stage_id_none_is_counted():
    acc = SpeechOutputTokenCounter()
    acc.observe(_FakeRes(stage_id=None, outputs=[_FakeCompletion(token_ids=[1, 2])]))
    assert acc.streaming_total() == 2


def test_final_total_uses_last_full_length():
    acc = SpeechOutputTokenCounter()
    # FINAL_ONLY: a single final res carrying the full sequence.
    acc.observe(_FakeRes(stage_id=0, outputs=[_FakeCompletion(token_ids=list(range(101)))]))
    assert acc.final_total() == 101


def test_final_output_token_count_helper():
    final = _FakeRes(stage_id=0, outputs=[_FakeCompletion(token_ids=list(range(64)))])
    assert final_output_token_count(final) == 64
    assert final_output_token_count(None) == 0
    assert final_output_token_count(_FakeRes(stage_id=2, outputs=[_FakeCompletion(token_ids=[1])])) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
