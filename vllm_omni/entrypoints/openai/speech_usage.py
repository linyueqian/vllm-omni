# SPDX-License-Identifier: Apache-2.0
"""Token-usage accounting for the Speech (``/v1/audio/speech``) API.

Why this module exists (issue #4646)
------------------------------------
For staged TTS models the engine prompt is a PLACEHOLDER: the serving layer
builds ``prompt_token_ids = [1] * prefill_len`` and lets the model rebuild the
real conditioning (text / ref_audio / ref_text) into ``inputs_embeds`` later.
So ``len(prompt_token_ids)`` is NOT a faithful count of what the caller sent.

For Qwen3-TTS specifically the placeholder length mirrors the model prefill:
  * CustomVoice/VoiceDesign: the full input text is embedded in the prefill, so
    the placeholder length scales with the input text.
  * Base in-context voice cloning: the prefill embeds only ``codec_bos`` + the
    reference-audio codec frames; the input text is consumed incrementally
    during DECODE, not prefill. So the placeholder length tracks the *reference
    audio* and is independent of the input text.

That is why ``usage.prompt_tokens`` (== ``len(prompt_token_ids)``) looked wrong
for Base: it counted reference-audio frames instead of the synthesized text.

This module computes usage from the *semantic* inputs instead:

    input_tokens  = text_tokens + audio_tokens
        text_tokens  -> tokens of ``input`` (+ ``instructions``); the text to speak
        audio_tokens -> reference-audio codec frames used as voice-clone
                        conditioning, counted ONLY when in-context cloning is
                        actually active (see ``gate_audio_tokens``)
    output_tokens = generated codec/audio tokens (stage-0 decode steps)
    total_tokens  = input_tokens + output_tokens

Naming (``input_tokens``/``output_tokens``) follows OpenAI's ``speech.audio.done``
event; the ``input_token_details`` breakdown follows OpenAI's realtime/chat
convention of never folding audio into an opaque text count.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vllm_omni.entrypoints.openai.protocol.audio import (
    SpeechInputTokenDetails,
    SpeechTokenUsage,
)


def _first(value: Any, default: Any = None) -> Any:
    """Unwrap the singleton-list convention used by ``tts_params``.

    ``tts_params`` wraps scalars in 1-element lists (e.g. ``task_type=["Base"]``)
    because the model side batches per request. This returns the inner scalar.
    """
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


def gate_audio_tokens(
    *,
    task_type: str | None,
    x_vector_only_mode: bool,
    icl_mode_override: bool | None,
    ref_code_length: Any,
) -> int:
    """Reference-audio codec frames that actually enter the prefill.

    Mirrors the ``in_context_mode`` decision in
    ``Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information``:
    reference codec frames are prefilled ONLY for Base in-context voice cloning.
    They are NOT prefilled for:
      * CustomVoice / VoiceDesign (no reference audio at all), or
      * Base ``x_vector_only_mode`` (the reference audio is reduced to a single
        speaker embedding vector; no per-frame codec context is inserted).

    Counting ``ref_code_length`` outside the in-context path would re-introduce
    the issue #4646 inconsistency in the other direction, so we gate it here.
    """
    in_context = (task_type == "Base") and not x_vector_only_mode
    # An explicit per-request ``voice_clone_prompt.icl_mode`` wins if present
    # (matches the estimator, which lets the payload override the default).
    if icl_mode_override is not None:
        in_context = bool(icl_mode_override)
    if not in_context:
        return 0
    try:
        return max(0, int(ref_code_length)) if ref_code_length is not None else 0
    except (TypeError, ValueError):
        return 0


def qwen3_tts_input_token_details(
    *,
    input_text: str,
    instructions: str | None,
    tts_params: dict[str, Any],
    count_text_tokens: Callable[[str], int],
) -> SpeechInputTokenDetails:
    """Compute the input-token breakdown for a Qwen3-TTS request.

    ``count_text_tokens`` tokenizes a string with the model's *text* tokenizer
    and returns the token count. ``tts_params`` is the finalized param dict from
    the adapter build (it already carries ``ref_code_length`` for ICL clones and
    the resolved ``task_type`` / ``x_vector_only_mode``).
    """
    # Text tokens = the text to synthesize plus the style/emotion instructions,
    # because Qwen3-TTS tokenizes and prepends the instruction block too.
    text = input_text or ""
    instr = instructions or ""
    text_tokens = count_text_tokens(text) if text else 0
    if instr.strip():
        text_tokens += count_text_tokens(instr)

    # Audio tokens = reference codec frames, gated to the in-context clone path.
    voice_clone_prompt = _first(tts_params.get("voice_clone_prompt"), None)
    icl_override = None
    if isinstance(voice_clone_prompt, dict):
        icl_flag = voice_clone_prompt.get("icl_mode")
        if isinstance(icl_flag, bool):
            icl_override = icl_flag
    audio_tokens = gate_audio_tokens(
        task_type=_first(tts_params.get("task_type"), "CustomVoice"),
        x_vector_only_mode=bool(_first(tts_params.get("x_vector_only_mode"), False)),
        icl_mode_override=icl_override,
        ref_code_length=_first(tts_params.get("ref_code_length"), None),
    )
    return SpeechInputTokenDetails(text_tokens=int(text_tokens), audio_tokens=int(audio_tokens))


def build_speech_usage(details: SpeechInputTokenDetails, output_tokens: int) -> SpeechTokenUsage:
    """Assemble the final usage object from the input breakdown + output count."""
    input_tokens = int(details.text_tokens) + int(details.audio_tokens)
    out = max(0, int(output_tokens))
    return SpeechTokenUsage(
        input_tokens=input_tokens,
        output_tokens=out,
        total_tokens=input_tokens + out,
        input_token_details=details,
    )


def usage_headers(usage: SpeechTokenUsage) -> dict[str, str]:
    """Render usage as response headers for the non-streaming raw-bytes path.

    The non-streaming ``/v1/audio/speech`` response body is raw audio, so usage
    cannot ride in JSON; expose it as ``x-usage-*`` headers instead.
    """
    return {
        "x-usage-input-tokens": str(usage.input_tokens),
        "x-usage-output-tokens": str(usage.output_tokens),
        "x-usage-total-tokens": str(usage.total_tokens),
        "x-usage-input-text-tokens": str(usage.input_token_details.text_tokens),
        "x-usage-input-audio-tokens": str(usage.input_token_details.audio_tokens),
    }


def _is_stage0(res: Any) -> bool:
    """True for stage-0 (the AR/codec-token-generating stage) outputs.

    Multi-stage pipelines tag outputs with ``stage_id``; the codec tokens come
    from stage 0. Single-stage models may not set ``stage_id`` (``None``), in
    which case we count their outputs too.
    """
    stage_id = getattr(res, "stage_id", None)
    return stage_id in (0, None)


@dataclass
class SpeechOutputTokenCounter:
    """Accumulates generated stage-0 (codec/audio) tokens off the engine stream.

    ``coerce_param_message_types`` configures the engine output shape:
      * streaming (DELTA): each ``res.outputs[0].token_ids`` is a *delta* slice;
        the total is the SUM of deltas -> use ``streaming_total()``.
      * non-streaming (FINAL_ONLY): a single final ``res`` carries the *full*
        token sequence -> use ``final_total()`` (the last length seen).

    We track both so the caller picks the correct one for its path instead of
    guessing. (A streaming run never sends FINAL_ONLY and vice-versa, so the two
    counters do not interfere within a single request.)
    """

    delta_sum: int = 0
    last_len: int = 0

    def observe(self, res: Any) -> None:
        if not _is_stage0(res):
            return
        outputs = getattr(res, "outputs", None)
        if not outputs:
            return
        token_ids = getattr(outputs[0], "token_ids", None)
        if not token_ids:
            return
        n = len(token_ids)
        self.delta_sum += n
        self.last_len = n

    def streaming_total(self) -> int:
        """Output token count for the streaming (DELTA) path: sum of deltas."""
        return self.delta_sum

    def final_total(self) -> int:
        """Output token count for the non-streaming (FINAL_ONLY) path.

        Prefer the full final length; fall back to the delta sum if only deltas
        were observed (defensive — should not happen under FINAL_ONLY).
        """
        return self.last_len or self.delta_sum


def final_output_token_count(final_output: Any) -> int:
    """Generated stage-0 token count from a single non-streaming final output."""
    if final_output is None or not _is_stage0(final_output):
        return 0
    outputs = getattr(final_output, "outputs", None)
    if not outputs:
        return 0
    token_ids = getattr(outputs[0], "token_ids", None)
    return len(token_ids) if token_ids else 0
