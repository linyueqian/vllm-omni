import logging
import os
import time

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.inputs.duplex_intermediate import (
    build_duplex_intermediate_buffer,
    set_ref_audio,
    set_tts_handoff,
)

logger = logging.getLogger(__name__)


def _extract_first_audio_ref(multi_modal_data):
    if not isinstance(multi_modal_data, dict):
        return None
    audio_data = multi_modal_data.get("audio")
    if audio_data is None:
        return None
    if isinstance(audio_data, list):
        if not audio_data:
            return None
        audio_data = audio_data[0]

    samples = None
    sample_rate = None
    if isinstance(audio_data, tuple) and len(audio_data) >= 2:
        samples, sample_rate = audio_data[0], audio_data[1]
    elif isinstance(audio_data, dict):
        sample_rate = audio_data.get("sample_rate") or audio_data.get("sampling_rate") or audio_data.get("sr")
        for key in ("audio", "wav", "samples", "array", "waveform"):
            if key in audio_data:
                samples = audio_data[key]
                break
    if samples is None or sample_rate is None:
        return None

    waveform = torch.as_tensor(samples, dtype=torch.float32)
    if waveform.ndim > 1:
        if waveform.shape[0] <= 2 and waveform.shape[-1] > waveform.shape[0]:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.mean(dim=-1)
    return waveform.reshape(-1).cpu(), int(sample_rate)


def _coerce_token_id_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return None
    out = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            return None
    return out


def _to_transport_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value


def _coerce_int(value):
    if hasattr(value, "detach"):
        flat = value.detach().cpu().reshape(-1)
        if flat.numel() == 0:
            return None
        value = flat[0].item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _special_token_ids_from_mm_output(mm_output):
    if not isinstance(mm_output, dict):
        return {}
    meta = mm_output.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    special_token_ids = mm_output.get("special_token_ids")
    if not isinstance(special_token_ids, dict):
        special_token_ids = {}
    flat_meta = {
        key.removeprefix("meta."): value
        for key, value in mm_output.items()
        if isinstance(key, str) and key.startswith("meta.")
    }
    return {
        key: value
        for key, value in (
            (key, _coerce_int(value))
            for source in (special_token_ids, meta, flat_meta)
            for key, value in source.items()
        )
        if value is not None and value >= 0
    }


def _has_native_duplex_prompt_metadata(mm_output):
    if not isinstance(mm_output, dict):
        return False
    return mm_output.get("duplex_prompt_token_ids") is not None or mm_output.get("ids.prompt") is not None


def _require_native_tts_boundary_metadata(special_token_ids):
    if special_token_ids.get("tts_bos_token_id") is None:
        raise ValueError(
            "MiniCPM-o native duplex TTS handoff requires tokenizer-derived "
            "<|tts_bos|> metadata; refusing to infer special token ids."
        )


def _native_tts_boundary_token_ids(special_token_ids):
    # Official duplex conditions the talker on mid-unit <|speak|> tokens AND
    # the <|turn_eos|> token+hidden (its embedding is the trained stop
    # signal); only chunk terminators and framing tokens bound the slice.
    return {
        token_id
        for token_id in (
            special_token_ids.get("tts_eos_token_id"),
            special_token_ids.get("tts_pad_token_id"),
            special_token_ids.get("listen_token_id"),
            special_token_ids.get("chunk_eos_token_id"),
            special_token_ids.get("chunk_tts_eos_token_id"),
            special_token_ids.get("unit_token_id"),
            special_token_ids.get("unit_end_token_id"),
        )
        if token_id is not None
    }


def _native_duplex_segment_output_ids(
    output_ids: list[int],
    output_text: str,
    streaming_context,
    *,
    request_id: str,
) -> tuple[list[int], str]:
    """Slice the cumulative thinker output down to the current segment.

    Tracks how many output tokens/characters were already handed to the
    talker in the orchestrator's streaming bridge state. A new request id
    (new epoch after barge-in) or a shrunken output list resets the counter.
    """
    bridge_states = getattr(streaming_context, "bridge_states", None)
    if not isinstance(bridge_states, dict):
        return output_ids, output_text
    state = bridge_states.setdefault("minicpmo45_tts_handoff", {})
    if state.get("request_id") != request_id:
        state["request_id"] = request_id
        state["sent_output_len"] = 0
        state["sent_text_len"] = 0
    sent_len = state.get("sent_output_len", 0)
    if not isinstance(sent_len, int) or sent_len < 0 or sent_len > len(output_ids):
        sent_len = 0
    sent_text_len = state.get("sent_text_len", 0)
    if not isinstance(sent_text_len, int) or sent_text_len < 0 or sent_text_len > len(output_text):
        sent_text_len = 0
    state["sent_output_len"] = len(output_ids)
    state["sent_text_len"] = len(output_text)
    return output_ids[sent_len:], output_text[sent_text_len:]


def _build_tts_scheduler_prompt_token_ids(
    tts_token_ids: torch.Tensor | None,
    llm_output_ids: list[int],
    prompt_token_ids: list[int],
) -> list[int]:
    if tts_token_ids is not None:
        ids = _coerce_token_id_list(tts_token_ids)
        if ids:
            return ids
    if llm_output_ids:
        return llm_output_ids
    if prompt_token_ids:
        return prompt_token_ids[-1:]
    raise ValueError("MiniCPM-o TTS stage requires at least one scheduler prompt token")


def llm2tts(
    source_outputs,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
    _streaming_context=None,
):
    """Convert thinker stage output to talker stage input for MiniCPMO Omni.

    Extracts from thinker output:
      - Full hidden states (prompt + generated) for speaker embedding extraction
      - Prompt token IDs (for finding spk_bos/spk_eos positions)
      - Generated token IDs (for decoding TTS text)

    The talker model will:
      1. Find <|spk_bos|>/<|spk_eos|> positions in prompt_token_ids
      2. Extract speaker embedding from hidden states at those positions
      3. Decode generated text and extract TTS content
      4. Run ConditionalChatTTS pipeline
    """
    llm_outputs = source_outputs
    tts_inputs = []

    if not isinstance(prompt, list):
        prompt = [prompt]

    profile_enabled = os.environ.get("MINICPMO45_PROFILE_LOGS") == "1"
    multi_modal_data = {}
    for llm_output, p in zip(llm_outputs, prompt):
        if isinstance(p, dict):
            multi_modal_data[llm_output.request_id] = p.get("multi_modal_data", None)
        else:
            multi_modal_data[llm_output.request_id] = getattr(p, "multi_modal_data", None)

    for i, llm_output in enumerate(llm_outputs):
        t0 = time.perf_counter()
        output = llm_output.outputs[0]
        mm_output = output.multimodal_output if isinstance(output.multimodal_output, dict) else {}
        special_token_ids = _special_token_ids_from_mm_output(mm_output)
        prompt_token_ids = (
            _coerce_token_id_list(mm_output.get("duplex_prompt_token_ids"))
            or _coerce_token_id_list(mm_output.get("ids.prompt"))
            or list(llm_output.prompt_token_ids)
        )
        llm_output_ids = getattr(output, "token_ids", None)
        if llm_output_ids is None:
            llm_output_ids = getattr(output, "cumulative_token_ids", [])
        # Always copy: CompletionOutput.token_ids can alias the upstream
        # detokenizer's live token list. Forwarding that exact object as the
        # talker prompt makes the stage-1 streaming update extend the list
        # with itself (state and update bind the same object), doubling the
        # thinker's recorded output every segment until the TTS engine input
        # buffer overflows.
        llm_output_ids = list(llm_output_ids)
        thinker_text = getattr(output, "text", "") or ""
        if _has_native_duplex_prompt_metadata(mm_output):
            # The thinker's resumable duplex request reports cumulative
            # output ids/text, but earlier segments are already folded into
            # the prompt by the scheduler session update, and the forwarded
            # hidden states only cover the current prompt + current segment.
            # Hand stage 1 exactly one segment per handoff so token/hidden
            # alignment holds, the talker prompt grows linearly with new
            # tokens instead of quadratically, and downstream transcripts
            # carry per-unit deltas instead of re-sending the whole reply.
            llm_output_ids, thinker_text = _native_duplex_segment_output_ids(
                llm_output_ids,
                thinker_text,
                _streaming_context,
                request_id=str(llm_output.request_id),
            )
        prompt_token_ids_len = len(prompt_token_ids)

        latent = mm_output.get("latent", None)
        if latent is None:
            latent = output.hidden_states if hasattr(output, "hidden_states") else None
            if latent is None:
                raise ValueError("No latent or hidden_states found in thinker output")

        thinker_hidden_states = latent.detach()
        if thinker_hidden_states.ndim == 3 and thinker_hidden_states.shape[0] == 1:
            thinker_hidden_states = thinker_hidden_states.squeeze(0)

        # Build full token sequence and extract TTS region
        full_token_ids = prompt_token_ids + llm_output_ids

        tts_bos_id = special_token_ids.get("tts_bos_token_id")
        is_native_duplex_handoff = _has_native_duplex_prompt_metadata(mm_output)
        if is_native_duplex_handoff:
            _require_native_tts_boundary_metadata(special_token_ids)
        tts_end_ids = _native_tts_boundary_token_ids(special_token_ids)

        # Plain-chat (use_tts_template) fallback: for non-duplex requests the thinker
        # does not surface special_token_ids, so resolve the MiniCPM-o 4.5 <|tts_bos|>
        # (151703) boundary directly and bound the spoken region at <|im_end|> (151645),
        # mirroring the pre-duplex code path so chat-completions audio still works.
        if tts_bos_id is None and not is_native_duplex_handoff:
            tts_bos_id = 151703
            tts_end_ids = set(tts_end_ids) | {151645}

        tts_bos_idx = None
        # For native duplex the resumable prompt folds every earlier unit, so
        # a <|tts_bos|> from an already-spoken reply can sit mid-prompt; only
        # a boundary folded as the FINAL prompt token (this unit's decision)
        # or one inside the current segment may start the slice, or stale
        # text would be re-handed to the talker on text-less continuations.
        search_start = max(0, prompt_token_ids_len - 1) if is_native_duplex_handoff else 0
        for idx_t in range(search_start, len(full_token_ids)):
            if full_token_ids[idx_t] == tts_bos_id:
                tts_bos_idx = idx_t + 1

        tts_eos_idx = None
        if tts_bos_idx is not None:
            for idx_t in range(tts_bos_idx, len(full_token_ids)):
                if full_token_ids[idx_t] in tts_end_ids:
                    tts_eos_idx = idx_t
                    break

        tts_token_ids_slice = tts_hidden_slice = None
        if tts_bos_idx is not None and thinker_hidden_states.shape[0] > tts_bos_idx:
            end_idx = tts_eos_idx if tts_eos_idx is not None else thinker_hidden_states.shape[0]
            tts_token_ids_slice = torch.tensor(full_token_ids[tts_bos_idx:end_idx], dtype=torch.long)
            tts_hidden_slice = thinker_hidden_states[tts_bos_idx:end_idx].to(torch.float32).contiguous()
        elif is_native_duplex_handoff:
            # Official MiniCPM-o duplex does not prefill an assistant
            # <|tts_bos|> boundary before generation.  The first generated
            # token is the listen/speak decision; hidden states for TTS start
            # after that decision token and stop at the next duplex state token.
            listen_id = special_token_ids.get("listen_token_id")
            generated_start = prompt_token_ids_len
            if llm_output_ids and llm_output_ids[0] != listen_id:
                tts_start_idx = generated_start + 1
                tts_end_idx = len(full_token_ids)
                for idx_t in range(tts_start_idx, len(full_token_ids)):
                    if full_token_ids[idx_t] in tts_end_ids:
                        tts_end_idx = idx_t
                        break
                tts_end_idx = min(tts_end_idx, int(thinker_hidden_states.shape[0]))
                if tts_end_idx > tts_start_idx:
                    tts_token_ids_slice = torch.tensor(full_token_ids[tts_start_idx:tts_end_idx], dtype=torch.long)
                    tts_hidden_slice = thinker_hidden_states[tts_start_idx:tts_end_idx].to(torch.float32).contiguous()
        if profile_enabled:
            logger.info(
                "llm2tts profile req=%s prompt_tokens=%d output_tokens=%d hidden_shape=%s tts_tokens=%d total_ms=%.3f",
                getattr(llm_output, "request_id", f"idx-{i}"),
                prompt_token_ids_len,
                len(llm_output_ids),
                tuple(thinker_hidden_states.shape),
                0 if tts_token_ids_slice is None else int(tts_token_ids_slice.shape[0]),
                (time.perf_counter() - t0) * 1000,
            )
            logger.info(
                "llm2tts token trace req=%s special=%s prompt_tail=%s output_head=%s output_tail=%s",
                getattr(llm_output, "request_id", f"idx-{i}"),
                special_token_ids,
                prompt_token_ids[-16:],
                llm_output_ids[:32],
                llm_output_ids[-16:],
            )

        model_intermediate_buffer = build_duplex_intermediate_buffer(
            request_id=str(llm_output.request_id),
            prompt_token_ids=prompt_token_ids,
            output_token_ids=llm_output_ids,
            output_text=thinker_text,
            stream_output=is_native_duplex_handoff,
            minicpmo45_native_duplex=is_native_duplex_handoff,
        )
        if is_native_duplex_handoff:
            turn_eos_id = special_token_ids.get("turn_eos_token_id")
            if turn_eos_id is not None:
                # The talker detects turn end from <|turn_eos|> inside the
                # handed condition (official conditions on its embedding).
                model_intermediate_buffer.setdefault("meta", {})["turn_eos_token_id"] = int(turn_eos_id)
        req_mm_data = multi_modal_data.get(llm_output.request_id)
        ref_audio = _extract_first_audio_ref(req_mm_data)
        if ref_audio is not None:
            ref_waveform, ref_sr = ref_audio
            set_ref_audio(model_intermediate_buffer, _to_transport_list(ref_waveform), ref_sr)
        set_tts_handoff(
            model_intermediate_buffer,
            _coerce_token_id_list(tts_token_ids_slice) if tts_token_ids_slice is not None else None,
            _to_transport_list(tts_hidden_slice) if tts_hidden_slice is not None else None,
        )

        scheduler_prompt_token_ids = _build_tts_scheduler_prompt_token_ids(
            tts_token_ids_slice,
            llm_output_ids,
            prompt_token_ids,
        )
        tts_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=scheduler_prompt_token_ids,
                model_intermediate_buffer=model_intermediate_buffer,
                multi_modal_data=(
                    multi_modal_data[llm_output.request_id]
                    if requires_multimodal_data and multi_modal_data.get(llm_output.request_id) is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )

    return tts_inputs


def tts2t2w(
    source_outputs,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
    _streaming_context=None,
):
    """Convert talker stage output to code2wav stage input for MiniCPMO Omni.

    Extracts mel_spec from talker's multimodal output and passes it to
    the code2wav stage for Vocos vocoder (mel → waveform) conversion.
    """
    tts_outputs = source_outputs
    t2w_inputs = []

    if not isinstance(prompt, list):
        prompt = [prompt]

    multi_modal_data = {}
    for tts_output, p in zip(tts_outputs, prompt):
        if isinstance(p, dict):
            multi_modal_data[tts_output.request_id] = p.get("multi_modal_data", None)
        else:
            multi_modal_data[tts_output.request_id] = getattr(p, "multi_modal_data", None)

    for i, tts_output in enumerate(tts_outputs):
        output = tts_output.outputs[0]

        mel_spec = None
        waveform = None
        if hasattr(output, "multimodal_output") and isinstance(output.multimodal_output, dict):
            import torch as _torch

            mel_spec = output.multimodal_output.get("mel_spec")
            waveform = output.multimodal_output.get("model_outputs")
            if waveform is None:
                waveform = output.multimodal_output.get("audio")
            # The 4.5 talker already runs DVAE+Vocos internally and produces a
            # 1-D waveform tensor; it is stored under `model_outputs` which the
            # output_processor renames to the stage's `engine_output_type`
            # (e.g. "latent"). Recover it here.
            latent = output.multimodal_output.get("latent")
            import logging as _logging

            _log = _logging.getLogger(__name__)
            if latent is not None:
                if isinstance(latent, _torch.Tensor):
                    _log.info(
                        "tts2t2w: latent tensor shape=%s dtype=%s numel=%d",
                        tuple(latent.shape),
                        latent.dtype,
                        latent.numel(),
                    )
                elif isinstance(latent, list):
                    _log.info(
                        "tts2t2w: latent is list len=%d type0=%s shape0=%s",
                        len(latent),
                        type(latent[0]).__name__ if latent else None,
                        tuple(latent[0].shape) if latent and isinstance(latent[0], _torch.Tensor) else None,
                    )
                else:
                    _log.info("tts2t2w: latent type=%s", type(latent).__name__)
            if isinstance(latent, list) and latent:
                cand = latent[0]
                if isinstance(cand, _torch.Tensor):
                    latent = cand
            if isinstance(latent, _torch.Tensor):
                if latent.dim() == 1 and latent.numel() > 1000:
                    if waveform is None:
                        waveform = latent
                elif latent.dim() == 2 and 1 in latent.shape and latent.numel() > 1000:
                    if waveform is None:
                        waveform = latent.reshape(-1)
                elif latent.dim() >= 2 and 100 in latent.shape and mel_spec is None:
                    mel_spec = latent

        if mel_spec is None and waveform is None:
            import logging

            logging.getLogger(__name__).warning(
                "tts2t2w: no mel_spec/waveform found in talker output (multimodal_output keys: %s)",
                list(output.multimodal_output.keys())
                if hasattr(output, "multimodal_output") and isinstance(output.multimodal_output, dict)
                else "N/A",
            )

        model_intermediate_buffer = {}
        if waveform is not None:
            model_intermediate_buffer["waveform"] = _to_transport_list(waveform)
        elif mel_spec is not None:
            model_intermediate_buffer["mel_spec"] = _to_transport_list(mel_spec)

        t2w_prompt_token_ids = _coerce_token_id_list(getattr(output, "token_ids", None))
        if not t2w_prompt_token_ids:
            t2w_prompt_token_ids = list(getattr(tts_output, "prompt_token_ids", []) or [])
        if not t2w_prompt_token_ids:
            raise ValueError("MiniCPM-o token2wav stage requires at least one scheduler prompt token")
        t2w_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=t2w_prompt_token_ids,
                model_intermediate_buffer=model_intermediate_buffer,
                multi_modal_data=(
                    multi_modal_data[tts_output.request_id]
                    if requires_multimodal_data and multi_modal_data.get(tts_output.request_id) is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )

    return t2w_inputs
