# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import torch

from vllm_omni.model_executor.stage_input_processors.cosyvoice3 import talker2code2wav_async_chunk, text2flow


def _source_output(request_id: str, prompt_ids: list[int], out_ids: list[int], mm: dict):
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=prompt_ids,
        outputs=[SimpleNamespace(token_ids=out_ids, multimodal_output=mm)],
    )


def test_text2flow_supports_batched_source_outputs():
    stage_list = [
        SimpleNamespace(
            engine_outputs=[
                _source_output("req-0", [10, 11], [1, 2, 3], {"speech_token": torch.tensor([[1, 2]])}),
                _source_output("req-1", [20, 21], [4, 5], {"speech_token": torch.tensor([[3, 4]])}),
            ]
        )
    ]

    outputs = text2flow(stage_list=stage_list, engine_input_source=[0], prompt=None)

    assert len(outputs) == 2
    assert outputs[0]["prompt_token_ids"] == [1, 2, 3]
    assert outputs[1]["prompt_token_ids"] == [4, 5]
    assert outputs[0]["additional_information"]["prefix_ids"] == [10, 11]
    assert outputs[1]["additional_information"]["prefix_ids"] == [20, 21]


def test_talker2code2wav_async_chunk_filters_special_tokens_and_emits_payload():
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": 2,
                    "codec_left_context_frames": 2,
                    "codec_vocab_size": 6561,
                }
            }
        ),
    )
    request = SimpleNamespace(
        external_req_id="rid-0",
        output_token_ids=[1, 2, 6562, 3],
        additional_information={
            "speech_token": [torch.tensor([[11, 12, 13]])],
            "speech_feat": [torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])],
            "embedding": [torch.tensor([[0.5, 0.6]])],
        },
        is_finished=lambda: True,
    )

    payload = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=True,
    )

    assert payload is not None
    assert payload["finished"].item() is True
    assert payload["code_predictor_codes"] == [1, 2, 3]
    assert payload["left_context_size"] == 2
    assert "speech_token" in payload
    assert "speech_feat" in payload
    assert "embedding" in payload


def test_talker2code2wav_async_chunk_emits_eof_when_finished_without_valid_codes():
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": 25,
                    "codec_left_context_frames": 25,
                    "codec_vocab_size": 6561,
                }
            }
        ),
    )
    request = SimpleNamespace(
        external_req_id="rid-eof",
        output_token_ids=[6561, 6562],  # all filtered out
        additional_information={},
        is_finished=lambda: True,
    )

    payload = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=True,
    )

    assert payload is not None
    assert payload["code_predictor_codes"] == []
    assert payload["finished"].item() is True


def test_talker2code2wav_async_chunk_does_not_reemit_without_new_tokens():
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": 2,
                    "codec_left_context_frames": 2,
                    "codec_vocab_size": 6561,
                }
            }
        ),
    )
    request = SimpleNamespace(
        external_req_id="rid-stable",
        output_token_ids=[1, 2],
        additional_information={},
        is_finished=lambda: False,
    )

    payload1 = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=False,
    )
    payload2 = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=False,
    )

    assert payload1 is not None
    assert payload1["code_predictor_codes"] == [1, 2]
    assert payload2 is None


def test_talker2code2wav_async_chunk_emits_terminal_eof_without_duplicate_audio():
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": 2,
                    "codec_left_context_frames": 2,
                    "codec_vocab_size": 6561,
                }
            }
        ),
    )
    request = SimpleNamespace(
        external_req_id="rid-tail",
        output_token_ids=[3, 4],
        additional_information={},
        is_finished=lambda: False,
    )

    payload_stream = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=False,
    )
    payload_final = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=True,
    )

    assert payload_stream is not None
    assert payload_stream["finished"].item() is False
    assert payload_stream["code_predictor_codes"] == [3, 4]
    assert payload_final is not None
    assert payload_final["finished"].item() is True
    assert payload_final["code_predictor_codes"] == []
