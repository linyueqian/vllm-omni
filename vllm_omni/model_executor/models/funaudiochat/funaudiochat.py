# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.modeling_outputs import BaseModelOutput
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.model_executor.models.funaudiochat.common import (
    ensure_funaudiochat_importable,
    register_funaudiochat_processor,
)

try:
    from vllm.model_executor.models.funaudiochat import (
        FunAudioChatForConditionalGeneration as VllmNativeFunAudioChatForConditionalGeneration,
    )
except ImportError:  # pragma: no cover - environment-specific dependency
    VllmNativeFunAudioChatForConditionalGeneration = None

_NativeFunAudioChatBase = (
    VllmNativeFunAudioChatForConditionalGeneration
    if VllmNativeFunAudioChatForConditionalGeneration is not None
    else nn.Module
)

logger = init_logger(__name__)

DEFAULT_SP_GEN_KWARGS = {
    "text_greedy": True,
    "only_crq_sampling": True,
    "disable_speech": False,
    "force_text_abos": True,
}

_OFFICIAL_CRQ_SAMPLING_DEFAULTS = {
    "repetition_penalty": 1.2,
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 0,
}

_AUDIO_TOKEN_IDS_KEY = "funaudiochat_audio_token_ids"
_CRQ_AUDIO_EMBEDS_KEY = "funaudiochat_crq_audio_embeds"
_CRQ_PAST_KEY_VALUES_KEY = "funaudiochat_crq_past_key_values"
_CURRENT_INPUT_TOKEN_ID_KEY = "funaudiochat_current_input_token_id"
_FORCE_AUDIO_BOS_KEY = "funaudiochat_force_audio_bos_pending"
_FINISH_SPEECH_KEY = "funaudiochat_finish_speech"
_GENERATE_SPEECH_KEY = "funaudiochat_generate_speech"
_SPEECH_IDS_KEY = "funaudiochat_speech_ids"
_TEXT_INPUT_IDS_KEY = "funaudiochat_text_input_ids"
_TEXT_SEQ_LEN_KEY = "funaudiochat_text_seq_len"


@register_funaudiochat_processor
class FunAudioChatForConditionalGeneration(_NativeFunAudioChatBase, SupportsMultiModal):
    supports_multimodal_raw_input_only = True
    supports_multimodal = True
    requires_raw_input_tokens = False
    input_modalities = "audio"
    pooler_output_buffer_keys = ("audio_token_ids",)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        if VllmNativeFunAudioChatForConditionalGeneration is None:
            raise ImportError(
                "Installed vLLM does not expose a native FunAudioChat model. "
                "Upgrade vLLM to a build that includes "
                "`vllm.model_executor.models.funaudiochat`."
            )

        super().__init__(vllm_config=vllm_config, prefix=prefix)
        ensure_funaudiochat_importable()
        from funaudiochat.modeling_funaudiochat import FunAudioChatDecoder  # type: ignore

        self.audio_invert_tower = FunAudioChatDecoder(self.config.audio_config)
        self._patch_audio_invert_tower_sampling_step()
        self.sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
        self.has_preprocess = True
        self.has_postprocess = True
        self.have_multimodal_outputs = False
        self._batch_preprocess_in_progress = False
        self._batch_req_infos: list[dict[str, Any]] = []
        self._batch_sidecar_results: list[dict[str, Any]] = []
        self._postprocess_cursor = 0
        self._logged_stage0_backend = False

    @staticmethod
    def _move_nested_to_cpu(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().to("cpu").contiguous()
        if isinstance(value, tuple):
            return tuple(FunAudioChatForConditionalGeneration._move_nested_to_cpu(v) for v in value)
        if isinstance(value, list):
            return [FunAudioChatForConditionalGeneration._move_nested_to_cpu(v) for v in value]
        return value

    @staticmethod
    def _move_nested_to_device(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        if isinstance(value, tuple):
            return tuple(FunAudioChatForConditionalGeneration._move_nested_to_device(v, device) for v in value)
        if isinstance(value, list):
            return [FunAudioChatForConditionalGeneration._move_nested_to_device(v, device) for v in value]
        return value

    @staticmethod
    def _as_2d_long_tensor(value: Any, device: torch.device) -> torch.Tensor:
        if value is None:
            return torch.empty((1, 0), dtype=torch.long, device=device)
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device, dtype=torch.long)
        else:
            tensor = torch.as_tensor(value, dtype=torch.long, device=device)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _patch_audio_invert_tower_sampling_step(self) -> None:
        if getattr(self.audio_invert_tower, "_vllm_omni_crq_generator_patched", False):
            return

        def _sampling_step_with_generator(
            decoder_self: nn.Module,
            logits: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            next_token_logits = logits[:, -1, :].to(copy=True, dtype=torch.float32, device=logits.device)
            next_token_scores = decoder_self.crq_logits_processor(
                torch.cat([decoder_self.crq_speech_ids, *decoder_self.crq_generate_tokens], dim=-1),
                next_token_logits,
            )

            if decoder_self.crq_do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            return next_tokens, logits

        self.audio_invert_tower.sampling_step = MethodType(_sampling_step_with_generator, self.audio_invert_tower)
        self.audio_invert_tower._vllm_omni_crq_generator_patched = True

    def _empty_audio_token_ids(self, device: torch.device) -> torch.Tensor:
        return torch.full(
            (1, int(self.config.audio_config.group_size)),
            -1,
            dtype=torch.long,
            device=device,
        )

    @staticmethod
    def _sampling_value_at(
        value: torch.Tensor | None,
        req_index: int,
        default: float,
    ) -> float:
        if value is None:
            return float(default)
        if value.ndim == 0:
            return float(value.item())
        if req_index >= value.shape[0]:
            return float(default)
        return float(value[req_index].item())

    @staticmethod
    def _resolve_text_seq_len(
        prev_text_seq_len: Any,
        span_len: int,
    ) -> tuple[int, int]:
        prev = int(prev_text_seq_len or 0)
        if span_len > 1:
            current = prev + span_len
            return current, current
        current = prev if prev > 0 else 1
        return current, current + 1

    @staticmethod
    def _resolve_next_speech_state(
        *,
        sampled_token_id: int,
        generate_speech: bool,
        finish_speech: bool,
        force_audio_bos_pending: bool,
        audio_bos_id: int,
        audio_eos_id: int,
    ) -> tuple[int, bool, bool]:
        if finish_speech:
            return audio_eos_id, False, False

        final_token_id = audio_bos_id if force_audio_bos_pending else sampled_token_id
        next_speech_active = generate_speech or final_token_id == audio_bos_id
        if final_token_id == audio_eos_id:
            next_speech_active = False

        return final_token_id, next_speech_active, False

    def _build_crq_sampling_config(
        self,
        sampling_metadata: Any,
        req_index: int,
    ) -> tuple[LogitsProcessorList, bool]:
        repetition_penalty = self._sampling_value_at(
            getattr(sampling_metadata, "repetition_penalties", None) if sampling_metadata is not None else None,
            req_index,
            _OFFICIAL_CRQ_SAMPLING_DEFAULTS["repetition_penalty"],
        )
        default_temperature = 0.0
        default_top_p = 1.0
        default_top_k = -1.0
        if self.sp_gen_kwargs["text_greedy"]:
            default_temperature = _OFFICIAL_CRQ_SAMPLING_DEFAULTS["temperature"]
            default_top_p = _OFFICIAL_CRQ_SAMPLING_DEFAULTS["top_p"]
            default_top_k = float(_OFFICIAL_CRQ_SAMPLING_DEFAULTS["top_k"])

        temperature = self._sampling_value_at(
            getattr(sampling_metadata, "temperature", None) if sampling_metadata is not None else None,
            req_index,
            default_temperature,
        )
        top_p = self._sampling_value_at(
            getattr(sampling_metadata, "top_p", None) if sampling_metadata is not None else None,
            req_index,
            default_top_p,
        )
        top_k = int(
            round(
                self._sampling_value_at(
                    getattr(sampling_metadata, "top_k", None) if sampling_metadata is not None else None,
                    req_index,
                    default_top_k,
                )
            )
        )

        if self.sp_gen_kwargs["text_greedy"] and temperature <= 0.0:
            temperature = float(_OFFICIAL_CRQ_SAMPLING_DEFAULTS["temperature"])
            if top_p >= 1.0:
                top_p = float(_OFFICIAL_CRQ_SAMPLING_DEFAULTS["top_p"])
            if top_k < 0:
                top_k = int(_OFFICIAL_CRQ_SAMPLING_DEFAULTS["top_k"])

        processors: list[Any] = []
        if repetition_penalty > 0.0 and abs(repetition_penalty - 1.0) > 1e-6:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))

        do_sample = temperature > 0.0
        if do_sample:
            if abs(temperature - 1.0) > 1e-6:
                processors.append(TemperatureLogitsWarper(temperature))
            if top_k > 0:
                processors.append(TopKLogitsWarper(top_k=top_k))
            if 0.0 < top_p < 1.0:
                processors.append(TopPLogitsWarper(top_p=top_p))

        return LogitsProcessorList(processors), do_sample

    def _get_stage0_backend(self) -> str:
        try:
            backend_cls = self.get_language_model().model.layers[0].self_attn.attn.get_attn_backend()
            backend_name = str(backend_cls.get_name())
        except Exception:
            backend_name = "UNKNOWN"
        if not self._logged_stage0_backend:
            logger.info("FunAudioChat stage-0 native language backend: %s", backend_name)
            self._logged_stage0_backend = True
        return backend_name

    def _run_audio_sidecar_step(
        self,
        hidden_state: torch.Tensor,
        current_input_token_id: int,
        speech_ids: torch.Tensor,
        cached_audio_embeds: Any,
        cached_past_key_values: Any,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        current_text_seq_len: int,
    ) -> dict[str, Any]:
        device = hidden_state.device
        text_embed = (
            self.get_language_model()
            .embed_input_ids(torch.tensor([current_input_token_id], device=device, dtype=torch.long))
            .reshape(1, 1, -1)
        )
        speech_inputs_embeds = hidden_state.reshape(1, 1, -1) + text_embed.detach()
        attention_mask = torch.ones((1, max(current_text_seq_len, 1)), dtype=torch.long, device=device)
        position_ids = torch.tensor([[max(current_text_seq_len - 1, 0)]], dtype=torch.long, device=device)

        self.audio_invert_tower.crq_audio_embeds = self._move_nested_to_device(cached_audio_embeds, device)
        self.audio_invert_tower.crq_past_key_values = self._move_nested_to_device(cached_past_key_values, device)
        self.audio_invert_tower.crq_do_sample = do_sample
        self.audio_invert_tower.crq_logits_processor = logits_processor
        self.audio_invert_tower.crq_speech_ids = speech_ids
        self.audio_invert_tower.crq_generate_forward(
            inputs_embeds=speech_inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )

        next_audio_tokens = self.audio_invert_tower.crq_generate_tokens.reshape(1, -1).to(dtype=torch.long)
        eos_token_id = int(self.config.audio_config.eos_token_id)
        finish_speech = bool((next_audio_tokens == eos_token_id).any().item())
        if finish_speech:
            next_audio_tokens = torch.full_like(next_audio_tokens, eos_token_id)

        updated_speech_ids = torch.cat([speech_ids, next_audio_tokens], dim=-1)
        return {
            _AUDIO_TOKEN_IDS_KEY: next_audio_tokens.detach(),
            _CRQ_AUDIO_EMBEDS_KEY: self._move_nested_to_cpu(self.audio_invert_tower.crq_audio_embeds),
            _CRQ_PAST_KEY_VALUES_KEY: self._move_nested_to_cpu(self.audio_invert_tower.crq_past_key_values),
            _FINISH_SPEECH_KEY: finish_speech,
            _SPEECH_IDS_KEY: updated_speech_ids.detach().to("cpu").contiguous(),
        }

    def _run_audio_sidecar_decode_warmup(
        self,
        hidden_state: torch.Tensor,
        current_input_token_id: int,
        speech_ids: torch.Tensor,
        cached_audio_embeds: Any,
        cached_past_key_values: Any,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
    ) -> dict[str, Any]:
        device = hidden_state.device
        text_embed = (
            self.get_language_model()
            .embed_input_ids(torch.tensor([current_input_token_id], device=device, dtype=torch.long))
            .reshape(1, 1, -1)
        )
        speech_inputs_embeds = hidden_state.reshape(1, 1, -1) + text_embed.detach()

        self.audio_invert_tower.crq_audio_embeds = self._move_nested_to_device(cached_audio_embeds, device)
        self.audio_invert_tower.crq_past_key_values = self._move_nested_to_device(cached_past_key_values, device)
        self.audio_invert_tower.crq_do_sample = do_sample
        self.audio_invert_tower.crq_logits_processor = logits_processor
        self.audio_invert_tower.crq_speech_ids = speech_ids
        self.audio_invert_tower.crq_generate_forward(
            inputs_embeds=speech_inputs_embeds,
            return_dict=True,
        )
        return {
            _CRQ_AUDIO_EMBEDS_KEY: self._move_nested_to_cpu(self.audio_invert_tower.crq_audio_embeds),
            _CRQ_PAST_KEY_VALUES_KEY: self._move_nested_to_cpu(self.audio_invert_tower.crq_past_key_values),
        }

    def _run_audio_sidecar_prefill_warmup(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        speech_ids: torch.Tensor,
        cached_audio_embeds: Any,
        cached_past_key_values: Any,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
    ) -> dict[str, Any]:
        device = hidden_states.device
        input_ids = input_ids.to(device=device, dtype=torch.long).reshape(1, -1)
        text_embeds = (
            self.get_language_model()
            .embed_input_ids(input_ids.reshape(-1))
            .reshape(
                1,
                -1,
                hidden_states.shape[-1],
            )
        )
        speech_inputs_embeds = hidden_states.reshape(1, -1, hidden_states.shape[-1]) + text_embeds.detach()

        self.audio_invert_tower.crq_audio_embeds = self._move_nested_to_device(cached_audio_embeds, device)
        self.audio_invert_tower.crq_past_key_values = self._move_nested_to_device(cached_past_key_values, device)
        self.audio_invert_tower.crq_do_sample = do_sample
        self.audio_invert_tower.crq_logits_processor = logits_processor
        self.audio_invert_tower.crq_speech_ids = speech_ids
        self.audio_invert_tower.crq_generate_forward(
            inputs_embeds=speech_inputs_embeds,
            return_dict=True,
        )
        return {
            _CRQ_AUDIO_EMBEDS_KEY: self._move_nested_to_cpu(self.audio_invert_tower.crq_audio_embeds),
            _CRQ_PAST_KEY_VALUES_KEY: self._move_nested_to_cpu(self.audio_invert_tower.crq_past_key_values),
        }

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        del input_embeds
        if not self._batch_preprocess_in_progress:
            self._batch_req_infos = []
            self._batch_sidecar_results = []
            self._postprocess_cursor = 0
            self._batch_preprocess_in_progress = True

        span_len = int(input_ids.shape[0])
        device = input_ids.device
        text_embeds = self.get_language_model().embed_input_ids(input_ids.reshape(-1))
        req_embeds = text_embeds

        generate_speech = bool(info_dict.get(_GENERATE_SPEECH_KEY, False))
        force_audio_bos_pending = bool(info_dict.get(_FORCE_AUDIO_BOS_KEY, self.sp_gen_kwargs["force_text_abos"]))
        speech_ids = self._as_2d_long_tensor(info_dict.get(_SPEECH_IDS_KEY), device)
        current_text_seq_len, next_text_seq_len = self._resolve_text_seq_len(
            info_dict.get(_TEXT_SEQ_LEN_KEY),
            span_len,
        )

        if span_len == 1:
            current_text_embed = text_embeds.reshape(1, -1)
            if generate_speech and speech_ids.shape[-1] >= int(self.config.audio_config.group_size):
                last_group = speech_ids[:, -int(self.config.audio_config.group_size) :]
                audio_features = self.audio_tower(last_group.to(device=device, dtype=torch.long))
                if isinstance(audio_features, BaseModelOutput):
                    audio_features = audio_features.last_hidden_state
                elif isinstance(audio_features, (tuple, list)):
                    audio_features = audio_features[0]
                req_embeds = (current_text_embed + audio_features.reshape(1, -1)) / 2

        current_input_token_id = int(input_ids.reshape(-1)[-1].item())
        self._get_stage0_backend()
        update_dict = {
            _CURRENT_INPUT_TOKEN_ID_KEY: current_input_token_id,
            _FORCE_AUDIO_BOS_KEY: force_audio_bos_pending,
            _GENERATE_SPEECH_KEY: generate_speech,
            _SPEECH_IDS_KEY: speech_ids.detach().to("cpu").contiguous(),
            _TEXT_SEQ_LEN_KEY: next_text_seq_len,
            "audio_token_ids": self._empty_audio_token_ids(device).to("cpu"),
        }

        self._batch_req_infos.append(
            {
                _CURRENT_INPUT_TOKEN_ID_KEY: current_input_token_id,
                _FORCE_AUDIO_BOS_KEY: force_audio_bos_pending,
                _GENERATE_SPEECH_KEY: generate_speech,
                _SPEECH_IDS_KEY: speech_ids.detach().to("cpu").contiguous(),
                _CRQ_AUDIO_EMBEDS_KEY: info_dict.get(_CRQ_AUDIO_EMBEDS_KEY),
                _CRQ_PAST_KEY_VALUES_KEY: info_dict.get(_CRQ_PAST_KEY_VALUES_KEY),
                _TEXT_INPUT_IDS_KEY: input_ids.detach().to("cpu").contiguous(),
                _TEXT_SEQ_LEN_KEY: current_text_seq_len,
            }
        )
        return input_ids, req_embeds, update_dict

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is None:
            self._batch_preprocess_in_progress = False
            return None

        self._batch_sidecar_results = []
        for idx, req_info in enumerate(self._batch_req_infos):
            force_audio_bos_pending = bool(req_info.get(_FORCE_AUDIO_BOS_KEY, False))
            speech_active = bool(req_info.get(_GENERATE_SPEECH_KEY, False))

            sidecar_result = {
                _AUDIO_TOKEN_IDS_KEY: self._empty_audio_token_ids(hidden_states.device).to("cpu"),
                _CRQ_AUDIO_EMBEDS_KEY: req_info.get(_CRQ_AUDIO_EMBEDS_KEY),
                _CRQ_PAST_KEY_VALUES_KEY: req_info.get(_CRQ_PAST_KEY_VALUES_KEY),
                _FORCE_AUDIO_BOS_KEY: force_audio_bos_pending,
                _FINISH_SPEECH_KEY: False,
                _GENERATE_SPEECH_KEY: speech_active,
                _SPEECH_IDS_KEY: req_info.get(_SPEECH_IDS_KEY),
                "audio_token_ids": self._empty_audio_token_ids(hidden_states.device).to("cpu"),
            }

            req_input_ids = self._as_2d_long_tensor(req_info.get(_TEXT_INPUT_IDS_KEY), hidden_states.device).reshape(-1)
            crq_logits_processor, do_sample = self._build_crq_sampling_config(
                sampling_metadata=sampling_metadata,
                req_index=idx,
            )
            if speech_active and not self.sp_gen_kwargs["disable_speech"]:
                sidecar_step = self._run_audio_sidecar_step(
                    hidden_state=hidden_states[idx],
                    current_input_token_id=int(req_info[_CURRENT_INPUT_TOKEN_ID_KEY]),
                    speech_ids=self._as_2d_long_tensor(req_info.get(_SPEECH_IDS_KEY), hidden_states.device),
                    cached_audio_embeds=req_info.get(_CRQ_AUDIO_EMBEDS_KEY),
                    cached_past_key_values=req_info.get(_CRQ_PAST_KEY_VALUES_KEY),
                    logits_processor=crq_logits_processor,
                    do_sample=do_sample,
                    current_text_seq_len=int(req_info.get(_TEXT_SEQ_LEN_KEY, 1)),
                )
                sidecar_result.update(sidecar_step)
                sidecar_result["audio_token_ids"] = sidecar_step[_AUDIO_TOKEN_IDS_KEY]
            elif not self.sp_gen_kwargs["disable_speech"]:
                if req_input_ids.numel() > 1:
                    sidecar_result["_run_prefill_crq_warmup"] = True
                    sidecar_result["_prefill_input_ids"] = req_info.get(_TEXT_INPUT_IDS_KEY)
                    sidecar_result["_prefill_crq_logits_processor"] = crq_logits_processor
                    sidecar_result["_prefill_crq_do_sample"] = do_sample
                else:
                    warmup_state = self._run_audio_sidecar_decode_warmup(
                        hidden_state=hidden_states[idx],
                        current_input_token_id=int(req_info[_CURRENT_INPUT_TOKEN_ID_KEY]),
                        speech_ids=self._as_2d_long_tensor(req_info.get(_SPEECH_IDS_KEY), hidden_states.device),
                        cached_audio_embeds=req_info.get(_CRQ_AUDIO_EMBEDS_KEY),
                        cached_past_key_values=req_info.get(_CRQ_PAST_KEY_VALUES_KEY),
                        logits_processor=crq_logits_processor,
                        do_sample=do_sample,
                    )
                    sidecar_result.update(warmup_state)

            self._batch_sidecar_results.append(sidecar_result)
        self._postprocess_cursor = 0
        self._batch_preprocess_in_progress = False
        return logits

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        if self._postprocess_cursor >= len(self._batch_sidecar_results):
            return {}
        sidecar_result = self._batch_sidecar_results[self._postprocess_cursor]
        self._postprocess_cursor += 1
        if bool(sidecar_result.pop("_run_prefill_crq_warmup", False)):
            prefill_input_ids = sidecar_result.pop("_prefill_input_ids", None)
            if prefill_input_ids is not None:
                warmup_state = self._run_audio_sidecar_prefill_warmup(
                    hidden_states=hidden_states,
                    input_ids=self._as_2d_long_tensor(prefill_input_ids, hidden_states.device).reshape(-1),
                    speech_ids=self._as_2d_long_tensor(sidecar_result.get(_SPEECH_IDS_KEY), hidden_states.device),
                    cached_audio_embeds=sidecar_result.get(_CRQ_AUDIO_EMBEDS_KEY),
                    cached_past_key_values=sidecar_result.get(_CRQ_PAST_KEY_VALUES_KEY),
                    logits_processor=sidecar_result.pop("_prefill_crq_logits_processor"),
                    do_sample=bool(sidecar_result.pop("_prefill_crq_do_sample", False)),
                )
                sidecar_result.update(warmup_state)
        return {
            _AUDIO_TOKEN_IDS_KEY: sidecar_result[_AUDIO_TOKEN_IDS_KEY],
            _CRQ_AUDIO_EMBEDS_KEY: sidecar_result[_CRQ_AUDIO_EMBEDS_KEY],
            _CRQ_PAST_KEY_VALUES_KEY: sidecar_result[_CRQ_PAST_KEY_VALUES_KEY],
            _FORCE_AUDIO_BOS_KEY: sidecar_result[_FORCE_AUDIO_BOS_KEY],
            _FINISH_SPEECH_KEY: sidecar_result[_FINISH_SPEECH_KEY],
            _GENERATE_SPEECH_KEY: sidecar_result[_GENERATE_SPEECH_KEY],
            _SPEECH_IDS_KEY: sidecar_result[_SPEECH_IDS_KEY],
            "audio_token_ids": sidecar_result["audio_token_ids"],
        }

    def postprocess_sampled_tokens(
        self,
        sampled_token_ids: torch.Tensor,
        req_ids: list[str],
        req_id_to_index: dict[str, int],
        model_intermediate_buffer: dict[str, dict[str, Any]],
    ) -> torch.Tensor:
        if sampled_token_ids.numel() == 0:
            return sampled_token_ids

        if sampled_token_ids.ndim == 2 and sampled_token_ids.shape[-1] != 1:
            return sampled_token_ids

        updated_token_ids = sampled_token_ids.clone()
        audio_bos_id = int(self.config.text_config.audio_bos_index)
        audio_eos_id = int(self.config.text_config.audio_eos_index)

        for rid in req_ids:
            req_buffer = model_intermediate_buffer.get(rid)
            if not isinstance(req_buffer, dict):
                continue

            idx = req_id_to_index.get(rid)
            if idx is None:
                continue

            token_slot = updated_token_ids[idx] if updated_token_ids.ndim == 1 else updated_token_ids[idx, 0]
            original_token_id = int(token_slot.item())
            speech_active = bool(req_buffer.get(_GENERATE_SPEECH_KEY, False))
            force_audio_bos_pending = bool(req_buffer.get(_FORCE_AUDIO_BOS_KEY, False))
            finish_speech = bool(req_buffer.pop(_FINISH_SPEECH_KEY, False))

            final_token_id, next_speech_active, next_force_audio_bos_pending = self._resolve_next_speech_state(
                sampled_token_id=original_token_id,
                generate_speech=speech_active,
                finish_speech=finish_speech,
                force_audio_bos_pending=force_audio_bos_pending,
                audio_bos_id=audio_bos_id,
                audio_eos_id=audio_eos_id,
            )

            if final_token_id != original_token_id:
                token_slot.fill_(final_token_id)

            req_buffer[_GENERATE_SPEECH_KEY] = next_speech_active
            req_buffer[_FORCE_AUDIO_BOS_KEY] = next_force_audio_bos_pending

        return updated_token_ids

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
