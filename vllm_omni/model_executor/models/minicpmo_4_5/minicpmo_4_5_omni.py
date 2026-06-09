# Copyright 2025 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
from functools import cached_property

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.minicpmo_4_5.duplex_policy import MiniCPMO45DuplexPolicy
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMDummyInputsBuilder,
    MiniCPMO45OmniLLMMultiModalProcessor,
    MiniCPMO45OmniLLMProcessingInfo,
    MiniCPMOConfig,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMO45OmniLLMMultiModalProcessor,
    info=MiniCPMO45OmniLLMProcessingInfo,
    dummy_inputs=MiniCPMO45OmniLLMDummyInputsBuilder,
)
class MiniCPMO45OmniForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, SupportsMRoPE):
    """MiniCPM-o 2.6 Omni model for conditional generation.

    This model supports multi-stage processing:
    - thinker: Image preprocessing + Vision encoder + 3D resampler
    - talker: LLM generation
    - code2wav: Speech output
    """

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "(<image>./</image>)"
        if modality.startswith("video"):
            return "(<video>./</video>)"
        if modality.startswith("audio"):
            return "(<audio>./</audio>)"
        raise ValueError("Only image, video or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        # keep vllm_config for later submodule init
        self.vllm_config = vllm_config

        # Store configs
        self.config = config
        self.multimodal_config = multimodal_config

        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "llm":
            # Initialize thinker model (image preprocessing + vision encoder + 3D resampler)
            self.thinker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=config,
                # Use registry architecture key
                architectures=["MiniCPMO45OmniLLMForConditionalGeneration"],
            )
            self.model = self.thinker
            self.talker = None
            self.code2wav = None

        elif self.model_stage == "tts":
            self.thinker = None
            # Initialize talker model (LLM generation)
            self.talker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                hf_config=config,
                # Use registry architecture key
                architectures=["MiniCPMO45OmniTTSForConditionalGeneration"],
            )
            # Initialize multimodal components if needed
            if hasattr(self.talker, "init_multi_modal"):
                self.talker.init_multi_modal(config)
            self.model = self.talker
            self.code2wav = None

        elif self.model_stage == "t2w":
            self.thinker = None
            self.talker = None
            # Code2wav only runs Vocos (mel → waveform);
            # use tts_config if available, otherwise use the main config.
            self.code2wav_config = getattr(config, "tts_config", None) or config
            self.code2wav = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "code2wav"),
                hf_config=self.code2wav_config,
                architectures=["MiniCPMO45OmniT2WModel"],
            )
            self.model = self.code2wav
        else:
            raise ValueError(f"Invalid model stage: {self.model_stage}. Must be one of: 'llm', 'tts', 't2w'")

        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            (self.thinker.make_empty_intermediate_tensors)
            if self.model_stage == "llm" and self.thinker is not None
            else lambda: None
        )

        self._language_model_names = ["model"]
        self.prefer_model_sampler = self.model_stage in {"llm", "tts"}
        self.has_preprocess = self.model_stage == "llm"

    @cached_property
    def sampler(self):
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        from vllm.v1.sample.sampler import Sampler

        return Sampler()

    # -------------------- Device utilities --------------------
    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            # No parameters; fall back to buffers or cpu
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def move_submodules_to_devices(
        self,
        *,
        thinker_device: str | torch.device | None = None,
        talker_device: str | torch.device | None = None,
        code2wav_device: str | torch.device | None = None,
    ) -> None:
        """Optionally move thinker/talker/code2wav to different devices.

        Example:
            model.move_submodules_to_devices(
                thinker_device='cuda:0',
                talker_device='cuda:1',
                code2wav_device='cpu',
            )
        """
        if thinker_device is not None and self.thinker is not None:
            self.thinker.to(thinker_device)
        if talker_device is not None and self.talker is not None:
            self.talker.to(talker_device)
        if code2wav_device is not None and self.code2wav is not None:
            self.code2wav.to(code2wav_device)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        if self.model_stage == "t2w":
            tts_cfg = getattr(self.config, "tts_config", None)
            hs = getattr(tts_cfg, "hidden_size", 768) if tts_cfg else 768
            return torch.zeros(
                input_ids.shape[0],
                hs,
                device=input_ids.device,
                dtype=torch.bfloat16,
            )
        embed_fn = getattr(self.model, "get_input_embeddings", None)
        if callable(embed_fn):
            try:
                return embed_fn(input_ids, multimodal_embeddings)
            except TypeError:
                embeddings = embed_fn()
                if callable(embeddings):
                    return embeddings(input_ids)
            except AttributeError:
                pass

        embed_tokens = getattr(getattr(getattr(self.model, "llm", None), "model", None), "embed_tokens", None)
        if callable(embed_tokens):
            return embed_tokens(input_ids)

        raise AttributeError(f"{type(self.model).__name__} does not expose token embeddings")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage in ("talker", "code2wav"):
            return self.get_input_embeddings(input_ids)
        return super().embed_input_ids(input_ids, multimodal_embeddings, is_multimodal=is_multimodal)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        """Model-runner data-plane hook for MiniCPM-o 4.5 duplex audio.

        The scheduler owns the request, block table, attention metadata, KV,
        and sampler. This hook only turns the current duplex audio append into
        the prompt embeddings consumed by the normal runner forward.
        """
        if self.model_stage != "llm":
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            return input_ids, embeds, {}

        duplex = kwargs.get("duplex")
        if not isinstance(duplex, dict) or duplex.get("data_plane") is not True:
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            return input_ids, embeds, {}

        helper = self._duplex_data_plane_helper()
        session_id = str(duplex.get("session_id") or "")
        payload = duplex.get("payload")
        if not session_id or not isinstance(payload, dict):
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            return input_ids, embeds, {"duplex": {"prefill_success": False, "reason": "bad_duplex_payload"}}

        state = helper.sessions.get(session_id)
        if state is None:
            from vllm_omni.model_executor.models.minicpmo_4_5.duplex_runtime import (
                _MiniCPMO45Stage0SessionState,
            )

            state = _MiniCPMO45Stage0SessionState(session_id=session_id)
            helper.sessions[session_id] = state
            session_config = duplex.get("session_config")
            helper.session_config = dict(session_config) if isinstance(session_config, dict) else {}
            if hasattr(helper.thinker, "audio_past_key_values"):
                helper.thinker.audio_past_key_values = None
            helper._configure_streaming_processor()
            helper._prepare_session_context(state, helper.session_config)

        audio_waveform = helper._decode_audio_payload(payload)
        seq = duplex.get("seq")
        try:
            seq = int(seq) if seq is not None else None
        except (TypeError, ValueError):
            seq = None
        result = helper._stage_prefill_embeddings_only(state, audio_waveform, seq=seq)
        update_result = dict(result)
        update_result.pop("inputs_embeds", None)
        if result.get("success") is not True:
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            return input_ids, embeds, {"duplex": update_result}

        target_dtype = (
            input_embeds.dtype if input_embeds is not None else self.get_input_embeddings(input_ids[:1]).dtype
        )
        full_req_embeds = result["inputs_embeds"].to(device=input_ids.device, dtype=target_dtype)
        full_input_token_ids = list(result.get("input_token_ids") or [])
        prompt_len = kwargs.get("duplex_prompt_len")
        try:
            prompt_len = int(prompt_len) if prompt_len is not None else int(full_req_embeds.shape[0])
        except (TypeError, ValueError):
            prompt_len = int(full_req_embeds.shape[0])
        pad_token_id = helper.stage_padding_token_id()
        if prompt_len > int(full_req_embeds.shape[0]):
            suffix_len = int(result.get("prompt_suffix_len") or 0)
            pad_len = prompt_len - int(full_req_embeds.shape[0])
            pad_ids = torch.full(
                (pad_len,),
                pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            pad_embeds = self.get_input_embeddings(pad_ids).to(dtype=full_req_embeds.dtype)
            if suffix_len > 0 and suffix_len < int(full_req_embeds.shape[0]):
                split_at = int(full_req_embeds.shape[0]) - suffix_len
                full_req_embeds = torch.cat(
                    [full_req_embeds[:split_at], pad_embeds, full_req_embeds[split_at:]],
                    dim=0,
                )
                full_input_token_ids = (
                    full_input_token_ids[:split_at] + [pad_token_id] * pad_len + full_input_token_ids[split_at:]
                )
            else:
                full_req_embeds = torch.cat([pad_embeds, full_req_embeds], dim=0)
                full_input_token_ids = [pad_token_id] * pad_len + full_input_token_ids

        span_len = int(input_ids.shape[0])
        token_offset = kwargs.get("duplex_token_offset", 0)
        try:
            token_offset = max(0, int(token_offset))
        except (TypeError, ValueError):
            token_offset = 0
        req_embeds = full_req_embeds[token_offset : token_offset + span_len]
        if req_embeds.shape[0] < span_len:
            pad_ids = torch.full(
                (span_len - req_embeds.shape[0],),
                pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            pad_embeds = self.get_input_embeddings(pad_ids).to(dtype=req_embeds.dtype)
            req_embeds = torch.cat([req_embeds, pad_embeds], dim=0)
        elif req_embeds.shape[0] > span_len:
            req_embeds = req_embeds[:span_len]

        input_token_ids = full_input_token_ids[token_offset : token_offset + span_len]
        if len(input_token_ids) < span_len:
            input_token_ids.extend([pad_token_id] * (span_len - len(input_token_ids)))
        if input_token_ids:
            req_input_ids = torch.tensor(input_token_ids, dtype=input_ids.dtype, device=input_ids.device)
            update_result["duplex_prompt_token_ids"] = full_input_token_ids
        else:
            req_input_ids = torch.full_like(input_ids, helper._required_token_id("unit_token_id"))
        return req_input_ids, req_embeds, {"duplex": update_result}

    def _duplex_data_plane_helper(self):
        helper = getattr(self, "_minicpmo45_duplex_data_plane_helper", None)
        if helper is not None:
            return helper
        from vllm_omni.model_executor.models.minicpmo_4_5.duplex_runtime import MiniCPMO45Stage0DuplexRuntime

        model_path = getattr(getattr(self.vllm_config, "model_config", None), "model", None)
        device = str(self._module_device(self.thinker if self.thinker is not None else self))
        helper = MiniCPMO45Stage0DuplexRuntime(self, model_path=model_path, device=device)
        self._minicpmo45_duplex_data_plane_helper = helper
        return helper

    def get_multimodal_embeddings(self, **kwargs):
        # Delegate to the active stage submodule when it implements MM encoding.
        mm_fn = getattr(self.model, "get_multimodal_embeddings", None)
        if mm_fn is not None:
            return mm_fn(**kwargs)
        return []

    def embed_multimodal(self, **kwargs: object):
        """vLLM V1 encoder profiling calls this; the inherited Protocol stub returns None."""
        return self.get_multimodal_embeddings(**kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        sampling_metadata: SamplingMetadata | None = None,
        logits_index: int | None = None,
        sampler=None,
        additional_information: dict[str, object] | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        """
        Forward pass for MiniCPM-o Omni model.

        Workflow:
        1) Thinker: Image preprocessing + Vision encoder + 3D resampler → hidden states
        2) Talker: LLM generation from hidden states → text tokens
        3) Code2Wav: Text tokens → speech waveform
        """
        if self.model_stage == "llm":
            # Normalize to batched inputs if caller provides 1D/2D unbatched tensors
            # TODO: Remove this hack when NPU supports batched inputs properly
            added_batch_dim = False
            if input_ids is not None and input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
                added_batch_dim = True
            if positions is not None and positions.ndim == 1:
                positions = positions.unsqueeze(0)
                added_batch_dim = True
            if inputs_embeds is not None and inputs_embeds.ndim == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
                added_batch_dim = True
            thinker_dev = self._module_device(self.thinker)

            # if input_ids is None, set it to a zero tensor
            if input_ids is None:
                input_ids = torch.zeros(inputs_embeds.shape[1], dtype=torch.long, device=thinker_dev).unsqueeze(0)
                added_batch_dim = True

            # Ensure inputs on thinker's device
            if input_ids is not None and input_ids.device != thinker_dev:
                input_ids = input_ids.to(thinker_dev)
            if positions is not None and positions.device != thinker_dev:
                positions = positions.to(thinker_dev)
            if inputs_embeds is not None and inputs_embeds.device != thinker_dev:
                inputs_embeds = inputs_embeds.to(thinker_dev)

            if current_omni_platform.is_npu():
                # TODO: remove this hack when NPU supports batched inputs properly
                thinker_input_ids = input_ids[0] if input_ids is not None and added_batch_dim else input_ids
                thinker_positions = positions[0] if positions.ndim > 1 else positions
                thinker_inputs_embeds = (
                    inputs_embeds[0] if inputs_embeds is not None and added_batch_dim else inputs_embeds
                )
            else:
                thinker_input_ids = input_ids[0] if input_ids is not None and added_batch_dim else input_ids
                thinker_positions = positions[0] if positions is not None and added_batch_dim else positions
                thinker_inputs_embeds = (
                    inputs_embeds[0] if inputs_embeds is not None and added_batch_dim else inputs_embeds
                )

            # Run thinker
            thinker_output = self.thinker(
                input_ids=thinker_input_ids,
                positions=thinker_positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=thinker_inputs_embeds,
                **kwargs,
            )

            if isinstance(thinker_output, tuple):
                embeds, text_hidden_states = thinker_output
            else:
                text_hidden_states = thinker_output

            # Prepare hidden states for downstream stages
            # Ensure correct shape: (batch_size, seq_len, hidden_dim)
            if added_batch_dim:
                text_hidden_states = text_hidden_states.squeeze(0)

            # Return hidden states with latent in multimodal_outputs for stage_input_processors
            multimodal_outputs = {"latent": text_hidden_states}
            runtime_info = kwargs.get("runtime_additional_information")
            if runtime_info and isinstance(runtime_info, list) and len(runtime_info) > 0:
                req_info = runtime_info[0] if isinstance(runtime_info[0], dict) else {}
                duplex_info = req_info.get("duplex") if isinstance(req_info, dict) else None
                if isinstance(duplex_info, dict):
                    prompt_token_ids = duplex_info.get("duplex_prompt_token_ids")
                    if isinstance(prompt_token_ids, list):
                        multimodal_outputs["duplex_prompt_token_ids"] = torch.tensor(
                            [prompt_token_ids],
                            dtype=torch.long,
                            device=text_hidden_states.device,
                        )
                    special_token_ids = duplex_info.get("special_token_ids")
                    if isinstance(special_token_ids, dict):
                        multimodal_outputs["meta"] = {
                            key: torch.tensor(
                                [int(value)],
                                dtype=torch.long,
                                device=text_hidden_states.device,
                            )
                            for key, value in special_token_ids.items()
                            if isinstance(key, str) and isinstance(value, int) and value >= 0
                        }
            return OmniOutput(
                text_hidden_states=text_hidden_states,
                multimodal_outputs=multimodal_outputs,
            )

        # Talker stage: runs ConditionalChatTTS + DVAE → mel_spec (+ optional Vocos → waveform)
        if self.model_stage == "tts":
            if input_ids is not None:
                num_tokens = input_ids.shape[0]
                device = input_ids.device
            elif inputs_embeds is not None:
                num_tokens = inputs_embeds.shape[0]
                device = inputs_embeds.device
            else:
                num_tokens = 1
                device = torch.device("cuda")
            hidden_dim = self.config.hidden_size if hasattr(self.config, "hidden_size") else 2560

            # Profile/dummy run: both input_ids and inputs_embeds are None.
            # Note: SupportsMultiModal preprocessing converts input_ids to
            # inputs_embeds, so input_ids=None alone does NOT indicate a dummy run.
            if input_ids is None and inputs_embeds is None:
                dummy_hidden = torch.zeros(num_tokens, hidden_dim, device=device)
                return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs=None)

            runtime_info = kwargs.get("runtime_additional_information")
            talker_info = {}
            if runtime_info and isinstance(runtime_info, list) and len(runtime_info) > 0:
                talker_info = runtime_info[0] if isinstance(runtime_info[0], dict) else {}
            tts_text = talker_info.get("llm_output_text", "")
            if isinstance(tts_text, list):
                tts_text = tts_text[0] if tts_text else ""
            if not isinstance(tts_text, str):
                tts_text = ""

            with torch.inference_mode():
                talker_result = self.talker(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                    additional_information=talker_info,
                )

            dummy_hidden = torch.zeros(num_tokens, hidden_dim, device=device)

            # talker returns (mel_spec, waveform_or_None) tuple
            if isinstance(talker_result, tuple) and len(talker_result) == 2:
                mel_spec, waveform = talker_result
                mm_out = {}
                chunk_flags = getattr(self.talker, "_ar_last_chunk_flags", [True])
                chunk_is_last = bool(chunk_flags[-1]) if chunk_flags else True
                mm_out["meta.tts_is_last_chunk"] = torch.tensor(
                    [int(chunk_is_last)],
                    dtype=torch.int32,
                    device=device,
                )
                if isinstance(tts_text, str) and tts_text:
                    mm_out["meta.llm_output_text_utf8"] = torch.tensor(
                        list(tts_text.encode("utf-8")),
                        dtype=torch.uint8,
                        device=device,
                    )
                    mm_out["meta.audio_text_total_chars"] = torch.tensor(
                        [len(tts_text)],
                        dtype=torch.int32,
                        device=device,
                    )
                if mel_spec is not None:
                    mm_out["mel_spec"] = [mel_spec]
                if waveform is not None:
                    mm_out["model_outputs"] = [waveform]
                elif mel_spec is not None:
                    mm_out["model_outputs"] = [mel_spec]
                return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs=mm_out)

            return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs=None)

        # Code2Wav stage: Vocos mel → waveform
        if self.model_stage == "t2w":
            if input_ids is not None:
                n_tokens = input_ids.shape[0]
                device = input_ids.device
            elif inputs_embeds is not None:
                n_tokens = inputs_embeds.shape[0]
                device = inputs_embeds.device
            else:
                n_tokens = 1
                device = torch.device("cuda")
            hidden_dim = self.config.hidden_size if hasattr(self.config, "hidden_size") else 2560

            # Profile/dummy run: both input_ids and inputs_embeds are None.
            if input_ids is None and inputs_embeds is None:
                dummy_hidden = torch.zeros(n_tokens, hidden_dim, device=device)
                return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs=None)

            runtime_info = kwargs.get("runtime_additional_information")
            code2wav_info = {}
            if runtime_info and isinstance(runtime_info, list) and len(runtime_info) > 0:
                code2wav_info = runtime_info[0] if isinstance(runtime_info[0], dict) else {}

            waveform_passthrough = code2wav_info.get("waveform")
            mel_spec = code2wav_info.get("mel_spec")
            dummy_hidden = torch.zeros(n_tokens, hidden_dim, device=device)

            if waveform_passthrough is not None:
                return OmniOutput(
                    text_hidden_states=dummy_hidden, multimodal_outputs={"model_outputs": [waveform_passthrough]}
                )

            if mel_spec is not None and isinstance(mel_spec, torch.Tensor) and mel_spec.dim() == 1:
                return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs={"model_outputs": [mel_spec]})

            if mel_spec is not None and self.code2wav is not None:
                with torch.inference_mode():
                    waveform = self.code2wav(
                        input_ids=input_ids,
                        positions=positions,
                        inputs_embeds=mel_spec if isinstance(mel_spec, torch.Tensor) else None,
                        additional_information=code2wav_info,
                    )
                return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs={"model_outputs": [waveform]})

            logger.warning("Code2Wav: no mel_spec or code2wav model, returning empty")
            return OmniOutput(text_hidden_states=dummy_hidden, multimodal_outputs={"model_outputs": [torch.zeros(0)]})

        raise ValueError(f"Unsupported model stage: {self.model_stage}")

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
        # Handle OmniOutput type
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        # Use model for logits computation
        return self.model.compute_logits(hidden_states)

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        if hasattr(self.model, "on_requests_finished"):
            self.model.on_requests_finished(finished_req_ids)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        *,
        duplex_rows: list[int] | None = None,
    ) -> SamplerOutput | None:
        native_duplex = self._sample_minicpmo45_native_duplex_stage0(
            logits,
            sampling_metadata,
            duplex_rows=duplex_rows,
        )
        if native_duplex is not None:
            return native_duplex
        if self.model_stage == "tts":
            return self.model.sample(logits, sampling_metadata)
        return None

    def _sample_minicpmo45_native_duplex_stage0(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        *,
        duplex_rows: list[int] | None = None,
    ) -> SamplerOutput | None:
        if self.model_stage != "llm" or logits.ndim != 2 or logits.shape[0] == 0:
            return None
        token_ids = self._minicpmo45_native_duplex_token_ids()
        unit_id = token_ids.get("unit_token_id", -1)
        if unit_id < 0:
            return None
        native_rows = self._native_duplex_prompt_rows(
            sampling_metadata,
            unit_id,
            logits.shape[0],
            duplex_rows=duplex_rows,
        )
        if not native_rows or len(native_rows) != logits.shape[0]:
            return None

        sampled_ids: list[int] = []
        for row_idx in range(logits.shape[0]):
            row_logits = logits[row_idx : row_idx + 1].clone()
            sampled_ids.append(
                self._sample_minicpmo45_native_duplex_row(
                    row_logits,
                    sampling_metadata,
                    row_idx=row_idx,
                    token_ids=token_ids,
                )
            )
        return SamplerOutput(
            sampled_token_ids=torch.tensor(sampled_ids, device=logits.device, dtype=torch.int32).unsqueeze(-1),
            logprobs_tensors=None,
        )

    def _sample_minicpmo45_native_duplex_row(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        *,
        row_idx: int,
        token_ids: dict[str, int],
    ) -> int:
        chunk_eos_id = token_ids.get("chunk_eos_token_id", -1)
        generator = getattr(sampling_metadata, "generators", {}).get(row_idx)
        output_token_ids = getattr(sampling_metadata, "output_token_ids", None) or []
        raw_recent_tokens = output_token_ids[row_idx] if row_idx < len(output_token_ids) else []
        recent_tokens = [int(token_id) for token_id in raw_recent_tokens if isinstance(token_id, int) and token_id >= 0]
        if MiniCPMO45DuplexPolicy.profile_logs_enabled() and not recent_tokens:
            top_values, top_indices = torch.topk(logits[0], k=min(12, logits.shape[-1]))
            special_scores = {
                name: float(logits[0, token_id].detach().cpu())
                for name, token_id in token_ids.items()
                if 0 <= token_id < logits.shape[-1]
            }
            logger.info(
                "MiniCPM-o native duplex first-step logits: top_ids=%s top_values=%s special_scores=%s",
                top_indices.detach().cpu().tolist(),
                [round(float(v), 4) for v in top_values.detach().cpu().tolist()],
                {key: round(value, 4) for key, value in special_scores.items()},
            )
        if chunk_eos_id >= 0 and chunk_eos_id < logits.shape[-1]:
            max_speak_tokens = int(getattr(self, "max_new_speak_tokens_per_chunk", 20) or 20)
            if len(recent_tokens) >= max(1, max_speak_tokens - 1):
                if MiniCPMO45DuplexPolicy.profile_logs_enabled():
                    logger.info(
                        "MiniCPM-o native duplex force chunk_eos: row=%s "
                        "recent_len=%d max_speak_tokens=%d chunk_eos_id=%s "
                        "recent_tail=%s",
                        row_idx,
                        len(recent_tokens),
                        max_speak_tokens,
                        chunk_eos_id,
                        recent_tokens[-8:],
                    )
                return int(chunk_eos_id)
            original_probs = F.softmax(logits[0], dim=-1)
            sampled = torch.multinomial(original_probs, num_samples=1, generator=generator).item()
            if MiniCPMO45DuplexPolicy.profile_logs_enabled():
                logger.info(
                    "MiniCPM-o native duplex sampled raw token: row=%s "
                    "sampled=%s chunk_eos_id=%s listen_id=%s recent_len=%d",
                    row_idx,
                    sampled,
                    chunk_eos_id,
                    token_ids.get("listen_token_id", -1),
                    len(recent_tokens),
                )
            if sampled == chunk_eos_id:
                return int(chunk_eos_id)

        forbidden = self._minicpmo45_native_forbidden_token_ids(token_ids)
        if forbidden:
            valid_forbidden = [token_id for token_id in forbidden if 0 <= token_id < logits.shape[-1]]
            if valid_forbidden:
                logits[:, valid_forbidden] = float("-inf")

        special_ids = self._minicpmo45_native_special_token_ids(token_ids)
        repetition_penalty = 1.05
        if repetition_penalty != 1.0 and recent_tokens:
            for token_id in set(recent_tokens[-512:]):
                if token_id in special_ids or token_id < 0 or token_id >= logits.shape[-1]:
                    continue
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Official MiniCPM-o StreamDecoder.decode listen handling (utils.py): bias
        # the listen token and optionally force-keep it only when it ranks in the
        # top-k. Defaults (scale=1.0, top_k=None) are a no-op; a scale < 1 makes
        # the model speak more readily (the official knob for listen/speak balance).
        listen_id = token_ids.get("listen_token_id", -1)
        listen_prob_scale, listen_top_k = self._minicpmo45_listen_decode_params()
        if 0 <= listen_id < logits.shape[-1]:
            if listen_prob_scale != 1.0:
                logits[0, listen_id] = logits[0, listen_id] * listen_prob_scale
            if listen_top_k is not None:
                listen_rank = int((logits[0] > logits[0, listen_id]).sum().item())
                if listen_rank < int(listen_top_k):
                    return int(listen_id)

        temperature = float(self._sampling_metadata_value(sampling_metadata, "temperature", row_idx, 0.7))
        top_k = int(self._sampling_metadata_value(sampling_metadata, "top_k", row_idx, 100))
        top_p = float(self._sampling_metadata_value(sampling_metadata, "top_p", row_idx, 0.8))
        if getattr(sampling_metadata, "all_greedy", False) or temperature <= 0:
            return int(torch.argmax(logits, dim=-1).item())

        logits = logits / temperature
        logits = self._top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1, generator=generator).item())

    def _minicpmo45_listen_decode_params(self) -> tuple[float, int | None]:
        """Listen-token decode controls, matching official StreamDecoder.decode.

        Read from config/env so the listen/speak balance is tunable without code
        changes. ``listen_prob_scale`` multiplies the listen-token logit (<1 ->
        speak more); ``listen_top_k`` forces listen only when it ranks in top-k.
        Defaults (1.0, None) preserve current behavior.
        """
        cached = getattr(self, "_minicpmo45_listen_decode_params_cache", None)
        if cached is not None:
            return cached
        import os

        try:
            scale = float(os.environ.get("MINICPMO45_LISTEN_PROB_SCALE", "1.0"))
        except (TypeError, ValueError):
            scale = 1.0
        top_k: int | None = None
        raw_k = os.environ.get("MINICPMO45_LISTEN_TOP_K")
        if raw_k not in (None, ""):
            try:
                top_k = int(raw_k)
            except (TypeError, ValueError):
                top_k = None
        params = (scale, top_k)
        self._minicpmo45_listen_decode_params_cache = params
        return params

    def _minicpmo45_native_duplex_token_ids(self) -> dict[str, int]:
        cached = getattr(self, "_minicpmo45_native_duplex_token_ids_cache", None)
        if isinstance(cached, dict):
            return cached
        tokenizer = None
        get_tokenizer = getattr(getattr(self, "thinker", None), "get_tokenizer", None)
        if callable(get_tokenizer):
            tokenizer = get_tokenizer()
        if tokenizer is None:
            try:
                from vllm.tokenizers import cached_tokenizer_from_config

                tokenizer = cached_tokenizer_from_config(self.vllm_config.model_config)
            except Exception as exc:
                if MiniCPMO45DuplexPolicy.profile_logs_enabled():
                    logger.info("MiniCPM-o native duplex sampler could not load tokenizer: %s", exc)
        cached = MiniCPMO45DuplexPolicy.token_ids_from_tokenizer(tokenizer)
        self._minicpmo45_native_duplex_token_ids_cache = cached
        return cached

    def _native_duplex_prompt_rows(
        self,
        sampling_metadata: SamplingMetadata,
        unit_id: int,
        batch_size: int,
        *,
        duplex_rows: list[int] | None = None,
    ) -> list[int]:
        if duplex_rows is not None:
            rows: list[int] = []
            for row in duplex_rows:
                try:
                    row_idx = int(row)
                except (TypeError, ValueError):
                    continue
                if 0 <= row_idx < batch_size:
                    rows.append(row_idx)
            return rows

        prompt_token_ids = getattr(sampling_metadata, "prompt_token_ids", None)
        if prompt_token_ids is None:
            return []
        if prompt_token_ids.ndim == 1:
            prompt_token_ids = prompt_token_ids.unsqueeze(0)
        rows: list[int] = []
        for row_idx in range(min(batch_size, int(prompt_token_ids.shape[0]))):
            row = prompt_token_ids[row_idx]
            if torch.count_nonzero(row == unit_id).item() >= 2:
                rows.append(row_idx)
        return rows

    def _minicpmo45_native_forbidden_token_ids(self, token_ids: dict[str, int]) -> list[int]:
        tokenizer = None
        get_tokenizer = getattr(getattr(self, "thinker", None), "get_tokenizer", None)
        if callable(get_tokenizer):
            tokenizer = get_tokenizer()
        bad_token_ids = getattr(tokenizer, "bad_token_ids", []) if tokenizer is not None else []
        return MiniCPMO45DuplexPolicy.native_forbidden_token_ids(token_ids, bad_token_ids=bad_token_ids)

    def _minicpmo45_native_special_token_ids(self, token_ids: dict[str, int]) -> set[int]:
        tokenizer = None
        get_tokenizer = getattr(getattr(self, "thinker", None), "get_tokenizer", None)
        if callable(get_tokenizer):
            tokenizer = get_tokenizer()
        return MiniCPMO45DuplexPolicy.native_special_token_ids(
            token_ids,
            tokenizer_special_ids=getattr(tokenizer, "all_special_ids", []) if tokenizer is not None else [],
        )

    @staticmethod
    def _sampling_metadata_value(
        sampling_metadata: SamplingMetadata,
        name: str,
        row_idx: int,
        default: float,
    ) -> float:
        value = getattr(sampling_metadata, name, None)
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            if value.ndim == 0:
                return float(value.item())
            idx = min(row_idx, int(value.numel()) - 1)
            return float(value.reshape(-1)[idx].item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _top_k_top_p_filter(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
        if top_k > 0 and top_k < logits.shape[-1]:
            kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(logits, dtype=torch.bool)
            remove.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
            logits = logits.masked_fill(remove, float("-inf"))
        return logits

    def generate_audio(self, code: torch.Tensor, voice_type: str = "default") -> torch.Tensor:
        """
        Generate audio from code tokens using the code2wav model.

        Args:
            code: Code tokens from talker model
            voice_type: Voice type for speech generation (optional for MiniCPM-o)

        Returns:
            Audio tensor
        """
        if self.code2wav is None:
            logger.warning("Code2Wav model not initialized, cannot generate audio")
            return torch.zeros(0)

        code2wav_dev = self._module_device(self.code2wav)
        if isinstance(code, torch.Tensor):
            code_tensor = code.to(dtype=torch.long, device=code2wav_dev)
        else:
            code_tensor = torch.as_tensor(code, dtype=torch.long, device=code2wav_dev)
        if code_tensor.ndim == 2 and code_tensor.shape[0] == 1:
            code_tensor = code_tensor.squeeze(0)

        # Generate audio using code2wav model
        # TODO: Implement actual audio generation based on MiniCPM-o's code2wav implementation
        with torch.inference_mode():
            audio_tensor = self.code2wav(code_tensor)

        return audio_tensor

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for all components of the omni model."""
        loaded_weights = set()
        thinker_weights = []
        talker_weights = []
        code2wav_weights = []

        # MiniCPM-o checkpoint prefixes → stage mapping:
        #   thinker: vpm, resampler, llm, apm, audio_projection_layer
        #   talker:  tts (ConditionalChatTTS)
        #   code2wav: (vocos loaded separately, not from main checkpoint)
        for k, v in weights:
            if k.startswith(("vpm.", "resampler.", "llm.", "apm.", "audio_projection_layer.")):
                thinker_weights.append((k, v))
            elif k.startswith("tts."):
                talker_weights.append((k, v))
            else:
                logger.warning("Unknown weight prefix: %s, skipping", k)

        # Load thinker weights
        if self.thinker is not None and thinker_weights:
            thinker_loaded = self.thinker.load_weights(thinker_weights)
            thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, "thinker")
            loaded_weights.update(thinker_loaded)

        # Load talker weights
        if self.talker is not None and talker_weights:
            talker_loaded = self.talker.load_weights(talker_weights)
            talker_loaded = add_prefix_to_loaded_weights(talker_loaded, "talker")
            loaded_weights.update(talker_loaded)

        # Load code2wav weights
        if self.code2wav is not None and code2wav_weights:
            code2wav_loaded = self.code2wav.load_weights(code2wav_weights)
            code2wav_loaded = add_prefix_to_loaded_weights(code2wav_loaded, "code2wav")
            loaded_weights.update(code2wav_loaded)

        return loaded_weights
