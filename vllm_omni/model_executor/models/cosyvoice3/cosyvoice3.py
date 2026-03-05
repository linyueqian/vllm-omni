# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Iterable, Mapping, Sequence
from functools import partial

import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.cosyvoice3.config import CosyVoice3Config
from vllm_omni.model_executor.models.cosyvoice3.utils import (
    concat_text_with_prompt_ids,
    extract_speech_feat,
    extract_speech_token,
    extract_spk_embedding,
    extract_text_token,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


class CosyVoice3MultiModalProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        """If the config is not already present pass it
        as a class and it will try to find it in your
        model directory just copy the config class there also.
        """
        return self.ctx.get_hf_config(CosyVoice3Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        """How many audio can you pass. I think I should keep it as 1
        For now I have kept it None.
        """
        return {"audio": None}

    def get_data_parser(self):
        return MultiModalDataParser(
            target_sr=self.ctx.get_hf_config().target_sr,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class CosyVoice3MultiModalProcessor(BaseMultiModalProcessor[CosyVoice3MultiModalProcessingInfo]):
    def _ensure_cached_runtime_components(self, model_dir: str, config: CosyVoice3Config) -> None:
        cached_model_dir = getattr(self, "_cached_model_dir", None)
        if cached_model_dir == model_dir:
            return

        # If model_dir is an HF repo ID (not a local path), resolve to cache
        if not os.path.isdir(model_dir):
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(model_dir)

        import onnxruntime

        from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer
        from vllm_omni.model_executor.models.cosyvoice3.utils import mel_spectrogram

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1

        self.tokenizer = get_qwen_tokenizer(
            token_path=os.path.join(model_dir, config.qwen_pretrain_path),
            skip_special_tokens=config.skip_special_tokens,
            version=config.version,
        )
        self.speech_tokenizer = onnxruntime.InferenceSession(
            os.path.join(model_dir, config.speech_tokenizer_path),
            sess_options=option,
            providers=["CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"],
        )
        self.feat_extractor = partial(mel_spectrogram, **getattr(config, "feat_extractor", {}))
        self.campplus_session = onnxruntime.InferenceSession(
            os.path.join(model_dir, config.campplus_onxx_path),
            sess_options=option,
            providers=["CPUExecutionProvider"],
        )
        self._cached_model_dir = model_dir

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """
        apply-> cached_apply_hf_processor -> apply_hf_processor_mm ->
        _call_hf_processor.
        _call_hf_processor takes input prompt and mm_data and returns
        token ids and tensors
        """
        config = self.info.ctx.get_hf_config()
        model_dir = self.info.ctx.model_config.model
        self._ensure_cached_runtime_components(model_dir, config)

        audio = mm_data.get("audio", None)

        if audio is None:
            audio = mm_data.get("audios")
            if audio is not None:
                audio = audio[0], config.target_sr

        text_token, text_token_len = extract_text_token(prompt, self.tokenizer, config.allowed_special)
        if audio is None:
            # Text-only path for profiling/cache
            return BatchFeature({"input_ids": text_token, "input_len": [text_token_len]})

        prompt_text = mm_kwargs.get("prompt_text")

        if not isinstance(prompt_text, str):
            raise ValueError(f"prompt text is None : {prompt_text}")

        prompt_text_token, prompt_text_token_len = extract_text_token(
            prompt_text, self.tokenizer, config.allowed_special
        )

        input_ids, input_len = concat_text_with_prompt_ids(
            text_token,
            text_token_len,
            prompt_text_token,
            prompt_text_token_len,
        )
        logger.debug(
            "cosyvoice _call_hf_processor: prompt_text_token=%s text_token=%s input_ids=%s "
            "prompt_text_len=%s text_len=%s input_len=%s",
            prompt_text_token.tolist(),
            text_token.tolist(),
            input_ids.tolist(),
            int(prompt_text_token_len),
            int(text_token_len),
            int(input_len),
        )
        device = "cpu"

        speech_token, speech_token_len = extract_speech_token(audio, self.speech_tokenizer, device)
        speech_feat, speech_feat_len = extract_speech_feat(audio, self.feat_extractor, device)

        if config.sample_rate == 24000:
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat, speech_feat_len[:] = speech_feat[:, : 2 * token_len], 2 * token_len
            speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len

        embedding = extract_spk_embedding(audio, self.campplus_session, device)

        ft = BatchFeature(
            {
                "input_ids": input_ids,
                "speech_feat": speech_feat,
                "speech_token": speech_token,
                "speech_token_len": [speech_token_len],
                "embedding": embedding,
            }
        )

        return ft

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "speech_feat": MultiModalFieldConfig.batched("audio"),
            "speech_token": MultiModalFieldConfig.batched("audio"),
            "speech_token_len": MultiModalFieldConfig.batched("audio"),
            "embedding": MultiModalFieldConfig.batched("audio"),
        }

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        def insertion_end(item_idx):
            # TODO: Think if this can be done better
            # sos + task + audio token ... ideally this needs to be split into
            # two start and end but somehow I couldn't pass two of these
            # wutg target .start() and .end()
            token_len = out_mm_kwargs["audio"][0]["speech_token_len"].data[0].item()
            return [1] * (1 + 1 + token_len)

        return [
            PromptInsertion(
                modality="audio",
                target=PromptIndexTargets.start(),
                insertion=insertion_end,
            ),
        ]


class CosyVoice3DummyInputsBuilder(BaseDummyInputsBuilder[CosyVoice3MultiModalProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "Hello, this is a test of the CosyVoice3 system capability."

    def get_dummy_mm_data(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions] | None = None
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio")
        max_prompt_seconds = 30
        prompt_sample_rate = 24000
        target_audio_length = max_prompt_seconds * prompt_sample_rate

        audio_overrides = mm_options.get("audio") if mm_options else None
        mm_data = {
            "audio": (
                self._get_dummy_audios(
                    length=target_audio_length,
                    num_audios=num_audios,
                    overrides=audio_overrides,
                )[0],
                24000,
            ),
        }
        return mm_data

    def get_dummy_processor_inputs(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions] | None = None
    ) -> ProcessorInputs:
        inputs = super().get_dummy_processor_inputs(seq_len, mm_counts, mm_options)
        inputs.hf_processor_mm_kwargs = {"prompt_text": "Testing my voices. Why should I not?"}
        return inputs


@MULTIMODAL_REGISTRY.register_processor(
    CosyVoice3MultiModalProcessor,
    info=CosyVoice3MultiModalProcessingInfo,
    dummy_inputs=CosyVoice3DummyInputsBuilder,
)
class CosyVoice3Model(
    nn.Module,
    SupportsMultiModal,
):
    supports_multimodal_raw_input_only = True
    supports_multimodal = True
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.have_multimodal_outputs = True
        self.model_stage = vllm_config.model_config.model_stage
        model_dir = vllm_config.model_config.model
        if not os.path.isdir(model_dir):
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(model_dir)
        self.model_dir = model_dir
        self.model = None
        if self.model_stage == "cosyvoice3_talker":
            # Initialize talker stage (text to speech tokens)
            from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_talker import CosyVoice3LM, VLLMQwen2Encoder

            llm_vllm_config = self._create_llm_vllm_config(vllm_config)
            llm = VLLMQwen2Encoder(vllm_config=llm_vllm_config, prefix="model")
            self.talker = CosyVoice3LM(
                llm_input_size=self.config.llm["llm_input_size"],
                llm_output_size=self.config.llm["llm_output_size"],
                speech_token_size=self.config.llm["speech_token_size"],
                llm=llm,
                length_normalized_loss=self.config.llm["length_normalized_loss"],
                lsm_weight=self.config.llm["lsm_weight"],
                mix_ratio=self.config.llm["mix_ratio"],
            )
            # KV cache is now managed externally by vLLM's PagedAttention
            # No need for self.llm_cache
            self.model = self.talker
        elif self.model_stage == "cosyvoice3_code2wav":
            # Initialize code2wav stage (flow matching + vocoder)
            from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

            self.code2wav = CosyVoice3Code2Wav(self.config)
            self.model = self.code2wav.flow_model
            self.hift = self.code2wav.hift
            # Keep additional information synchronized for async_chunk updates.
            self.enable_update_additional_information = True

            # Expose streaming parameters
            self.token_overlap_len = self.code2wav.token_overlap_len
            self.mel_overlap_len = self.code2wav.mel_overlap_len
            self.mel_window = self.code2wav.mel_window
            self.mel_cache_len = self.code2wav.mel_cache_len
            self.source_cache_len = self.code2wav.source_cache_len
            self.speech_window = self.code2wav.speech_window
        else:
            raise ValueError(f"Model stage not supported {self.model_stage}")

    def _create_llm_vllm_config(self, parent_config: VllmConfig) -> VllmConfig:
        """Create VllmConfig for the inner Qwen2 LLM.

        This creates a modified VllmConfig with the Qwen2 HF config loaded from
        the pretrained model directory. The cache config is inherited from the parent
        to enable PagedAttention with the same memory configuration.
        """
        from transformers import Qwen2Config

        qwen_config_path = os.path.join(self.model_dir, self.config.llm["llm"]["pretrain_path"])
        qwen_hf_config = Qwen2Config.from_pretrained(qwen_config_path)

        # Use parent's cache config - critical for PagedAttention to work correctly
        return parent_config.with_hf_config(qwen_hf_config, architectures=["Qwen2Model"])

    @staticmethod
    def _as_tensor(value: object) -> torch.Tensor | None:
        """Extract tensor payload from runtime info fields."""
        if isinstance(value, list):
            if not value:
                return None
            value = value[0]
        if isinstance(value, torch.Tensor):
            return value
        return None

    @staticmethod
    def _split_request_ids(ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        """Split concatenated input_ids into per-request segments."""
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + int(count))
            total = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], total)] for i in range(len(seq_token_counts))]

        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + int(s))
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]

        return [ids]

    def _sanitize_codec_tokens(self, req_ids: torch.Tensor) -> torch.Tensor:
        """Filter non-code tokens before feeding flow token embedding."""
        vocab_size = int(self.code2wav.input_embedding.num_embeddings)
        valid_mask = (req_ids >= 0) & (req_ids < vocab_size)
        return req_ids[valid_mask]

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if self.model_stage == "cosyvoice3_talker":
            logits = self.model.llm_decoder(hidden_states)
            vocab_size = self.config.vocab_size
            pad_size = vocab_size - logits.size(-1)
            pad_shape = logits.shape[:-1] + (pad_size,)
            pad = logits.new_full(pad_shape, float("-inf"))
            eos_token_val = logits[..., self.config.llm["eos_token_id"]].clone()
            logits[..., -200:] = float("-inf")
            logits[..., self.config.llm["eos_token_id"]] = eos_token_val
            logits = torch.cat([logits, pad], dim=-1)
            return logits
        else:
            raise RuntimeError(f"compute_logits is only valid for {self.model_stage}.")

    def embed_multimodal(self, **kwargs: object) -> torch.Tensor:
        if self.model_stage == "cosyvoice3_talker":
            speech_token = kwargs["speech_token"]
            speech_token_emb = self.model.speech_embedding(speech_token)
            return speech_token_emb
        else:
            raise RuntimeError(f"embed_multimodal is only valid for {self.model_stage}.")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "cosyvoice3_talker":
            if is_multimodal is not None and any(is_multimodal):
                embed_tokens = self.model.llm.model.embed_tokens(input_ids)
                sos = self.model.speech_embedding.weight[self.model.sos].reshape(1, -1)
                task_id = self.model.speech_embedding.weight[self.model.task_id].reshape(1, -1)
                prompt_speech_token_emb = multimodal_embeddings[0]
                pstoken_len = prompt_speech_token_emb.shape[0]  # Get length from tensor shape
                embed_tokens = torch.cat(
                    [sos, embed_tokens[2 + pstoken_len :], task_id, prompt_speech_token_emb], dim=0
                )
            else:
                embed_tokens = self.model.speech_embedding.weight[input_ids]
            return embed_tokens
        elif self.model_stage == "cosyvoice3_code2wav":
            assert input_ids.dim() == 1
            hidden = int(self.config.hidden_size)
            return torch.zeros(
                (input_ids.shape[0], hidden),
                device=input_ids.device,
            )
        else:
            raise RuntimeError(f"embed_input_ids is not valid for {self.model_stage}.")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict[str, object] | None = None,
        **kwargs: object,
    ) -> OmniOutput:
        if self.model_stage == "cosyvoice3_talker":
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)

            # [total_tokens, hidden]
            hidden_states = self.model.llm(inputs_embeds, positions)

            multimodal_outputs = {}

            if "speech_token" in kwargs:
                # Wrap in lists to pass through gpu_ar_model_runner shape filtering
                multimodal_outputs = {
                    "speech_token": [kwargs.get("speech_token")],
                    "speech_feat": [kwargs.get("speech_feat")],
                    "embedding": [kwargs.get("embedding")],
                }

            return OmniOutput(text_hidden_states=hidden_states, multimodal_outputs=multimodal_outputs)
        elif self.model_stage == "cosyvoice3_code2wav":
            runtime_info = kwargs.get("model_intermediate_buffer")
            if runtime_info is None:
                runtime_info = kwargs.get("runtime_additional_information", [])
            if "runtime_additional_information" in kwargs and "model_intermediate_buffer" not in kwargs:
                logger.warning_once("runtime_additional_information is deprecated, use model_intermediate_buffer")

            seq_token_counts = kwargs.get("seq_token_counts")
            flat_ids = input_ids.reshape(-1).to(dtype=torch.long)
            request_ids_list = self._split_request_ids(flat_ids, seq_token_counts)

            num_reqs = max(1, len(request_ids_list))
            sample_rate = torch.tensor(int(self.config.sample_rate), dtype=torch.int32)
            empty_audio = torch.zeros((0,), dtype=torch.float32, device=input_ids.device)
            audios: list[torch.Tensor] = [empty_audio] * num_reqs
            srs: list[torch.Tensor] = [sample_rate] * num_reqs
            samples_per_token: int | None = None
            try:
                # Prefer vocoder-derived stride: prod(upsample_rates) * hop_len * token_mel_ratio.
                hift_cfg = getattr(self.config, "hift", {}) or {}
                up_rates = list(hift_cfg.get("upsample_rates") or [])
                hop_len = int((hift_cfg.get("istft_params") or {}).get("hop_len", 0))
                token_mel_ratio = int(getattr(self.config, "token_mel_ratio", 0))
                if up_rates and hop_len > 0 and token_mel_ratio > 0:
                    stride = 1
                    for u in up_rates:
                        stride *= int(u)
                    samples_per_token = stride * hop_len * token_mel_ratio
                else:
                    token_hz = int(getattr(self.config, "token_frame_rate", 0))
                    sr_val = int(self.config.sample_rate)
                    if token_hz > 0 and sr_val % token_hz == 0:
                        samples_per_token = sr_val // token_hz
            except Exception:
                samples_per_token = None

            if not isinstance(runtime_info, list):
                runtime_info = []

            for idx, req_ids in enumerate(request_ids_list):
                info = runtime_info[idx] if idx < len(runtime_info) and isinstance(runtime_info[idx], dict) else {}
                speech_token = self._as_tensor(info.get("speech_token")) if info else None
                speech_feat = self._as_tensor(info.get("speech_feat")) if info else None
                embedding = self._as_tensor(info.get("embedding")) if info else None
                if speech_token is None or speech_feat is None or embedding is None:
                    if req_ids.numel() > 0 and info and ("left_context_size" in info or "generated_len" in info):
                        info_keys = ",".join(sorted(info.keys())) if info else ""
                        logger.warning_once(
                            "CosyVoice3 code2wav missing prompt conditioning for non-empty codec tokens: "
                            "raw_len=%d info_keys=%s",
                            int(req_ids.numel()),
                            info_keys,
                        )
                    continue

                token = self._sanitize_codec_tokens(req_ids)
                if token.numel() == 0:
                    if req_ids.numel() > 0:
                        logger.warning_once(
                            "CosyVoice3 code2wav received no valid codec tokens after filtering: "
                            "raw_len=%d raw_range=[%d,%d] vocab_size=%d",
                            req_ids.numel(),
                            int(req_ids.min().item()),
                            int(req_ids.max().item()),
                            int(self.code2wav.input_embedding.num_embeddings),
                        )
                    continue
                left_context_size = 0
                try:
                    left_context_size = max(0, int(info.get("left_context_size", 0))) if info else 0
                except (TypeError, ValueError):
                    left_context_size = 0

                tts_speech = self.code2wav(
                    token=token.unsqueeze(0),
                    prompt_token=speech_token[:1],
                    prompt_feat=speech_feat[:1],
                    embedding=embedding[:1],
                    n_timesteps=10,
                )
                audio = tts_speech.reshape(-1).to(dtype=torch.float32)
                if left_context_size > 0 and samples_per_token is None and audio.numel() > 0:
                    logger.warning_once(
                        "CosyVoice3 code2wav cannot trim left context because samples_per_token is unavailable: "
                        "left_context_size=%d sample_rate=%d",
                        left_context_size,
                        int(self.config.sample_rate),
                    )
                if left_context_size > 0 and samples_per_token is not None and audio.numel() > 0:
                    crop = left_context_size * samples_per_token
                    if crop > 0:
                        audio = audio[crop:] if crop < audio.numel() else audio[:0]
                audios[idx] = audio

            return OmniOutput(text_hidden_states=None, multimodal_outputs={"audio": audios, "sr": srs})
        else:
            raise ValueError(f"Unsupported model_stage: {self.model_stage}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.model_stage == "cosyvoice3_talker":
            # Load weights for text to speech LM stage using vLLM's weight loading
            llm_weight_path = os.path.join(self.model_dir, "llm.pt")
            device = next(self.parameters()).device
            checkpoint = torch.load(llm_weight_path, map_location=device)

            # 1. Load Qwen2 model weights into vLLM's Qwen2Model
            # The checkpoint has prefix "llm.model.model." for the transformer weights
            # vLLM's Qwen2Model expects just the model structure without extra prefixes
            qwen_weights = []
            for name, weight in checkpoint.items():
                if name.startswith("llm.model.model."):
                    # Strip prefix: llm.model.model.X -> X (for vLLM's Qwen2Model)
                    vllm_name = name.replace("llm.model.model.", "")
                    qwen_weights.append((vllm_name, weight))

            # Use vLLM's built-in load_weights which handles stacked params
            # (q_proj+k_proj+v_proj -> qkv_proj, gate_proj+up_proj -> gate_up_proj)
            self.model.llm.model.load_weights(iter(qwen_weights))

            # 2. Load CosyVoice3LM-specific weights (speech_embedding, llm_decoder)
            speech_emb_state = {
                k.replace("speech_embedding.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("speech_embedding.")
            }
            self.model.speech_embedding.load_state_dict(speech_emb_state)

            llm_decoder_state = {
                k.replace("llm_decoder.", ""): v for k, v in checkpoint.items() if k.startswith("llm_decoder.")
            }
            self.model.llm_decoder.load_state_dict(llm_decoder_state)

            self.model.to(device).eval()
        elif self.model_stage == "cosyvoice3_code2wav":
            # Load weights for code2wav stage (flow + hift)
            device = next(self.parameters()).device
            self.code2wav.load_weights(self.model_dir, device)
        else:
            raise ValueError(f"{self.model_stage} not supported yet!")
