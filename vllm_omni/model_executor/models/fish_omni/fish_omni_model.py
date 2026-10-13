# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fish Audio Drama 3, stage 0: text (+ voice references) -> codec frames.

The backbone is vLLM's Qwen3.5 MoE (hybrid gated-delta-net + full attention,
block-FP8 routed experts). vLLM samples each frame's main token from the
backbone logits; ``talker_mtp`` then runs the codebook tower for that frame and
mixes codebook 1 into the next backbone input.

Step timing for one request with a prompt of ``n`` rows:

    prefill   backbone rows 0..n-1      -> h[0..n-1]; tower teacher-forced on rows 0..n-2
    decode k  input m[n+k] (sampled)    -> tower at row n-1+k with h[n-1+k] -> codes of frame n+k
              backbone row n+k          -> h[n+k]
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput

from vllm_omni.model_executor.models.fish_speech.dac_encoder import encode_reference_audio_codes
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.utils.speaker_cache import get_speaker_cache

from .codebook_tower import CodebookTower, TowerBatch, TowerDecodeGraphs, TowerState, sample_rows
from .configuration_fish_omni import FishOmniConfig
from .prompt import Reference, build_prompt_frames, prompt_length

logger = init_logger(__name__)

_TOWER_PREFIXES = ("audio_decoder.", "audio_decoder_mtp_shared.", "audio_decoder_mtp.")
# DAC: 44.1 kHz audio, 2048 samples per frame.
_CODEC_SAMPLE_RATE = 44100
_CODEC_HOP = 2048
# Codebook-head sampling (model card defaults).
_HEAD_SAMPLING = (1.0, 30, 0.9)
# Repetition-aware re-draw (RAS) of the main token: window and (temperature, top_k, top_p).
_RAS_WINDOW = 16
_RAS_SAMPLING = (1.0, 30, 0.95)


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


class FishOmniForConditionalGeneration(Qwen3_5MoeForCausalLM):
    """Qwen3.5 MoE backbone + Drama 3 codebook tower, with vLLM-Omni AR hooks."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config: FishOmniConfig = vllm_config.model_config.hf_config  # type: ignore[assignment]
        self.model_path = vllm_config.model_config.model
        decoder_config = config.audio_decoder_config
        self._semantic_begin_id = int(config.semantic_start_token_id)
        self._semantic_end_id = int(config.semantic_end_token_id)
        self._eos_token_id = int(config.eos_token_id)
        self._num_codebooks = int(decoder_config.num_codebooks)

        max_seqs = int(vllm_config.scheduler_config.max_num_seqs)
        self.tower = CodebookTower(
            decoder_config,
            semantic_begin_id=self._semantic_begin_id,
            cache_slots=2 * max_seqs,
            max_len=int(vllm_config.model_config.max_model_len),
        )

        vocab = int(config.text_config.vocab_size)
        allowed = torch.zeros((vocab,), dtype=torch.bool)
        allowed[self._semantic_begin_id : self._semantic_end_id + 1] = True
        allowed[self._eos_token_id] = True
        self.register_buffer("_main_token_allowed", allowed, persistent=False)
        self.register_buffer("_main_token_candidates", allowed.nonzero().squeeze(-1), persistent=False)
        self.prefer_model_sampler = True

        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self.mtp_hidden_size = int(config.text_config.hidden_size)
        self.talker_mtp_output_key = ("codes", "audio")
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {("hidden_states", "last")}
        # The tower loop has data-dependent lengths and Python control flow.
        self.talker_mtp_graph_safe = False
        # The tower step keeps its own CUDA graphs (captured on first decode).
        self._tower_graphs = (
            None
            if vllm_config.model_config.enforce_eager
            else TowerDecodeGraphs(
                self.tower,
                tuple(sorted({b for b in (1, 2, 4, 8, 16, 32, 64) if b < max_seqs} | {max_seqs})),
                _HEAD_SAMPLING,
            )
        )
        self.talker_mtp_accepts_req_infos = True

        self._speaker_cache = get_speaker_cache()
        self._tokenizer = None
        logger.info(
            "fish_omni: mamba_ssm_cache_dtype=%s, mamba_cache_dtype=%s",
            vllm_config.cache_config.mamba_ssm_cache_dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    # -------------------- vLLM hooks --------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, *args: Any, **kwargs: Any):
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        logits = super().compute_logits(hidden_states)
        if logits is None:
            return None
        return logits.masked_fill(~self._main_token_allowed[: logits.shape[-1]], float("-inf"))

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []
        codes = [
            info["codes"]["audio"]
            for info in info_dicts
            if isinstance(info, dict) and isinstance(info.get("codes", {}).get("audio"), torch.Tensor)
        ]
        if not codes:
            return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs={})
        audio_codes = torch.cat(codes, dim=0)
        return OmniOutput(
            text_hidden_states=model_outputs[: int(audio_codes.shape[0])],
            multimodal_outputs={"audio_codes": audio_codes},
        )

    # -------------------- prompt --------------------

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self._tokenizer

    def _reference_codes(self, info: dict[str, Any], device: torch.device) -> torch.Tensor | None:
        codes = _first(info.get("ref_codes"))
        if isinstance(codes, torch.Tensor):
            return codes.to(dtype=torch.long)
        wav = _first(info.get("ref_audio_wav"))
        if wav is None:
            return None
        voice_name = _first(info.get("voice_name"))
        cache_key = None
        if isinstance(voice_name, str) and voice_name:
            cache_key = self._speaker_cache.make_cache_key(
                voice_name, model_type="fish_omni", created_at=int(_first(info.get("voice_created_at")) or 0)
            )
            cached = self._speaker_cache.get(cache_key)
            if cached is not None:
                return cached["ref_codes_fq"]
        sample_rate = int(_first(info.get("ref_audio_sr")))
        wav_np = wav.cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav, dtype=np.float32)
        codes = encode_reference_audio_codes(self.model_path, wav_np, sample_rate, device=device).cpu()
        if cache_key is not None:
            self._speaker_cache.put(cache_key, {"ref_codes_fq": codes})
        return codes

    def build_frames(self, info: dict[str, Any], device: torch.device) -> torch.Tensor:
        text = _first(info.get("text"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError("fish_omni needs a non-empty 'text'")
        references = []
        codes = self._reference_codes(info, device)
        if codes is not None:
            transcript = _first(info.get("ref_text"))
            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError("fish_omni voice cloning needs 'ref_text' with the reference audio")
            references.append(Reference(transcript, codes))
        return build_prompt_frames(
            self._get_tokenizer(),
            text,
            references,
            semantic_begin_id=self._semantic_begin_id,
            num_codebooks=self._num_codebooks,
        )

    @staticmethod
    def estimate_prompt_len_from_additional_information(
        additional_information: dict[str, Any] | None,
        *,
        tokenizer: Any = None,
        model_path: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Exact prompt rows: text tokens plus one row per reference frame."""
        info = additional_information or {}
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        transcripts, frames = [], []
        codes = _first(info.get("ref_codes"))
        wav = _first(info.get("ref_audio_wav"))
        if isinstance(codes, torch.Tensor):
            frames.append(int(codes.shape[0]))
        elif wav is not None:
            samples = len(wav) * _CODEC_SAMPLE_RATE / int(_first(info.get("ref_audio_sr")))
            frames.append(math.ceil(math.ceil(samples) / _CODEC_HOP))
        if frames:
            transcripts.append(str(_first(info.get("ref_text")) or ""))
        return prompt_length(tokenizer, str(_first(info.get("text")) or ""), transcripts, frames)

    # -------------------- preprocess / postprocess --------------------

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        extra = info_dict.get("additional_information")
        if isinstance(extra, dict):
            info_dict = {**extra, **{k: v for k, v in info_dict.items() if k != "additional_information"}}
        span = int(input_ids.shape[0])
        device = input_ids.device
        if span == 0:
            return input_ids, self.embed_input_ids(input_ids), {}

        if span > 1:
            # Prefill (chunked prefill is off, so one span holds the whole prompt).
            frames = self.build_frames(info_dict, device)
            if int(frames.shape[0]) != span:
                raise ValueError(
                    f"fish_omni prompt has {int(frames.shape[0])} rows but the request reserved {span} "
                    "placeholder tokens; size prompt_token_ids with "
                    "estimate_prompt_len_from_additional_information"
                )
            frames = frames.to(device)
            is_semantic = self.tower.is_semantic(frames[:, 0])
            embeds = self.tower.backbone_embedding(self.embed_input_ids(frames[:, 0]), frames[:, 1:], is_semantic)
            zeros = torch.zeros((span, self._num_codebooks), dtype=torch.long, device=device)
            return (
                frames[:, 0].clone(),
                embeds.to(torch.bfloat16),
                {"drama3_prompt": {"frames": frames}, "codes": {"audio": zeros}},
            )

        last_hidden = info_dict.get("hidden_states", {}).get("last")
        if not isinstance(last_hidden, torch.Tensor):
            raise RuntimeError("fish_omni decode step has no backbone hidden state from the previous step")
        embeds = self.embed_input_ids(input_ids.reshape(-1).to(torch.long)).reshape(1, -1)
        return (
            input_ids,
            embeds.to(torch.bfloat16),
            {
                "mtp_inputs": (
                    last_hidden.to(device=device, dtype=torch.bfloat16).reshape(1, -1),
                    torch.zeros(1, self.mtp_hidden_size, device=device, dtype=torch.bfloat16),
                )
            },
        )

    @torch.inference_mode()
    def postprocess(self, hidden_states: torch.Tensor, **info: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            return {}
        update: dict[str, Any] = {"hidden_states": {"last": hidden_states[-1].detach().contiguous()}}
        prompt = info.get("drama3_prompt")
        state = info.get("drama3_tower")
        if hidden_states.shape[0] > 1 and isinstance(prompt, dict) and not isinstance(state, TowerState):
            update["drama3_tower"] = self._prefill_tower(hidden_states, prompt["frames"])
            update["drama3_prompt"] = None
        return update

    def _prefill_tower(self, hidden_states: torch.Tensor, frames: torch.Tensor) -> TowerState:
        """Teacher-force the tower over prompt rows 0..n-2 (row p's next frame is p+1)."""
        frames = frames.to(hidden_states.device)
        rows = int(frames.shape[0])
        state = TowerState(
            slot=-1,
            next_position=rows - 1,
            input_frame=frames[-1, 1:].clone(),
            input_is_semantic=self.tower.is_semantic(frames[-1, 0]),
        )
        state.slot = self.tower.cache.acquire(state, in_use=set())
        if rows > 1:
            n = rows - 1
            device = hidden_states.device
            self.tower.run(
                hidden=hidden_states[:n].to(torch.bfloat16),
                input_codes=frames[:n, 1:],
                input_is_semantic=self.tower.is_semantic(frames[:n, 0]),
                next_main_tokens=frames[1:, 0],
                next_codes=frames[1:, 1:],
                batch=TowerBatch.prefill(state.slot, n, device),
                sampling=_HEAD_SAMPLING,
            )
        return state

    # -------------------- codebook tower per decode step --------------------

    @torch.inference_mode()
    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        req_ids: list[str] | None = None,
        req_infos: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = int(input_ids.shape[0])
        device = input_embeds.device
        main_tokens = input_ids.reshape(rows).to(torch.long)
        embeds = input_embeds.reshape(rows, -1)
        states = [info.get("drama3_tower") if isinstance(info, dict) else None for info in (req_infos or [])]
        live = [i for i, s in enumerate(states) if isinstance(s, TowerState) and self.tower.cache.owns(s.slot, s)]
        codes = torch.zeros((rows, self._num_codebooks), dtype=torch.long, device=device)
        if not live:
            return embeds, codes
        if len(live) != len(states):
            logger.warning("fish_omni: %d of %d rows have no tower state", len(states) - len(live), len(states))

        live_states = [states[i] for i in live]
        for s in live_states:
            self.tower.cache.touch(s.slot)
        # Host-built indices go up in one pinned, non-blocking copy.
        host = torch.tensor(
            [live, [s.slot for s in live_states], [s.next_position for s in live_states]], dtype=torch.long
        ).pin_memory()
        index, slots, positions = host.to(device, non_blocking=True).unbind(0)
        embeds = embeds.clone()
        step_inputs = {
            "hidden": last_talker_hidden[index].to(torch.bfloat16),
            "input_codes": torch.stack([s.input_frame for s in live_states]),
            "input_is_semantic": torch.stack([s.input_is_semantic for s in live_states]),
            "next_main_tokens": main_tokens[index],
        }
        if self._tower_graphs is not None and not self._tower_graphs.covers(1):
            self._tower_graphs.capture(device, torch.bfloat16)
        if self._tower_graphs is not None and self._tower_graphs.covers(len(live)):
            frame_codes = self._tower_graphs.run(**step_inputs, slots=slots, positions=positions)
        else:
            frame_codes = self.tower.run(
                **step_inputs, next_codes=None, batch=TowerBatch.decode(slots, positions), sampling=_HEAD_SAMPLING
            )
        is_semantic = self.tower.is_semantic(main_tokens[index])
        embeds[index] = self.tower.backbone_embedding(embeds[index], frame_codes, is_semantic).to(embeds.dtype)
        codes[index] = frame_codes
        for row, state in enumerate(live_states):
            state.next_position += 1
            state.input_frame = frame_codes[row]
            state.input_is_semantic = is_semantic[row]
        return embeds, codes

    # -------------------- main-token sampling --------------------

    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> SamplerOutput | None:
        """Main-token sampling with repetition-aware re-draw (RAS).

        Each row draws among end-of-speech and the semantic tokens with its own
        temperature / top-k / top-p. When a semantic draw repeats one of the
        request's last 16 main tokens, the row takes a second draw with fixed
        parameters instead; that draw may end the speech.
        """
        rows = int(logits.shape[0])
        if rows == 0:
            return None
        device = logits.device
        candidates = self._main_token_candidates
        candidate_logits = logits.index_select(-1, candidates).float()
        width = int(candidates.numel())

        def per_row(value: torch.Tensor | None, default: float, dtype: torch.dtype) -> torch.Tensor:
            if value is None:
                return torch.full((rows,), default, device=device, dtype=dtype)
            return value[:rows].to(device=device, dtype=dtype)

        temperature = per_row(sampling_metadata.temperature, 1.0, torch.float32)
        top_k = per_row(sampling_metadata.top_k, width, torch.long)
        top_k = torch.where((top_k <= 0) | (top_k > width), torch.full_like(top_k, width), top_k)
        top_p = per_row(sampling_metadata.top_p, 1.0, torch.float32)
        first = candidates[sample_rows(candidate_logits, temperature, top_k, top_p)]
        ras_temperature, ras_top_k, ras_top_p = _RAS_SAMPLING
        redraw = candidates[
            sample_rows(
                candidate_logits,
                torch.full((rows,), ras_temperature, device=device),
                torch.full((rows,), ras_top_k, device=device, dtype=torch.long),
                torch.full((rows,), ras_top_p, device=device),
            )
        ]
        history = sampling_metadata.output_token_ids or [[] for _ in range(rows)]
        window = torch.full((rows, _RAS_WINDOW), -1, dtype=torch.long)
        for row, tokens in enumerate(history[:rows]):
            recent = list(tokens[-_RAS_WINDOW:])
            if recent:
                window[row, -len(recent) :] = torch.tensor(recent, dtype=torch.long)
        window = window.to(device, non_blocking=True)
        repeats = (window == first.unsqueeze(-1)).any(-1) & self.tower.is_semantic(first)
        main = torch.where(repeats, redraw, first)
        return SamplerOutput(sampled_token_ids=main.to(torch.int32).unsqueeze(-1), logprobs_tensors=None)

    # -------------------- weights --------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        tower_params = dict(self.tower.named_parameters())
        tower_loaded: set[str] = set()

        def backbone_weights():
            for name, tensor in weights:
                if name.startswith(_TOWER_PREFIXES):
                    param = tower_params.get(name)
                    if param is None:
                        raise ValueError(f"fish_omni: unexpected tower weight {name}")
                    if tuple(param.shape) != tuple(tensor.shape):
                        raise ValueError(f"fish_omni: {name} shape {tuple(tensor.shape)} != {tuple(param.shape)}")
                    with torch.no_grad():
                        param.copy_(tensor)
                    tower_loaded.add(name)
                else:
                    yield name, tensor

        loaded = super().load_weights(backbone_weights())
        missing = set(tower_params) - tower_loaded
        if missing:
            raise ValueError(f"fish_omni: {len(missing)} tower weights missing, e.g. {sorted(missing)[:4]}")
        return loaded | {f"tower.{name}" for name in tower_loaded}
