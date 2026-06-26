# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex talker: the temporal transformer as a vLLM-native omni AR stage.

This is the stage-0 (``LLM_AR``) model the OmniGPUModelRunner drives. It composes
three pieces, each verified in isolation against Moshi:

* the Helium temporal transformer (:class:`HeliumModel`) on vLLM paged attention,
  consuming per-frame ``inputs_embeds`` and producing the per-frame hidden state;
* the input embeddings (:class:`PersonaPlexInputEmbeddings`, ``embed_codes``) that
  build those ``inputs_embeds`` from the delayed 17-row token stack;
* the depformer (:class:`PersonaPlexDepformer`) that, conditioned on the temporal
  hidden state and the sampled text token, predicts the per-frame audio codes.

Per-frame protocol (OmniGPUModelRunner, gpu_model_runner.py):

1. ``compute_logits`` produces the text logits; the engine samples the text token.
2. ``preprocess`` (per-request, with that request's ``additional_information``)
   carries Moshi's acoustic-delay cache and the precomputed user-audio code stream,
   builds the base ``inputs_embeds`` for the current frame, and exposes the
   previous frame's temporal hidden + text-step embedding via ``mtp_inputs``.
3. ``talker_mtp`` (batched, stateless) runs the depformer to predict the agent
   codes and finishes the next frame's ``inputs_embeds``; the codes are stored
   under ``talker_mtp_output_key=("codes","audio")`` for the Mimi code2wav stage.

Phase 1 is turn-based: the user-audio rows (9..16) come from a precomputed Mimi
encode of the input WAV (built in ``preprocess``); live duplex is Phase 2.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
    PersonaPlexConfig,
)
from vllm_omni.model_executor.models.personaplex.modeling_helium import HeliumModel
from vllm_omni.model_executor.models.personaplex.personaplex_depformer import (
    PersonaPlexDepformer,
)
from vllm_omni.model_executor.models.personaplex.personaplex_embeddings import (
    PersonaPlexInputEmbeddings,
)

__all__ = ["PersonaPlexTalkerForConditionalGeneration"]


class PersonaPlexTalkerForConditionalGeneration(nn.Module, SupportsPP):
    """vLLM-native PersonaPlex talker (temporal transformer + depformer)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        config: PersonaPlexConfig = vllm_config.model_config.hf_config  # type: ignore[assignment]
        self.config = config
        self.temporal_config = config.temporal_config
        hidden = self.temporal_config.hidden_size

        # Temporal backbone on vLLM paged attention (consumes inputs_embeds).
        self.model = HeliumModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            config=self.temporal_config,
        )
        # Text head (Moshi text_linear -> lm_head), vocab = text_vocab_size (32000).
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.text_vocab_size,
                hidden,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.text_vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Verified custom components: embed_codes + depformer.
        self.input_embeddings = PersonaPlexInputEmbeddings(config)
        self.depformer = PersonaPlexDepformer(
            config.depformer_config,
            temporal_hidden_size=hidden,
            text_card=config.text_vocab_size,
        )

        # Omni AR runner contract.
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.mtp_hidden_size = hidden
        self.talker_mtp_output_key = ("codes", "audio")
        # dep_q audio codebooks per frame; only cb 0..num_active are vocoded.
        self.dep_q = config.depformer_config.dep_q
        self.num_active_codebooks = config.depformer_config.num_active_codebooks

    # ------------------------------------------------------------------
    # Core forward / logits
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        if hidden_states.dim() == 3:
            b, s, h = hidden_states.shape
            logits = self.logits_processor(self.lm_head, hidden_states.reshape(b * s, h))
            return None if logits is None else logits.reshape(b, s, -1)
        return self.logits_processor(self.lm_head, hidden_states)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        audio_codes_list: list[torch.Tensor] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            ac = info.get("codes", {}).get("audio")
            if isinstance(ac, torch.Tensor):
                audio_codes_list.append(ac)
        if not audio_codes_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})
        audio_codes = torch.cat(audio_codes_list, dim=0)
        hidden = hidden[: int(audio_codes.shape[0])]
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={"codes": {"audio": audio_codes}})

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Route the Moshi checkpoint into the three components.

        * ``transformer.*`` / ``out_norm.alpha`` -> the Helium temporal backbone
          (same q/k-split + gate/up-split + alpha-squeeze map as HeliumForCausalLM).
        * ``text_linear.weight`` -> ``lm_head``.
        * ``emb.*`` / ``text_emb.weight`` -> input embeddings.
        * ``depformer*`` / ``linears.*`` -> depformer.
        """
        weights = list(weights)
        params = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()

        emb_w: list[tuple[str, torch.Tensor]] = []
        dep_w: list[tuple[str, torch.Tensor]] = []
        for name, w in weights:
            if name.startswith(("emb.", "text_emb.")):
                emb_w.append((name, w))
            elif name.startswith(("depformer.", "depformer_in.", "depformer_emb.", "depformer_text_emb", "linears.")):
                dep_w.append((name, w))
            elif name == "text_linear.weight":
                self._load_direct("lm_head.weight", w, params, loaded)
            else:
                loaded |= self._load_temporal(name, w, params)

        # Delegate to the verified component loaders, prefixed into this module.
        for sub, sub_w in (("input_embeddings", emb_w), ("depformer", dep_w)):
            module = getattr(self, sub)
            for tgt in module.load_weights(sub_w):
                loaded.add(f"{sub}.{tgt}")
        return loaded

    def _load_temporal(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params: dict[str, nn.Parameter],
    ) -> set[str]:
        out: set[str] = set()
        if name == "out_norm.alpha":
            self._load_direct("model.norm.weight", loaded_weight.squeeze(), params, out)
            return out
        prefix = "transformer.layers."
        if not name.startswith(prefix):
            return out
        rest = name.removeprefix(prefix)
        layer_index, _, suffix = rest.partition(".")
        if not layer_index.isdigit() or not suffix:
            return out
        base = f"model.layers.{layer_index}"
        if suffix == "self_attn.in_proj_weight":
            q, k, v = loaded_weight.chunk(3, dim=0)
            pname = f"{base}.self_attn.qkv_proj.weight"
            self._load_shard(pname, q, "q", params)
            self._load_shard(pname, k, "k", params)
            self._load_shard(pname, v, "v", params)
            out.add(pname)
        elif suffix == "self_attn.out_proj.weight":
            self._load_direct(f"{base}.self_attn.o_proj.weight", loaded_weight, params, out)
        elif suffix == "gating.linear_in.weight":
            gate, up = loaded_weight.chunk(2, dim=0)
            pname = f"{base}.mlp.gate_up_proj.weight"
            self._load_shard(pname, gate, 0, params)
            self._load_shard(pname, up, 1, params)
            out.add(pname)
        elif suffix == "gating.linear_out.weight":
            self._load_direct(f"{base}.mlp.down_proj.weight", loaded_weight, params, out)
        elif suffix == "norm1.alpha":
            self._load_direct(f"{base}.input_layernorm.weight", loaded_weight.squeeze(), params, out)
        elif suffix == "norm2.alpha":
            self._load_direct(f"{base}.post_attention_layernorm.weight", loaded_weight.squeeze(), params, out)
        return out

    @staticmethod
    def _load_direct(
        name: str,
        loaded_weight: torch.Tensor,
        params: dict[str, nn.Parameter],
        loaded: set[str],
    ) -> None:
        if name not in params:
            return
        param = params[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        loaded.add(name)

    @staticmethod
    def _load_shard(
        name: str,
        loaded_weight: torch.Tensor,
        shard_id: str | int,
        params: dict[str, nn.Parameter],
    ) -> None:
        if name not in params:
            return
        param = params[name]
        param.weight_loader(param, loaded_weight, shard_id)
