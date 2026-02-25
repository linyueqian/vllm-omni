from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TopKLogitsWarper,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import is_pp_missing_parameter

from .configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerConfig

logger = init_logger(__name__)


# ============================================================================
# Code Predictor Attention Layer (HF-style, batch-major, no KV cache)
# ============================================================================


class Qwen3TTSCodePredictorAttention(nn.Module):
    """Multi-head self-attention for TTS code predictor.

    Uses HF attention backends (SDPA/xformers/eager) instead of vLLM Attention,
    so that the code predictor can run inside a CUDA graph without needing
    per-step ``set_forward_context`` calls.
    """

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        layer_idx: int,
        quant_config: Any | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.hidden_size = config.hidden_size

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=True,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=True,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=None,
            dual_chunk_attention_config=None,
        )

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        # Query/Key normalization (Qwen3 style)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.is_causal = True
        self.config = config
        self.layer_idx = layer_idx

        self.attention_backends = ["flash_attention_2", "xformers", "eager", "sdpa"]
        cudagraph_mode = get_current_vllm_config().compilation_config.cudagraph_mode
        if "flash_attention_2" in ALL_ATTENTION_FUNCTIONS and cudagraph_mode.has_full_cudagraphs():
            logger.warning(
                "CUDAGraphMode.%s is currently not supported with flash attention "
                "for Qwen3-TTS code predictor. Removing flash_attention_2 from backends.",
                cudagraph_mode.name,
            )
            self.attention_backends.remove("flash_attention_2")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape for normalization
        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.reshape(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply Q/K normalization
        q = self.q_norm(q).contiguous()
        k = self.k_norm(k).contiguous()
        q = q.reshape(-1, self.q_size)
        k = k.reshape(-1, self.kv_size)

        # Apply RoPE
        q, k = self.rotary_emb(position_ids, q, k)

        # Reshape for attention
        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        v_heads = v.transpose(1, 2).contiguous()
        q_heads = q.transpose(1, 2).contiguous()
        k_heads = k.transpose(1, 2).contiguous()

        # Try attention backends in order of preference
        attn_output = None
        last_error = None

        for backend_name in self.attention_backends:
            if backend_name not in ALL_ATTENTION_FUNCTIONS:
                continue

            try:
                attention_interface = ALL_ATTENTION_FUNCTIONS[backend_name]
                attn_output, _ = attention_interface(
                    self,
                    q_heads,
                    k_heads,
                    v_heads,
                    None,
                    dropout=0.0 if not self.training else getattr(self, "attention_dropout", 0.0),
                    scaling=self.head_dim**-0.5,
                    sliding_window=None,
                    use_cache=False,
                    position_ids=position_ids[:seq_len].unsqueeze(0),
                    output_hidden_states=True,
                    output_attentions=False,
                )
                break
            except (ValueError, ImportError, RuntimeError, AttributeError) as e:
                last_error = e
                continue

        if attn_output is None:
            raise RuntimeError(
                f"All attention backends failed. Last error: {last_error}. "
                "Please install flash-attn, or ensure PyTorch's scaled_dot_product_attention is available."
            )
        attn_output = attn_output.reshape(*(hidden_states.shape[:-1]), -1).contiguous()

        attn_output, _ = self.o_proj(attn_output)
        return attn_output


# ============================================================================
# Code Predictor MLP Layer
# ============================================================================


class Qwen3TTSCodePredictorMLP(nn.Module):
    """Feed-forward network for TTS code predictor with fused gate/up projection."""

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        quant_config: Any | None = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=True,
        )

        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        down, _ = self.down_proj(F.silu(gate) * up)
        return down


# ============================================================================
# Code Predictor Transformer Layer
# ============================================================================


class Qwen3TTSCodePredictorMTPLayer(nn.Module):
    """Transformer layer for TTS code predictor — self-attention + MLP."""

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        layer_idx: int,
        quant_config: Any | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.self_attn = Qwen3TTSCodePredictorAttention(
            config,
            layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3TTSCodePredictorMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================================
# Code Predictor Model (Transformer backbone)
# ============================================================================


class Qwen3TTSTalkerCodePredictorModelVLLM(nn.Module):
    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        *,
        talker_hidden_size: int | None = None,
        quant_config: Any | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.layers = nn.ModuleList(
            [
                Qwen3TTSCodePredictorMTPLayer(
                    config,
                    layer_idx=i,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Official code_predictor uses one embedding table per residual group.
        # Some Qwen3-TTS checkpoints store codec embeddings in the talker hidden
        # space, even when `code_predictor_config.hidden_size` is smaller.
        # We keep the embedding dim aligned with the checkpoint and project down
        # via `small_to_mtp_projection` in the wrapper module.
        emb_dim = int(talker_hidden_size) if talker_hidden_size is not None else int(config.hidden_size)
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
        )

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.codec_embedding

    def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Batch-major forward: inputs_embeds is [B, seq_len, H]."""
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Match vLLM Qwen2/Qwen3 packing conventions: q_proj/k_proj/v_proj -> qkv_proj,
        # gate_proj/up_proj -> gate_up_proj.
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.quant_config is not None and (scale_name := self.quant_config.get_cache_scale(name)):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(".bias") and mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped, self):
                    continue
                if mapped.endswith("scale"):
                    mapped = maybe_remap_kv_scale_name(mapped, params_dict)
                    if mapped is None:
                        continue
                param = params_dict.get(mapped)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                mapped = maybe_remap_kv_scale_name(name, params_dict)
                if mapped is None:
                    continue
                if name.endswith(".bias") and mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped, self):
                    continue
                param = params_dict.get(mapped)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped)
        return loaded_params


# ============================================================================
# Code Predictor Wrapper (eager — AR loop has dynamic shapes)
# ============================================================================


class Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(nn.Module):
    """vLLM-native code_predictor used by the AR talker (residual codebooks).

    Uses HF-style attention (batch-major, full recompute, no KV cache).
    The internal AR loop has dynamic tensor shapes (sequence grows each step),
    so this module runs eagerly — torch.compile and CUDA graphs are not used.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen3TTSTalkerCodePredictorConfig,
        talker_config: Qwen3TTSTalkerConfig,
        prefix: str = "code_predictor",
    ) -> None:
        super().__init__()
        self.config = config
        self.talker_config = talker_config

        # Keep module/weight names aligned with official checkpoint (talker.code_predictor.model.*).
        self.model = Qwen3TTSTalkerCodePredictorModelVLLM(
            config,
            talker_hidden_size=int(talker_config.hidden_size),
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.model",
        )

        # One head per residual group.
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )

        if config.hidden_size != talker_config.hidden_size:
            self.small_to_mtp_projection = nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()

        # Sampling parameters (temperature + top_k).
        self.temperature = 0.9
        self.logits_processors = LogitsProcessorList(
            [
                TopKLogitsWarper(top_k=50),
            ]
        )

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.model.get_input_embeddings()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        model_weights: list[tuple[str, torch.Tensor]] = []
        other_weights: list[tuple[str, torch.Tensor]] = []
        for name, w in weights:
            if name.startswith("model."):
                model_weights.append((name[len("model.") :], w))
            else:
                other_weights.append((name, w))

        loaded_model = self.model.load_weights(model_weights)
        loaded |= {f"model.{n}" for n in loaded_model}

        params = dict(self.named_parameters(remove_duplicate=False))
        for name, w in other_weights:
            if name not in params:
                continue
            default_weight_loader(params[name], w)
            loaded.add(name)
        return loaded

    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Full autoregressive prediction of residual codebooks 1..Q-1.

        Uses full-recompute attention (no KV cache) at each AR step.

        Args:
            layer0_code: [B, 1] first-layer codec token ids.
            layer0_embed: [B, 1, H] embedding of layer0_code (talker hidden space).
            last_talker_hidden: [B, 1, H] hidden state from the talker.

        Returns:
            audio_codes: [B, Q] all codebook tokens (layer0 + residuals).
        """
        bsz = int(layer0_code.shape[0])
        num_groups = int(self.config.num_code_groups)

        # Start with [last_talker_hidden, layer0_embed], project to predictor hidden size.
        current_input = torch.cat([last_talker_hidden, layer0_embed], dim=1)  # [B, 2, talker_H]
        current_input = current_input.to(dtype=torch.bfloat16)
        current_input = self.small_to_mtp_projection(current_input)  # [B, 2, predictor_H]

        all_codes = [layer0_code.reshape(bsz, 1)]

        for layer_idx in range(num_groups - 1):
            seq_len = layer_idx + 2
            # Position IDs for full recompute (flat, repeated per batch element)
            position_ids = torch.arange(seq_len, device=current_input.device, dtype=torch.int64).repeat(bsz)

            # Forward through transformer (batch-major, full recompute)
            hidden_state = self.model(current_input, position_ids)  # [B, seq_len, H]

            # Get logits from last token via corresponding lm_head
            logits = self.lm_head[layer_idx](hidden_state[:, -1:, :])

            # Sample (temperature + top_k)
            scaled = logits[:, -1] / self.temperature
            scaled = self.logits_processors(None, scaled)
            probs = F.softmax(scaled, dim=-1)
            code = torch.multinomial(probs, num_samples=1)  # [B, 1]
            all_codes.append(code)

            # Embed new code and concat for next step (skip on last iteration)
            if layer_idx < num_groups - 2:
                new_embed = self.model.codec_embedding[layer_idx](code)  # [B, 1, talker_H]
                new_embed = self.small_to_mtp_projection(new_embed)  # [B, 1, predictor_H]
                current_input = torch.cat([current_input, new_embed], dim=1)  # [B, seq_len+1, H]

        return torch.cat(all_codes, dim=1)  # [B, Q]
