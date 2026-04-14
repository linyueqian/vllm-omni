"""Fast AR (audio_decoder) for Fish Speech S2 Pro.

4-layer transformer that predicts residual codebook codes (1..num_codebooks-1)
autoregressively after each Slow AR step.  Analogous to Qwen3 TTS's
CodePredictor but with its own embedding table, RoPE, and output head.

Uses KV-cached incremental decode: each codebook step only forwards the
new token [B, 1, H] while caching K/V from prior positions.  The KV cache
is reset at each Slow AR step (max 10+1 positions, negligible memory).

Optimisations:
  - KV-cached incremental decode (O(n) vs O(n^2) per step)
  - torch.compile on model forward (kernel fusion)
  - Pre-allocated KV cache + embedding buffer
  - Pre-allocated position_ids
  - Inline sampling
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config
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

from .configuration_fish_speech import FishSpeechFastARConfig, FishSpeechSlowARConfig

logger = init_logger(__name__)


# ===================================================================
#  Standalone Fast AR Layers (KV-cached incremental decode)
# ===================================================================


class _FastARAttention(nn.Module):
    """Multi-head attention with static KV cache for incremental decode.

    Supports fused QKV, RoPE, optional q/k normalization, and native GQA.
    Input: [B, seq_len, hidden_size].

    KV cache layout per layer: (k_cache, v_cache) each [B, num_kv_heads, max_seq, head_dim].
    """

    def __init__(self, config: FishSpeechFastARConfig, *, prefix: str = "") -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self._use_gqa = self.num_kv_heads != self.num_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=True,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            prefix=f"{prefix}.o_proj",
            disable_tp=True,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            is_neox_style=False,  # Fish Speech uses interleaved (GPT-J) RoPE
            rope_parameters={"rope_theta": config.rope_theta, "rope_type": "default"},
        )
        # Fish Speech Fast AR typically has attention_qk_norm=false.
        if config.attention_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_position: int = 0,
    ) -> torch.Tensor:
        """Forward with optional KV cache.

        Args:
            hidden_states: [B, seq_len, H]
            position_ids: [B, seq_len]
            kv_cache: (k_cache, v_cache) each [B, num_kv_heads, max_seq, head_dim].
                      If provided, writes new K/V at cache_position and attends
                      over positions [0, cache_position + seq_len).
            cache_position: write offset into KV cache (number of tokens
                            already cached).
        """
        bsz, seq_len, _ = hidden_states.shape

        qkv, _ = self.qkv_proj(hidden_states.reshape(bsz * seq_len, -1))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.q_norm is not None:
            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)

        q, k = self.rotary_emb(position_ids, q, k)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Write new K/V into cache at the current position.
            k_cache[:bsz, :, cache_position : cache_position + seq_len, :] = k
            v_cache[:bsz, :, cache_position : cache_position + seq_len, :] = v
            # Attend over all cached positions [0, cache_position + seq_len).
            total_len = cache_position + seq_len
            k = k_cache[:bsz, :, :total_len, :]
            v = v_cache[:bsz, :, :total_len, :]

        # For prefill (seq_len > 1), use causal mask so each position only
        # attends to itself and earlier positions.  For single-token decode
        # (seq_len == 1), no mask needed since there is only one query.
        use_causal = seq_len > 1
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scaling,
            is_causal=use_causal,
            enable_gqa=self._use_gqa,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz * seq_len, -1)
        output, _ = self.o_proj(attn_out)
        return output.view(bsz, seq_len, -1)


class _FastARMLP(nn.Module):
    """SiLU-gated MLP, matching Qwen3/LLaMA MLP structure."""

    def __init__(self, config: FishSpeechFastARConfig, *, prefix: str = "") -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=True,
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
            disable_tp=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.silu(gate) * up
        x, _ = self.down_proj(x)
        return x


class _FastARDecoderLayer(nn.Module):
    """Transformer decoder layer for Fast AR with optional KV cache."""

    def __init__(self, config: FishSpeechFastARConfig, *, prefix: str = "") -> None:
        super().__init__()
        self.self_attn = _FastARAttention(config, prefix=f"{prefix}.self_attn")
        self.mlp = _FastARMLP(config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_position: int = 0,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_ids,
            kv_cache=kv_cache, cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ===================================================================
#  Fast AR Transformer Model
# ===================================================================


class FishSpeechFastARModel(nn.Module):
    """4-layer transformer for residual codebook prediction (KV-cached)."""

    def __init__(self, config: FishSpeechFastARConfig, *, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [_FastARDecoderLayer(config, prefix=f"{prefix}.layers.{i}") for i in range(config.num_hidden_layers)]
        )
        # NOTE: final norm is handled by FishSpeechFastAR.fast_norm (one norm weight
        # in checkpoint: audio_decoder.norm.weight → fast_ar.fast_norm.weight).

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        cache_position: int = 0,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            hidden_states = layer(
                hidden_states, position_ids,
                kv_cache=kv, cache_position=cache_position,
            )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
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


# ===================================================================
#  Fast AR Wrapper (KV-cached incremental decode + torch.compile)
# ===================================================================


class FishSpeechFastAR(nn.Module):
    """vLLM-native Fast AR for Fish Speech (residual codebooks).

    KV-cached incremental decode: each codebook step only forwards
    the new token [B, 1, H] while KV from prior positions stays cached.
    The cache is reset at the start of each Slow AR step.

    This is O(n) per step vs O(n^2) for re-prefill, and enables better
    batching across concurrent requests.

    Optimisations:
      1. KV-cached incremental decode (each step is [B, 1, H]).
      2. torch.compile on model forward -- kernel fusion.
      3. Pre-allocated KV cache + embedding buffer.
      4. Pre-allocated position_ids.
      5. Inline sampling.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: FishSpeechFastARConfig,
        slow_ar_config: FishSpeechSlowARConfig,
        prefix: str = "fast_ar",
    ) -> None:
        super().__init__()
        self._vllm_config = vllm_config
        self.config = config
        self.slow_ar_config = slow_ar_config

        self.model = FishSpeechFastARModel(config, prefix=f"{prefix}.model")

        # Codebook embeddings for Fast AR (separate from Slow AR's codebook_embeddings).
        self.fast_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # Output projection for residual codes.
        self.fast_output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Layer norm for Fast AR output.
        self.fast_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Projection from Slow AR hidden dim → Fast AR hidden dim.
        # s2-pro has same dim for both (2560) and no fast_project_in in checkpoint,
        # so use identity when dims match.
        if slow_ar_config.hidden_size != config.hidden_size:
            self.fast_project_in = nn.Linear(slow_ar_config.hidden_size, config.hidden_size, bias=True)
        else:
            self.fast_project_in = nn.Identity()

        self._num_codebooks = config.num_codebooks
        self._num_layers = config.num_hidden_layers
        self._fast_dim = config.hidden_size
        self._num_kv_heads = config.num_key_value_heads
        self._head_dim = config.head_dim
        self._max_seq = config.num_codebooks + 1  # projected hidden + codebook tokens

        # Pre-allocated buffers (lazily initialised on first forward).
        self._embed_buf: torch.Tensor | None = None
        self._pos_ids: torch.Tensor | None = None
        self._kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._compiled_model_fwd: object | None = None
        self._compile_attempted = False
        self._compile_failed = False
        self._disable_compile_for_graph = False

    def _ensure_buffers(self, bsz: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self._embed_buf is not None
            and self._embed_buf.shape[0] >= bsz
            and self._embed_buf.device == device
            and self._embed_buf.dtype == dtype
        ):
            return
        self._embed_buf = torch.zeros(bsz, self._max_seq, self._fast_dim, dtype=dtype, device=device)
        self._pos_ids = torch.arange(self._max_seq, dtype=torch.long, device=device)
        # Allocate KV caches for all layers.
        # Each: (k_cache, v_cache) with shape [bsz, num_kv_heads, max_seq, head_dim]
        self._kv_caches = [
            (
                torch.zeros(bsz, self._num_kv_heads, self._max_seq, self._head_dim, dtype=dtype, device=device),
                torch.zeros(bsz, self._num_kv_heads, self._max_seq, self._head_dim, dtype=dtype, device=device),
            )
            for _ in range(self._num_layers)
        ]

    def reset_caches(self) -> None:
        """Reset KV caches (called at the start of each Slow AR step)."""
        if self._kv_caches is not None:
            for k_cache, v_cache in self._kv_caches:
                k_cache.zero_()
                v_cache.zero_()

    def _setup_compile(self) -> None:
        if self._compile_attempted:
            return
        self._compile_attempted = True
        if self._disable_compile_for_graph:
            try:
                self._compiled_model_fwd = torch.compile(
                    self.model.forward,
                    dynamic=True,
                    options={"epilogue_fusion": False},
                )
            except Exception as exc:
                logger.warning("Fast AR torch.compile (graph mode) failed: %s", exc)
                self._compiled_model_fwd = self.model.forward
            return
        try:
            self._compiled_model_fwd = torch.compile(
                self.model.forward,
                mode="default",
                dynamic=True,
                fullgraph=False,
            )
        except Exception as exc:
            self._compile_failed = True
            logger.warning("Failed to enable torch.compile for Fish Speech Fast AR: %s", exc)
            self._compiled_model_fwd = self.model.forward
        else:
            logger.info("Enabled torch.compile for Fish Speech Fast AR forward (mode=default)")

    @torch.inference_mode()
    def warmup_compile(
        self,
        device: torch.device,
        dtype: torch.dtype,
        batch_sizes: tuple[int, ...] = (1,),
    ) -> None:
        self._setup_compile()
        if self._compiled_model_fwd is self.model.forward or self._compile_failed:
            return
        for batch_size in batch_sizes:
            hidden = torch.zeros((batch_size, self.slow_ar_config.hidden_size), device=device, dtype=dtype)
            semantic = torch.full(
                (batch_size,),
                self.slow_ar_config.semantic_begin_id,
                device=device,
                dtype=torch.long,
            )
            self(hidden, semantic, do_sample=False)
        torch.cuda.synchronize(device)

    @torch.inference_mode()
    def _run_model(
        self,
        step_input: torch.Tensor,
        step_pos_ids: torch.Tensor,
        bsz: int,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        cache_position: int = 0,
    ) -> torch.Tensor:
        model_fwd = self._compiled_model_fwd or self.model.forward
        if not self._disable_compile_for_graph and bsz != 1:
            model_fwd = self.model.forward
        try:
            return model_fwd(step_input, step_pos_ids, kv_caches=kv_caches, cache_position=cache_position)
        except Exception as exc:
            if model_fwd is self.model.forward or self._compile_failed:
                raise
            self._compile_failed = True
            self._compiled_model_fwd = self.model.forward
            logger.warning("Fish Speech Fast AR torch.compile fallback to eager after runtime failure: %s", exc)
            return self.model.forward(step_input, step_pos_ids, kv_caches=kv_caches, cache_position=cache_position)

    @torch.inference_mode()
    def forward(
        self,
        slow_ar_hidden: torch.Tensor,
        semantic_token_id: torch.Tensor,
        *,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_k: int = 30,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Predict residual codebook codes 0..num_codebooks-1 via KV-cached decode.

        Args:
            slow_ar_hidden: [B, hidden_size] last hidden state from Slow AR.
            semantic_token_id: [B] or [B, 1] sampled semantic token IDs (in vocab space).

        Returns:
            all_codes: [B, num_codebooks] codes for codebooks 0..num_codebooks-1.
                       Code 0 is the semantic code (token - semantic_begin_id).
        """
        bsz = int(slow_ar_hidden.shape[0])
        num_cb = self._num_codebooks
        device = slow_ar_hidden.device
        dtype = slow_ar_hidden.dtype

        semantic_begin = self.slow_ar_config.semantic_begin_id
        semantic_end = self.slow_ar_config.semantic_end_id
        codebook_size = semantic_end - semantic_begin + 1  # 4096
        # Convert vocab-space semantic token to codebook index.
        semantic_code = (semantic_token_id.reshape(bsz) - semantic_begin).clamp(min=0, max=codebook_size - 1)

        all_codes = torch.empty(bsz, num_cb, dtype=torch.long, device=device)
        all_codes[:, 0] = semantic_code

        self._ensure_buffers(bsz, device, dtype)
        self._setup_compile()

        embed_buf = self._embed_buf
        pos_ids = self._pos_ids

        # Reset KV caches for this new Slow AR step.
        self.reset_caches()

        # Position 0: projected Slow AR hidden state.
        projected = self.fast_project_in(slow_ar_hidden.reshape(bsz, -1))
        embed_buf[:bsz, 0, :] = projected

        # Position 1: embedding of semantic code.
        code_embed = self.fast_embeddings(semantic_code.clamp(min=0))
        embed_buf[:bsz, 1, :] = code_embed

        use_sampling = do_sample and temperature > 0
        inv_temperature = 1.0 / max(temperature, 1e-6) if use_sampling else 0.0

        # Residual codebook size (1024) vs semantic codebook size (4096).
        residual_codebook_size = 1024

        # Prefill step: forward positions [0, 1] together to populate KV cache.
        prefill_input = embed_buf[:bsz, :2, :]  # [B, 2, H]
        prefill_pos = pos_ids[:2].unsqueeze(0).expand(bsz, -1)  # [B, 2]
        hidden_out = self._run_model(
            prefill_input, prefill_pos, bsz,
            kv_caches=self._kv_caches, cache_position=0,
        )
        # Sample first residual code from position 1's output.
        logits = self.fast_output(self.fast_norm(hidden_out[:, -1, :]))
        logits = logits[:, :residual_codebook_size]
        all_codes[:, 1] = self._sample(logits, use_sampling, inv_temperature, top_k, top_p)

        # Incremental decode: steps 2..num_cb-1, each forwarding only [B, 1, H].
        for step in range(2, num_cb):
            # Embed the code just sampled at step-1.
            new_embed = self.fast_embeddings(all_codes[:, step - 1])
            embed_buf[:bsz, step, :] = new_embed

            step_input = embed_buf[:bsz, step : step + 1, :]  # [B, 1, H]
            step_pos = pos_ids[step : step + 1].unsqueeze(0).expand(bsz, -1)  # [B, 1]

            hidden_out = self._run_model(
                step_input, step_pos, bsz,
                kv_caches=self._kv_caches, cache_position=step,
            )
            logits = self.fast_output(self.fast_norm(hidden_out[:, -1, :]))
            logits = logits[:, :residual_codebook_size]
            all_codes[:, step] = self._sample(logits, use_sampling, inv_temperature, top_k, top_p)

        return all_codes

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        use_sampling: bool,
        inv_temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Sample from logits with temperature/top-k/top-p."""
        if not use_sampling:
            return logits.argmax(dim=-1)

        scaled = logits * inv_temperature
        if top_k > 0:
            topk_vals, _ = scaled.topk(min(top_k, scaled.shape[-1]), dim=-1)
            scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_indices_to_remove] = float("-inf")
            scaled = sorted_logits.scatter(1, sorted_indices, sorted_logits)
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).reshape(-1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights -- Fast AR weights have already been remapped by the parent."""
        with set_current_vllm_config(self._vllm_config):
            params_dict = dict(self.named_parameters(remove_duplicate=False))
            loaded: set[str] = set()

            stacked_params_mapping = [
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ]

            for name, loaded_weight in weights:
                if "rotary_emb.inv_freq" in name:
                    continue

                handled = False
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped = name.replace(weight_name, param_name)
                    if mapped not in params_dict:
                        continue
                    param = params_dict[mapped]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    if weight_loader == default_weight_loader:
                        weight_loader(param, loaded_weight)
                    else:
                        weight_loader(param, loaded_weight, shard_id)
                    loaded.add(mapped)
                    handled = True
                    break

                if handled:
                    continue

                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded.add(name)

            return loaded
