# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 codebook tower: predicts codes 1..num_codebooks-1 of each frame.

Every tower layer attends causally over the whole sequence, so each layer
keeps its own key/value history per request. The tower at sequence position
``p`` reads the backbone's final-norm hidden state at ``p`` and the frame at
``p`` (input), and predicts the codes of frame ``p + 1``.

Module names follow the checkpoint (``audio_decoder*``) so weights load by name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .configuration_fish_omni import FishOmniAudioDecoderConfig


class GemmaRMSNorm(nn.Module):
    """RMSNorm with a zero-centered gain (scale = 1 + weight); with ``residual``
    it normalizes ``x + residual`` and also returns that sum."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is not None:
            x = x + residual
            residual = x
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        out = (h * (1.0 + self.weight.float())).to(x.dtype)
        return out if residual is None else (out, residual)


def interleaved_rope(states: torch.Tensor, positions: torch.Tensor, rotary_dim: int, base: float) -> torch.Tensor:
    """GPT-J rope on the first ``rotary_dim`` channels of [rows, heads, head_dim]."""
    inv_freq = 1.0 / base ** (torch.arange(0, rotary_dim, 2, device=states.device, dtype=torch.float32) / rotary_dim)
    angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
    rotary = states[..., :rotary_dim].float()
    even, odd = rotary[..., 0::2], rotary[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    return torch.cat((rotated.to(states.dtype), states[..., rotary_dim:]), dim=-1)


class TowerKVCache:
    """Dense per-request key/value history for every tower layer.

    One slot per live request: [slots, layers, 2, max_len, kv_heads, head_dim].
    Allocated on first use; a request takes a slot at prefill and keeps it until
    the slot is reclaimed for a newer request (least recently used first).
    """

    def __init__(self, *, slots: int, layers: int, max_len: int, kv_heads: int, head_dim: int) -> None:
        self.slots = slots
        self.layers = layers
        self.max_len = max_len
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.buffer: torch.Tensor | None = None
        self._owner: list[object | None] = [None] * slots
        self._last_use = [0] * slots
        self._clock = 0

    def ensure(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.buffer is None:
            self.buffer = torch.zeros(
                (self.slots, self.layers, 2, self.max_len, self.kv_heads, self.head_dim),
                device=device,
                dtype=dtype,
            )
        return self.buffer

    def acquire(self, owner: object, in_use: set[int]) -> int:
        self._clock += 1
        free = [s for s in range(self.slots) if self._owner[s] is None]
        if not free:
            free = sorted((s for s in range(self.slots) if s not in in_use), key=lambda s: self._last_use[s])
        if not free:
            raise RuntimeError(f"all {self.slots} tower cache slots are in use")
        slot = free[0]
        self._owner[slot] = owner
        self._last_use[slot] = self._clock
        return slot

    def owns(self, slot: int, owner: object) -> bool:
        return self._owner[slot] is owner

    def touch(self, slot: int) -> None:
        self._clock += 1
        self._last_use[slot] = self._clock


@dataclass(eq=False)
class TowerState:
    """Per-request tower progress, kept in the request's intermediate buffer."""

    slot: int
    next_position: int
    input_frame: torch.Tensor  # [num_codebooks] codes of the frame at next_position
    input_is_semantic: bool


class TowerAttention(nn.Module):
    def __init__(self, config: FishOmniAudioDecoderConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv = config.n_local_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.rope_base = config.rope_base
        q_width = self.n_head * self.head_dim
        kv_width = self.n_kv * self.head_dim
        # Checkpoint rows: [query | output gate | key | value].
        self.wqkv = nn.Linear(config.dim, 2 * q_width + 2 * kv_width, bias=False)
        self.wo = nn.Linear(q_width, config.dim, bias=False)
        self.q_norm = GemmaRMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        layer_cache: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """x [rows, dim]; positions [rows]; slots [rows] index ``layer_cache``
        [slots, 2, max_len, kv_heads, head_dim]. Rows of one slot must be in
        position order, and a slot's cached positions must all precede them."""
        rows = x.shape[0]
        q_width = self.n_head * self.head_dim
        kv_width = self.n_kv * self.head_dim
        query, gate, key, value = self.wqkv(x).split([q_width, q_width, kv_width, kv_width], dim=-1)
        query = self.q_norm(query.reshape(-1, self.head_dim)).view(rows, self.n_head, self.head_dim)
        key = self.k_norm(key.reshape(-1, self.head_dim)).view(rows, self.n_kv, self.head_dim)
        query = interleaved_rope(query, positions, self.rotary_dim, self.rope_base)
        key = interleaved_rope(key, positions, self.rotary_dim, self.rope_base)
        value = value.view(rows, self.n_kv, self.head_dim)

        layer_cache[slots, 0, positions] = key.to(layer_cache.dtype)
        layer_cache[slots, 1, positions] = value.to(layer_cache.dtype)

        # Group rows by slot: [groups, rows_per_group]. Decode has one row per
        # slot; prefill has one slot with many rows.
        unique_slots, inverse = torch.unique(slots, return_inverse=True)
        if unique_slots.numel() == rows:
            key_len = int(positions.max().item()) + 1
            keys = layer_cache[slots, 0, :key_len].transpose(1, 2)  # [rows, kv, len, d]
            values = layer_cache[slots, 1, :key_len].transpose(1, 2)
            mask = torch.arange(key_len, device=x.device).unsqueeze(0) <= positions.unsqueeze(1)
            attended = F.scaled_dot_product_attention(
                query.unsqueeze(2),
                keys,
                values,
                attn_mask=mask[:, None, None, :],
                enable_gqa=True,
            ).squeeze(2)
        elif unique_slots.numel() == 1:
            slot = int(unique_slots.item())
            key_len = int(positions.max().item()) + 1
            keys = layer_cache[slot, 0, :key_len].transpose(0, 1).unsqueeze(0)
            values = layer_cache[slot, 1, :key_len].transpose(0, 1).unsqueeze(0)
            mask = torch.arange(key_len, device=x.device).unsqueeze(0) <= positions.unsqueeze(1)
            attended = F.scaled_dot_product_attention(
                query.transpose(0, 1).unsqueeze(0),
                keys,
                values,
                attn_mask=mask[None, None],
                enable_gqa=True,
            )[0].transpose(0, 1)
        else:
            raise ValueError("tower attention takes one row per slot, or many rows of one slot")
        attended = attended.reshape(rows, q_width)
        return self.wo(attended * torch.sigmoid(gate))


class TowerFeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TowerBlock(nn.Module):
    def __init__(self, config: FishOmniAudioDecoderConfig) -> None:
        super().__init__()
        self.attention = TowerAttention(config)
        self.feed_forward = TowerFeedForward(config.dim, config.intermediate_size)
        self.attention_norm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = GemmaRMSNorm(config.dim, config.norm_eps)


class TowerLayer(nn.Module):
    """Fuse a code embedding with the incoming hidden state (eh_proj), then apply
    one attention + feed-forward block."""

    def __init__(self, config: FishOmniAudioDecoderConfig, embedding_name: str) -> None:
        super().__init__()
        self.embedding_name = embedding_name
        self.add_module(embedding_name, nn.Embedding(config.vocab_size, config.dim))
        self.enorm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.hnorm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.eh_proj = nn.Linear(2 * config.dim, config.dim, bias=False)
        self.transformer_layer = TowerBlock(config)
        self.final_layernorm = GemmaRMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        code_ids: torch.Tensor,
        code_is_valid: torch.Tensor,
        extra_input: torch.Tensor | None,
        positions: torch.Tensor,
        layer_cache: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        table: nn.Embedding = getattr(self, self.embedding_name)
        embedded = torch.where(
            code_is_valid.unsqueeze(-1),
            table(code_ids.clamp(0, table.num_embeddings - 1)),
            0.0,
        )
        if extra_input is not None:
            embedded = embedded + extra_input.to(embedded.dtype)
        mixed = self.eh_proj(torch.cat((self.enorm(embedded), self.hnorm(hidden)), dim=-1))
        block = self.transformer_layer
        attended = block.attention(block.attention_norm(mixed), positions, layer_cache, slots)
        ffn_input, residual = block.ffn_norm(attended, mixed)
        out, _ = self.final_layernorm(block.feed_forward(ffn_input), residual)
        return out


class _LayerList(nn.Module):
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _ResidualTower(nn.Module):
    def __init__(self, config: FishOmniAudioDecoderConfig) -> None:
        super().__init__()
        count = config.num_codebooks - 1
        self.layers = nn.ModuleList(TowerLayer(config, "codebook_embedding") for _ in range(count))
        self.heads = nn.ModuleList(nn.Linear(config.dim, config.vocab_size, bias=False) for _ in range(count))


class _FrameEmbeddings(nn.Module):
    def __init__(self, config: FishOmniAudioDecoderConfig) -> None:
        super().__init__()
        self.codebook_embeddings = nn.Embedding(config.vocab_size * config.num_codebooks, config.dim)


def sample_rows(
    logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor
) -> torch.Tensor:
    """Sample a column index per row, with per-row temperature, top-k and top-p.

    A candidate survives when its rank is below top_k and the probability mass
    before and including it stays within top_p; the most likely candidate is
    never filtered out."""
    scaled = logits.float() / temperature.float().clamp(min=1e-5).unsqueeze(-1)
    sorted_logits, order = torch.sort(scaled, dim=-1, descending=True)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    ranks = torch.arange(sorted_logits.shape[-1], device=logits.device).unsqueeze(0)
    drop = (cumulative > top_p.float().unsqueeze(-1)) | (ranks >= top_k.unsqueeze(-1))
    drop[:, 0] = False
    probs = torch.softmax(sorted_logits.masked_fill(drop, float("-inf")), dim=-1)
    return torch.gather(order, -1, torch.multinomial(probs, 1)).squeeze(-1)


def sample_top_k_top_p(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """Sample one token per row with scalar parameters (same filter as sample_rows)."""
    scaled = logits.float() / max(temperature, 1e-5)
    k = min(top_k, scaled.shape[-1]) if top_k > 0 else scaled.shape[-1]
    top_logits, top_ids = torch.topk(scaled, k, dim=-1)
    probs = torch.softmax(top_logits, dim=-1)
    drop = torch.cumsum(probs, dim=-1) > top_p
    drop[:, 0] = False
    probs = torch.softmax(top_logits.masked_fill(drop, float("-inf")), dim=-1)
    choice = torch.multinomial(probs, 1)
    return torch.gather(top_ids, -1, choice).squeeze(-1)


class CodebookTower(nn.Module):
    """Frame embeddings, 4 shared layers, and 9 residual layers with heads."""

    def __init__(self, config: FishOmniAudioDecoderConfig, *, semantic_begin_id: int, cache_slots: int, max_len: int):
        super().__init__()
        self.config = config
        self.semantic_begin_id = semantic_begin_id
        self.codebook_size = config.vocab_size
        self.num_codebooks = config.num_codebooks
        self.backbone_codebooks = config.vq_backbone_codebooks
        self.audio_decoder = _FrameEmbeddings(config)
        self.audio_decoder_mtp_shared = _LayerList(
            [TowerLayer(config, "semantic_embedding") for _ in range(config.shared_enrichment_layers)]
        )
        self.audio_decoder_mtp = _ResidualTower(config)
        self.register_buffer("row_offsets", torch.arange(config.num_codebooks) * config.vocab_size, persistent=False)
        self.cache = TowerKVCache(
            slots=cache_slots,
            layers=config.shared_enrichment_layers + config.num_codebooks - 1,
            max_len=max_len,
            kv_heads=config.n_local_heads,
            head_dim=config.head_dim,
        )

    def is_semantic(self, token_ids: torch.Tensor) -> torch.Tensor:
        return (token_ids >= self.semantic_begin_id) & (token_ids < self.semantic_begin_id + self.codebook_size)

    def code_embedding_sum(self, codes: torch.Tensor, first: int, end: int) -> torch.Tensor:
        """Sum of codebook embeddings of codes[:, first:end] (codebook-offset rows)."""
        rows = codes[:, first:end].clamp(0, self.codebook_size - 1) + self.row_offsets[first:end]
        return F.embedding_bag(rows, self.audio_decoder.codebook_embeddings.weight, mode="sum")

    def backbone_embedding(self, token_embeds: torch.Tensor, codes: torch.Tensor, is_semantic: torch.Tensor):
        """Backbone input: for audio frames, the token embedding plus the embeddings of
        the first backbone_codebooks residual codes, rescaled."""
        extra = self.code_embedding_sum(codes, 1, 1 + self.backbone_codebooks).to(token_embeds.dtype)
        mixed = (token_embeds + extra) * (1.0 / math.sqrt(1.0 + self.backbone_codebooks))
        return torch.where(is_semantic.unsqueeze(-1), mixed, token_embeds)

    def _shared_input(self, input_codes: torch.Tensor, input_is_semantic: torch.Tensor) -> torch.Tensor:
        extra = self.code_embedding_sum(input_codes, 1 + self.backbone_codebooks, self.num_codebooks)
        return torch.where(input_is_semantic.unsqueeze(-1), extra, 0.0)

    def run(
        self,
        hidden: torch.Tensor,
        input_codes: torch.Tensor,
        input_is_semantic: torch.Tensor,
        next_main_tokens: torch.Tensor,
        next_codes: torch.Tensor | None,
        positions: torch.Tensor,
        slots: torch.Tensor,
        sampling: tuple[float, int, float],
    ) -> torch.Tensor:
        """Run all tower layers. With ``next_codes`` (prefill) the codes are teacher
        forced; without it (decode) codes 1.. are sampled. Returns next codes [rows, Q]."""
        cache = self.cache.ensure(hidden.device, hidden.dtype)
        label_is_semantic = self.is_semantic(next_main_tokens)
        code0 = (next_main_tokens - self.semantic_begin_id).clamp(0, self.codebook_size - 1)
        shared_input = self._shared_input(input_codes, input_is_semantic)
        layer_index = 0
        for i, layer in enumerate(self.audio_decoder_mtp_shared.layers):
            hidden = layer(
                hidden,
                code0,
                label_is_semantic,
                shared_input if i == 0 else None,
                positions,
                cache[:, layer_index],
                slots,
            )
            layer_index += 1
        codes = [code0]
        for r, (layer, head) in enumerate(zip(self.audio_decoder_mtp.layers, self.audio_decoder_mtp.heads)):
            current = codes[-1] if next_codes is None else next_codes[:, r]
            hidden = layer(hidden, current, label_is_semantic, None, positions, cache[:, layer_index], slots)
            layer_index += 1
            if next_codes is None:
                codes.append(sample_top_k_top_p(head(hidden), *sampling))
        if next_codes is not None:
            return next_codes
        return torch.stack(codes, dim=-1)
