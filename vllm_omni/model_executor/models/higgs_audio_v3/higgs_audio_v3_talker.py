# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-0 talker for higgs-audio v3 (Qwen3 backbone, fused multi-codebook).

Architecture:
- Backbone: Qwen3 (~4B, 36 layers, 2560 hidden, GQA 32/8). No DualFFN.
- Fused multi-codebook embedding: [N*V, D] weight, offset lookup, sum across N
- Fused multi-codebook head: same weight (tied), reshape to [L, N, V]
- MusicGen-style delay pattern [0,1,...,7] with BOC/EOC
- Audio feedback: replace continuation-token embedding with fused codebook embed

Weight loading maps from the HF checkpoint's prefixes:
  tied.embedding.text_embedding. -> model.embed_tokens.
  body.layers.                   -> model.layers.
  body.norm.                     -> model.norm.
  tied.head.text_head.           -> lm_head.
  tied.embedding.modality_embeddings.0.embedding. -> multimodal_embedding.
  tied.embedding.modality_embeddings.0.model.*    -> skipped (codec for code2wav)
  tied.head.modality_heads.0.*                    -> skipped when tied
"""

from __future__ import annotations

import copy
import os
import time
from collections import Counter
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.platforms import current_platform
from vllm.v1.outputs import SamplerOutput

from vllm_omni.model_executor.models.higgs_audio_v3.configuration_higgs_audio_v3 import (
    HiggsAudioV3Config,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

__all__ = ["HiggsAudioV3TalkerForConditionalGeneration"]

logger = init_logger(__name__)

_PROFILE_ENABLED = bool(os.getenv("HIGGS_AUDIO_V3_PROFILE"))
_PROFILE_SYNC = os.getenv("HIGGS_AUDIO_V3_PROFILE_SYNC", "1").lower() not in {"0", "false", "no"}
_PROFILE_SUMMARY_EVERY = max(1, int(os.getenv("HIGGS_AUDIO_V3_PROFILE_SUMMARY_EVERY", "200")))
_PROFILE_LAYER_STEPS = max(0, int(os.getenv("HIGGS_AUDIO_V3_PROFILE_LAYER_STEPS", "0")))
_PROFILE_STATS: dict[str, list[float]] = {}
_PROFILE_EVENTS = 0
_LAYER_CUDAGRAPH_ENABLED = os.getenv("HIGGS_AUDIO_V3_LAYER_CUDAGRAPH", "").lower() in {"1", "true", "yes"}
_LAYER_CUDAGRAPH_STRICT_SIGNATURE = os.getenv("HIGGS_AUDIO_V3_LAYER_CUDAGRAPH_STRICT_SIGNATURE", "").lower() in {
    "1",
    "true",
    "yes",
}
_LAYER_CUDAGRAPH_STATS_ENABLED = os.getenv("HIGGS_AUDIO_V3_LAYER_CUDAGRAPH_STATS", "").lower() in {
    "1",
    "true",
    "yes",
}
_LAYER_CUDAGRAPH_STATS_EVERY = max(0, int(os.getenv("HIGGS_AUDIO_V3_LAYER_CUDAGRAPH_STATS_EVERY", "1000") or 0))
_LAYER_CUDAGRAPH_CACHE_PER_BATCH = max(1, int(os.getenv("HIGGS_AUDIO_V3_LAYER_CUDAGRAPH_CACHE_PER_BATCH", "16") or 1))
_LAYER_CUDAGRAPH_WARMUP_RUNS = max(0, int(os.getenv("HIGGS_AUDIO_V3_LAYER_CUDAGRAPH_WARMUP_RUNS", "0") or 0))
_FAST_AUDIO_SAMPLER_ENABLED = os.getenv("HIGGS_AUDIO_V3_FAST_AUDIO_SAMPLER", "1").lower() not in {
    "0",
    "false",
    "no",
}
_FAST_AUDIO_CPU_METADATA_FALLBACK = os.getenv("HIGGS_AUDIO_V3_CPU_AUDIO_MODE_FALLBACK", "").lower() in {
    "1",
    "true",
    "yes",
}
_FAST_AUDIO_SAMPLER_STATS_ENABLED = os.getenv("HIGGS_AUDIO_V3_FAST_AUDIO_SAMPLER_STATS", "").lower() in {
    "1",
    "true",
    "yes",
}
_FAST_AUDIO_SAMPLER_STATS_EVERY = max(
    0,
    int(os.getenv("HIGGS_AUDIO_V3_FAST_AUDIO_SAMPLER_STATS_EVERY", "1000") or 0),
)
_FAST_AUDIO_TOPK_SAMPLING_ENABLED = os.getenv("HIGGS_AUDIO_V3_FAST_AUDIO_TOPK_SAMPLING", "").lower() in {
    "1",
    "true",
    "yes",
}
_FAST_AUDIO_ASSUME_FULL_DECODE = os.getenv("HIGGS_AUDIO_V3_FAST_AUDIO_ASSUME_FULL_DECODE", "").lower() in {
    "1",
    "true",
    "yes",
}
_SKIP_TEXT_LOGITS_ENABLED = os.getenv("HIGGS_AUDIO_V3_SKIP_TEXT_LOGITS", "").lower() in {
    "1",
    "true",
    "yes",
}
_TERMINATION_DEBUG = os.getenv("HIGGS_AUDIO_V3_TERMINATION_DEBUG", "").lower() in {
    "1",
    "true",
    "yes",
}
_TERMINATION_DEBUG_INTERVAL = int(os.getenv("HIGGS_AUDIO_V3_TERMINATION_DEBUG_INTERVAL", "100") or "100")


class _ProfileScope:
    def __init__(self, name: str):
        self.name = name
        self.start = 0.0

    def __enter__(self) -> None:
        if _PROFILE_SYNC and torch.cuda.is_available():
            torch.accelerator.synchronize()
        self.start = time.perf_counter()
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if _PROFILE_SYNC and torch.cuda.is_available():
            torch.accelerator.synchronize()
        elapsed_ms = (time.perf_counter() - self.start) * 1000.0
        _record_profile(self.name, elapsed_ms)


def _record_profile(name: str, elapsed_ms: float) -> None:
    global _PROFILE_EVENTS
    if not _PROFILE_ENABLED:
        return
    stats = _PROFILE_STATS.setdefault(name, [0.0, 0.0, 0.0])
    stats[0] += 1.0
    stats[1] += elapsed_ms
    stats[2] = max(stats[2], elapsed_ms)
    _PROFILE_EVENTS += 1
    if _PROFILE_EVENTS % _PROFILE_SUMMARY_EVERY != 0:
        return
    _log_profile_summary()


def _log_profile_summary() -> None:
    for name, (count, total_ms, max_ms) in sorted(_PROFILE_STATS.items()):
        logger.info(
            "[HiggsAudioV3Profile] name=%s count=%d total_ms=%.3f mean_ms=%.3f max_ms=%.3f",
            name,
            int(count),
            total_ms,
            total_ms / max(count, 1.0),
            max_ms,
        )


# Delay pattern constants
BOC_ID = 1024  # beginning of codebook
EOC_ID = 1025  # end of codebook

# Checkpoint prefix mapping
_BACKBONE_PREFIX_MAP = {
    "tied.embedding.text_embedding.": "model.embed_tokens.",
    "body.layers.": "model.layers.",
    "body.norm.": "model.norm.",
    "tied.head.text_head.": "lm_head.",
}
_MODALITY_EMBEDDING_PREFIX = "tied.embedding.modality_embeddings.0.embedding."
_MODALITY_HEAD_PREFIX = "tied.head.modality_heads.0."
_CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."


class HiggsFusedMultiTextEmbedding(nn.Module):
    """Fused multi-codebook embedding: [N*V, D] weight + offset lookup."""

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.register_buffer(
            "_offsets",
            torch.arange(num_codebooks, dtype=torch.long) * vocab_size,
            persistent=False,
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        fused_ids = codes + self._offsets
        return F.embedding(fused_ids, self.weight).sum(dim=-2)


class HiggsFusedMultiTextHead(nn.Module):
    """Fused multi-codebook head: [L, D] -> [L, N, V] via one linear."""

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

    def generate(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden, self.weight)
        return logits.reshape(hidden.shape[0], self.num_codebooks, self.vocab_size)


class HiggsAudioV3TalkerForConditionalGeneration(nn.Module):
    """Stage-0 talker for higgs-audio v3.

    Wraps Qwen3Model backbone + fused multi-codebook modules for TTS generation
    with MusicGen-style delay pattern sampling and audio feedback embedding.
    """

    # Tell the AR runner to call model.sample() instead of the stock sampler.
    prefer_model_sampler: bool = True
    # Tell the runner to call postprocess() to emit per-step audio codes.
    have_multimodal_outputs: bool = True
    has_postprocess: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        hf_config = vllm_config.model_config.hf_config
        if isinstance(hf_config, HiggsAudioV3Config):
            self.config = hf_config
        else:
            self.config = HiggsAudioV3Config(**hf_config.to_dict())

        self.vllm_config = vllm_config
        self.num_codebooks = int(self.config.num_codebooks)
        self.codebook_size = int(self.config.codebook_size)
        hidden_size = int(self.config.audio_hidden_size)
        self.tie_modality = self.config.tie_modality_embeddings

        # Fused multi-codebook modules
        self.multimodal_embedding = HiggsFusedMultiTextEmbedding(self.num_codebooks, self.codebook_size, hidden_size)
        self.modality_head = HiggsFusedMultiTextHead(self.num_codebooks, self.codebook_size, hidden_size)
        if self.tie_modality:
            self.modality_head.weight = self.multimodal_embedding.weight

        # Qwen3 backbone
        self._backbone_config = self.config.text_config
        backbone_vllm_config = copy.copy(vllm_config)
        backbone_model_config = copy.copy(vllm_config.model_config)
        backbone_model_config.hf_config = self._backbone_config
        backbone_vllm_config.model_config = backbone_model_config

        self.model = Qwen3Model(
            vllm_config=backbone_vllm_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )

        if self._backbone_config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                self._backbone_config.vocab_size,
                self._backbone_config.hidden_size,
                prefix=f"{prefix}.lm_head" if prefix else "lm_head",
            )

        self.logits_processor = LogitsProcessor(self._backbone_config.vocab_size)
        self._text_vocab_size = int(self._backbone_config.vocab_size)
        self._dummy_logits: torch.Tensor | None = None
        self._skip_text_logits_total = 0
        self._skip_text_logits_hits = 0

        # Audio continuation token ID — resolved lazily from tokenizer.
        # This is the <|audio|> token that serves as the LM-level continuation
        # marker during audio generation (equivalent to v2's audio_token_id).
        self._audio_continuation_id: int | None = None
        self._eos_token_id: int | None = None
        self._resolved_tokens = False

        # Per-request audio state keyed by batch row index.
        # Reset per slot via _slot_output_len tracking (same pattern as v2).
        self._audio_state: dict[int, dict[str, Any]] = {}
        self._slot_output_len: dict[int, int] = {}
        self._last_logits_hidden: torch.Tensor | None = None
        self._last_step_input_ids: torch.Tensor | None = None
        self._last_step_query_start_loc: torch.Tensor | None = None
        self._last_first_audio_after_start: torch.Tensor | None = None
        self._last_seed_audio_rows: list[int] = []
        self._last_active_audio_rows: list[int] = []
        self._last_audio_codes: torch.Tensor | None = None
        self._last_audio_code_valid: list[bool] = []
        self._postprocess_cursor: int = 0
        self._last_audio_codes_buffer: torch.Tensor | None = None
        self._last_audio_host_staging: torch.Tensor | None = None
        self._last_audio_gpu_staging: torch.Tensor | None = None
        self._last_audio_staging_event: torch.cuda.Event | None = None
        self._audio_staging_event: torch.cuda.Event | None = None
        self._row_index_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._codebook_index_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._boc_frame_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._fast_audio_direct_rows: int = 0
        self._fast_audio_probe_rows: int = 0
        self._postprocess_audio_rows: int = 0
        self._postprocess_audio_active_rows: int = 0
        self._layer_graphs: dict[int, dict[tuple, dict[str, Any]]] = {}
        self._layer_graph_disabled: set[int] = set()
        self._last_sig_ptr_hash: int = 0
        self._last_sig_value: tuple = ()
        self._layer_graph_total = 0
        self._layer_graph_hits = 0
        self._layer_graph_captures = 0
        self._layer_graph_fallbacks = 0
        self._layer_graph_signature_misses = 0
        self._layer_graph_cache_full = 0
        self._layer_graph_nested_capture_fallbacks = 0
        self._layer_graph_capture_failures = 0
        self._layer_graph_invalid_fallbacks = 0
        self._layer_graph_batch_requests: Counter[int] = Counter()
        self._layer_graph_batch_hits: Counter[int] = Counter()
        self._layer_graph_batch_fallbacks: Counter[int] = Counter()
        self._layer_graph_signature_miss_keys: Counter[str] = Counter()
        self._fast_audio_sampler_total = 0
        self._fast_audio_sampler_hits = 0
        self._fast_audio_sampler_fallbacks = 0
        self._fast_audio_sampler_batch_requests: Counter[int] = Counter()
        self._fast_audio_sampler_batch_hits: Counter[int] = Counter()
        self._fast_audio_sampler_fallback_reasons: Counter[str] = Counter()

        # Pre-allocated decode-step audio feedback buffers (CUDA-graph safe).
        # Populated by sample(), read by forward() via torch.where (no dict).
        max_bs = 64  # safe upper bound; will grow if needed
        self._decode_last_codes = torch.zeros(max_bs, self.num_codebooks, dtype=torch.long)
        self._decode_has_codes = torch.zeros(max_bs, dtype=torch.bool)
        self._decode_delay_count = torch.zeros(max_bs, dtype=torch.int32)
        self._decode_eoc_countdown = torch.full((max_bs,), -1, dtype=torch.int32)
        self._decode_generation_done = torch.zeros(max_bs, dtype=torch.bool)
        self._td_step: int = 0
        self._td_eoc_detected: int = 0
        self._td_done_fired: int = 0
        self._td_eos_emitted: int = 0
        self._td_reset_fired: int = 0
        self._decode_active_audio_count: int = 0

        # PrefixCache opt-outs (mirror qwen3_tts pattern):
        # 1. The talker only consumes the last token's hidden state, so the
        #    runner can skip the per-step full hidden-state GPU->CPU merge
        #    that PrefixCache otherwise does.
        # 2. Per-step ``codes.audio`` rows stay GPU-resident; defer the CPU
        #    write of the prefix-cache mm-output copy to request finish so
        #    the per-step bookkeeping does not block batching. Stage 0 can
        #    then set ``enable_prefix_caching: true`` without the regression
        #    observed in qwen3_tts (#3665).
        self.requires_full_prefix_cached_hidden_states = False
        self.deferred_prefix_cache_mm_keys = {"codes.audio"}

    def _resolve_token_ids(self) -> None:
        """Resolve <|audio|> and eos token IDs.

        Prefers config's pre-resolved IDs (from ``resolve_special_tokens()``),
        falls back to loading the HF tokenizer directly.
        """
        if self._resolved_tokens:
            return
        self._resolved_tokens = True

        # Try config first (populated by resolve_special_tokens or from_pretrained)
        cfg_audio = getattr(self.config, "audio_continuation_id", None)
        cfg_eos = getattr(self.config, "eos_token_id", None)
        if cfg_audio is not None:
            self._audio_continuation_id = int(cfg_audio)
        if cfg_eos is not None:
            self._eos_token_id = int(cfg_eos)

        if self._audio_continuation_id is not None:
            logger.info(
                "Resolved v3 token IDs from config: audio_continuation=%s, eos=%s",
                self._audio_continuation_id,
                self._eos_token_id,
            )
            return

        # Fallback: load tokenizer directly
        model_path = getattr(self.vllm_config.model_config, "model", None)
        if model_path is None:
            return
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            vocab = dict(tokenizer.get_added_vocab())
            if "<|audio|>" in vocab:
                self._audio_continuation_id = vocab["<|audio|>"]
            if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                self._eos_token_id = int(tokenizer.eos_token_id)
            logger.info(
                "Resolved v3 token IDs from tokenizer: audio_continuation=%s, eos=%s",
                self._audio_continuation_id,
                self._eos_token_id,
            )
        except Exception as exc:
            logger.warning("Failed to resolve token IDs from tokenizer: %s", exc)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information")

        if inputs_embeds is None:
            # Mask -100 placeholders to 0 before embedding. Use torch.where
            # (no Python data-dependent branch) so this is CUDA-graph safe.
            safe_ids = torch.where(input_ids < 0, torch.zeros_like(input_ids), input_ids)
            if _PROFILE_ENABLED:
                with _ProfileScope("stage0.forward.embed_tokens"):
                    hidden_states = self.model.embed_tokens(safe_ids)
            else:
                hidden_states = self.model.embed_tokens(safe_ids)
        else:
            hidden_states = inputs_embeds

        if input_ids is not None:
            self._last_step_input_ids = input_ids

        # Stash query_start_loc for audio-state row mapping. Some attention
        # backends do not expose it in their metadata, so prefer backend
        # metadata when available and otherwise use the runner-supplied buffer.
        try:
            fallback_qsl = kwargs.get("omni_query_start_loc")
            from vllm.forward_context import get_forward_context

            attn_metadata = get_forward_context().attn_metadata
            if isinstance(attn_metadata, dict) and attn_metadata:
                attn = next(iter(attn_metadata.values()))
            else:
                attn = attn_metadata
            qsl = getattr(attn, "query_start_loc", None)
            if isinstance(qsl, torch.Tensor):
                self._last_step_query_start_loc = qsl.detach().clone()
            elif isinstance(fallback_qsl, torch.Tensor):
                self._last_step_query_start_loc = fallback_qsl.detach().clone()
            else:
                self._last_step_query_start_loc = None
        except Exception:
            self._last_step_query_start_loc = None

        # Prefill-only operations: ref audio substitution and audio feedback
        # require Python dict/list ops that break CUDA graph capture.
        # Detect prefill (sequence length > batch size heuristic) vs decode.
        is_prefill = input_ids is not None and inputs_embeds is None and int(input_ids.numel()) > 1
        if is_prefill and info_dicts:
            # Voice clone: replace -100 placeholder positions with ref audio embeddings
            if _PROFILE_ENABLED:
                with _ProfileScope("stage0.forward.ref_audio_substitution"):
                    hidden_states = self._apply_ref_audio_substitution(hidden_states, input_ids, info_dicts)
            else:
                hidden_states = self._apply_ref_audio_substitution(hidden_states, input_ids, info_dicts)

        # Audio feedback at decode: replace continuation token embeddings
        if input_ids is not None and inputs_embeds is None:
            if _PROFILE_ENABLED:
                with _ProfileScope("stage0.forward.audio_feedback"):
                    hidden_states = self._apply_audio_feedback(hidden_states, input_ids)
            else:
                hidden_states = self._apply_audio_feedback(hidden_states, input_ids)

        if _LAYER_CUDAGRAPH_ENABLED and not _PROFILE_ENABLED:
            graph_out = self._run_qwen3_layers_graph(positions, hidden_states)
            if graph_out is not None:
                return graph_out

        return self._run_qwen3_layers_eager(positions, hidden_states)

    def _run_qwen3_layers_eager(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor | None = None
        if _PROFILE_ENABLED:
            layer_profile = _PROFILE_LAYER_STEPS > 0 and _PROFILE_EVENTS < _PROFILE_LAYER_STEPS * len(self.model.layers)
            with _ProfileScope("stage0.forward.qwen3_layers_total"):
                for layer_idx, layer in enumerate(self.model.layers):
                    if layer_profile:
                        with _ProfileScope(f"stage0.forward.layer_{layer_idx:02d}"):
                            hidden_states, residual = layer(positions, hidden_states, residual)
                    else:
                        hidden_states, residual = layer(positions, hidden_states, residual)
            with _ProfileScope("stage0.forward.norm"):
                norm_out = self.model.norm(hidden_states, residual)
        else:
            for layer in self.model.layers:
                hidden_states, residual = layer(positions, hidden_states, residual)
            norm_out = self.model.norm(hidden_states, residual)
        if isinstance(norm_out, tuple):
            norm_out = norm_out[0]
        return norm_out

    def _run_qwen3_layers_graph(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor | None:
        self._record_layer_graph_attempt(
            "attempt",
            batch=int(hidden_states.shape[0]) if hidden_states.ndim >= 1 else None,
        )
        if self._uses_flashinfer_attention():
            self._record_layer_graph_attempt("invalid")
            return None
        if (
            not torch.cuda.is_available()
            or not hidden_states.is_cuda
            or hidden_states.ndim != 2
            or positions.ndim != 1
            or int(hidden_states.shape[0]) != int(positions.shape[0])
        ):
            self._record_layer_graph_attempt("invalid")
            return None

        batch = int(hidden_states.shape[0])
        if batch <= 0 or batch > 16 or batch in self._layer_graph_disabled:
            self._record_layer_graph_attempt("invalid", batch=batch)
            return None
        if not self._is_decode_only_graph_batch(batch):
            self._record_layer_graph_attempt("invalid", batch=batch)
            return None
        try:
            if torch.cuda.is_current_stream_capturing():
                self._record_layer_graph_attempt("nested_capture", batch=batch)
                return None
        except Exception:
            self._record_layer_graph_attempt("invalid", batch=batch)
            return None

        signature = self._attention_metadata_signature()
        if not signature:
            self._record_layer_graph_attempt("invalid", batch=batch)
            return None

        sig_cache = self._layer_graphs.get(batch)
        if sig_cache is not None:
            entry = sig_cache.get(signature)
            if entry is not None:
                entry["static_hidden"].copy_(hidden_states)
                entry["static_positions"].copy_(positions)
                entry["graph"].replay()
                self._record_layer_graph_attempt("hit", batch=batch)
                return entry["static_output"]
            self._record_layer_graph_attempt("signature_miss", batch=batch)
            first_sig = next(iter(sig_cache), None)
            if first_sig is not None:
                self._record_layer_graph_signature_delta(first_sig, signature)
            if len(sig_cache) >= _LAYER_CUDAGRAPH_CACHE_PER_BATCH:
                self._record_layer_graph_attempt("cache_full", batch=batch)
                return None

        try:
            static_hidden = torch.empty_like(hidden_states)
            static_positions = torch.empty_like(positions)
            static_hidden.copy_(hidden_states)
            static_positions.copy_(positions)
            ctx = get_forward_context()
            patched_ctx = self._nullify_volatile_metadata(ctx)
            with override_forward_context(patched_ctx):
                self._warmup_qwen3_layers_graph_capture(static_positions, static_hidden)
                graph = torch.cuda.CUDAGraph()
                with torch.inference_mode(), torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                    h = static_hidden
                    residual: torch.Tensor | None = None
                    for layer in self.model.layers:
                        h, residual = layer(static_positions, h, residual)
                    static_output = self.model.norm(h, residual)
                    if isinstance(static_output, tuple):
                        static_output = static_output[0]
            sig_cache = self._layer_graphs.setdefault(batch, {})
            sig_cache[signature] = {
                "graph": graph,
                "static_hidden": static_hidden,
                "static_positions": static_positions,
                "static_output": static_output,
            }
            self._record_layer_graph_attempt("capture", batch=batch)
            logger.info(
                "HiggsAudioV3Talker: captured layer CUDA graph for decode batch=%d cache_entries=%d/%d",
                batch,
                len(sig_cache),
                _LAYER_CUDAGRAPH_CACHE_PER_BATCH,
            )
            return static_output
        except Exception as exc:
            self._layer_graph_disabled.add(batch)
            self._record_layer_graph_attempt("capture_failure", batch=batch)
            logger.warning("HiggsAudioV3Talker: layer CUDA graph disabled for batch=%d: %s", batch, exc)
            return None

    def _uses_flashinfer_attention(self) -> bool:
        layer = next(iter(getattr(self.model, "layers", [])), None)
        impl = getattr(getattr(getattr(layer, "self_attn", None), "attn", None), "impl", None)
        if impl is None:
            return False
        impl_name = f"{impl.__class__.__module__}.{impl.__class__.__qualname__}".lower()
        return "flashinfer" in impl_name

    def _is_decode_only_graph_batch(self, batch: int) -> bool:
        q_start = self._last_step_query_start_loc
        return isinstance(q_start, torch.Tensor) and int(q_start.numel()) == batch + 1

    @staticmethod
    def _nullify_volatile_metadata(ctx: Any) -> Any:
        """Remove FA3 scheduler_metadata from graph capture context.

        The scheduler metadata tensor is reallocated as request scheduling
        changes. FA3 can run without it by using default scheduling, while the
        persistent tensors needed by attention remain updated in-place by the
        model runner.
        """
        if not isinstance(getattr(ctx, "attn_metadata", None), dict):
            return ctx
        ctx = copy.copy(ctx)
        patched: dict[str, Any] = {}
        for layer_name, meta in ctx.attn_metadata.items():
            if getattr(meta, "scheduler_metadata", None) is not None:
                meta = copy.copy(meta)
                meta.scheduler_metadata = None
            patched[layer_name] = meta
        ctx.attn_metadata = patched
        return ctx

    def _warmup_qwen3_layers_graph_capture(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> None:
        if _LAYER_CUDAGRAPH_WARMUP_RUNS <= 0:
            return
        stream = torch.cuda.Stream(device=hidden_states.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(stream):
            for _ in range(_LAYER_CUDAGRAPH_WARMUP_RUNS):
                h = hidden_states
                residual: torch.Tensor | None = None
                for layer in self.model.layers:
                    h, residual = layer(positions, h, residual)
                norm_out = self.model.norm(h, residual)
                if isinstance(norm_out, tuple):
                    norm_out = norm_out[0]
        torch.cuda.current_stream().wait_stream(stream)

    def _record_layer_graph_attempt(self, kind: str, batch: int | None = None) -> None:
        if not _LAYER_CUDAGRAPH_STATS_ENABLED:
            return
        if kind == "attempt":
            self._layer_graph_total += 1
            if batch is not None:
                self._layer_graph_batch_requests[int(batch)] += 1
        elif kind == "hit":
            self._layer_graph_hits += 1
            if batch is not None:
                self._layer_graph_batch_hits[int(batch)] += 1
        elif kind == "capture":
            self._layer_graph_captures += 1
        elif kind == "signature_miss":
            self._layer_graph_signature_misses += 1
        elif kind == "cache_full":
            self._layer_graph_cache_full += 1
            self._layer_graph_fallbacks += 1
            if batch is not None:
                self._layer_graph_batch_fallbacks[int(batch)] += 1
        elif kind == "nested_capture":
            self._layer_graph_nested_capture_fallbacks += 1
            self._layer_graph_fallbacks += 1
            if batch is not None:
                self._layer_graph_batch_fallbacks[int(batch)] += 1
        elif kind == "capture_failure":
            self._layer_graph_capture_failures += 1
            self._layer_graph_fallbacks += 1
            if batch is not None:
                self._layer_graph_batch_fallbacks[int(batch)] += 1
        elif kind == "invalid":
            self._layer_graph_invalid_fallbacks += 1
            self._layer_graph_fallbacks += 1
            if batch is not None:
                self._layer_graph_batch_fallbacks[int(batch)] += 1

        if (
            kind == "attempt"
            and _LAYER_CUDAGRAPH_STATS_EVERY > 0
            and self._layer_graph_total % _LAYER_CUDAGRAPH_STATS_EVERY == 0
        ):
            self._log_layer_graph_stats()

    def _record_layer_graph_signature_delta(self, old_signature: Any, new_signature: Any) -> None:
        if not _LAYER_CUDAGRAPH_STATS_ENABLED:
            return
        if not isinstance(old_signature, tuple) or not isinstance(new_signature, tuple):
            self._layer_graph_signature_miss_keys["unavailable"] += 1
            return

        old_by_name = {entry[0]: entry for entry in old_signature if isinstance(entry, tuple) and entry}
        new_by_name = {entry[0]: entry for entry in new_signature if isinstance(entry, tuple) and entry}
        all_names = set(old_by_name) | set(new_by_name)
        changed = 0
        for name in sorted(all_names):
            if old_by_name.get(name) == new_by_name.get(name):
                continue
            self._layer_graph_signature_miss_keys[str(name)] += 1
            changed += 1
            if changed >= 16:
                break
        if changed == 0:
            self._layer_graph_signature_miss_keys["unknown"] += 1

    def _log_layer_graph_stats(self) -> None:
        if not _LAYER_CUDAGRAPH_STATS_ENABLED or self._layer_graph_total <= 0:
            return
        hit_rate = 100.0 * self._layer_graph_hits / max(1, self._layer_graph_total)
        cache_entries = {batch: len(entries) for batch, entries in sorted(self._layer_graphs.items())}
        logger.info(
            "HiggsAudioV3Talker layer CUDA Graph stats: total=%d hits=%d captures=%d "
            "fallbacks=%d signature_misses=%d cache_full=%d nested_capture=%d "
            "capture_failures=%d invalid=%d hit_rate=%.2f%% cache_entries=%s "
            "top_requests=%s top_hits=%s top_fallbacks=%s top_signature_miss_keys=%s",
            self._layer_graph_total,
            self._layer_graph_hits,
            self._layer_graph_captures,
            self._layer_graph_fallbacks,
            self._layer_graph_signature_misses,
            self._layer_graph_cache_full,
            self._layer_graph_nested_capture_fallbacks,
            self._layer_graph_capture_failures,
            self._layer_graph_invalid_fallbacks,
            hit_rate,
            cache_entries,
            self._layer_graph_batch_requests.most_common(8),
            self._layer_graph_batch_hits.most_common(8),
            self._layer_graph_batch_fallbacks.most_common(8),
            self._layer_graph_signature_miss_keys.most_common(16),
        )

    def _attention_metadata_signature(self) -> tuple:
        try:
            metadata = get_forward_context().attn_metadata
        except Exception:
            return ()

        seen: set[int] = set()
        ptrs: list[int] = []
        values: list[tuple] = []

        def visit(name: str, obj: Any, depth: int = 0) -> None:
            if depth > 3:
                return
            if name.endswith(".scheduler_metadata"):
                return
            if isinstance(obj, (bool, int, float, str)) or obj is None:
                if _LAYER_CUDAGRAPH_STRICT_SIGNATURE:
                    values.append((name, hash(obj)))
                return
            if isinstance(obj, torch.Tensor):
                dp = int(obj.data_ptr())
                ptrs.append(dp)
                values.append((name, dp, tuple(int(x) for x in obj.shape)))
                return
            obj_id = id(obj)
            if obj_id in seen:
                return
            seen.add(obj_id)
            if isinstance(obj, dict):
                for key, value in obj.items():
                    visit(f"{name}.{key}", value, depth + 1)
                return
            if isinstance(obj, (list, tuple)):
                for idx, value in enumerate(obj):
                    visit(f"{name}.{idx}", value, depth + 1)
                return
            obj_dict = getattr(obj, "__dict__", None)
            if isinstance(obj_dict, dict):
                for key, value in obj_dict.items():
                    if key.startswith("_"):
                        continue
                    visit(f"{name}.{key}", value, depth + 1)

        visit("attn", metadata)

        ptr_hash = hash(tuple(ptrs))
        if ptr_hash == self._last_sig_ptr_hash and self._last_sig_value:
            return self._last_sig_value

        values.sort()
        sig = tuple(values)
        self._last_sig_ptr_hash = ptr_hash
        self._last_sig_value = sig
        return sig

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any = None) -> torch.Tensor:
        self._last_logits_hidden = hidden_states

        if _SKIP_TEXT_LOGITS_ENABLED and self._can_skip_text_logits(hidden_states, sampling_metadata):
            self._skip_text_logits_hits += 1
            return self._get_dummy_logits(hidden_states)

        if _PROFILE_ENABLED:
            with _ProfileScope("stage0.compute_logits.text"):
                return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

    def _can_skip_text_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any) -> bool:
        self._skip_text_logits_total += 1
        if self._audio_continuation_id is None:
            return False
        num_rows = int(hidden_states.shape[0])
        if num_rows <= 0 or self._fast_audio_direct_rows != num_rows or not self._is_single_token_decode_step(num_rows):
            return False
        if getattr(sampling_metadata, "max_num_logprobs", None) is not None:
            return False
        self._ensure_decode_state_capacity(num_rows, hidden_states.device)
        return True

    def _get_dummy_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_rows = int(hidden_states.shape[0])
        needs_init = (
            self._dummy_logits is None
            or self._dummy_logits.device != hidden_states.device
            or self._dummy_logits.shape[0] < num_rows
        )
        if needs_init:
            buf_rows = max(num_rows, 16)
            self._dummy_logits = torch.full(
                (buf_rows, self._text_vocab_size),
                float("-inf"),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            if self._audio_continuation_id is not None:
                self._dummy_logits[:, self._audio_continuation_id] = 0.0
        else:
            # audio_mode_bias may have modified rows in-place (termination path
            # writes eos mask); reset the slice we're about to return.
            self._dummy_logits[:num_rows].fill_(float("-inf"))
            if self._audio_continuation_id is not None:
                self._dummy_logits[:num_rows, self._audio_continuation_id] = 0.0
        return self._dummy_logits[:num_rows]

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        safe_ids = torch.where(input_ids < 0, torch.zeros_like(input_ids), input_ids)
        text_embed = self.model.embed_tokens(safe_ids)
        return self._apply_audio_feedback(text_embed, input_ids)

    # ------------------------------------------------------------------ ref audio substitution
    def _apply_ref_audio_substitution(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        info_dicts: list[dict[str, Any]] | None,
    ) -> torch.Tensor:
        """Replace -100 placeholder positions with fused multi-codebook embeddings
        of the delay-pattern-encoded reference audio codes.

        Called at prefill to inject voice clone reference. ``info_dicts`` is a
        list of per-request dicts from ``model_intermediate_buffer``, each
        containing ``audio_input_ids`` ([T, N] delayed codes) and
        ``audio_input_ids_mask`` ([T] bool mask).
        """
        if not info_dicts:
            return hidden_states

        PLACEHOLDER = -100
        flat_ids = input_ids.reshape(-1)
        placeholder_mask = flat_ids == PLACEHOLDER
        if not placeholder_mask.any():
            return hidden_states

        # Use query_start_loc to map placeholders to per-request spans
        q_start = self._last_step_query_start_loc
        if not isinstance(q_start, torch.Tensor) or q_start.numel() < 2:
            # Fallback: single-request batch
            q_start_list = [0, int(flat_ids.numel())]
        else:
            q_start_list = q_start.detach().to("cpu").tolist()

        new_hidden: torch.Tensor | None = None
        num_requests = min(len(info_dicts), len(q_start_list) - 1)

        for i in range(num_requests):
            info = info_dicts[i]
            if not isinstance(info, dict):
                continue

            codes = info.get("audio_input_ids")
            mask = info.get("audio_input_ids_mask")

            # Handle msgspec serialization (may be list-wrapped)
            if isinstance(codes, list):
                codes = codes[0] if codes else None
            if isinstance(mask, list):
                mask = mask[0] if mask else None
            if not isinstance(codes, torch.Tensor):
                continue

            # codes shape: [T, num_codebooks] delayed reference codes
            if codes.ndim == 3:
                codes = codes[0]
            if codes.ndim != 2:
                continue

            if isinstance(mask, torch.Tensor):
                if mask.ndim == 2:
                    mask = mask[0]
                codes = codes[mask.to(dtype=torch.bool)]

            if codes.numel() == 0:
                continue

            # Find placeholder positions in this request's span
            s = int(q_start_list[i])
            e = int(q_start_list[i + 1])
            if e - s <= 1:
                continue  # Decode step, skip

            span_mask = placeholder_mask[s:e]
            placeholders = span_mask.nonzero(as_tuple=True)[0]
            n_codes = int(codes.shape[0])

            if int(placeholders.numel()) < n_codes:
                continue  # Mismatch

            # Embed delayed codes via fused multi-codebook embedding
            target = placeholders[:n_codes] + s
            codes_device = codes.to(device=hidden_states.device, dtype=torch.long)
            embeds = self.multimodal_embedding(codes_device)  # [n_codes, hidden]

            if new_hidden is None:
                new_hidden = hidden_states.clone()
            flat_hidden = new_hidden.reshape(-1, new_hidden.shape[-1])
            flat_hidden[target] = embeds.to(new_hidden.dtype)

        return new_hidden if new_hidden is not None else hidden_states

    # ------------------------------------------------------------------ audio feedback
    def _apply_audio_feedback(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Replace decode-step embeddings with audio feedback from pre-allocated buffers.

        CUDA-graph safe: reads from pre-allocated _decode_last_codes and
        _decode_has_codes tensors using torch.where (no Python dict lookup).
        The buffers are populated by sample() after each step.

        For decode steps (1 token per request): position i maps to row i.
        For prefill: audio feedback is not needed (ref audio substitution
        handles the prefill path separately).
        """
        if self._decode_active_audio_count == 0 or not self._is_single_token_decode_step(int(hidden_states.shape[0])):
            return hidden_states

        bs = hidden_states.shape[0]

        # Ensure buffers are on the right device and large enough.
        self._ensure_decode_state_capacity(int(bs), hidden_states.device)

        # Compute audio embeddings from last_codes for ALL rows (graph-safe)
        codes_slice = self._decode_last_codes[:bs]  # [bs, N]
        has_codes = self._decode_has_codes[:bs].unsqueeze(-1)  # [bs, 1]
        audio_embeds = self.multimodal_embedding(codes_slice)  # [bs, D]
        audio_embeds = audio_embeds.to(dtype=hidden_states.dtype)

        # Select: where has_codes, use audio embed; else keep text embed
        return torch.where(has_codes, audio_embeds, hidden_states)

    def _is_single_token_decode_step(self, num_rows: int) -> bool:
        ids = self._last_step_input_ids
        return isinstance(ids, torch.Tensor) and int(ids.numel()) == int(num_rows)

    def _prefill_row_mask(self, num_rows: int, device: torch.device) -> torch.Tensor:
        q_start = self._last_step_query_start_loc
        if isinstance(q_start, torch.Tensor) and int(q_start.numel()) == num_rows + 1:
            q = q_start.to(device=device, dtype=torch.long)
            return (q[1:] - q[:-1]) > 1
        return torch.ones(num_rows, dtype=torch.bool, device=device)

    def _reset_decode_state_rows(self, mask: torch.Tensor, num_rows: int, device: torch.device) -> None:
        self._ensure_decode_state_capacity(num_rows, device)
        row_mask = mask.to(device=device, dtype=torch.bool)
        self._decode_has_codes[:num_rows] = torch.where(
            row_mask,
            torch.zeros_like(self._decode_has_codes[:num_rows]),
            self._decode_has_codes[:num_rows],
        )
        self._decode_generation_done[:num_rows] = torch.where(
            row_mask,
            torch.zeros_like(self._decode_generation_done[:num_rows]),
            self._decode_generation_done[:num_rows],
        )
        self._decode_delay_count[:num_rows] = torch.where(
            row_mask,
            torch.zeros_like(self._decode_delay_count[:num_rows]),
            self._decode_delay_count[:num_rows],
        )
        self._decode_eoc_countdown[:num_rows] = torch.where(
            row_mask,
            torch.full_like(self._decode_eoc_countdown[:num_rows], -1),
            self._decode_eoc_countdown[:num_rows],
        )

    def _ensure_decode_state_capacity(self, num_rows: int, device: torch.device | None = None) -> None:
        """Keep GPU-resident decode state tensors aligned and large enough."""
        if device is not None and self._decode_last_codes.device != device:
            self._decode_last_codes = self._decode_last_codes.to(device)
            self._decode_has_codes = self._decode_has_codes.to(device)
            self._decode_delay_count = self._decode_delay_count.to(device)
            self._decode_eoc_countdown = self._decode_eoc_countdown.to(device)
            self._decode_generation_done = self._decode_generation_done.to(device)

        cur_rows = int(self._decode_last_codes.shape[0])
        if num_rows <= cur_rows:
            return

        new_size = max(num_rows, cur_rows * 2)
        state_device = self._decode_last_codes.device

        new_last_codes = torch.zeros(new_size, self.num_codebooks, dtype=torch.long, device=state_device)
        new_last_codes[:cur_rows].copy_(self._decode_last_codes)
        self._decode_last_codes = new_last_codes

        new_has_codes = torch.zeros(new_size, dtype=torch.bool, device=state_device)
        new_has_codes[:cur_rows].copy_(self._decode_has_codes)
        self._decode_has_codes = new_has_codes

        new_delay_count = torch.zeros(new_size, dtype=torch.int32, device=state_device)
        new_delay_count[:cur_rows].copy_(self._decode_delay_count)
        self._decode_delay_count = new_delay_count

        new_eoc_countdown = torch.full((new_size,), -1, dtype=torch.int32, device=state_device)
        new_eoc_countdown[:cur_rows].copy_(self._decode_eoc_countdown)
        self._decode_eoc_countdown = new_eoc_countdown

        new_generation_done = torch.zeros(new_size, dtype=torch.bool, device=state_device)
        new_generation_done[:cur_rows].copy_(self._decode_generation_done)
        self._decode_generation_done = new_generation_done

    def _audio_seed_mask_from_step_input(self, num_rows: int, device: torch.device) -> torch.Tensor | None:
        """Return rows whose previous token is <|audio|> using current step input_ids.

        This is the hot-path replacement for scanning sampling_metadata
        output_token_ids. In decode, vLLM feeds the previous sampled token as
        the current single-token input for each row.
        """
        audio_id = self._audio_continuation_id
        ids = self._last_step_input_ids
        if audio_id is None or not isinstance(ids, torch.Tensor) or int(ids.numel()) <= 0:
            return None

        flat_ids = ids.reshape(-1).to(device=device)
        if int(flat_ids.numel()) == num_rows:
            tail_ids = flat_ids
        else:
            q_start = self._last_step_query_start_loc
            if not isinstance(q_start, torch.Tensor) or int(q_start.numel()) != num_rows + 1:
                return None
            tail_idx = q_start.to(device=device, dtype=torch.long)[1:] - 1
            valid_tail = tail_idx >= 0
            tail_ids = flat_ids.index_select(0, tail_idx.clamp_min(0))
            return (tail_ids == int(audio_id)) & valid_tail
        return tail_ids == int(audio_id)

    def _fast_audio_sampler_gpu_fallback_reason(
        self,
        *,
        logits: torch.Tensor,
        sampling_metadata: Any,
        num_rows: int,
    ) -> str | None:
        if not _FAST_AUDIO_SAMPLER_ENABLED:
            return "disabled"
        self._record_fast_audio_sampler_attempt("attempt", batch=num_rows)
        if logits is None or logits.ndim != 2 or int(logits.shape[0]) != num_rows:
            return "invalid_logits"
        if num_rows <= 0:
            return "empty_batch"
        if getattr(sampling_metadata, "max_num_logprobs", None) is not None:
            return "logprobs"
        if getattr(sampling_metadata, "allowed_token_ids_mask", None) is not None:
            return "allowed_token_ids"
        if bool(getattr(sampling_metadata, "bad_words_token_ids", None)):
            return "bad_words"
        return None

    def _apply_audio_mode_bias_batched(
        self,
        logits: torch.Tensor,
        audio_mask: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> None:
        """Force audio/EOS rows with GPU masks, avoiding metadata scans."""
        audio_id = self._audio_continuation_id
        if audio_id is None or logits is None or logits.ndim != 2:
            return

        vocab = int(logits.shape[-1])
        if audio_id < 0 or audio_id >= vocab:
            return

        target = torch.full(
            (int(logits.shape[0]),),
            int(audio_id),
            dtype=torch.long,
            device=logits.device,
        )
        if self._eos_token_id is not None and 0 <= int(self._eos_token_id) < vocab:
            eos_target = torch.full_like(target, int(self._eos_token_id))
            target = torch.where(done_mask.to(device=logits.device), eos_target, target)

        target_values = logits.gather(1, target.unsqueeze(1))
        forced = torch.full_like(logits, float("-inf"))
        forced.scatter_(1, target.unsqueeze(1), target_values)
        logits.copy_(torch.where(audio_mask.to(device=logits.device).unsqueeze(1), forced, logits))

    # ------------------------------------------------------------------ sampling
    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> Any:
        """Model-owned sampler with delay-pattern audio dispatch.

        Mirrors v2's pattern: bias LM logits to force audio continuation,
        sample multi-codebook codes via the fused head, apply delay pattern,
        and accumulate per-request state.
        """
        self._resolve_token_ids()

        audio_id = self._audio_continuation_id

        def run_stock_sampler() -> Any:
            sampler = getattr(self, "_stock_sampler", None)
            if sampler is None:
                from vllm.v1.sample.sampler import Sampler

                sampler = Sampler()
                self._stock_sampler = sampler
            if _PROFILE_ENABLED:
                with _ProfileScope("stage0.sample.stock_sampler"):
                    return sampler(logits=logits, sampling_metadata=sampling_metadata)
            return sampler(logits=logits, sampling_metadata=sampling_metadata)

        hidden = self._last_logits_hidden
        self._last_logits_hidden = None
        if hidden is None or audio_id is None:
            sampler_output = run_stock_sampler()
            self._last_audio_codes = None
            self._last_audio_code_valid = []
            return sampler_output

        num_rows = int(hidden.shape[0])
        self._ensure_decode_state_capacity(num_rows, hidden.device)
        self._td_step += 1
        decode_only = self._is_single_token_decode_step(num_rows)
        if not decode_only:
            self._fast_audio_direct_rows = 0
            self._fast_audio_probe_rows = 0
            pfmask = self._prefill_row_mask(num_rows, hidden.device)
            if _TERMINATION_DEBUG:
                n_pf = int(pfmask.sum().item())
                self._td_reset_fired += 1
                if n_pf == num_rows:
                    qsl_numel = (
                        str(int(self._last_step_query_start_loc.numel()))
                        if isinstance(self._last_step_query_start_loc, torch.Tensor)
                        else "None"
                    )
                    logger.info(
                        "TD[step=%d] RESET_ALL_ROWS num_rows=%d qsl_numel=%s", self._td_step, num_rows, qsl_numel
                    )
                elif n_pf > 0:
                    ramp_before = self._decode_eoc_countdown[:num_rows] >= 0
                    done_before = self._decode_generation_done[:num_rows].to(torch.bool)
                    conflict = pfmask.to(hidden.device) & (
                        ramp_before.to(hidden.device) | done_before.to(hidden.device)
                    )
                    if int(conflict.sum().item()) > 0:
                        logger.warning(
                            "TD[step=%d] RESET_CONFLICTS prefill_rows=%d conflict_ramp_or_done=%d",
                            self._td_step,
                            n_pf,
                            int(conflict.sum().item()),
                        )
            self._reset_decode_state_rows(pfmask, num_rows, hidden.device)
        prev_audio_mask = self._audio_seed_mask_from_step_input(num_rows, hidden.device)
        if prev_audio_mask is None:
            prev_audio_mask = torch.zeros(num_rows, dtype=torch.bool, device=hidden.device)
        active_mask = self._decode_has_codes[:num_rows].to(device=hidden.device, dtype=torch.bool)
        done_mask = self._decode_generation_done[:num_rows].to(device=hidden.device, dtype=torch.bool)
        if _TERMINATION_DEBUG and int(done_mask.sum().item()) > 0:
            self._td_eos_emitted += int(done_mask.sum().item())
            done_rows = torch.nonzero(done_mask, as_tuple=False).reshape(-1)
            for ri in done_rows[:4].tolist():
                logger.info("TD[step=%d] EOS_EMIT row=%d", self._td_step, ri)
        self._last_seed_audio_rows = []
        self._last_active_audio_rows = []
        self._last_first_audio_after_start = None

        fast_fallback_reason = self._fast_audio_sampler_gpu_fallback_reason(
            logits=logits,
            sampling_metadata=sampling_metadata,
            num_rows=num_rows,
        )
        assume_full_audio_decode = fast_fallback_reason is None and decode_only and _FAST_AUDIO_ASSUME_FULL_DECODE
        if assume_full_audio_decode:
            # Higgs v3 TTS decode enters an all-audio continuation phase after
            # prefill. Avoid a GPU->CPU sync just to prove that every row is
            # already at <|audio|>; rows without state are seeded on GPU.
            seed_mask = (~active_mask) & ~done_mask
            audio_mask = torch.ones(num_rows, dtype=torch.bool, device=hidden.device)
            code_row_mask = ~done_mask
        else:
            seed_mask = prev_audio_mask & ~active_mask & ~done_mask
            audio_mask = prev_audio_mask | active_mask | done_mask
            code_row_mask = (seed_mask | active_mask) & ~done_mask
        gpu_stock_sampler_reasons = {"logprobs", "allowed_token_ids", "bad_words"}
        use_gpu_audio_mode = fast_fallback_reason is None or fast_fallback_reason in gpu_stock_sampler_reasons
        if fast_fallback_reason == "disabled" and not _FAST_AUDIO_CPU_METADATA_FALLBACK:
            use_gpu_audio_mode = True

        if fast_fallback_reason is None:
            direct_audio_batch = decode_only and (self._fast_audio_direct_rows == num_rows or assume_full_audio_decode)

            if direct_audio_batch:
                self._fast_audio_direct_rows = num_rows
                audio_tokens = torch.full((num_rows,), int(audio_id), dtype=torch.int32, device=hidden.device)
                if self._eos_token_id is not None:
                    eos_tokens = torch.full_like(audio_tokens, int(self._eos_token_id))
                    audio_tokens = torch.where(done_mask, eos_tokens, audio_tokens)
                sampled = audio_tokens.unsqueeze(-1)
                sampler_output = SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None)
                self._record_fast_audio_sampler_attempt("hit", batch=num_rows)
            else:
                if _PROFILE_ENABLED:
                    with _ProfileScope("stage0.sample.audio_mode_bias_gpu"):
                        self._apply_audio_mode_bias_batched(logits, audio_mask, done_mask)
                else:
                    self._apply_audio_mode_bias_batched(logits, audio_mask, done_mask)
                sampler_output = run_stock_sampler()
                sampled = getattr(sampler_output, "sampled_token_ids", None)
                if sampled is None:
                    self._last_audio_codes = None
                    self._last_audio_code_valid = []
                    self._last_audio_host_staging = None
                    return sampler_output
        elif use_gpu_audio_mode:
            self._fast_audio_direct_rows = 0
            self._fast_audio_probe_rows = 0
            if _PROFILE_ENABLED:
                with _ProfileScope("stage0.sample.audio_mode_bias_gpu_stock"):
                    self._apply_audio_mode_bias_batched(logits, audio_mask, done_mask)
            else:
                self._apply_audio_mode_bias_batched(logits, audio_mask, done_mask)
            self._record_fast_audio_sampler_attempt(
                "fallback",
                batch=num_rows,
                reason=f"gpu_stock_{fast_fallback_reason}",
            )
            sampler_output = run_stock_sampler()
            sampled = getattr(sampler_output, "sampled_token_ids", None)
            if sampled is None:
                self._last_audio_codes = None
                self._last_audio_code_valid = []
                self._last_audio_host_staging = None
                return sampler_output
        else:
            self._fast_audio_direct_rows = 0
            self._fast_audio_probe_rows = 0
            if _FAST_AUDIO_CPU_METADATA_FALLBACK:
                if _PROFILE_ENABLED:
                    with _ProfileScope("stage0.sample.audio_mode_bias_fallback"):
                        self._apply_audio_mode_bias(logits, sampling_metadata, mutate=True)
                else:
                    self._apply_audio_mode_bias(logits, sampling_metadata, mutate=True)
            else:
                self._last_seed_audio_rows = []
                self._last_active_audio_rows = []
            init_audio_rows = [r for r in self._last_seed_audio_rows if 0 <= r < num_rows]
            active_audio_rows = [r for r in self._last_active_audio_rows if 0 <= r < num_rows]
            audio_row_set = set(init_audio_rows)
            audio_row_set.update(active_audio_rows)
            audio_row_indices = [r for r in range(num_rows) if r in audio_row_set]
            self._record_fast_audio_sampler_attempt("fallback", batch=num_rows, reason=fast_fallback_reason)
            sampler_output = run_stock_sampler()
            sampled = getattr(sampler_output, "sampled_token_ids", None)
            if sampled is None:
                self._last_audio_codes = None
                self._last_audio_code_valid = []
                return sampler_output
            audio_row_tensor = torch.tensor(audio_row_indices, dtype=torch.long, device=hidden.device)

        sampled_flat = sampled.reshape(-1)
        if int(sampled_flat.numel()) != num_rows:
            self._last_audio_codes = None
            self._last_audio_code_valid = []
            return sampler_output

        if use_gpu_audio_mode:
            audio_row_tensor = self._get_row_indices(num_rows, hidden.device)
            seed_mask_1d = seed_mask.to(device=hidden.device, dtype=torch.bool)
            done_mask_1d = done_mask.to(device=hidden.device, dtype=torch.bool)
            self._decode_generation_done[:num_rows] = torch.where(
                done_mask_1d,
                torch.zeros_like(self._decode_generation_done[:num_rows]),
                self._decode_generation_done[:num_rows],
            )
            self._decode_has_codes[:num_rows] = torch.where(
                done_mask_1d,
                torch.zeros_like(self._decode_has_codes[:num_rows]),
                self._decode_has_codes[:num_rows],
            )
            self._decode_delay_count[:num_rows] = torch.where(
                done_mask_1d,
                torch.zeros_like(self._decode_delay_count[:num_rows]),
                self._decode_delay_count[:num_rows],
            )
            self._decode_eoc_countdown[:num_rows] = torch.where(
                done_mask_1d,
                torch.full_like(self._decode_eoc_countdown[:num_rows], -1),
                self._decode_eoc_countdown[:num_rows],
            )

            boc_frames = self._get_boc_frames(num_rows, hidden.device)
            seed_mask_2d = seed_mask_1d.unsqueeze(1)
            self._decode_last_codes[:num_rows] = torch.where(
                seed_mask_2d,
                boc_frames,
                self._decode_last_codes[:num_rows],
            )
            self._decode_has_codes[:num_rows] = torch.where(
                seed_mask_1d,
                torch.ones_like(self._decode_has_codes[:num_rows]),
                self._decode_has_codes[:num_rows],
            )
            self._decode_delay_count[:num_rows] = torch.where(
                seed_mask_1d,
                torch.zeros_like(self._decode_delay_count[:num_rows]),
                self._decode_delay_count[:num_rows],
            )
            self._decode_eoc_countdown[:num_rows] = torch.where(
                seed_mask_1d,
                torch.full_like(self._decode_eoc_countdown[:num_rows], -1),
                self._decode_eoc_countdown[:num_rows],
            )
            self._decode_generation_done[:num_rows] = torch.where(
                seed_mask_1d,
                torch.zeros_like(self._decode_generation_done[:num_rows]),
                self._decode_generation_done[:num_rows],
            )
            self._decode_active_audio_count = max(self._decode_active_audio_count, num_rows)

        if fast_fallback_reason is not None and audio_row_tensor.numel() == 0:
            self._last_audio_codes = None
            self._last_audio_code_valid = []
            self._last_audio_host_staging = None
            self._last_audio_staging_event = None
            return sampler_output

        # Per-codebook logits at audio positions
        if _PROFILE_ENABLED:
            with _ProfileScope("stage0.sample.audio_codebook_logits"):
                cb_logits = self._audio_codebook_logits_from_rows(hidden, audio_row_tensor, all_rows=use_gpu_audio_mode)
        else:
            cb_logits = self._audio_codebook_logits_from_rows(hidden, audio_row_tensor, all_rows=use_gpu_audio_mode)

        # Apply delay pattern masking BEFORE sampling
        if _PROFILE_ENABLED:
            with _ProfileScope("stage0.sample.delay_pattern_masking"):
                self._apply_delay_pattern_masking_batched(cb_logits, audio_row_tensor, all_rows=use_gpu_audio_mode)
        else:
            self._apply_delay_pattern_masking_batched(cb_logits, audio_row_tensor, all_rows=use_gpu_audio_mode)

        # Sample per-codebook
        cb_logits_2d = cb_logits.reshape(-1, cb_logits.shape[-1])
        if _PROFILE_ENABLED:
            with _ProfileScope("stage0.sample.audio_multinomial"):
                codes_2d = self._sample_audio_codes(cb_logits_2d)
        else:
            codes_2d = self._sample_audio_codes(cb_logits_2d)
        codes_flat = codes_2d.view(cb_logits.shape[0], cb_logits.shape[1]).to(torch.long)

        if _PROFILE_ENABLED:
            with _ProfileScope("stage0.sample.delay_state_update"):
                self._update_delay_state_batched(
                    codes_flat,
                    audio_row_tensor,
                    num_rows,
                    hidden.device,
                    code_row_mask=code_row_mask,
                    all_rows=use_gpu_audio_mode,
                )
        else:
            self._update_delay_state_batched(
                codes_flat,
                audio_row_tensor,
                num_rows,
                hidden.device,
                code_row_mask=code_row_mask,
                all_rows=use_gpu_audio_mode,
            )
        return sampler_output

    def _get_audio_codes_buffer(self, num_rows: int, device: torch.device) -> torch.Tensor:
        buf = self._last_audio_codes_buffer
        if buf is None or buf.device != device or int(buf.shape[0]) < num_rows:
            rows = max(num_rows, 16)
            buf = torch.empty((rows, self.num_codebooks), dtype=torch.long, device=device)
            self._last_audio_codes_buffer = buf
        return buf[:num_rows]

    def _get_audio_gpu_staging_buffer(self, num_rows: int, device: torch.device) -> torch.Tensor:
        width = self.num_codebooks + 2
        buf = self._last_audio_gpu_staging
        if buf is None or buf.device != device or int(buf.shape[0]) < num_rows:
            rows = max(num_rows, 16)
            buf = torch.empty((rows, width), dtype=torch.int32, device=device)
            self._last_audio_gpu_staging = buf
        return buf[:num_rows]

    def _get_audio_host_staging_buffer(self, num_rows: int) -> torch.Tensor:
        width = self.num_codebooks + 2
        buf = self._last_audio_host_staging
        if buf is None or buf.device.type != "cpu" or int(buf.shape[0]) < num_rows:
            rows = max(num_rows, 16)
            pin_memory = torch.cuda.is_available()
            buf = torch.empty((rows, width), dtype=torch.int32, device="cpu", pin_memory=pin_memory)
            self._last_audio_host_staging = buf
        return buf[:num_rows]

    @staticmethod
    def _device_cache_key(device: torch.device) -> tuple[str, int]:
        return (device.type, -1 if device.index is None else int(device.index))

    def _get_row_indices(self, num_rows: int, device: torch.device) -> torch.Tensor:
        key = self._device_cache_key(device)
        buf = self._row_index_cache.get(key)
        if buf is None or buf.device != device or int(buf.shape[0]) < num_rows:
            rows = max(num_rows, 16)
            buf = torch.arange(rows, dtype=torch.long, device=device)
            self._row_index_cache[key] = buf
        return buf[:num_rows]

    def _get_codebook_indices(self, device: torch.device) -> torch.Tensor:
        key = self._device_cache_key(device)
        buf = self._codebook_index_cache.get(key)
        if buf is None or buf.device != device or int(buf.shape[1]) != self.num_codebooks:
            buf = torch.arange(self.num_codebooks, dtype=torch.long, device=device).view(1, self.num_codebooks)
            self._codebook_index_cache[key] = buf
        return buf

    def _get_boc_frames(self, num_rows: int, device: torch.device) -> torch.Tensor:
        key = self._device_cache_key(device)
        buf = self._boc_frame_cache.get(key)
        if buf is None or buf.device != device or int(buf.shape[0]) < num_rows:
            rows = max(num_rows, 16)
            buf = torch.full((rows, self.num_codebooks), BOC_ID, dtype=torch.long, device=device)
            self._boc_frame_cache[key] = buf
        return buf[:num_rows]

    def _fast_audio_sampler_fallback_reason(
        self,
        *,
        logits: torch.Tensor,
        sampling_metadata: Any,
        num_rows: int,
        seeded_rows: list[int],
        audio_row_indices: list[int],
    ) -> str | None:
        if not _FAST_AUDIO_SAMPLER_ENABLED:
            return "disabled"
        self._record_fast_audio_sampler_attempt("attempt", batch=num_rows)
        if logits is None or logits.ndim != 2 or int(logits.shape[0]) != num_rows:
            return "invalid_logits"
        if num_rows <= 0:
            return "empty_batch"
        if getattr(sampling_metadata, "max_num_logprobs", None) is not None:
            return "logprobs"
        if getattr(sampling_metadata, "allowed_token_ids_mask", None) is not None:
            return "allowed_token_ids"
        if bool(getattr(sampling_metadata, "bad_words_token_ids", None)):
            return "bad_words"
        audio_rows = set(seeded_rows)
        audio_rows.update(audio_row_indices)
        if len(audio_rows) != num_rows:
            return "mixed_or_terminal"
        return None

    def _record_fast_audio_sampler_attempt(
        self,
        kind: str,
        *,
        batch: int | None = None,
        reason: str | None = None,
    ) -> None:
        if not _FAST_AUDIO_SAMPLER_STATS_ENABLED:
            return
        if kind == "attempt":
            self._fast_audio_sampler_total += 1
            if batch is not None:
                self._fast_audio_sampler_batch_requests[int(batch)] += 1
        elif kind == "hit":
            self._fast_audio_sampler_hits += 1
            if batch is not None:
                self._fast_audio_sampler_batch_hits[int(batch)] += 1
        elif kind == "fallback":
            self._fast_audio_sampler_fallbacks += 1
            if reason:
                self._fast_audio_sampler_fallback_reasons[str(reason)] += 1

        if (
            kind == "attempt"
            and _FAST_AUDIO_SAMPLER_STATS_EVERY > 0
            and self._fast_audio_sampler_total % _FAST_AUDIO_SAMPLER_STATS_EVERY == 0
        ):
            self._log_fast_audio_sampler_stats()

    def _log_fast_audio_sampler_stats(self) -> None:
        if not _FAST_AUDIO_SAMPLER_STATS_ENABLED or self._fast_audio_sampler_total <= 0:
            return
        hit_rate = 100.0 * self._fast_audio_sampler_hits / max(1, self._fast_audio_sampler_total)
        logger.info(
            "HiggsAudioV3Talker fast audio sampler stats: total=%d hits=%d "
            "fallbacks=%d hit_rate=%.2f%% top_requests=%s top_hits=%s "
            "top_fallback_reasons=%s",
            self._fast_audio_sampler_total,
            self._fast_audio_sampler_hits,
            self._fast_audio_sampler_fallbacks,
            hit_rate,
            self._fast_audio_sampler_batch_requests.most_common(8),
            self._fast_audio_sampler_batch_hits.most_common(8),
            self._fast_audio_sampler_fallback_reasons.most_common(8),
        )

    # ------------------------------------------------------------------ postprocess
    def postprocess(
        self,
        hidden_states_slice: torch.Tensor,
        multimodal_outputs: Any = None,
        **req_infos: Any,
    ) -> dict[str, Any]:
        """Publish per-request audio codes into model_intermediate_buffer.

        Called once per request in batch order. Indexes _last_audio_codes
        by a running cursor (one row per request per step).
        """
        _ = multimodal_outputs
        if _PROFILE_ENABLED:
            with _ProfileScope("stage0.postprocess.audio_codes"):
                return self._postprocess_impl()
        return self._postprocess_impl()

    def _postprocess_impl(self) -> dict[str, Any]:
        host_staging = self._last_audio_host_staging
        codes_full = self._last_audio_codes
        if host_staging is None and codes_full is None:
            return {}
        event = self._last_audio_staging_event
        if event is not None:
            event.synchronize()
            self._last_audio_staging_event = None

        cursor = int(self._postprocess_cursor)
        num_rows = int(host_staging.shape[0]) if host_staging is not None else int(codes_full.shape[0])
        if cursor >= num_rows:
            self._postprocess_cursor = 0
            return {}
        self._postprocess_cursor = cursor + 1

        if host_staging is not None:
            valid_flag = int(host_staging[cursor, self.num_codebooks].item())
            done_flag = int(host_staging[cursor, self.num_codebooks + 1].item())
            if valid_flag or done_flag:
                self._postprocess_audio_active_rows += 1
            if cursor + 1 >= num_rows:
                if self._postprocess_audio_rows == num_rows and self._postprocess_audio_active_rows == num_rows:
                    self._fast_audio_direct_rows = num_rows
                self._postprocess_audio_rows = 0
                self._postprocess_audio_active_rows = 0
            if valid_flag == 0:
                return {}
            new_codes = host_staging[cursor : cursor + 1, : self.num_codebooks]
            return {"codes": {"audio": new_codes}}

        if cursor >= len(self._last_audio_code_valid) or not self._last_audio_code_valid[cursor]:
            return {}
        slice_codes = codes_full[cursor : cursor + 1]
        new_codes = slice_codes.to(torch.int32)
        return {"codes": {"audio": new_codes}}

    # ------------------------------------------------------------------ helpers
    def _audio_codebook_logits(self, hidden_states: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        mask = audio_mask.reshape(-1).to(hidden_states.device)
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        if not mask.any():
            return torch.empty(
                (0, self.num_codebooks, self.codebook_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        return self.modality_head.generate(hidden_flat[mask])

    def _audio_codebook_logits_from_rows(
        self, hidden_states: torch.Tensor, audio_rows: torch.Tensor, *, all_rows: bool = False
    ) -> torch.Tensor:
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        if audio_rows.numel() == 0:
            return torch.empty(
                (0, self.num_codebooks, self.codebook_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        if all_rows:
            return self.modality_head.generate(hidden_flat[: int(audio_rows.numel())])
        return self.modality_head.generate(hidden_flat.index_select(0, audio_rows.to(hidden_flat.device)))

    def _apply_delay_pattern_masking(self, cb_logits: torch.Tensor, audio_row_indices: list[int]) -> None:
        """Mask per-codebook logits according to delay pattern state, in-place.

        During delay phase: codebooks beyond delay_count only allow BOC.
        During ramp-down: locked codebooks only allow EOC.
        Normal generation: BOC disallowed; only cb0 allows EOC.
        """
        bos_pre = BOC_ID
        eos_pre = EOC_ID
        num_codebooks = self.num_codebooks
        for local_i, batch_i in enumerate(audio_row_indices):
            state = self._audio_state.get(int(batch_i))
            num_delay = int(state["num_delay"]) if state else 0
            num_rem = state.get("num_remaining_delays") if state else None

            if num_rem is not None:
                lock_until = num_codebooks - int(num_rem)
                if lock_until > 0:
                    locked = cb_logits[local_i, :lock_until]
                    eoc_logits = locked[:, eos_pre].clone()
                    locked.fill_(float("-inf"))
                    locked[:, eos_pre] = eoc_logits
                if lock_until < num_codebooks:
                    cb_logits[local_i, lock_until:, bos_pre] = float("-inf")
                    cb_logits[local_i, lock_until:, eos_pre] = float("-inf")
            else:
                allowed_until = min(num_delay + 1, num_codebooks)
                if allowed_until > 0:
                    cb_logits[local_i, :allowed_until, bos_pre] = float("-inf")
                    if allowed_until > 1:
                        cb_logits[local_i, 1:allowed_until, eos_pre] = float("-inf")
                if allowed_until < num_codebooks:
                    delayed = cb_logits[local_i, allowed_until:]
                    boc_logits = delayed[:, bos_pre].clone()
                    delayed.fill_(float("-inf"))
                    delayed[:, bos_pre] = boc_logits

    def _apply_delay_pattern_masking_batched(
        self, cb_logits: torch.Tensor, audio_rows: torch.Tensor, *, all_rows: bool = False
    ) -> None:
        """Vectorized delay-pattern masking using GPU-resident sampler state."""
        if cb_logits.numel() == 0:
            return

        rows = audio_rows.to(device=cb_logits.device, dtype=torch.long)
        num_audio_rows = int(cb_logits.shape[0])
        num_codebooks = self.num_codebooks
        q = self._get_codebook_indices(cb_logits.device)
        if all_rows:
            delay = self._decode_delay_count[:num_audio_rows].to(torch.long)
            rem = self._decode_eoc_countdown[:num_audio_rows].to(torch.long)
        else:
            delay = self._decode_delay_count.index_select(0, rows).to(torch.long)
            rem = self._decode_eoc_countdown.index_select(0, rows).to(torch.long)
        ramp = rem >= 0

        bos = BOC_ID
        eos = EOC_ID

        # Ramp-down: codebooks already behind the EOC wave are locked to EOC.
        lock_until = num_codebooks - rem
        locked = ramp.unsqueeze(1) & (q < lock_until.unsqueeze(1))
        eoc_logits = cb_logits[:, :, eos].clone()
        cb_logits.masked_fill_(locked.unsqueeze(-1), float("-inf"))
        cb_logits[:, :, eos] = torch.where(locked, eoc_logits, cb_logits[:, :, eos])

        # Ramp-down tail: active codebooks cannot emit stream specials.
        ramp_tail = ramp.unsqueeze(1) & (q >= lock_until.unsqueeze(1))
        cb_logits[:, :, bos] = torch.where(
            ramp_tail,
            torch.full_like(cb_logits[:, :, bos], float("-inf")),
            cb_logits[:, :, bos],
        )
        cb_logits[:, :, eos] = torch.where(
            ramp_tail,
            torch.full_like(cb_logits[:, :, eos], float("-inf")),
            cb_logits[:, :, eos],
        )

        # Normal generation: delayed codebooks are forced to BOC.
        normal = ~ramp
        allowed_until = torch.clamp(delay + 1, max=num_codebooks)
        delayed = normal.unsqueeze(1) & (q >= allowed_until.unsqueeze(1))
        boc_logits = cb_logits[:, :, bos].clone()
        cb_logits.masked_fill_(delayed.unsqueeze(-1), float("-inf"))
        cb_logits[:, :, bos] = torch.where(delayed, boc_logits, cb_logits[:, :, bos])

        # Normal active codebooks: BOC is disallowed; EOC is only allowed on cb0.
        allowed = normal.unsqueeze(1) & (q < allowed_until.unsqueeze(1))
        cb_logits[:, :, bos] = torch.where(
            allowed,
            torch.full_like(cb_logits[:, :, bos], float("-inf")),
            cb_logits[:, :, bos],
        )
        nonzero_allowed = allowed & (q > 0)
        cb_logits[:, :, eos] = torch.where(
            nonzero_allowed,
            torch.full_like(cb_logits[:, :, eos], float("-inf")),
            cb_logits[:, :, eos],
        )

    def _update_delay_state_batched(
        self,
        codes_flat: torch.Tensor,
        audio_rows: torch.Tensor,
        num_rows: int,
        device: torch.device,
        *,
        code_row_mask: torch.Tensor | None = None,
        all_rows: bool = False,
    ) -> None:
        """Update delay/ramp-down state in batch and stage per-request outputs."""
        self._ensure_decode_state_capacity(num_rows, device)
        if codes_flat.numel() == 0:
            self._last_audio_codes = None
            self._last_audio_code_valid = []
            self._last_audio_host_staging = None
            self._last_audio_staging_event = None
            return

        rows = audio_rows.to(device=device, dtype=torch.long)
        num_audio_rows = int(codes_flat.shape[0])
        num_codebooks = self.num_codebooks
        q = self._get_codebook_indices(device)

        if all_rows:
            prev_delay = self._decode_delay_count[:num_audio_rows].to(torch.long)
            prev_rem = self._decode_eoc_countdown[:num_audio_rows].to(torch.long)
            prev_done = self._decode_generation_done[:num_audio_rows].to(torch.bool)
            prev_has_codes = self._decode_has_codes[:num_audio_rows].to(torch.bool)
            prev_codes = self._decode_last_codes[:num_audio_rows]
        else:
            prev_delay = self._decode_delay_count.index_select(0, rows).to(torch.long)
            prev_rem = self._decode_eoc_countdown.index_select(0, rows).to(torch.long)
            prev_done = self._decode_generation_done.index_select(0, rows).to(torch.bool)
            prev_has_codes = self._decode_has_codes.index_select(0, rows).to(torch.bool)
            prev_codes = self._decode_last_codes.index_select(0, rows)
        if code_row_mask is None:
            update_mask = torch.ones_like(prev_has_codes)
        elif all_rows:
            update_mask = code_row_mask.to(device=device, dtype=torch.bool)[:num_audio_rows]
        else:
            update_mask = code_row_mask.to(device=device, dtype=torch.bool).index_select(0, rows)
        ramp = prev_rem >= 0

        codes = codes_flat.to(device=device, dtype=torch.long).clone()

        # Leading BOC delay pad.
        delay_active = (~ramp) & ((prev_delay + 1) < num_codebooks)
        delay_mask = delay_active.unsqueeze(1) & (q > prev_delay.unsqueeze(1))
        codes = torch.where(delay_mask, torch.full_like(codes, BOC_ID), codes)
        next_delay = torch.where(delay_active, prev_delay + 1, prev_delay)

        # Trailing EOC ramp-down.
        lock_until = num_codebooks - prev_rem
        ramp_mask = ramp.unsqueeze(1) & (q < lock_until.unsqueeze(1))
        codes = torch.where(ramp_mask, torch.full_like(codes, EOC_ID), codes)
        ramp_next_rem = prev_rem - 1

        # New EOC detection in normal mode. This avoids tensor->Python control
        # flow; ties are impossible for the index because q is monotonic.
        eos_mask = (~ramp).unsqueeze(1) & (codes == EOC_ID)
        eos_idx_values = torch.where(eos_mask, q.expand_as(codes), torch.full_like(codes, -1))
        last_eos_idx = eos_idx_values.max(dim=1).values
        has_eos = last_eos_idx >= 0
        eos_prefix = has_eos.unsqueeze(1) & (q <= last_eos_idx.unsqueeze(1))
        codes = torch.where(eos_prefix, torch.full_like(codes, EOC_ID), codes)
        normal_next_rem = torch.where(has_eos, num_codebooks - last_eos_idx - 1, torch.full_like(last_eos_idx, -1))

        next_rem = torch.where(ramp, ramp_next_rem, normal_next_rem)
        done = (next_rem >= 0) & (next_rem <= 0)
        valid = (~done) & update_mask
        done = done & update_mask

        if _TERMINATION_DEBUG:
            n_eos = int(has_eos.sum().item())
            n_ramp = int(ramp.sum().item())
            n_done = int(done.sum().item())
            if n_eos > 0:
                self._td_eoc_detected += n_eos
                eos_rows_idx = torch.nonzero(has_eos & update_mask, as_tuple=False).reshape(-1)
                for ri in eos_rows_idx[:4].tolist():
                    logger.info(
                        "TD[step=%d] EOC_DETECTED row=%d last_eos_idx=%d normal_next_rem=%d cb0_code=%d",
                        self._td_step,
                        int(rows[ri].item()),
                        int(last_eos_idx[ri].item()),
                        int(normal_next_rem[ri].item()),
                        int(codes[ri, 0].item()),
                    )
            if n_done > 0:
                self._td_done_fired += n_done
                done_rows_idx = torch.nonzero(done, as_tuple=False).reshape(-1)
                for ri in done_rows_idx[:4].tolist():
                    logger.info(
                        "TD[step=%d] DONE_FIRED row=%d prev_rem=%d next_rem=%d",
                        self._td_step,
                        int(rows[ri].item()),
                        int(prev_rem[ri].item()),
                        int(next_rem[ri].item()),
                    )
            if self._td_step % _TERMINATION_DEBUG_INTERVAL == 0:
                logger.info(
                    "TD[step=%d] batch=%d ramp=%d eos_det=%d done=%d | cum: eoc=%d done=%d eos_emit=%d reset=%d",
                    self._td_step,
                    int(rows.numel()),
                    n_ramp,
                    n_eos,
                    n_done,
                    self._td_eoc_detected,
                    self._td_done_fired,
                    self._td_eos_emitted,
                    self._td_reset_fired,
                )

        next_delay = torch.where(done, torch.zeros_like(next_delay), next_delay)
        next_rem = torch.where(done, torch.full_like(next_rem, -1), next_rem)
        output_codes = torch.where(done.unsqueeze(1), torch.full_like(codes, -1), codes)

        write_delay = torch.where(update_mask, next_delay, prev_delay)
        write_rem = torch.where(update_mask, next_rem, prev_rem)
        write_done = torch.where(update_mask, done, prev_done)
        write_codes = torch.where(update_mask.unsqueeze(1), codes, prev_codes)
        write_has_codes = torch.where(update_mask, valid, prev_has_codes)

        if all_rows:
            self._decode_delay_count[:num_audio_rows].copy_(write_delay.to(torch.int32))
            self._decode_eoc_countdown[:num_audio_rows].copy_(write_rem.to(torch.int32))
            self._decode_generation_done[:num_audio_rows].copy_(write_done)
            self._decode_last_codes[:num_audio_rows].copy_(write_codes)
            self._decode_has_codes[:num_audio_rows].copy_(write_has_codes)
        else:
            self._decode_delay_count.index_copy_(0, rows, write_delay.to(torch.int32))
            self._decode_eoc_countdown.index_copy_(0, rows, write_rem.to(torch.int32))
            self._decode_generation_done.index_copy_(0, rows, write_done)
            self._decode_last_codes.index_copy_(0, rows, write_codes)
            self._decode_has_codes.index_copy_(0, rows, write_has_codes)

        codes_full = self._get_audio_codes_buffer(num_rows, device)
        codes_full.fill_(-1)
        staged_codes = torch.where(update_mask.unsqueeze(1), output_codes, torch.full_like(output_codes, -1))
        if all_rows:
            codes_full[:num_audio_rows].copy_(staged_codes)
            valid_full = valid
            done_full = done
        else:
            codes_full.index_copy_(0, rows, staged_codes)
            valid_full = torch.zeros(num_rows, dtype=torch.bool, device=device)
            done_full = torch.zeros(num_rows, dtype=torch.bool, device=device)
            valid_full.index_copy_(0, rows, valid)
            done_full.index_copy_(0, rows, done)

        staging = self._get_audio_gpu_staging_buffer(num_rows, device)
        staging[:, :num_codebooks].copy_(codes_full.to(torch.int32))
        staging[:, num_codebooks].copy_(valid_full.to(torch.int32))
        staging[:, num_codebooks + 1].copy_(done_full.to(torch.int32))

        host = self._get_audio_host_staging_buffer(num_rows)
        host.copy_(staging, non_blocking=True)
        if device.type == "cuda":
            event = self._audio_staging_event
            if event is None or self._last_audio_staging_event is not None:
                event = torch.cuda.Event()
                if self._audio_staging_event is None:
                    self._audio_staging_event = event
            event.record(torch.cuda.current_stream(device))
            self._last_audio_staging_event = event
        else:
            self._last_audio_staging_event = None

        self._decode_active_audio_count = max(self._decode_active_audio_count, num_rows)
        self._last_audio_codes = codes_full
        self._last_audio_host_staging = host[:num_rows]
        self._last_audio_code_valid = []
        self._postprocess_cursor = 0
        self._postprocess_audio_rows = num_rows
        self._postprocess_audio_active_rows = 0

    def _sample_audio_codes(self, logits_2d: torch.Tensor) -> torch.Tensor:
        """Replicate upstream sampling: temperature → top-k → top-p → multinomial."""
        x = logits_2d.float()
        top_k = 50
        top_p = 0.95
        if _FAST_AUDIO_TOPK_SAMPLING_ENABLED and 0 < top_k < x.shape[-1] and 0.0 < top_p < 1.0:
            fallback = x.argmax(dim=-1)
            sorted_logits, sorted_idx = x.topk(top_k, dim=-1, largest=True, sorted=True)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_mask = cumprobs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            has_finite = torch.isfinite(sorted_logits).any(dim=-1)
            all_masked = ~has_finite
            safe_logits = torch.where(all_masked.unsqueeze(-1), torch.zeros_like(sorted_logits), sorted_logits)
            probs = safe_logits.softmax(dim=-1)
            sampled_local = torch.multinomial(probs, num_samples=1)
            sampled = sorted_idx.gather(-1, sampled_local).squeeze(-1)
            return torch.where(all_masked, fallback, sampled)
        if 0 < top_k < x.shape[-1]:
            kth = x.topk(top_k, dim=-1).values[..., -1:]
            x = x.masked_fill(x < kth, float("-inf"))
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = x.sort(dim=-1, descending=True)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_mask = cumprobs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask.scatter_(-1, sorted_idx, sorted_mask)
            x = x.masked_fill(mask, float("-inf"))
        # Detect all-masked rows BEFORE softmax (softmax of all-inf yields NaN).
        has_finite = torch.isfinite(x).any(dim=-1)
        all_masked = ~has_finite
        # Do not branch on tensor bools here; that synchronizes every audio step.
        fallback = x.argmax(dim=-1)
        safe_x = torch.where(all_masked.unsqueeze(-1), torch.zeros_like(x), x)
        probs = safe_x.softmax(dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return torch.where(all_masked, fallback, sampled)

    def _apply_audio_mode_bias(self, logits: torch.Tensor, sampling_metadata: Any, *, mutate: bool = True) -> None:
        """Detect <|audio|> transition, force continuation, force eos at ramp-down.

        Mirrors v2's _apply_audio_mode_bias: walks per-request to find the
        previous token, and if it was <|audio|> or the continuation token,
        forces the next emit to <|audio|>. On ramp-down completion, forces eos.
        """
        if logits is None or logits.ndim != 2:
            return

        audio_id = self._audio_continuation_id
        eos_id = self._eos_token_id
        if audio_id is None:
            return

        num_rows = int(logits.shape[0])
        self._ensure_decode_state_capacity(num_rows, logits.device)
        prompt_ids = getattr(sampling_metadata, "prompt_token_ids", None)
        output_ids = getattr(sampling_metadata, "output_token_ids", None)

        # Fallback prev-token source from stashed input_ids
        stash_ids = self._last_step_input_ids
        stash_tail: list[int] | None = None
        stash_tail_computed = False

        def get_stash_tail() -> list[int] | None:
            nonlocal stash_tail, stash_tail_computed
            if stash_tail_computed:
                return stash_tail
            stash_tail_computed = True
            if not isinstance(stash_ids, torch.Tensor) or stash_ids.numel() == 0:
                return None
            q_start = self._last_step_query_start_loc
            if isinstance(q_start, torch.Tensor) and int(q_start.numel()) == num_rows + 1:
                q_start_cpu = q_start.detach().to("cpu").tolist()
                tail_idx = [max(0, int(q_start_cpu[i + 1]) - 1) for i in range(num_rows)]
                flat_ids = stash_ids.detach().to("cpu").tolist()
                stash_tail = [int(flat_ids[idx]) if idx < len(flat_ids) else -1 for idx in tail_idx]
            elif int(stash_ids.numel()) >= num_rows:
                stash_tail = stash_ids[-num_rows:].detach().to("cpu").tolist()
            return stash_tail

        seed_audio_rows: list[int] = []
        active_audio_rows: list[int] = []

        for i in range(num_rows):
            prev: int | None = None
            if output_ids is not None and i < len(output_ids):
                hist = output_ids[i]
                if hist:
                    prev = int(hist[-1])
            if prev is None and prompt_ids is not None:
                try:
                    p_i = prompt_ids[i]
                    if hasattr(p_i, "tolist"):
                        p_i = p_i.tolist()
                    if p_i:
                        prev = int(p_i[-1])
                except (IndexError, TypeError):
                    prev = None
            if prev is None:
                tail = get_stash_tail()
                if tail is not None and i < len(tail):
                    prev = int(tail[i])
            if prev is None:
                continue

            # Only bias if previous token was <|audio|> (the continuation token)
            if prev != audio_id:
                continue

            # Check if this is the FIRST step after <|audio|> appears
            # (i.e., transitioning from prompt to audio generation)
            has_codes = bool(self._decode_has_codes[i].item())
            should_terminate = bool(self._decode_generation_done[i].item())
            if not has_codes and not should_terminate:
                # No state yet — this is the first audio step
                self._decode_has_codes[i] = False
                seed_audio_rows.append(i)

            # Check for ramp-down termination
            if should_terminate and eos_id is not None and 0 <= eos_id < int(logits.shape[-1]):
                if mutate:
                    row = logits[i]
                    eos_logit = row[eos_id].clone()
                    row.fill_(float("-inf"))
                    row[eos_id] = eos_logit
                    if self._decode_has_codes[i]:
                        self._decode_active_audio_count = max(0, self._decode_active_audio_count - 1)
                    self._decode_has_codes[i] = False
                    self._decode_generation_done[i] = False
                    self._decode_delay_count[i] = 0
                    self._decode_eoc_countdown[i] = -1
                continue

            if has_codes:
                active_audio_rows.append(i)

            # Force audio continuation token
            if not mutate:
                continue
            row = logits[i]
            if 0 <= audio_id < row.shape[-1]:
                audio_logit = row[audio_id].clone()
                row.fill_(float("-inf"))
                row[audio_id] = audio_logit
            else:
                row.fill_(float("-inf"))

        self._last_seed_audio_rows = seed_audio_rows
        self._last_active_audio_rows = active_audio_rows
        self._last_first_audio_after_start = None

    # ------------------------------------------------------------------ omni output
    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs

        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information")
        if info_dicts is None:
            info_dicts = []

        audio_codes_list: list[torch.Tensor] = []
        any_nonempty = False
        for info in info_dicts:
            ac: torch.Tensor | None = None
            if isinstance(info, dict):
                codes_field = info.get("codes")
                if isinstance(codes_field, dict):
                    ac = codes_field.get("audio")
                else:
                    ac = info.get("audio_codes")
            if isinstance(ac, torch.Tensor) and ac.numel() > 0:
                audio_codes_list.append(ac)
                any_nonempty = True
            else:
                audio_codes_list.append(torch.empty(0, dtype=torch.long))

        if any_nonempty:
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={"codes": {"audio": audio_codes_list}},
            )
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs=None)

    # ------------------------------------------------------------------ weight loading

    # Per-layer suffixes from the actual V3 checkpoint (results/higgs_v3_checkpoint_analysis.txt)
    _V3_LAYER_SUFFIXES = (
        "input_layernorm.weight",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "post_attention_layernorm.weight",
        "self_attn.k_norm.weight",
        "self_attn.k_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.v_proj.weight",
    )

    @classmethod
    def _build_required_keys(cls, num_layers: int) -> set[str]:
        """Build the exact set of required V3 checkpoint keys."""
        keys = {
            "tied.embedding.text_embedding.weight",
            "body.norm.weight",
            f"{_MODALITY_EMBEDDING_PREFIX}weight",
        }
        for i in range(num_layers):
            for suffix in cls._V3_LAYER_SUFFIXES:
                keys.add(f"body.layers.{i}.{suffix}")
        return keys

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        loaded_params: set[str] = set()
        own_params = dict(self.named_parameters())
        seen_checkpoint_keys: set[str] = set()

        for name, tensor in weights:
            seen_checkpoint_keys.add(name)

            mapped = self._map_weight_name(name)
            if mapped is None:
                continue

            if mapped.startswith("model.") or mapped.startswith("lm_head."):
                backbone_weights.append((mapped, tensor))
            elif mapped in own_params:
                param = own_params[mapped]
                if param.shape != tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {mapped}: expected {param.shape}, "
                        f"got {tensor.shape} (checkpoint key: {name})"
                    )
                param.data.copy_(tensor.to(param.dtype))
                loaded_params.add(mapped)

        if backbone_weights:
            backbone_module = _BackboneWrapper(self.model, self.lm_head, self._backbone_config)
            loaded = backbone_module.load_weights(iter(backbone_weights))
            loaded_params.update(loaded)

        # Resolve special token IDs from the tokenizer
        model_path = getattr(self.vllm_config.model_config, "model", None)
        if model_path:
            self.config.resolve_special_tokens(model_path)
        self._resolve_token_ids()

        # Verify every required checkpoint key was seen.
        num_layers = int(self._backbone_config.num_hidden_layers)
        required = self._build_required_keys(num_layers)
        missing = required - seen_checkpoint_keys
        if missing:
            raise RuntimeError(
                f"HiggsAudioV3Talker: {len(missing)} required checkpoint keys missing: {sorted(missing)[:5]}..."
            )

        logger.info(
            "HiggsAudioV3Talker: loaded %d params, modality_embedding=%s, tied=%s",
            len(loaded_params),
            tuple(self.multimodal_embedding.weight.shape),
            self.tie_modality,
        )
        return loaded_params

    def _map_weight_name(self, name: str) -> str | None:
        if name.startswith(_CODEC_PREFIX):
            return None
        if name.startswith(_MODALITY_HEAD_PREFIX):
            if self.tie_modality:
                return None
            return name.replace(_MODALITY_HEAD_PREFIX, "modality_head.")
        if name.startswith(_MODALITY_EMBEDDING_PREFIX):
            return name.replace(_MODALITY_EMBEDDING_PREFIX, "multimodal_embedding.")
        for ckpt_prefix, model_prefix in _BACKBONE_PREFIX_MAP.items():
            if name.startswith(ckpt_prefix):
                return name.replace(ckpt_prefix, model_prefix, 1)
        # Reject unexpected non-codec Higgs checkpoint keys
        raise ValueError(
            f"Unexpected checkpoint key with no known mapping: {name!r}. "
            f"Known prefixes: {list(_BACKBONE_PREFIX_MAP.keys())}, "
            f"{_MODALITY_EMBEDDING_PREFIX!r}, {_MODALITY_HEAD_PREFIX!r}, {_CODEC_PREFIX!r}"
        )


class _BackboneWrapper(nn.Module):
    """Wrapper to use AutoWeightsLoader for Qwen3 backbone."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, model, lm_head, config):
        super().__init__()
        self.model = model
        self.lm_head = lm_head
        self.config = config

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.models.utils import AutoWeightsLoader

        skip = ["lm_head."] if getattr(self.config, "tie_word_embeddings", False) else None
        loader = AutoWeightsLoader(self, skip_prefixes=skip)
        return loader.load_weights(weights)
