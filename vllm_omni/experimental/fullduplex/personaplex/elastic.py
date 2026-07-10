# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Elastic (continuous) batching for the PersonaPlex ``moshi`` fork, as runtime patches.

The stock PersonaPlex serving stack is one conversation per process: the moshi
streaming state machine batches fine for decode, but three GLOBAL scalars make it
impossible to recycle ONE batch row for a new caller mid-flight:

* ``RingKVCache.end_offset`` is a single scalar -- no per-row KV validity window.
* ``_LMGenState.offset`` is a plain ``int`` driving the per-codebook delay warmup --
  a recycled row cannot replay its warmup independently.
* Mimi's streaming conv buffers carry per-row history with only whole-batch resets.

This module ports the per-slot recycle design from the ``cb-fix`` work in
https://github.com/Hcoder10/personaplex-business-ft (MIT, license preserved there),
which verified it bit-exact upstream: untouched slots are bit-identical to a
no-recycle run, and a recycled slot is bit-identical to a fresh conversation,
across all staging-ring phases. Key ideas (all shapes static, CUDA-graph safe):

* KV: keep writes uniform (rows tick in lockstep) and add a per-row
  ``start_offset[B]`` that only affects the attention MASK. ``reset_slot(b)`` sets
  ``start_offset[b] = end_offset + 1``: a fresh conversation skips the model at its
  ``o == 0`` early-return, but a mid-flight recycled row cannot skip a batch tick,
  so it makes one spurious KV write on its first post-reset tick; the ``+1`` masks
  exactly that entry.
* LMGen: split ``offset`` into a shared ``phys_offset`` (staging-ring indexing,
  uniform because lockstep) and a per-row ``local_offset[B]`` (delay-warmup gates +
  output suppression). ``step(..., force_rows=mask)`` forces agent tokens on
  selected rows only, so a recycled slot can re-prefill its persona while the rest
  of the batch keeps conversing. ``valid_mask()`` reports which rows of the last
  ``step()`` output are past their own warmup.
* Everything is applied as method patches on the installed ``moshi`` package, so no
  fork of the dependency is required; with no recycling and ``start_offset == 0``
  the patched code paths are bit-identical to upstream.

Apply with :func:`apply_elastic_patches` (idempotent) before building ``LMGen``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_elastic_patches() -> None:
    """Patch the installed moshi fork for per-slot elastic batching (idempotent)."""
    global _APPLIED
    if _APPLIED:
        return

    import torch
    from moshi.models import lm as moshi_lm
    from moshi.modules import streaming as moshi_streaming
    from moshi.modules import transformer as moshi_transformer

    _patch_ring_kv_cache(torch, moshi_transformer)
    _patch_attention_forward(torch, moshi_transformer)
    _patch_streaming_reset_slot(moshi_streaming, moshi_transformer)
    _patch_lm_gen(torch, moshi_lm)

    _APPLIED = True
    logger.info("PersonaPlex elastic-batching patches applied to moshi")


# ---------------------------------------------------------------------------
# RingKVCache: per-row valid-window start (mask-only; write path unchanged)
# ---------------------------------------------------------------------------


def _patch_ring_kv_cache(torch, tr) -> None:
    RingKVCache = tr.RingKVCache
    orig_init = RingKVCache.__init__

    def ring_init(
        self, batch_size, num_heads, dim_per_head, capacity, device=torch.device("cuda"), dtype=torch.bfloat16
    ):
        orig_init(self, batch_size, num_heads, dim_per_head, capacity, device=device, dtype=dtype)
        # Per-row "valid window start": a row only attends to cache cells whose
        # ABSOLUTE position is >= start_offset[b]. All zeros == upstream behaviour.
        self.start_offset = torch.zeros(batch_size, device=device, dtype=torch.long)

    def reset(self):
        self.end_offset.zero_()
        self.start_offset.zero_()

    def reset_slot(self, b: int):
        # Mask (don't zero) the previous caller's K/V: move this row's window
        # start to end_offset, so everything written so far is invisible and the
        # row's NEXT write is the first visible entry -- the correct semantics
        # for Mimi's encoder/decoder streams, whose first post-reset frame is
        # real data. The LM stream additionally masks one spurious write (see
        # LMGen.reset_slot: a fresh conversation skips its o == 0 tick, a
        # mid-flight recycled row cannot).
        self.start_offset[b] = self.end_offset.clone()

    def complete(self, k, v):
        assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
        B, H, T, D = k.shape
        indexes = torch.arange(T, device=self.end_offset.device, dtype=self.end_offset.dtype) + self.end_offset
        indexes = indexes % self.capacity
        self.cache[0].index_copy_(2, indexes, k)
        self.cache[1].index_copy_(2, indexes, v)
        self.end_offset.add_(T)

        keys = self.cache[0]
        values = self.cache[1]

        indexes = torch.arange(self.capacity, device=self.end_offset.device, dtype=torch.long)
        invalid = indexes >= self.end_offset
        end_index = self.end_offset % self.capacity
        delta = indexes - end_index
        positions = torch.where(
            delta <= 0,
            self.end_offset + delta,
            self.end_offset + delta - self.capacity,
        )
        positions = torch.where(invalid, torch.full_like(positions, -1), positions)

        # Per-row gate: invalidate cells below this row's window start. When all
        # start_offset == 0 this equals the original [capacity] positions broadcast
        # over B (bit-identical, no recycling).
        positions = positions.view(1, -1)  # [1, capacity]
        below_start = positions < self.start_offset.view(-1, 1)  # [B, capacity]
        positions = torch.where(below_start, torch.full_like(positions, -1), positions)

        return tr.KVCacheResult(keys, values, positions)

    RingKVCache.__init__ = ring_init
    RingKVCache.reset = reset
    RingKVCache.reset_slot = reset_slot
    RingKVCache.complete = complete


def _patch_attention_forward(torch, tr) -> None:
    """Teach StreamingMultiheadAttention to consume a per-row [B, capacity] pos_k."""
    import torch.nn as nn
    import torch.nn.functional as F
    from einops import rearrange

    MHA = tr.StreamingMultiheadAttention
    multi_linear = tr.multi_linear

    def forward(self, query, key, value):
        state = self._streaming_state
        T = query.shape[1]

        if state is None:
            offset = torch.zeros(1, device=query.device, dtype=torch.long)
            offset_cpu = 0
        else:
            assert self.causal, "Streaming only available for causal"
            offset = state.offset
            offset_cpu = state.offset_cpu

        if self.weights_per_step:
            projected = multi_linear(self.weights_per_step, self.in_proj_weight, query, offset_cpu)
        else:
            projected = nn.functional.linear(query, self.in_proj_weight)
        q, k, v = rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads)

        if self.rope:
            q, k = self.rope(q, k, offset, time_before_heads=False)

        k, v, pos_k = self._complete_kv(k, v)
        if self.causal:
            # pos_k is [capacity] (non-streaming) or [B, capacity] (per-row window).
            if pos_k.dim() == 1:
                pos_k = pos_k.view(1, 1, -1)
            else:
                pos_k = pos_k.view(pos_k.shape[0], 1, pos_k.shape[1])
            pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(1, -1, 1)
            delta = pos_q - pos_k  # [B, T, capacity]
            attn_bias = (pos_k >= 0) & (delta >= 0)
            if self.context is not None:
                attn_bias = attn_bias & (delta < self.context)
            attn_bias = attn_bias.unsqueeze(1)  # [B, 1, T, capacity] for SDPA
        else:
            attn_bias = None
        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)

        x = rearrange(x, "b h t d -> b t (h d)")
        if self.weights_per_step:
            x = multi_linear(self.weights_per_step, self.out_proj.weight, x, offset_cpu)
        else:
            x = self.out_proj(x)
        if state is not None:
            state.offset.add_(T)
            state.offset_cpu += T
        return x

    MHA.forward = forward

    # RoPE / absolute offsets stay SHARED across a recycle (attention depends only
    # on relative distance); only the KV cache carries per-row history to clear.
    tr._MHAState.reset_slot = lambda self, b: self.kv_cache.reset_slot(b)


# ---------------------------------------------------------------------------
# StreamingModule: per-row reset propagation (mirror of reset_streaming)
# ---------------------------------------------------------------------------


def _patch_streaming_reset_slot(st, tr) -> None:
    def module_reset_slot(self, b: int):
        """Reset the streaming state of ONLY batch row ``b`` in this module tree."""

        def _reset_slot(name, module):
            state = module._streaming_state
            if state is None:
                raise ValueError(f"Trying to reset streaming slot, but {name} wasn't streaming.")
            fn = getattr(state, "reset_slot", None)
            # States without a reset_slot carry no per-row history (shared scalars
            # or per-step-recreated state); analyzed per class in the upstream port.
            if fn is not None:
                fn(b)

        self._apply_named_streaming(_reset_slot)

    st.StreamingModule.reset_slot = module_reset_slot

    def add_reset_slot(self, b: int):
        if self.previous_x is not None:
            self.previous_x[b].zero_()
        if self.previous_y is not None:
            self.previous_y[b].zero_()

    def conv_reset_slot(self, b: int):
        # Zero this row's carried left-context == "silence history" for a fresh stream.
        if self.previous is not None:
            self.previous[b].zero_()

    def convtr_reset_slot(self, b: int):
        # Zero this row's pending overlap-add tail.
        if self.partial is not None:
            self.partial[b].zero_()

    st._StreamingAddState.reset_slot = add_reset_slot
    st._StreamingConvState.reset_slot = conv_reset_slot
    st._StreamingConvTrState.reset_slot = convtr_reset_slot


# ---------------------------------------------------------------------------
# LMGen: shared phys_offset + per-row local_offset, force_rows, valid_mask
# ---------------------------------------------------------------------------


def _patch_lm_gen(torch, lm) -> None:
    from moshi.utils.compile import CUDAGraphed

    LMGen = lm.LMGen
    AUDIO_TOKENS_PER_STREAM = lm.AUDIO_TOKENS_PER_STREAM
    sample_token = lm.sample_token
    create_loss_report = lm.create_loss_report

    @dataclass
    class ElasticLMGenState:
        cache: torch.Tensor
        provided: torch.Tensor
        initial: torch.Tensor
        graphed_main: CUDAGraphed
        graphed_embeddings: CUDAGraphed
        graphed_depth: CUDAGraphed
        # Elastic batching splits the main forward into embed + temporal so a
        # prefill row's embedding can be overridden while live rows take the
        # numerically identical path on EVERY tick (mixed and normal ticks must
        # not differ bit-wise for untouched rows, or greedy near-ties flip).
        graphed_embed: CUDAGraphed = None
        # Shared PHYSICAL tick (staging-ring indexing; uniform, graph-friendly).
        phys_offset: int = 0
        # Per-row logical clock (CPU): delay-warmup gates + output suppression only.
        local_offset: torch.Tensor = None  # [B] long
        last_valid: torch.Tensor = None  # [B] bool

        def reset(self):
            self.phys_offset = 0
            self.local_offset.zero_()
            self.provided[:] = False

        def reset_slot(self, b: int):
            # Restart row b's logical clock and wipe its staging cells to the
            # caller-agnostic initial token so nothing from the previous caller
            # can be read. KV masking is done by RingKVCache.reset_slot via the
            # module hierarchy; phys_offset is shared and intentionally untouched.
            self.local_offset[b] = 0
            self.cache[b] = self.initial[0, :, 0:1].expand(-1, self.cache.shape[2]).clone()
            self.provided[b] = False

    lm._LMGenState = ElasticLMGenState  # keep type hints resolvable

    def _init_streaming_state(self, batch_size: int) -> ElasticLMGenState:
        lm_model = self.lm_model
        initial = lm_model._get_initial_token()
        cache = torch.full(
            (batch_size, self.lm_model.num_codebooks, self.max_delay + 3),
            lm_model.ungenerated_token_id,
            device=lm_model.device,
            dtype=torch.long,
        )
        provided = torch.full(
            (batch_size, self.lm_model.num_codebooks, self.max_delay + 3),
            False,
            device=lm_model.device,
            dtype=torch.bool,
        )

        disable = lm_model.device.type != "cuda"
        graphed_main = CUDAGraphed(lm_model.forward_codes, disable=disable)
        graphed_embeddings = CUDAGraphed(lm_model.forward_embeddings, disable=disable)
        graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)
        graphed_embed = CUDAGraphed(lm_model.embed_codes, disable=disable)

        return ElasticLMGenState(
            cache,
            provided,
            initial,
            graphed_main,
            graphed_embeddings,
            graphed_depth,
            graphed_embed=graphed_embed,
            phys_offset=0,
            local_offset=torch.zeros(batch_size, dtype=torch.long, device="cpu"),
            last_valid=torch.zeros(batch_size, dtype=torch.bool, device="cpu"),
        )

    def reset_slot(self, b: int):
        """Recycle conversation slot ``b`` mid-flight (call BETWEEN steps only)."""
        state = self._streaming_state
        if state is None:
            raise RuntimeError("reset_slot requires an active streaming state (use streaming()).")
        state.reset_slot(b)
        self.lm_model.reset_slot(b)

        # LM-specific skip semantics: a fresh conversation skips the model at its
        # o == 0 early-return, but a mid-flight recycled row cannot skip a batch
        # tick, so its first post-reset tick makes one spurious KV write. Mask
        # exactly that entry by bumping the LM tree's window starts one past the
        # generic reset (Mimi streams must NOT get this bump -- their first
        # post-reset frame is real data).
        def _bump(name, module):
            st = module._streaming_state
            kv = getattr(st, "kv_cache", None)
            if kv is not None and hasattr(kv, "start_offset"):
                kv.start_offset[b] += 1

        self.lm_model._apply_named_streaming(_bump)

    def valid_mask(self) -> torch.Tensor:
        """[B] bool (CPU): rows of the last ``step()`` output past their own warmup."""
        state = self._streaming_state
        assert state is not None
        return state.last_valid

    @torch.no_grad()
    def prepare_step_input(self, input_tokens=None, moshi_tokens=None, text_token=None, force_rows=None):
        # force_rows: optional [B] bool. When given, text_token/moshi_tokens are
        # forced only on True rows (others sample normally) -- persona injection
        # into a recycled slot while the rest of the batch keeps conversing.
        state = self._streaming_state
        if state is None:
            raise RuntimeError("You should wrap those calls with a `with lm_gen.streaming(): ...`.")
        lm_model = self.lm_model

        needed_tokens = lm_model.num_codebooks - AUDIO_TOKENS_PER_STREAM - 1
        CT = state.cache.shape[2]
        # Ring indexing runs on the shared physical tick; warmup gates run on the
        # per-row local clock. For a never-recycled batch local == phys for every
        # row, so behaviour is bit-identical to the original scalar-offset code.
        o = state.phys_offset
        lo = state.local_offset

        if input_tokens is not None:
            assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
            B, Ki, S = input_tokens.shape
            assert S == 1, "Only support being given steps one by one."
            assert Ki == needed_tokens, f"We expect {needed_tokens} tokens from the user stream, got {Ki}."
            for q_other in range(input_tokens.shape[1]):
                k = AUDIO_TOKENS_PER_STREAM + 1 + q_other
                delay = lm_model.delays[k]
                write_position = (o + delay) % CT
                state.cache[:, k, write_position : write_position + 1] = input_tokens[:, q_other]
                state.provided[:, k, write_position : write_position + 1] = True

        if moshi_tokens is not None:
            assert moshi_tokens.dim() == 3, "Shape should be [B, K, T]."
            B, Ki, S = moshi_tokens.shape
            assert S == 1, "Only support being given steps one by one."
            assert Ki == needed_tokens, f"We expect {needed_tokens} tokens from the moshi stream, got {Ki}."
            for q_moshi in range(moshi_tokens.shape[1]):
                k = 1 + q_moshi
                delay = lm_model.delays[k]
                write_position = (o + delay) % CT
                if force_rows is None:
                    state.cache[:, k, write_position : write_position + 1] = moshi_tokens[:, q_moshi]
                    state.provided[:, k, write_position : write_position + 1] = True
                else:
                    state.cache[force_rows, k, write_position : write_position + 1] = moshi_tokens[force_rows, q_moshi]
                    state.provided[force_rows, k, write_position : write_position + 1] = True

        if text_token is not None:
            write_position = (o + lm_model.delays[0]) % CT
            if force_rows is None:
                state.cache[:, 0, write_position] = text_token
                state.provided[:, 0, write_position] = True
            else:
                state.cache[force_rows, 0, write_position] = text_token[force_rows]
                state.provided[force_rows, 0, write_position] = True

        B = state.cache.shape[0]
        for k, delay in enumerate(lm_model.delays):
            # Delay warmup, gated PER ROW so a recycled row re-runs its own warmup.
            # The write position uses the shared physical ring index: at a row's
            # local warmup step the buffer phase lines up the same way it did at
            # the global cold start.
            warm = lo <= delay  # [B] bool (CPU)
            if warm.any():
                warm_dev = warm.to(state.cache.device)
                pos = o % CT
                init_k = state.initial[:, k, 0]
                state.cache[:, k, pos] = torch.where(warm_dev, init_k.expand(B), state.cache[:, k, pos])
                state.provided[:, k, pos] = state.provided[:, k, pos] | warm_dev

        if o == 0:
            # GENUINE batch cold start only (no prior physical frame to read). A
            # mid-flight recycled row never re-enters this branch; its local o == 0
            # warmup is handled by the per-row gates above + output suppression.
            state.cache[:, :, 0] = state.initial[:, :, 0]
            state.phys_offset += 1
            state.local_offset += 1
            return None

        model_input_position = (o - 1) % CT
        target_position = o % CT
        input_ = state.cache[:, :, model_input_position : model_input_position + 1]
        target_ = state.cache[:, :, target_position : target_position + 1]
        provided_ = state.provided[:, :, target_position : target_position + 1]

        if self.check:
            assert not (input_ == lm_model.ungenerated_token_id).any(), (state.phys_offset, input_)
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()
        return input_, provided_, target_, model_input_position, target_position

    @torch.no_grad()
    def step(self, input_tokens=None, moshi_tokens=None, text_token=None, return_embeddings=False, force_rows=None):
        state = self._streaming_state
        lm_model = self.lm_model
        prepared_inputs = self.prepare_step_input(input_tokens, moshi_tokens, text_token, force_rows=force_rows)
        if prepared_inputs is None:
            return (None, None) if self.report_loss or self.return_logits else None
        input_, provided_, target_, model_input_position, target_position = prepared_inputs
        if self.check:
            assert not (input_ == lm_model.ungenerated_token_id).any(), (state.phys_offset, input_)
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()
        embeddings = None
        if return_embeddings:
            embeddings = self.lm_model.embed_codes(input_)
        # Split path (embed graph, then temporal graph) on EVERY tick, so mixed
        # prefill ticks are numerically identical to normal ticks for live rows.
        emb = state.graphed_embed(input_)
        transformer_out, text_logits = state.graphed_embeddings(emb)
        output = self.process_transformer_output(
            transformer_out, text_logits, provided_, target_, model_input_position, target_position
        )
        if return_embeddings:
            return output, embeddings
        return output

    @torch.no_grad()
    def step_prefill_embedding(
        self, embeddings, embed_rows, input_tokens, moshi_tokens=None, text_token=None, force_rows=None
    ):
        """Mixed tick: some rows replay a stored voice-prompt embedding while live
        rows keep conversing on their staged tokens.

        ``embeddings`` is ``[B, 1, H]`` (values ignored outside ``embed_rows``);
        rows where ``embed_rows`` is True get it as their model input, with
        caller-agnostic dummy tokens forced into their staging cells exactly like
        ``step_embeddings``. Other rows run their normal token path through the
        SAME ``graphed_embeddings`` forward (``forward_codes(x) ==
        forward_embeddings(embed_codes(x))``). ``moshi_tokens``/``text_token``/
        ``force_rows`` allow token-forced prefill rows (silence/persona phases) to
        share the tick with embedding rows.
        """
        state = self._streaming_state
        lm_model = self.lm_model
        needed = lm_model.num_codebooks - AUDIO_TOKENS_PER_STREAM - 1
        B = input_tokens.shape[0]
        dummy = lm_model._get_initial_token().expand(B, -1, -1)
        if moshi_tokens is None:
            moshi_tokens = dummy[:, 1 : 1 + needed].clone()
        if text_token is None:
            text_token = torch.full((B,), self.zero_text_code, dtype=torch.long, device=lm_model.device)
        mask = embed_rows if force_rows is None else (embed_rows | force_rows)
        prepared = self.prepare_step_input(
            input_tokens=input_tokens, moshi_tokens=moshi_tokens, text_token=text_token, force_rows=mask
        )
        if prepared is None:
            return None
        input_, provided_, target_, model_input_position, target_position = prepared
        emb = state.graphed_embed(input_)  # [B, S, H], same graph as normal ticks
        emb = torch.where(embed_rows.view(B, 1, 1), embeddings.to(emb.dtype), emb)
        transformer_out, text_logits = state.graphed_embeddings(emb)
        return self.process_transformer_output(
            transformer_out, text_logits, provided_, target_, model_input_position, target_position
        )

    @torch.no_grad()
    def process_transformer_output(
        self, transformer_out, text_logits, provided_, target_, model_input_position, target_position
    ):
        state = self._streaming_state
        lm_model = self.lm_model

        sampled_text_token = sample_token(text_logits.float(), self.use_sampling, self.temp_text, self.top_k_text)
        assert sampled_text_token.dim() == 3, sampled_text_token.shape
        assert sampled_text_token.shape[2] == 1
        assert sampled_text_token.shape[1] == 1, "Only one text stream supported."
        sampled_text_token = sampled_text_token[:, 0, 0]

        next_text_token = torch.where(provided_[:, 0, 0], target_[:, 0, 0], sampled_text_token)

        if self.return_logits:
            sampled_audio_tokens, audio_logits = state.graphed_depth(
                next_text_token,
                transformer_out,
                target_[:, lm_model.audio_offset :, 0],
                provided_[:, lm_model.audio_offset :, 0],
            )
        else:
            sampled_audio_tokens = state.graphed_depth(
                next_text_token,
                transformer_out,
                target_[:, lm_model.audio_offset :, 0],
                provided_[:, lm_model.audio_offset :, 0],
            )

        state.provided[:, :, model_input_position] = False

        state.cache[:, 0, target_position] = torch.where(
            ~state.provided[:, 0, target_position], sampled_text_token, state.cache[:, 0, target_position]
        )
        state.cache[:, 1 : lm_model.dep_q + 1, target_position] = torch.where(
            ~state.provided[:, 1 : lm_model.dep_q + 1, target_position],
            sampled_audio_tokens,
            state.cache[:, 1 : lm_model.dep_q + 1, target_position],
        )

        report = {}
        if self.report_loss:
            report = create_loss_report(
                state_cache=state.cache,
                lm_model=lm_model,
                text_logits=text_logits,
                audio_logits=audio_logits,
                target=target_,
                sampled_text_token=sampled_text_token,
                sampled_audio_tokens=sampled_audio_tokens,
                target_position=target_position,
            )

        # Output validity is PER ROW: a row emits only once its local clock is past
        # the delay warmup. Rows still in warmup carry caller-agnostic warmup
        # tokens (nothing from a previous caller), but callers should drop them via
        # valid_mask(). Return contract unchanged for non-elastic callers.
        lo = state.local_offset
        valid = lo > self.max_delay

        if not bool(valid.any()):
            state.phys_offset += 1
            state.local_offset += 1
            if self.report_loss:
                return None, report
            if self.return_logits:
                return None, None
            return None

        B = state.cache.shape[0]
        CT = state.cache.shape[2]
        gen_delays_cuda = self.delays_cuda[: lm_model.dep_q + 1]
        index = ((state.phys_offset - self.max_delay + gen_delays_cuda) % CT).view(1, -1, 1).expand(B, -1, 1)
        out = state.cache.gather(dim=2, index=index)

        state.last_valid = valid.clone()

        state.phys_offset += 1
        state.local_offset += 1
        if self.report_loss:
            return out, report
        elif self.return_logits and not self.report_loss:
            return out, (text_logits.clone(), audio_logits.clone())
        else:
            return out

    LMGen._init_streaming_state = _init_streaming_state
    LMGen.reset_slot = reset_slot
    LMGen.valid_mask = valid_mask
    LMGen.prepare_step_input = prepare_step_input
    LMGen.step = step
    LMGen.step_prefill_embedding = step_prefill_embedding
    LMGen.process_transformer_output = process_transformer_output
