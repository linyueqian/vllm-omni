# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical parity check for PersonaPlex Moshi -> vLLM Helium.

Run on an H200 box with the Moshi fork and vLLM runtime installed:

    HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 python tools/personaplex/parity_vllm.py \
        --moshi-checkpoint <path> --seq-len 16

If the checkpoint is in a private HF repo, download/cache it first or make sure
`HF_TOKEN` is set for the Moshi loader's own hub access path.
"""

from __future__ import annotations

import argparse
import copy
import os
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from moshi.models import loaders
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config

from vllm_omni.model_executor.models.personaplex.configuration_helium import (
    HeliumConfig,
)
from vllm_omni.model_executor.models.personaplex.modeling_helium import (
    HeliumForCausalLM,
)


class _EagerSlidingWindowAttention(torch.nn.Module):
    """Engine-free attention op for the parity harness only.

    vLLM's production `Attention` layer requires scheduler-built forward
    context and KV cache metadata. For direct eager parity we keep Helium's
    projections, RoPE, norms, MLP, and LM head intact, and replace only the
    context-bound attention op with PyTorch SDPA using Moshi's causal
    sliding-window mask. This is the direct-construction alternative to running
    a full vLLM engine/model runner.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int,
        scale: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.scale = scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if query.dim() != 2:
            raise ValueError(
                "The parity eager attention expects flattened [T, H] tensors; "
                f"got {tuple(query.shape)}."
            )
        seq_len = query.shape[0]
        q = query.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        k = key.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        v = value.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1).unsqueeze(0)

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        pos = torch.arange(seq_len, device=query.device)
        delta = pos[:, None] - pos[None, :]
        mask = (delta >= 0) & (delta < self.sliding_window)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            scale=self.scale,
        )
        return out.squeeze(0).transpose(0, 1).reshape(seq_len, self.num_heads * self.head_dim)


def _install_eager_attention_for_parity(model: HeliumForCausalLM) -> None:
    for layer in model.model.layers:
        if not hasattr(layer, "self_attn"):
            continue
        attn = layer.self_attn
        attn.attn = _EagerSlidingWindowAttention(
            num_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_dim=attn.head_dim,
            sliding_window=model.config.sliding_window,
            scale=attn.scaling,
        )


def _make_vllm_config(config: HeliumConfig):
    try:
        vllm_config = VllmConfig()
        return vllm_config.with_hf_config(
            config,
            architectures=["HeliumForCausalLM"],
        )
    except Exception:
        # Minimal construction fallback for direct parity. Production loading
        # should use a real VllmConfig from the engine.
        compilation_config = SimpleNamespace(static_forward_context={})
        model_config = SimpleNamespace(
            hf_config=config,
            is_mm_prefix_lm=False,
            use_mla=False,
            dtype=torch.get_default_dtype(),
        )
        cache_config = SimpleNamespace(
            sliding_window=config.sliding_window,
            cache_dtype="auto",
            calculate_kv_scales=False,
            kv_cache_dtype_skip_layers=None,
            enable_prefix_caching=False,
        )
        attention_config = SimpleNamespace(
            flex_attn_block_m=None,
            flex_attn_block_n=None,
        )
        return SimpleNamespace(
            model_config=model_config,
            cache_config=cache_config,
            quant_config=None,
            lora_config=None,
            compilation_config=compilation_config,
            attention_config=attention_config,
        )


@contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def _build_helium(device: torch.device, dtype: torch.dtype) -> HeliumForCausalLM:
    config = HeliumConfig()
    with _default_dtype(dtype):
        vllm_config = _make_vllm_config(config)
        if hasattr(vllm_config, "compilation_config"):
            vllm_config.compilation_config = copy.copy(vllm_config.compilation_config)
            vllm_config.compilation_config.static_forward_context = {}
        with set_current_vllm_config(vllm_config):
            model = HeliumForCausalLM(vllm_config=vllm_config)
    model.to(device=device, dtype=dtype)
    model.eval()
    _install_eager_attention_for_parity(model)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moshi-checkpoint", required=True)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seq_len > HeliumConfig().sliding_window:
        raise ValueError("--seq-len must be <= HeliumConfig.sliding_window")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    lm = loaders.get_moshi_lm(args.moshi_checkpoint, device=str(device))
    lm.eval()
    dtype = next(lm.parameters()).dtype

    helium = _build_helium(device, dtype)
    helium.load_weights(lm.state_dict())

    inputs_embeds = torch.randn(
        1,
        args.seq_len,
        HeliumConfig().hidden_size,
        device=device,
        dtype=dtype,
    )
    positions = torch.arange(args.seq_len, device=device, dtype=torch.long).unsqueeze(0)

    with torch.inference_mode():
        ref_hidden, ref_text_logits = lm.forward_embeddings(inputs_embeds)
        got_hidden = helium(
            input_ids=None,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
        got_text_logits = helium.compute_logits(got_hidden)

    assert got_text_logits is not None
    hidden_diff = (ref_hidden - got_hidden).float().abs().max().item()
    logits_diff = (
        ref_text_logits[:, 0].contiguous() - got_text_logits.contiguous()
    ).float().abs().max().item()

    print(f"checkpoint={os.path.abspath(args.moshi_checkpoint)}")
    print(f"seq_len={args.seq_len} dtype={dtype} device={device}")
    print(f"hidden_max_abs_diff={hidden_diff:.8g}")
    print(f"text_logits_max_abs_diff={logits_diff:.8g}")

    if hidden_diff >= args.threshold or logits_diff >= args.threshold:
        raise AssertionError(
            "Parity check failed: "
            f"hidden_diff={hidden_diff:.8g}, logits_diff={logits_diff:.8g}, "
            f"threshold={args.threshold:.8g}"
        )


if __name__ == "__main__":
    main()
