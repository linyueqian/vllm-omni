# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export PersonaPlex's Moshi temporal transformer to a stock HF Llama checkpoint.

The temporal transformer is a Llama-variant (RoPE + RMSNorm + SwiGLU, no bias).
The only non-standard bit is RoPE: Moshi uses the **interleaved / GPT-J** pairing
(`rope.py` views `[..., D//2, 2]`), while HF Llama uses rotate-half. The classic
Meta->HF `permute` on q/k weights converts the former to the latter, so the
converted checkpoint runs on vLLM's **stock** ``LlamaForCausalLM`` (is_neox_style
=True) — no custom model class. Config is measured (see tools/personaplex/parity_temporal.py).

Run on a box with the moshi fork + cached weights:
    HF_TOKEN=... python tools/personaplex/export_helium_hf.py --out /home/yueqian/helium_hf
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from safetensors.torch import save_file

REPO = "nvidia/personaplex-7b-v1"
H, L, NH = 4096, 32, 32  # hidden, layers, heads
HEAD = H // NH  # 128
FF = 11264  # measured intermediate
VOCAB = 32000  # lm_head / text head
EMB_VOCAB = 32001  # text embedding rows (32000 + 1 special)
EPS = 1e-8
THETA = 10000.0
SLIDING = 3000


def permute(w: torch.Tensor, n_heads: int = NH, dim: int = H) -> torch.Tensor:
    """Meta(interleaved)->HF(rotate-half) q/k permutation."""
    return w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-permute", action="store_true", help="skip q/k permute (debug)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    w = hf_hub_download(REPO, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(w, device="cpu")
    sd = lm.state_dict()
    out: dict[str, torch.Tensor] = {}

    def norm(k):  # alpha [1,1,H] -> [H]
        return sd[k].reshape(-1).contiguous()

    for i in range(L):
        p, v = f"transformer.layers.{i}.", f"model.layers.{i}."
        qkv = sd[p + "self_attn.in_proj_weight"]  # [3H, H], order q,k,v
        q, k, vv = qkv[:H], qkv[H : 2 * H], qkv[2 * H :]
        if not args.no_permute:
            q, k = permute(q.contiguous()), permute(k.contiguous())
        out[v + "self_attn.q_proj.weight"] = q.contiguous()
        out[v + "self_attn.k_proj.weight"] = k.contiguous()
        out[v + "self_attn.v_proj.weight"] = vv.contiguous()
        out[v + "self_attn.o_proj.weight"] = sd[p + "self_attn.out_proj.weight"].contiguous()
        # Moshi gating.linear_in = [2*FF, H]. ActivationGating splits into (gate, value):
        # gating_forward_kernel uses the FIRST half as the SiLU gate. HF SwiGLU also puts
        # gate first (gate_proj, up_proj), so the halves map directly.
        gin = sd[p + "gating.linear_in.weight"]  # [2FF, H]
        out[v + "mlp.gate_proj.weight"] = gin[:FF].contiguous()
        out[v + "mlp.up_proj.weight"] = gin[FF:].contiguous()
        out[v + "mlp.down_proj.weight"] = sd[p + "gating.linear_out.weight"].contiguous()
        out[v + "input_layernorm.weight"] = norm(p + "norm1.alpha")
        out[v + "post_attention_layernorm.weight"] = norm(p + "norm2.alpha")

    out["model.norm.weight"] = norm("out_norm.alpha")
    out["lm_head.weight"] = sd["text_linear.weight"].contiguous()
    # embed_tokens: pad the 32000-vocab text embedding to a [VOCAB,H] table (rows 0..31999);
    # the real input path uses inputs_embeds, so embed_tokens is only a placeholder for vLLM.
    te = sd["text_emb.weight"]  # [32001, H]
    out["model.embed_tokens.weight"] = te[:VOCAB].contiguous()

    out = {kk: vv.to(torch.bfloat16) for kk, vv in out.items()}
    save_file(out, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})

    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": H,
        "intermediate_size": FF,
        "num_hidden_layers": L,
        "num_attention_heads": NH,
        "num_key_value_heads": NH,
        "head_dim": HEAD,
        "hidden_act": "silu",
        "max_position_embeddings": SLIDING,
        "sliding_window": SLIDING,
        "rms_norm_eps": EPS,
        "rope_theta": THETA,
        "vocab_size": VOCAB,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
        "torch_dtype": "bfloat16",
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"exported {len(out)} tensors -> {args.out} (permute={'off' if args.no_permute else 'on'})")


if __name__ == "__main__":
    main()
