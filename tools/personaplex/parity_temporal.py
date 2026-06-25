# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-0 de-risk for the PersonaPlex vLLM-native port (see
``plan/personaplex_vllm_native_plan.md``).

Maps PersonaPlex's Moshi temporal transformer onto a vLLM Llama/Mistral param
layout and verifies 100% tensor coverage + shape consistency, and pins the
conventions (interleaved RoPE -> ``is_neox_style=False``, fp32 RMSNorm, SwiGLU
gate+up, fused QKV). It also captures a ``forward_codes`` reference as the oracle
for the future one-step numerical parity.

Does NOT need the vLLM engine — it proves the port is mechanically complete and
nails the exact config the native model class must declare.

Run (on a box with the moshi fork + cached nvidia/personaplex-7b-v1):
    HF_TOKEN=... CUDA_VISIBLE_DEVICES=1 python tools/personaplex/parity_temporal.py

Verified on H200 2026-06-25: 195/195 temporal tensors map 1:1, all shapes consistent
(only the expected text_emb 32001 vs lm_head 32000 asymmetry). Measured config:
dim 4096, 32L/32H, head 128, intermediate 11264, vocab 32000.
"""

import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders

REPO = "nvidia/personaplex-7b-v1"
DEV = "cuda"

H, L, NH = 4096, 32, 32  # dim, layers, heads
FF = 11264  # measured from gating.linear_out (NOT 4.125*H — the _lm_kwargs scale is misleading)
TEXT_HEAD = 32000  # lm_head / text_linear vocab
TEXT_EMB = 32001  # text embedding rows (32000 + 1 special); asymmetric with the head, by design


def main() -> None:
    weight = hf_hub_download(REPO, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(weight, device=DEV)
    lm.eval()
    sd = lm.state_dict()

    # moshi temporal key -> (vLLM Llama key, expected shape)
    mapping: dict[str, tuple[str, tuple[int, ...]]] = {}
    for i in range(L):
        p, v = f"transformer.layers.{i}.", f"model.layers.{i}."
        mapping[p + "self_attn.in_proj_weight"] = (v + "self_attn.qkv_proj.weight", (3 * H, H))
        mapping[p + "self_attn.out_proj.weight"] = (v + "self_attn.o_proj.weight", (H, H))
        mapping[p + "gating.linear_in.weight"] = (v + "mlp.gate_up_proj.weight", (2 * FF, H))
        mapping[p + "gating.linear_out.weight"] = (v + "mlp.down_proj.weight", (H, FF))
        mapping[p + "norm1.alpha"] = (v + "input_layernorm.weight", (H,))
        mapping[p + "norm2.alpha"] = (v + "post_attention_layernorm.weight", (H,))
    mapping["out_norm.alpha"] = ("model.norm.weight", (H,))
    mapping["text_emb.weight"] = ("model.embed_tokens.weight", (TEXT_EMB, H))
    mapping["text_linear.weight"] = ("lm_head.weight", (TEXT_HEAD, H))

    temporal = [
        k
        for k in sd
        if k.startswith("transformer.") or k in ("out_norm.alpha", "text_emb.weight", "text_linear.weight")
    ]
    mapped = shape_ok = 0
    bad: list[tuple] = []
    for mk in temporal:
        if mk not in mapping:
            bad.append(("UNMAPPED", mk))
            continue
        mapped += 1
        _, exp = mapping[mk]
        got = tuple(s for s in sd[mk].shape if s != 1) or tuple(sd[mk].shape)
        exp_sq = tuple(s for s in exp if s != 1) or exp
        if got == exp_sq:
            shape_ok += 1
        else:
            bad.append((mk, got, exp))

    print(f"temporal tensors: {len(temporal)} | mapped: {mapped} | shape-consistent: {shape_ok}")
    print(f"discrepancies (expected: only text_emb 32001 vs head 32000): {bad}")
    print(
        f"non-Llama parts: emb.N audio codebooks={sum(k.startswith('emb.') for k in sd)}, "
        f"depformer tensors={sum(k.startswith('depformer') for k in sd)}"
    )
    print(
        "conventions: rope=interleaved(is_neox_style=False) | norm=fp32 RMSNorm eps1e-8 | "
        "mlp=SwiGLU gate+up | attn=fused QKV no-bias (MHA)"
    )

    with torch.no_grad():
        seq = lm._get_initial_token().to(DEV).expand(1, lm.num_codebooks, 4).contiguous()
        tr_out, text_logits = lm.forward_codes(seq)
    print(
        f"reference forward_codes: transformer_out{tuple(tr_out.shape)} std={tr_out.float().std():.4f} "
        f"text_logits{tuple(text_logits.shape)}"
    )


if __name__ == "__main__":
    main()
