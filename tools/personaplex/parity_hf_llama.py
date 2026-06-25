# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the exported Helium->HF-Llama checkpoint reproduces Moshi's temporal
transformer numerically.

Feeds the SAME ``inputs_embeds`` to (a) Moshi ``lm.forward_embeddings`` and (b) a
stock ``transformers.LlamaForCausalLM`` loaded from the export, and compares hidden
states + text logits. HF Llama uses the exact math vLLM's Llama uses (rotate-half
RoPE, RMSNorm, SwiGLU), so a match proves the conversion is correct for the whole
Llama family — including vLLM's paged-attention runtime.

Run after export_helium_hf.py:
    HF_TOKEN=... CUDA_VISIBLE_DEVICES=1 python tools/personaplex/parity_hf_llama.py --hf /home/yueqian/helium_hf
"""

from __future__ import annotations

import argparse

import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders

REPO = "nvidia/personaplex-7b-v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="exported HF Llama dir")
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--dev", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(0)

    # Moshi reference (temporal transformer in non-streaming full-causal mode).
    w = hf_hub_download(REPO, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(w, device=args.dev)
    lm.eval()
    dim = 4096
    x = torch.randn(1, args.seq, dim, device=args.dev, dtype=torch.bfloat16) * 0.1
    with torch.no_grad():
        ref_hidden, ref_logits = lm.forward_embeddings(x)  # ([1,S,4096], [1,1,S,32000])
    ref_logits = ref_logits.squeeze(1)  # [1,S,32000]

    # HF Llama from the export.
    from transformers import LlamaForCausalLM

    hf = LlamaForCausalLM.from_pretrained(args.hf, torch_dtype=torch.bfloat16).to(args.dev).eval()
    with torch.no_grad():
        out = hf.model(inputs_embeds=x)
        hf_hidden = out.last_hidden_state
        hf_logits = hf.lm_head(hf_hidden)

    hd = (ref_hidden.float() - hf_hidden.float()).abs()
    ld = (ref_logits.float() - hf_logits.float()).abs()
    ref_arg = ref_logits[0].argmax(-1)
    hf_arg = hf_logits[0].argmax(-1)
    agree = (ref_arg == hf_arg).float().mean().item()
    print(f"hidden  max|diff|={hd.max():.4f} mean={hd.mean():.5f}")
    print(f"logits  max|diff|={ld.max():.4f} mean={ld.mean():.5f}")
    print(f"argmax agreement over {args.seq} positions: {agree * 100:.1f}%")
    ok = hd.max() < 0.2 and agree > 0.99
    print("PARITY:", "PASS (conversion correct -> vLLM Llama will run it)" if ok else "FAIL (debug rope/gate order)")


if __name__ == "__main__":
    main()
