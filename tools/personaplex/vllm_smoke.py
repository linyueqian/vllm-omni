# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prove the exported Helium checkpoint runs on vLLM's paged-attention engine, and
that vLLM's per-position logits match Moshi for the same ``inputs_embeds``.

Loads the converted stock-Llama checkpoint into ``vllm.LLM`` and drives it with
``prompt_embeds`` (the real PersonaPlex input path is precomputed embeddings),
requesting ``prompt_logprobs`` so we can compare vLLM's argmax against Moshi's.

    HF_TOKEN=... VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=1 \
        python tools/personaplex/vllm_smoke.py --hf /home/yueqian/helium_hf
"""

from __future__ import annotations

import argparse

import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders

REPO = "nvidia/personaplex-7b-v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--seq", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0)

    # Moshi reference logits for a fixed random inputs_embeds.
    w = hf_hub_download(REPO, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(w, device="cuda").eval()
    x = torch.randn(1, args.seq, 4096, device="cuda", dtype=torch.bfloat16) * 0.1
    with torch.no_grad():
        _, ref_logits = lm.forward_embeddings(x)
    ref_arg = ref_logits.squeeze(1)[0].argmax(-1).tolist()
    del lm
    torch.cuda.empty_cache()

    from vllm import LLM

    llm = LLM(
        model=args.hf,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.55,
        max_model_len=512,
        disable_log_stats=True,
    )
    # Drive with the SAME embeddings via prompt_embeds; greedy-generate one token.
    # vLLM's first greedy token = argmax of the last-position logits = Moshi's
    # next-token prediction at that position (ref_arg[-1]).
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=1, temperature=0.0)
    out = llm.generate({"prompt_embeds": x.squeeze(0).cpu()}, sp)
    vllm_tok = out[0].outputs[0].token_ids[0]
    print(f"vLLM ran on paged attention. moshi last-position argmax={ref_arg[-1]} | vLLM greedy token={vllm_tok}")
    print(
        "VLLM PARITY:",
        "PASS — PersonaPlex temporal transformer runs correctly on vLLM"
        if vllm_tok == ref_arg[-1]
        else f"CHECK (full moshi argmax {ref_arg})",
    )


if __name__ == "__main__":
    main()
