# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical parity: the vLLM-native PersonaPlex depformer vs Moshi.

Verifies that ``PersonaPlexDepformer`` reproduces Moshi's per-frame depformer
exactly. For a real temporal hidden state it runs both predictors greedily over
all ``dep_q`` inner steps and asserts the sampled codes match (and reports the
per-step logit max-abs-diff). This is the keystone gate for the native port:
the depformer is the only fully-custom component (the temporal transformer is a
stock Llama; Mimi is the external codec).

    HF_TOKEN=... VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=2 \
        python tools/personaplex/parity_depformer.py --batch 2 --frames 8
"""

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/personaplex-7b-v1")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--frames", type=int, default=8, help="temporal prefix length")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
        PersonaPlexConfig,
    )
    from vllm_omni.model_executor.models.personaplex.personaplex_depformer import (
        PersonaPlexDepformer,
    )

    dev = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    moshi_w = hf_hub_download(args.repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_w, device=dev, dtype=dtype).eval()
    n_q, dep_q, card, text_card = lm.n_q, lm.dep_q, lm.card, lm.text_card
    print(f"loaded moshi LM: dim={lm.dim} n_q={n_q} dep_q={dep_q} card={card} text_card={text_card}")

    # A realistic temporal hidden state from a random-but-valid frame sequence.
    B, S = args.batch, args.frames
    text = torch.randint(0, text_card, (B, 1, S), device=dev)
    audio = torch.randint(0, card, (B, n_q, S), device=dev)
    codes = torch.cat([text, audio], dim=1)  # [B, 17, S]
    with torch.inference_mode():
        transformer_out_full, text_logits = lm.forward_codes(codes)
    transformer_out = transformer_out_full[:, -1:, :].contiguous()  # [B, 1, dim]
    text_token = text_logits[:, 0, -1].float().argmax(dim=-1)  # [B]

    # --- reference: Moshi greedy depformer (== depformer_step, use_sampling=False) ---
    ref_codes, ref_logits = [], []
    with torch.inference_mode(), lm.depformer.streaming(B):
        prev = text_token
        for cb in range(dep_q):
            lg = lm.forward_depformer(cb, prev[:, None, None], transformer_out)[:, 0, 0].float()
            nxt = lg.argmax(dim=-1)
            ref_codes.append(nxt)
            ref_logits.append(lg)
            prev = nxt
    ref = torch.stack(ref_codes, dim=1)  # [B, dep_q]
    ref_lg = torch.stack(ref_logits, dim=1)  # [B, dep_q, card]

    # --- candidate: the vLLM-native port, loaded from the same checkpoint ---
    cfg = PersonaPlexConfig()
    dep = PersonaPlexDepformer(
        cfg.depformer_config, temporal_hidden_size=lm.dim, text_card=text_card
    ).to(dev, dtype)
    loaded = dep.load_weights(lm.state_dict())
    dep.eval()
    total = sum(1 for _ in dep.named_parameters())
    print(f"depformer params loaded: {len(loaded)}/{total}")
    assert len(loaded) == total, "not all depformer params were loaded"

    got, got_lg = dep(text_token, transformer_out, return_logits=True)

    n = ref.numel()
    match = int((got == ref).sum().item())
    logit_diff = (got_lg - ref_lg).abs().max().item()
    print(f"codes match: {match}/{n} = {100 * match / n:.1f}%")
    print(f"per-step logit max-abs-diff: {logit_diff:.4g}")
    if match != n:
        mism = (got != ref).nonzero(as_tuple=False).tolist()
        print(f"FIRST MISMATCHES (b, step): {mism[:8]}")
        print(f"ref[0]={ref[0].tolist()}")
        print(f"got[0]={got[0].tolist()}")
        raise SystemExit("FAIL: native depformer diverges from Moshi reference")
    print("PASS: native PersonaPlex depformer matches Moshi exactly.")


if __name__ == "__main__":
    main()
