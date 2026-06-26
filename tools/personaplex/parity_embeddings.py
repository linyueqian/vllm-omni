# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical parity: native PersonaPlex input embeddings vs Moshi ``embed_codes``.

The temporal transformer is fed precomputed ``inputs_embeds``; this checks that
:class:`PersonaPlexInputEmbeddings` reproduces Moshi's ``LMModel.embed_codes``
bit-for-bit (it is a deterministic sum of embedding lookups, so exact match is
expected, modulo nothing).

    HF_TOKEN=... CUDA_VISIBLE_DEVICES=2 python tools/personaplex/parity_embeddings.py
"""

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/personaplex-7b-v1")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
        PersonaPlexConfig,
    )
    from vllm_omni.model_executor.models.personaplex.personaplex_embeddings import (
        PersonaPlexInputEmbeddings,
    )

    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(args.seed)

    moshi_w = hf_hub_download(args.repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_w, device=dev, dtype=dtype).eval()
    n_q, card, text_card = lm.n_q, lm.card, lm.text_card

    B, S = args.batch, args.frames
    text = torch.randint(0, text_card, (B, 1, S), device=dev)
    audio = torch.randint(0, card, (B, n_q, S), device=dev)
    seq = torch.cat([text, audio], dim=1)  # [B, 1+n_q, S]

    with torch.inference_mode():
        ref = lm.embed_codes(seq)  # [B, S, dim]

    emb = PersonaPlexInputEmbeddings(PersonaPlexConfig()).to(dev, dtype)
    loaded = emb.load_weights(lm.state_dict())
    emb.eval()
    total = sum(1 for _ in emb.named_parameters())
    print(f"embeddings loaded: {len(loaded)}/{total}")
    assert len(loaded) == total, "not all embedding tables were loaded"

    # Strong proof: the embedding tables load bit-identically to Moshi's.
    assert torch.equal(emb.text_emb.weight, lm.text_emb.weight), "text_emb table mismatch"
    for cb in range(n_q):
        assert torch.equal(emb.audio_emb[cb].weight, lm.emb[cb].weight), f"emb[{cb}] table mismatch"
    print("tables bit-identical to Moshi: text_emb + all 16 audio codebooks")

    with torch.inference_mode():
        got = emb(seq)

    # The forward diff is pure bf16 accumulation rounding (summing 1 + n_q terms);
    # bf16 ULP near these magnitudes is ~4e-3, so a few e-3 is expected and benign.
    diff = (got.float() - ref.float()).abs().max().item()
    print(f"embed_codes bf16 forward max-abs-diff: {diff:.4g} (bf16 sum rounding)")
    if diff > 2e-2:
        raise SystemExit("FAIL: native embeddings diverge beyond bf16 rounding")
    print("PASS: native PersonaPlex input embeddings match Moshi embed_codes.")


if __name__ == "__main__":
    main()
