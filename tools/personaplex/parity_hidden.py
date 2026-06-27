# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Teacher-forced hidden parity: engine HeliumModel (paged-KV) vs Moshi temporal.

The engine dumps, per frame, its input 17-row stack (PPLEX_DUMP_HID + ".stacks")
and the temporal hidden it produced (PPLEX_DUMP_HID). This replays the SAME stacks
through Moshi's own temporal (lm.forward_embeddings, a full-sequence forward = the
verified-correct reference) and compares the hidden frame-by-frame. If they diverge
given identical input, the engine's incremental paged-KV temporal is the bug.

    HF_TOKEN=... CUDA_VISIBLE_DEVICES=2 python tools/personaplex/parity_hidden.py \
        --repo nvidia/personaplex-7b-v1 \
        --engine-hid /home/yueqian/engine_hid.pt
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/personaplex-7b-v1")
    ap.add_argument("--engine-hid", default="/home/yueqian/engine_hid.pt")
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
    lm = loaders.get_moshi_lm(hf_hub_download(args.repo, loaders.MOSHI_NAME), device=dev, dtype=dtype).eval()
    cfg = PersonaPlexConfig()
    emb_mod = PersonaPlexInputEmbeddings(cfg).to(dev, dtype)
    emb_mod.load_weights(lm.state_dict())
    emb_mod.eval()

    engine_hid = torch.load(args.engine_hid).float()  # [F_total, H] (prefill + decode)
    decode_stacks = torch.load(args.engine_hid + ".stacks").long()  # [F_dec, 17]
    n_q = cfg.num_audio_codebooks
    card = cfg.audio_vocab_size
    text_card = cfg.text_vocab_size
    initial = torch.tensor([text_card] + [card] * n_q, dtype=torch.long).reshape(1, 1 + n_q)  # [1,17]
    full = torch.cat([initial, decode_stacks], dim=0)  # [1+F_dec, 17]
    n = min(full.shape[0], engine_hid.shape[0])
    full = full[:n]
    print(f"frames: engine_hid={tuple(engine_hid.shape)} stacks={tuple(decode_stacks.shape)} compared={n}")

    seq = full.transpose(0, 1).unsqueeze(0).to(dev)  # [1, 17, n]
    with torch.inference_mode():
        emb = emb_mod(seq)  # [1, n, H]
        ref_hidden, _ = lm.forward_embeddings(emb)  # full-forward reference
    ref = ref_hidden.reshape(n, -1).float().cpu()
    eng = engine_hid[:n]

    print("per-frame max|engine-moshi| / cosine:")
    for t in range(min(n, 12)):
        d = (eng[t] - ref[t]).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(eng[t], ref[t], dim=0).item()
        print(f"  f{t}: maxabs={d:.4f} cos={cos:.5f}")
    overall = (eng - ref).abs().max().item()
    meancos = torch.nn.functional.cosine_similarity(eng, ref, dim=1).mean().item()
    print(f"OVERALL maxabs={overall:.4f} mean_cos={meancos:.5f}")
    print("VERDICT:", "ENGINE TEMPORAL MATCHES MOSHI (bug is elsewhere)" if meancos > 0.99
          else "ENGINE TEMPORAL DIVERGES FROM MOSHI (paged-KV temporal is the bug)")


if __name__ == "__main__":
    main()
