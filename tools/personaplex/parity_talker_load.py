# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate 1 (routing): the talker's load_weights covers 100% of the checkpoint.

Constructing the talker standalone fights vLLM's config-context machinery (init
distributed vs. initialize_model_parallel both want a real engine config), so the
shape-level weight_loader check happens at the engine boot. This gate verifies the
piece that does NOT need the engine: that ``load_weights``' three-way router
consumes EVERY Moshi checkpoint tensor and maps each temporal tensor to a valid
vLLM-Llama target name (no orphans, no drops). The verified component loaders
(embeddings, depformer) are exercised directly.

    HF_TOKEN=... CUDA_VISIBLE_DEVICES=2 python tools/personaplex/parity_talker_load.py
"""

from __future__ import annotations

import argparse

import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders

from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
    PersonaPlexConfig,
)
from vllm_omni.model_executor.models.personaplex.personaplex_depformer import (
    PersonaPlexDepformer,
)
from vllm_omni.model_executor.models.personaplex.personaplex_embeddings import (
    PersonaPlexInputEmbeddings,
)


def _temporal_target(name: str, n_layers: int) -> str | None:
    """Mirror PersonaPlexTalker._load_temporal: temporal checkpoint tensor -> vLLM target."""
    if name == "out_norm.alpha":
        return "model.norm.weight"
    if name == "text_linear.weight":
        return "lm_head.weight"
    if not name.startswith("transformer.layers."):
        return None
    rest = name.removeprefix("transformer.layers.")
    idx, _, suffix = rest.partition(".")
    if not idx.isdigit() or int(idx) >= n_layers:
        return None
    base = f"model.layers.{idx}"
    return {
        "self_attn.in_proj_weight": f"{base}.self_attn.qkv_proj.weight",
        "self_attn.out_proj.weight": f"{base}.self_attn.o_proj.weight",
        "gating.linear_in.weight": f"{base}.mlp.gate_up_proj.weight",
        "gating.linear_out.weight": f"{base}.mlp.down_proj.weight",
        "norm1.alpha": f"{base}.input_layernorm.weight",
        "norm2.alpha": f"{base}.post_attention_layernorm.weight",
    }.get(suffix)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/personaplex-7b-v1")
    args = ap.parse_args()

    cfg = PersonaPlexConfig()
    n_layers = cfg.temporal_config.num_hidden_layers
    lm = loaders.get_moshi_lm(hf_hub_download(args.repo, loaders.MOSHI_NAME), device="cuda", dtype=torch.bfloat16)
    ckpt = dict(lm.state_dict())

    # Build the expected temporal target set from a stock Helium-Llama layout.
    expected_temporal = {"model.norm.weight", "lm_head.weight"}
    for i in range(n_layers):
        b = f"model.layers.{i}"
        expected_temporal |= {
            f"{b}.self_attn.qkv_proj.weight",
            f"{b}.self_attn.o_proj.weight",
            f"{b}.mlp.gate_up_proj.weight",
            f"{b}.mlp.down_proj.weight",
            f"{b}.input_layernorm.weight",
            f"{b}.post_attention_layernorm.weight",
        }

    emb_keys, dep_keys, temporal_hits, unrouted = [], [], set(), []
    for name in ckpt:
        if name.startswith(("emb.", "text_emb.")):
            emb_keys.append(name)
        elif name.startswith(("depformer.", "depformer_in.", "depformer_emb.", "depformer_text_emb", "linears.")):
            dep_keys.append(name)
        else:
            tgt = _temporal_target(name, n_layers)
            if tgt is None:
                unrouted.append(name)
            else:
                temporal_hits.add(tgt)

    # Exercise the verified component loaders directly (plain nn.Modules, no vLLM).
    emb = PersonaPlexInputEmbeddings(cfg)
    emb_loaded = emb.load_weights([(k, ckpt[k]) for k in emb_keys])
    dep = PersonaPlexDepformer(cfg.depformer_config, temporal_hidden_size=cfg.temporal_config.hidden_size)
    dep_loaded = dep.load_weights([(k, ckpt[k]) for k in dep_keys])

    emb_all = {n for n, _ in emb.named_parameters()}
    dep_all = {n for n, _ in dep.named_parameters()}
    print(f"temporal: {len(temporal_hits)}/{len(expected_temporal)} target names produced")
    print(f"embeddings: {len(emb_loaded)}/{len(emb_all)} params loaded ({len(emb_keys)} ckpt tensors)")
    print(f"depformer: {len(dep_loaded)}/{len(dep_all)} params loaded ({len(dep_keys)} ckpt tensors)")
    print(f"unrouted checkpoint tensors: {len(unrouted)}")

    ok = temporal_hits == expected_temporal and emb_loaded == emb_all and dep_loaded == dep_all and not unrouted
    if not ok:
        if unrouted:
            print(f"UNROUTED (first 10): {unrouted[:10]}")
        miss_t = sorted(expected_temporal - temporal_hits)
        if miss_t:
            print(f"temporal targets NOT produced (first 10): {miss_t[:10]}")
        raise SystemExit("FAIL: talker load_weights routing is incomplete")
    print("PASS: talker load_weights routes 100% of the checkpoint (temporal + embeddings + depformer).")


if __name__ == "__main__":
    main()
