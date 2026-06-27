# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end composition gate: native PersonaPlex components in the real loop.

The unit parity harnesses (`parity_depformer.py`, `parity_embeddings.py`) verify
each native component in isolation. This drives Moshi's actual per-frame
generation loop -- with its acoustic-delay buffers, code feedback, and Mimi
decode -- but swaps in BOTH native components:

* the temporal forward's ``embed_codes`` -> :class:`PersonaPlexInputEmbeddings`
* the depformer step (``graphed_depth``) -> :class:`PersonaPlexDepformer`

and compares the generated 17-row token stream against pure Moshi over the same
input. This confirms the native components compose correctly inside the exact
per-frame machinery that the talker's ``talker_mtp`` must replicate. The temporal
transformer stays Moshi's here (already proven on vLLM in Milestone B), isolating
the two newly-ported pieces.

    HF_TOKEN=... CUDA_VISIBLE_DEVICES=2 python tools/personaplex/end2end_native.py \
        --input-wav <wav> --seconds 6
"""

from __future__ import annotations

import argparse

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/personaplex-7b-v1")
    ap.add_argument("--input-wav", required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--out-wav", default=None)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from moshi.models import LMGen, loaders
    from moshi.models.lm import _iterate_audio as iter_audio
    from moshi.models.lm import encode_from_sphn, load_audio

    from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
        PersonaPlexConfig,
    )
    from vllm_omni.model_executor.models.personaplex.personaplex_depformer import (
        PersonaPlexDepformer,
    )
    from vllm_omni.model_executor.models.personaplex.personaplex_embeddings import (
        PersonaPlexInputEmbeddings,
    )

    dev, dtype = "cuda", torch.bfloat16
    mimi = loaders.get_mimi(hf_hub_download(args.repo, loaders.MIMI_NAME), dev)
    lm = loaders.get_moshi_lm(hf_hub_download(args.repo, loaders.MOSHI_NAME), device=dev, dtype=dtype).eval()
    frame = int(mimi.sample_rate / mimi.frame_rate)
    mimi.streaming_forever(1)  # enter streaming mode so reset_streaming() is valid

    # Build + load the native components from the same checkpoint.
    cfg = PersonaPlexConfig()
    nat_emb = PersonaPlexInputEmbeddings(cfg).to(dev, dtype)
    nat_emb.load_weights(lm.state_dict())
    nat_emb.eval()
    nat_dep = PersonaPlexDepformer(cfg.depformer_config, temporal_hidden_size=lm.dim, text_card=lm.text_card).to(
        dev, dtype
    )
    nat_dep.load_weights(lm.state_dict())
    nat_dep.eval()

    def native_forward_codes(seq: torch.Tensor):
        # Swap embed_codes for the native embeddings; keep Moshi's temporal forward.
        emb = nat_emb(seq)
        return lm.forward_embeddings(emb)

    def native_depformer_step(text_token, transformer_out, audio_tokens, audio_provided):
        return nat_dep(text_token, transformer_out, audio_tokens=audio_tokens, audio_provided=audio_provided)

    # Pre-encode user audio once so both runs see identical inputs.
    user = load_audio(args.input_wav, mimi.sample_rate)[:, : int(args.seconds * mimi.sample_rate)]
    enc_cols: list[torch.Tensor] = []
    for enc in encode_from_sphn(mimi, iter_audio(user, sample_interval_size=frame, pad=True), max_batch=1):
        for c in range(enc.shape[-1]):
            enc_cols.append(enc[:, :, c : c + 1].clone())

    def make_lmgen() -> LMGen:
        g = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
            sample_rate=mimi.sample_rate,
            device=dev,
            frame_rate=mimi.frame_rate,
            use_sampling=False,  # greedy
        )
        g.streaming_forever(1)
        g.reset_streaming()
        return g

    def run(step_fn, decode: bool):
        toks, frames = [], []
        for col in enc_cols:
            tokens = step_fn(col)
            if tokens is None:
                continue
            toks.append(tokens[:, :, 0].clone())  # [1, 17]
            if decode:
                frames.append(mimi.decode(tokens[:, 1:9]).detach().cpu().numpy()[0, 0].astype(np.float32))
        return toks, frames

    # --- native components driving the loop ---
    mimi.reset_streaming()
    nat_gen = make_lmgen()
    nat_gen._streaming_state.graphed_main = native_forward_codes
    nat_gen._streaming_state.graphed_depth = native_depformer_step
    nat_tokens, frames = run(nat_gen.step, decode=args.out_wav is not None)

    # --- pure Moshi reference on identical input ---
    mimi.reset_streaming()
    ref_gen = make_lmgen()
    ref_tokens, _ = run(ref_gen.step, decode=False)

    n = min(len(nat_tokens), len(ref_tokens))
    nat = torch.cat(nat_tokens[:n], 0)  # [n, 17]
    ref = torch.cat(ref_tokens[:n], 0)
    text_agree = (nat[:, 0] == ref[:, 0]).float().mean().item()
    code_agree = (nat[:, 1:9] == ref[:, 1:9]).float().mean().item()
    print(f"frames compared: {n}")
    print(f"text-token agreement:  {text_agree * 100:.1f}%")
    print(f"agent-code agreement (cb0..7): {code_agree * 100:.1f}%")

    import os as _os

    if _os.environ.get("PPLEX_DUMP_CODES"):
        # Save pure-Moshi agent codes [F, 8] (rows 1..8) as the parity reference.
        torch.save(ref[:, 1:9].cpu(), _os.environ["PPLEX_DUMP_CODES"])
        print(f"saved moshi agent codes {tuple(ref[:, 1:9].shape)} -> {_os.environ['PPLEX_DUMP_CODES']}")

    if args.out_wav and frames:
        import sphn

        audio = np.concatenate(frames)
        sphn.write_wav(args.out_wav, audio, mimi.sample_rate)
        print(f"wrote {audio.shape[0] / mimi.sample_rate:.2f}s -> {args.out_wav}")

    # Gate: same bar as Milestone B (text near-exact, codes >=95%); residual is
    # bf16 greedy-tie cascade, not a port error (the unit tests prove fidelity).
    ok = text_agree > 0.98 and code_agree > 0.95
    print("PASS: native components compose correctly in the full loop" if ok else "CHECK: agreement below bar")
    if not ok:
        raise SystemExit("FAIL: native composition diverges beyond bf16-tie tolerance")


if __name__ == "__main__":
    main()
