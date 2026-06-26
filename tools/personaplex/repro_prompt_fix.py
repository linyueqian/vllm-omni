# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce + fix the live-serving garble (voice-prompt offset/buffer misalignment).

The live moshi.server injects a voice prompt via `LMGen.step_embeddings` -> the
SEPARATE `graphed_embeddings` graph (not `graphed_main`). Our vLLM swap only
replaced `graphed_main`, so the voice frames advance LMGen's streaming offset
WITHOUT entering the vLLM re-prefill buffer -> the temporal positions misalign
with the delay/depformer bookkeeping -> the model free-runs incoherently.

This harness injects a voice (.pt) + persona prompt then feeds a user WAV, with
three modes, and compares agent text vs pure-Moshi:
  --mode moshi   pure moshi (reference ground truth)
  --mode broken  vLLM swaps graphed_main only (reproduces the garble)
  --mode fixed   vLLM swaps graphed_main AND graphed_embeddings into one buffer

    HF_TOKEN=... VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=1 \
        python tools/personaplex/repro_prompt_fix.py --hf /home/yueqian/helium_hf \
            --voices <voices_dir> --mode fixed --seconds 12
"""

import argparse
import os

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--voices", required=True, help="dir with NATF2.pt etc.")
    ap.add_argument("--voice", default="NATF2.pt")
    ap.add_argument("--input-wav", required=True)
    ap.add_argument("--mode", choices=["moshi", "broken", "fixed"], required=True)
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()

    import sentencepiece
    from huggingface_hub import hf_hub_download
    from moshi.models import LMGen, loaders
    from moshi.models.lm import _iterate_audio as iter_audio
    from moshi.models.lm import encode_from_sphn, load_audio

    dev = "cuda"
    mimi = loaders.get_mimi(hf_hub_download("nvidia/personaplex-7b-v1", loaders.MIMI_NAME), dev)
    lm = loaders.get_moshi_lm(hf_hub_download("nvidia/personaplex-7b-v1", loaders.MOSHI_NAME), device=dev).eval()
    tok = sentencepiece.SentencePieceProcessor(hf_hub_download("nvidia/personaplex-7b-v1", loaders.TEXT_TOKENIZER_NAME))
    frame = int(mimi.sample_rate / mimi.frame_rate)

    llm = None
    if args.mode != "moshi":
        from vllm import LLM

        llm = LLM(
            model=args.hf,
            runner="pooling",
            convert="embed",
            dtype="bfloat16",
            enforce_eager=True,
            gpu_memory_utilization=0.25,
            max_model_len=3000,
            enable_prompt_embeds=True,
            pooler_config={"pooling_type": "LAST", "use_activation": False},
        )

        import moshi.models.lm as lm_mod

        orig_init = lm_mod.LMGen._init_streaming_state
        fix = args.mode == "fixed"

        def patched_init(self, bsz):
            state = orig_init(self, bsz)
            buf: list[torch.Tensor] = []

            def vfwd(emb):
                buf.append(emb.detach())
                full = torch.cat(buf, dim=1)
                out = llm.embed({"prompt_embeds": full.squeeze(0).to(torch.bfloat16).cpu()})
                h = torch.tensor(np.asarray(out[0].outputs.embedding), device=emb.device, dtype=emb.dtype).reshape(
                    1, 1, -1
                )
                return h, self.lm_model.text_linear(h)[:, None]

            state.graphed_main = lambda codes: vfwd(self.lm_model.embed_codes(codes))
            if fix:
                # ALSO route the voice-prompt path (step_embeddings -> graphed_embeddings)
                # through vLLM into the SAME buffer, keeping offset and buffer aligned.
                state.graphed_embeddings = vfwd
            return state

        lm_mod.LMGen._init_streaming_state = patched_init

    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=dev,
        frame_rate=mimi.frame_rate,
        use_sampling=False,
    )
    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    # Prompt injection (voice .pt + persona), exactly like the live server.
    lm_gen.load_voice_prompt_embeddings(os.path.join(args.voices, args.voice))
    persona = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
    lm_gen.text_prompt_tokens = tok.encode(f"<system> {persona} <system>")
    mimi.reset_streaming()
    lm_gen.reset_streaming()
    lm_gen.step_system_prompts(mimi)
    mimi.reset_streaming()

    user = load_audio(args.input_wav, mimi.sample_rate)[:, : int(args.seconds * mimi.sample_rate)]
    texts = []
    for enc in encode_from_sphn(mimi, iter_audio(user, sample_interval_size=frame, pad=True), max_batch=1):
        for c in range(enc.shape[-1]):
            tokens = lm_gen.step(enc[:, :, c : c + 1])
            if tokens is None:
                continue
            tid = int(tokens[0, 0, 0].item())
            if tid not in (0, 3):
                texts.append(tok.id_to_piece(tid).replace("▁", " "))
    print(f"[mode={args.mode}] agent text: {''.join(texts)!r}")


if __name__ == "__main__":
    main()
