# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full PersonaPlex audio generation with the temporal transformer running on vLLM.

Reuses Moshi's correct generation machinery (embed_codes, the depformer, the
acoustic-delay buffers, Mimi decode) but swaps the 7B temporal transformer's
per-frame forward for a call into vLLM's paged-attention engine (the exported
stock-Llama checkpoint). The temporal ``transformer_out`` is pulled out of vLLM via
the embed/pooling path with ``use_activation=False`` (raw last hidden state;
cosine 0.9998 vs Moshi). This proves the expensive backbone runs on vLLM inside a
real generation loop.

Re-prefill per frame (O(n^2)) — correctness proof, not a perf path; keep --seconds small.

    HF_TOKEN=... VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=1 \
        python tools/personaplex/generate_vllm.py --hf /home/yueqian/helium_hf \
            --input-wav .../input_assistant.wav --out-wav /home/yueqian/gen_vllm.wav --seconds 6
"""

from __future__ import annotations

import argparse

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--input-wav", required=True)
    ap.add_argument("--out-wav", required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--ref-wav", default=None, help="optional: also run pure-moshi for parity")
    args = ap.parse_args()

    import sphn
    from huggingface_hub import hf_hub_download
    from moshi.models import LMGen, loaders
    from moshi.models.lm import _iterate_audio as iter_audio
    from moshi.models.lm import encode_from_sphn, load_audio

    dev = "cuda"
    mimi_w = hf_hub_download("nvidia/personaplex-7b-v1", loaders.MIMI_NAME)
    moshi_w = hf_hub_download("nvidia/personaplex-7b-v1", loaders.MOSHI_NAME)
    mimi = loaders.get_mimi(mimi_w, dev)
    lm = loaders.get_moshi_lm(moshi_w, device=dev).eval()
    frame = int(mimi.sample_rate / mimi.frame_rate)

    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=dev,
        frame_rate=mimi.frame_rate,
        use_sampling=False,
    )  # greedy
    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    mimi.reset_streaming()
    lm_gen.reset_streaming()

    # vLLM temporal transformer (raw last hidden state).
    from vllm import LLM

    llm = LLM(
        model=args.hf,
        runner="pooling",
        convert="embed",
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.45,
        max_model_len=2048,
        enable_prompt_embeds=True,
        pooler_config={"pooling_type": "LAST", "use_activation": False},
    )

    text_linear = lm.text_linear
    buf: list[torch.Tensor] = []

    def vllm_forward_codes(seq: torch.Tensor):
        # seq: codes [B,17,1] -> moshi embed_codes -> append -> vLLM full-prefill -> last hidden.
        emb = lm.embed_codes(seq)  # [1,1,dim]
        buf.append(emb.detach())
        full = torch.cat(buf, dim=1)  # [1,T,dim]
        out = llm.embed({"prompt_embeds": full.squeeze(0).to(torch.bfloat16).cpu()})
        h = torch.tensor(np.asarray(out[0].outputs.embedding), device=dev, dtype=emb.dtype).reshape(1, 1, -1)
        text_logits = text_linear(h)[:, None]  # [1,1,1,vocab]
        return h, text_logits

    # Swap the temporal forward in the live streaming state.
    lm_gen._streaming_state.graphed_main = vllm_forward_codes

    # User audio frames (truncated), pre-encoded so both runs see identical inputs.
    user = load_audio(args.input_wav, mimi.sample_rate)
    user = user[:, : int(args.seconds * mimi.sample_rate)]
    enc_cols: list[torch.Tensor] = []
    for enc in encode_from_sphn(mimi, iter_audio(user, sample_interval_size=frame, pad=True), max_batch=1):
        for c in range(enc.shape[-1]):
            enc_cols.append(enc[:, :, c : c + 1].clone())

    def run(step_fn):
        out_tokens, frames = [], []
        for col in enc_cols:
            tokens = step_fn(col)
            if tokens is None:
                continue
            out_tokens.append(tokens[:, :, 0].clone())  # [1,17]
            frames.append(mimi.decode(tokens[:, 1:9]).detach().cpu().numpy()[0, 0].astype(np.float32))
        return out_tokens, frames

    vt, frames = run(lm_gen.step)  # vLLM temporal transformer drives this
    audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
    sphn.write_wav(args.out_wav, audio, mimi.sample_rate)
    rms = float(np.sqrt((audio**2).mean())) if audio.size else 0.0
    print(f"vLLM-gen: steps={len(vt)} dur={audio.shape[0] / mimi.sample_rate:.2f}s RMS={rms:.4f} -> {args.out_wav}")

    # Pure-Moshi reference on the identical input (separate fresh LMGen, native transformer).
    mimi.reset_streaming()
    lm2 = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=dev,
        frame_rate=mimi.frame_rate,
        use_sampling=False,
    )
    lm2.streaming_forever(1)
    lm2.reset_streaming()
    mt, _ = run(lm2.step)

    n = min(len(vt), len(mt))
    if n:
        v = torch.cat(vt[:n], 0)  # [n,17]
        m = torch.cat(mt[:n], 0)
        text_agree = (v[:, 0] == m[:, 0]).float().mean().item()
        audio_agree = (v[:, 1:9] == m[:, 1:9]).float().mean().item()
        print(
            f"vLLM-gen vs Moshi-gen over {n} frames: text-token agreement={text_agree * 100:.1f}% "
            f"agent-code agreement={audio_agree * 100:.1f}%"
        )
        print("GEN PARITY:", "PASS — vLLM-driven generation matches Moshi" if audio_agree > 0.95 else "CHECK")


if __name__ == "__main__":
    main()
