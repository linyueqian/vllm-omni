# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end native generation through the vLLM-Omni engine (offline driver).

Drives the registered 2-stage PersonaPlex pipeline (talker on paged-KV + Mimi
code2wav) via the offline ``Omni`` API, bypassing the HTTP/serving-adapter layer.
The talker generates an agent opening from the initial frame (silence user input,
Phase-1 turn-based), and the Mimi stage vocodes the agent codes to PCM.

    HF_TOKEN=... VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=2 \
        python tools/personaplex/end2end_native_omni.py --model <snapshot> --frames 50
"""

import argparse
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import soundfile as sf
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="PersonaPlex snapshot dir")
    ap.add_argument("--frames", type=int, default=50, help="agent frames to generate (~80ms each)")
    ap.add_argument("--out", default="/home/yueqian/pplex_native_omni.wav")
    args = ap.parse_args()

    from vllm_omni import Omni

    omni = Omni(
        model=args.model,
        skip_tokenizer_init=True,
        trust_remote_code=True,
        async_chunk=False,  # sync path for the first e2e
    )

    inputs = {
        "prompt_token_ids": [0],  # single initial frame; preprocess builds the embed
        "additional_information": {"max_new_tokens": [int(args.frames)]},
    }

    got_audio = False
    for stage_outputs in omni.generate([inputs]):
        out = stage_outputs.request_output
        mm = out.outputs[0].multimodal_output
        audio = mm.get("audio")
        sr_raw = mm.get("sr")
        if audio is None:
            print(f"[req {out.request_id}] no audio in multimodal_output keys={list(mm.keys())}")
            continue
        sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
        sr = int(sr_val.item() if hasattr(sr_val, "item") else sr_val)
        wav = torch.cat(audio, dim=-1) if isinstance(audio, list) else audio
        wav = wav.float().cpu().numpy().flatten()
        sf.write(args.out, wav, samplerate=sr, format="WAV")
        dur = len(wav) / sr if sr else 0.0
        rms = float((wav**2).mean() ** 0.5) if wav.size else 0.0
        print(f"[req {out.request_id}] wrote {dur:.2f}s @ {sr}Hz rms={rms:.4f} -> {args.out}")
        got_audio = True

    print("E2E PASS: native PersonaPlex pipeline produced audio." if got_audio else "E2E FAIL: no audio produced.")


if __name__ == "__main__":
    main()
