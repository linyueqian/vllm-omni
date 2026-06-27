# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end native generation through the vLLM-Omni engine (offline driver).

Drives the registered 2-stage PersonaPlex pipeline (talker on paged-KV + Mimi
code2wav) via the offline ``Omni`` API, bypassing the HTTP/serving layer. The
input WAV is Mimi-encoded to the user-audio code stream and fed (turn-based,
Phase 1) so the agent responds coherently; the Mimi stage vocodes the agent codes.

    HF_TOKEN=... VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=2 \
        python tools/personaplex/end2end_native_omni.py --model <snap> \
            --input-wav <wav> --seconds 6
"""

import argparse
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import soundfile as sf
import torch


def _encode_user_codes(model: str, input_wav: str, seconds: float) -> list[list[int]]:
    """Mimi-encode the user WAV -> per-frame 8-codebook user codes [F][8]."""
    from moshi.models import loaders
    from moshi.models.lm import _iterate_audio as iter_audio
    from moshi.models.lm import encode_from_sphn, load_audio

    # Resolve the Mimi weight file: local snapshot dir, else HF repo id.
    local = os.path.join(model, loaders.MIMI_NAME)
    if os.path.isfile(local):
        mimi_path = local
    else:
        from huggingface_hub import hf_hub_download

        mimi_path = hf_hub_download(model, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_path, "cuda")
    mimi.set_num_codebooks(8)
    frame = int(mimi.sample_rate / mimi.frame_rate)
    user = load_audio(input_wav, mimi.sample_rate)[:, : int(seconds * mimi.sample_rate)]
    mimi.streaming_forever(1)
    mimi.reset_streaming()
    cols: list[list[int]] = []
    for enc in encode_from_sphn(mimi, iter_audio(user, sample_interval_size=frame, pad=True), max_batch=1):
        for c in range(enc.shape[-1]):
            cols.append(enc[0, :8, c].to(torch.long).tolist())
    del mimi
    return cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="PersonaPlex snapshot dir")
    ap.add_argument("--input-wav", default=None, help="user audio; if unset, silence opening")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--frames", type=int, default=50, help="frames when no input-wav")
    ap.add_argument("--out", default="/home/yueqian/pplex_native_omni.wav")
    args = ap.parse_args()

    add_info: dict = {}
    if args.input_wav:
        user_codes = _encode_user_codes(args.model, args.input_wav, args.seconds)
        n_frames = len(user_codes)
        # Pass as a tensor (round-trips cleanly through the omni payload serializer;
        # a nested list mangles through list_data).
        add_info["pplex_user_codes"] = torch.tensor(user_codes, dtype=torch.long)
        print(f"encoded user stream: {n_frames} frames")
    else:
        n_frames = int(args.frames)
    add_info["max_new_tokens"] = [n_frames]

    from vllm_omni import Omni

    omni = Omni(model=args.model, skip_tokenizer_init=True, trust_remote_code=True)
    inputs = {"prompt_token_ids": [0], "additional_information": add_info}

    got_audio = False
    for stage_outputs in omni.generate([inputs]):
        out = stage_outputs.request_output
        mm = out.outputs[0].multimodal_output
        audio = mm.get("audio")
        sr_raw = mm.get("sr")
        if audio is None:
            print(f"[req {out.request_id}] no audio; mm keys={list(mm.keys())}")
            continue
        sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
        sr = int(sr_val.item() if hasattr(sr_val, "item") else sr_val)
        wav = torch.cat(audio, dim=-1) if isinstance(audio, list) else audio
        wav = wav.float().cpu().numpy().flatten()
        sf.write(args.out, wav, samplerate=sr, format="WAV")
        dur = len(wav) / sr if sr else 0.0
        rms = float((wav**2).mean() ** 0.5) if wav.size else 0.0
        peak = float(abs(wav).max()) if wav.size else 0.0
        print(f"[req {out.request_id}] wrote {dur:.2f}s @ {sr}Hz rms={rms:.4f} peak={peak:.3f} -> {args.out}")
        got_audio = True

    print("E2E PASS: native PersonaPlex pipeline produced audio." if got_audio else "E2E FAIL: no audio produced.")


if __name__ == "__main__":
    main()
