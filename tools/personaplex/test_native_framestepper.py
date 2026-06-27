# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone test of the NATIVE PersonaPlex FrameStepper (duplex serve backend).

Drives PersonaPlexEngine(use_native_components=True) frame-by-frame over an input
WAV and ASR-checks the agent audio -- the duplex-serve analog of the omni-engine
offline e2e. Verifies the native embed_codes + depformer (swapped into LMGen)
produce coherent speech in the lockstep loop.

    HF_TOKEN=... CUDA_VISIBLE_DEVICES=2 python tools/personaplex/test_native_framestepper.py \
        --input-wav <wav> --seconds 6
"""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-wav", required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--tail", type=float, default=10.0, help="trailing silence seconds for the agent reply")
    ap.add_argument("--out", default="/home/yueqian/pplex_native_fs.wav")
    ap.add_argument("--persona", default=None)
    ap.add_argument("--native", type=int, default=1)
    args = ap.parse_args()

    from vllm_omni.experimental.fullduplex.personaplex.config import PersonaPlexConfig
    from vllm_omni.experimental.fullduplex.personaplex.engine import PersonaPlexEngine

    cfg = PersonaPlexConfig(use_native_components=bool(args.native), greedy=True, seed=42)
    eng = PersonaPlexEngine(cfg).load()
    eng.open_session(persona=args.persona)
    fs = eng.frame_size
    sr = eng.sample_rate

    # Load user audio at the codec rate (Moshi's loader handles resampling), mono.
    from moshi.models.lm import load_audio

    wav = np.asarray(load_audio(args.input_wav, sr), dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    wav = wav[: int(args.seconds * sr)]
    # Lockstep duplex: after the user finishes, keep stepping with silence so the
    # agent can speak its response (the agent listens during the user turn, then talks).
    wav = np.concatenate([wav, np.zeros(int(args.tail * sr), dtype=np.float32)])
    n_frames = len(wav) // fs
    print(f"native={bool(args.native)} frames={n_frames} (user {int(args.seconds * sr) // fs} + tail) sr={sr}")

    out: list[np.ndarray] = []
    pieces: list[str] = []
    for i in range(n_frames):
        frame = wav[i * fs : (i + 1) * fs]
        fo = eng.step(frame)
        if fo.audio is not None:
            out.append(fo.audio)
        if fo.text:
            pieces.append(fo.text)
    audio = np.concatenate(out) if out else np.zeros(fs, dtype=np.float32)
    sf.write(args.out, audio, sr)
    rms = float(np.sqrt((audio**2).mean()))
    peak = float(np.abs(audio).max())
    print(f"wrote {len(audio)/sr:.2f}s rms={rms:.4f} peak={peak:.3f} -> {args.out}")
    print(f"inner-monologue text: {''.join(pieces)[:200]!r}")

    import os
    import tempfile

    import whisper

    m = whisper.load_model("small", device="cuda")
    for a, b in [(0, 12), (0, 30)]:
        seg = audio[int(a * sr) : int(b * sr)]
        if seg.size == 0:
            continue
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tf.name, seg, sr)
        r = m.transcribe(tf.name, language="en", fp16=True)
        os.unlink(tf.name)
        print(f"ASR[{a}:{b}s]={r['text'].strip()[:200]!r}")


if __name__ == "__main__":
    main()
