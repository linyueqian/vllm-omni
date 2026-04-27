# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference example for MOSS-TTS-Nano via vLLM-Omni.

Single-stage pipeline: the 0.1B AR LM and MOSS-Audio-Tokenizer-Nano codec
both run inside one generation stage. Output is 48 kHz stereo WAV.

MOSS-TTS-Nano is voice-cloning-only — every request needs a reference audio
clip (--prompt-audio) and its transcript (--prompt-text).

Usage:
  # Voice clone with reference audio (required)
  python end2end.py \\
    --text "Hello!" \\
    --prompt-audio /path/to/ref.wav \\
    --prompt-text "Transcript of the reference clip."

  # Sample reference clips ship in the upstream repo:
  #   https://github.com/OpenMOSS/MOSS-TTS-Nano/tree/main/assets/audio
  # e.g. zh_1.wav (Chinese), en_2.wav (English), jp_2.wav (Japanese).
"""

from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch
from vllm import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

# Prevent multiprocessing from re-importing CUDA in the wrong context.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm_omni import Omni  # noqa: E402

MODEL = "OpenMOSS-Team/MOSS-TTS-Nano"


def build_request(
    text: str,
    prompt_audio_path: str,
    prompt_text: str,
    mode: str = "voice_clone",
    max_new_frames: int = 375,
    seed: int | None = None,
    audio_temperature: float = 0.8,
    audio_top_k: int = 25,
    audio_top_p: float = 0.95,
    text_temperature: float = 1.0,
) -> dict:
    """Build an Omni request payload for MOSS-TTS-Nano."""
    additional: dict = {
        "text": [text],
        "mode": [mode],
        "prompt_audio_path": [str(prompt_audio_path)],
        "prompt_text": [prompt_text],
        "max_new_frames": [max_new_frames],
        "audio_temperature": [audio_temperature],
        "audio_top_k": [audio_top_k],
        "audio_top_p": [audio_top_p],
        "text_temperature": [text_temperature],
    }
    if seed is not None:
        additional["seed"] = [seed]

    return {
        "prompt": "<|im_start|>assistant\n",  # minimal placeholder prompt
        "additional_information": additional,
    }


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int = 48000) -> None:
    audio_np = waveform.float().numpy()
    # Reshape stereo: inference_stream yields interleaved [samples*2]; reshape to [samples, 2]
    if audio_np.ndim == 1 and audio_np.shape[0] % 2 == 0:
        audio_np = audio_np.reshape(-1, 2)
    sf.write(path, audio_np, sample_rate)
    print(f"  Saved {path} ({audio_np.shape}, {sample_rate} Hz)")


def main(args) -> None:
    omni = Omni(
        model=MODEL,
        deploy_config=args.deploy_config,
        stage_init_timeout=args.stage_init_timeout,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        max_tokens=4096,
        seed=args.seed if args.seed is not None else 42,
        detokenize=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Synthesizing: {args.text!r}")
    print(f"  ref_audio: {args.prompt_audio}")
    inputs = build_request(
        text=args.text,
        prompt_audio_path=args.prompt_audio,
        prompt_text=args.prompt_text,
        mode=args.mode,
        max_new_frames=args.max_new_frames,
        seed=args.seed,
        audio_temperature=args.audio_temperature,
        audio_top_k=args.audio_top_k,
        audio_top_p=args.audio_top_p,
        text_temperature=args.text_temperature,
    )
    params_list = sampling_params

    for stage_outputs in omni.generate(inputs, params_list):
        for i, req_output in enumerate(stage_outputs.request_output):
            for j, out in enumerate(req_output.outputs):
                mm = out.multimodal_output
                if mm is None:
                    print(f"  [req {i}] No audio output.")
                    continue
                audio = mm.get("audio")
                sr_tensor = mm.get("sr")
                if audio is None:
                    print(f"  [req {i}] No waveform in multimodal_output.")
                    continue
                sr = int(sr_tensor.item()) if sr_tensor is not None else 48000
                out_path = str(output_dir / f"output_{i}_{j}.wav")
                save_audio(audio.cpu(), out_path, sr)

    print("Done.")


def parse_args():
    parser = FlexibleArgumentParser(description="MOSS-TTS-Nano offline inference")
    parser.add_argument("--text", default="Hello, this is MOSS-TTS-Nano speaking.", help="Text to synthesize.")
    parser.add_argument(
        "--prompt-audio",
        required=True,
        help="Path to reference audio for voice cloning (required — MOSS-TTS-Nano is voice-cloning-only).",
    )
    parser.add_argument(
        "--prompt-text",
        required=True,
        help="Exact transcript of --prompt-audio (required for voice cloning).",
    )
    parser.add_argument("--mode", default="voice_clone", choices=["voice_clone", "continuation"])
    parser.add_argument("--max-new-frames", type=int, default=375, help="Max AR frames (~14s at default).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--audio-top-k", type=int, default=25)
    parser.add_argument("--audio-top-p", type=float, default=0.95)
    parser.add_argument("--text-temperature", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
            "moss_tts_nano_output",
        ),
        help="Directory for WAV outputs (default: ~/.cache/moss_tts_nano_output).",
    )
    parser.add_argument(
        "--deploy-config",
        default=None,
        help="Path to a deploy YAML; leave unset to auto-load vllm_omni/deploy/moss_tts_nano.yaml.",
    )
    parser.add_argument("--stage-init-timeout", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
