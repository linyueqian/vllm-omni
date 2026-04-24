#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import soundfile as sf
from vllm import SamplingParams
from vllm.multimodal.media.audio import load_audio

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.models.cosyvoice3.config import CosyVoice3Config
from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer


def _ensure_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    try:
        return list(x)
    except TypeError:
        return [x]


def _concat_audio(audio_val: Any) -> np.ndarray:
    if isinstance(audio_val, list):
        parts: list[np.ndarray] = []
        for item in audio_val:
            arr = _concat_audio(item)
            if arr.size > 0:
                parts.append(arr)
        if not parts:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(parts, axis=0)

    if hasattr(audio_val, "detach"):
        audio_val = audio_val.detach()
    if hasattr(audio_val, "cpu"):
        audio_val = audio_val.cpu()
    if hasattr(audio_val, "float"):
        audio_val = audio_val.float()
    if hasattr(audio_val, "numpy"):
        audio_val = audio_val.numpy()
    return np.asarray(audio_val, dtype=np.float32).reshape(-1)


def _get_sample_rate(audio_mm: dict[str, Any]) -> int:
    sr_val = audio_mm.get("sr", 24000)
    if isinstance(sr_val, list) and sr_val:
        sr_val = sr_val[-1]
    if hasattr(sr_val, "item"):
        sr_val = sr_val.item()
    return int(sr_val)


def _tail_peak(audio: np.ndarray, sr: int, tail_ms: float = 50.0) -> float:
    if audio.size == 0 or sr <= 0:
        return 0.0
    tail = max(1, int(round(sr * tail_ms / 1000.0)))
    return float(np.max(np.abs(audio[-tail:])))


def _stage_outputs(omni: Omni, stage_id: int) -> list[Any]:
    stage_clients = getattr(getattr(omni, "engine", None), "stage_clients", None)
    if stage_clients is None:
        raise RuntimeError("Unable to locate stage_clients on Omni engine")
    outputs = getattr(stage_clients[stage_id], "engine_outputs", None)
    return _ensure_list(outputs)


def _stage0_completion_map(omni: Omni) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in _stage_outputs(omni, 0):
        request_id = getattr(item, "request_id", None)
        outputs = _ensure_list(getattr(item, "outputs", None))
        if request_id is None or not outputs:
            continue
        completion = outputs[0]
        token_ids = [int(tok) for tok in _ensure_list(getattr(completion, "token_ids", None))]
        result[str(request_id)] = {
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "last_10_token_ids": token_ids[-10:],
            "finish_reason": getattr(completion, "finish_reason", None),
            "stop_reason": getattr(completion, "stop_reason", None),
        }
    return result


def _bucket(
    duration_s: float,
    tail_peak: float,
    *,
    quiet_peak_threshold: float,
    cut_peak_threshold: float,
) -> str:
    if duration_s < 2.0 and tail_peak <= quiet_peak_threshold:
        return "short_quiet"
    if 4.0 <= duration_s <= 8.0 and tail_peak <= quiet_peak_threshold:
        return "normal"
    if tail_peak >= cut_peak_threshold:
        return "remaining_cut"
    return "other"


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    token_counts = [int(r["token_count"]) for r in rows]
    finish_reasons: dict[str, int] = {}
    for row in rows:
        key = str(row.get("finish_reason"))
        finish_reasons[key] = finish_reasons.get(key, 0) + 1
    return {
        "count": len(rows),
        "token_count_min": min(token_counts),
        "token_count_max": max(token_counts),
        "token_count_mean": round(mean(token_counts), 2),
        "finish_reasons": finish_reasons,
        "request_ids": [row["request_id"] for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe residual CosyVoice3 issue #3090 stage-0 behavior.")
    parser.add_argument("--model", required=True, help="CosyVoice3 model directory")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer directory, usually <model>/CosyVoice-BlankEN")
    parser.add_argument("--audio-path", required=True, help="Reference prompt wav path")
    parser.add_argument("--prompt", default="Hello, this is a test of the CosyVoice system capability.")
    parser.add_argument(
        "--prompt-text",
        default="You are a helpful assistant.<|endofprompt|>Testing my voices. Why should I not?",
    )
    parser.add_argument("--deploy-config", default=None)
    parser.add_argument("-n", "--num-requests", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--repetition-penalty", type=float, default=1.0001)
    parser.add_argument("--stop-token-id", type=int, default=6562)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--allowed-token-cap", type=int, default=0)
    parser.add_argument("--quiet-peak-threshold", type=float, default=0.02)
    parser.add_argument("--cut-peak-threshold", type=float, default=0.5)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--save-dir", default=None, help="Optional directory to save per-request wavs")
    args = parser.parse_args()

    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(args.tokenizer)
    if not os.path.exists(args.audio_path):
        raise FileNotFoundError(args.audio_path)
    if args.deploy_config is not None and not os.path.exists(args.deploy_config):
        raise FileNotFoundError(args.deploy_config)

    audio_signal, sr = load_audio(args.audio_path, sr=None)
    if sr < 16000:
        raise ValueError(f"Reference audio sample rate too low: {sr}")
    audio_data = (audio_signal.astype(np.float32), sr)

    config = CosyVoice3Config()
    tokenizer = get_qwen_tokenizer(
        token_path=args.tokenizer,
        skip_special_tokens=config.skip_special_tokens,
        version=config.version,
    )
    base_len = max(1, len(tokenizer.encode(args.prompt, allowed_special=config.allowed_special)))
    min_tokens = int(base_len * config.min_token_text_ratio)

    omni = Omni(
        model=args.model,
        deploy_config=args.deploy_config,
        tokenizer=args.tokenizer,
        log_stats=False,
    )
    try:
        sampling_params_list = copy.deepcopy(omni.default_sampling_params_list)
        stage0 = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            min_tokens=min_tokens,
            max_tokens=args.max_tokens,
            stop_token_ids=[args.stop_token_id],
            allowed_token_ids=list(range(args.allowed_token_cap)) if args.allowed_token_cap > 0 else None,
            detokenize=False,
        )
        sampling_params_list[0] = stage0

        prompts = [
            {
                "prompt": args.prompt,
                "multi_modal_data": {"audio": audio_data},
                "modalities": ["audio"],
                "mm_processor_kwargs": {
                    "prompt_text": args.prompt_text,
                    "sample_rate": sr,
                },
            }
            for _ in range(args.num_requests)
        ]

        outputs = list(omni.generate(prompts, sampling_params_list=sampling_params_list))
        stage0_map = _stage0_completion_map(omni)

        if args.save_dir:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        for idx, output in enumerate(outputs):
            request_id = str(getattr(output, "request_id", f"req-{idx}"))
            audio_mm = getattr(output, "multimodal_output", None) or {}
            audio = _concat_audio(audio_mm.get("audio"))
            out_sr = _get_sample_rate(audio_mm)
            duration_s = float(audio.size / out_sr) if out_sr > 0 else 0.0
            last50_peak = _tail_peak(audio, out_sr)
            stage0 = stage0_map.get(request_id, {})
            token_ids = stage0.get("token_ids", [])
            final_token = token_ids[-1] if token_ids else None
            row = {
                "index": idx,
                "request_id": request_id,
                "duration_s": round(duration_s, 6),
                "last50_peak": round(last50_peak, 6),
                "token_count": int(stage0.get("token_count", 0)),
                "last_10_token_ids": stage0.get("last_10_token_ids", []),
                "finish_reason": stage0.get("finish_reason"),
                "stop_reason": stage0.get("stop_reason"),
                "final_token_id": final_token,
                "final_token_is_6562": final_token == args.stop_token_id,
                "final_token_ge_6562": final_token is not None and int(final_token) >= args.stop_token_id,
            }
            row["bucket"] = _bucket(
                duration_s,
                last50_peak,
                quiet_peak_threshold=args.quiet_peak_threshold,
                cut_peak_threshold=args.cut_peak_threshold,
            )
            rows.append(row)

            if args.save_dir:
                out_path = Path(args.save_dir) / f"{idx:02d}_{request_id}.wav"
                sf.write(out_path, audio, out_sr)

        groups = {
            "remaining_cut": [r for r in rows if r["bucket"] == "remaining_cut"],
            "short_quiet": [r for r in rows if r["bucket"] == "short_quiet"],
            "normal": [r for r in rows if r["bucket"] == "normal"],
            "other": [r for r in rows if r["bucket"] == "other"],
        }
        report = {
            "prompt": args.prompt,
            "base_len": base_len,
            "min_tokens": min_tokens,
            "max_tokens": args.max_tokens,
            "allowed_token_cap": args.allowed_token_cap,
            "num_requests": args.num_requests,
            "groups": {name: _summarize_group(rows) for name, rows in groups.items()},
            "rows": rows,
        }
    finally:
        omni.shutdown()

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
