"""Tests for DFX runner metadata field exclusion."""
import json


def test_task_excluded_from_cli_args():
    """'task' field must not become --task CLI arg."""
    params = {
        "task": "voice_clone",
        "dataset_name": "seed-tts",
        "backend": "openai-audio-speech",
        "endpoint": "/v1/audio/speech",
        "percentile-metrics": "audio_rtf,audio_ttfp",
        "baseline": {"mean_audio_rtf": [0.5]},
    }
    exclude_keys = {"request_rate", "baseline", "num_prompts", "max_concurrency", "task", "enabled", "eval_phase"}
    args = []
    for key, value in params.items():
        if key in exclude_keys or value is None:
            continue
        arg_name = f"--{key.replace('_', '-')}"
        if isinstance(value, bool) and value:
            args.append(arg_name)
        elif isinstance(value, dict):
            args.extend([arg_name, json.dumps(value)])
        elif not isinstance(value, bool):
            args.extend([arg_name, str(value)])
    assert "--task" not in args
    assert "--enabled" not in args
    assert "--dataset-name" in args
