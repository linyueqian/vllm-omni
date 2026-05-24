# Family-specific knobs

Beyond the generic 7-tuple `(model, task, deploy_yaml, dataset, num_prompts,
max_concurrency, num_warmups)` in `SKILL.md`, each model family carries its
own knobs that change the measured cell. Setting one wrong is a common
path to a wrong-cell bench.

## TTS (Qwen3-TTS, MOSS-TTS-Nano, GLM-TTS)

| Knob               | Where                                             | Notes |
|--------------------|---------------------------------------------------|-------|
| `task_type`        | bench `--extra-body`                              | CustomVoice variants only: `CustomVoice` / `VoiceDesign`. Voice_clone (Base) reads ref-audio from the dataset row and needs no `extra_body`. |
| `voice`            | bench `--extra-body`                              | Required when `task_type=CustomVoice`. Common values: `Vivian`, `Cherry`, etc. |
| `language`         | bench `--extra-body`                              | `English` / `Chinese` — affects the front-end text normalizer and Stage-0 token count. |
| `deploy_yaml`      | `vllm serve --deploy-config`                      | `qwen3_tts.yaml` vs `qwen3_tts_high_concurrency.yaml`. The high-c YAML enables `code_predictor_prefix_graphs` AND a larger Stage-0 batch (S0=64 vs S0=8). Many regressions appear ONLY under this YAML. |
| `seed_tts_locale`  | bench (voice_clone only)                          | `en` / `zh` — selects the seed-tts subset. |

### Headline metrics
- `median_audio_ttfp_ms` — first-packet latency. Most sensitive metric.
- `median_audio_rtf` — real-time factor (audio_duration / total_time).
- `audio_throughput` — aggregate tokens/sec at the configured concurrency.

## Diffusion-image (HunyuanImage3, etc.)

| Knob                  | Where                                        | Notes |
|-----------------------|----------------------------------------------|-------|
| `width` / `height`    | bench `--extra-body`                         | 512 / 1024 / 2048. Step-time scales ~quadratically. |
| `num_inference_steps` | bench `--extra-body`                         | Affects total latency linearly; default 28-50. |
| `guidance_scale`      | bench `--extra-body`                         | Numeric value; affects sampler path but not perf much. |
| `stage_overrides`     | `vllm serve --deploy-config`                 | Partitions VAE / DiT / text-encoder across GPUs. The number of replicas per stage changes the cell. |

### Headline metrics
- `image_latency_ms` (e2e generation time).
- `gpu_busy_pct` — throughput proxy.

## Omni-audio (Qwen2.5-Omni, etc.)

| Knob                  | Where                                        | Notes |
|-----------------------|----------------------------------------------|-------|
| Modality combination  | dataset prep (`audio_only` / `audio+video`)  | Lives in the dataset, not in `extra_body`. |
| `max_num_audios`      | bench `--extra-body`                         | Caps per-request audio inputs. |
| Audio tokenizer       | upstream of bench                            | Different vocoder paths have different perf characteristics. |

### Headline metrics
- `median_audio_ttfp_ms`, `audio_throughput` — shared with TTS.
- `e2el_ms` for end-to-end multimodal latency.

## How to extract knobs from a regression report

1. Look for `extra_body` JSON or `--extra-body` arg in the bench command.
2. Look for `--deploy-config` or any `vllm serve` flag that points at a YAML.
3. Look for dataset-name and any locale flag.
4. If any knob is unspecified in the report, echo a default back to the user
   and confirm before benching — defaults frequently change between reports.
