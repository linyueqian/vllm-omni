# Cells

Each YAML in this directory captures one perf cell — the 7-tuple of
`(model, task, deploy_yaml, dataset, num_prompts, max_concurrency, num_warmups)`
plus any family-specific knobs (`extra_body`, `stage_overrides`, sampler options).

These files are inputs to `../run_bisect.sh`. Copy the values into the EDIT-HERE
block at the top of the script.

## Naming

Files use a `<family>_<descriptor>.yaml` prefix so families don't collide as the
list grows:

- `tts_*` — Qwen3-TTS, MOSS-TTS-Nano, GLM-TTS, etc.
- `diffusion_*` — HunyuanImage3, and any future text-to-image / image-to-video model.
- `omni_*` — Qwen2.5-Omni and other audio-in/audio-out stacks.

## Cells in this directory

| File | Cell | Why it exists |
|------|------|----------------|
| `tts_default_voice_high_c.yaml` | CustomVoice `default_voice` c=64 N=512 + `qwen3_tts_high_concurrency.yaml` | PR #3839's regression cell. The high-concurrency YAML enables `code_predictor_prefix_graphs`; the default YAML does not exercise the regressed code path. |
| `tts_voice_clone_nightly.yaml` | Base `voice_clone` c=64 N=128 + default `stage_overrides` | Kanban nightly parity. Use it to cross-check a local bench against the public H100 daily series. |
| `diffusion_hunyuan_t2i_1024.yaml` | HunyuanImage-3.0 t2i @ 1024² c=4 N=32 | Representative diffusion-image cell. Image latency scales quadratically with resolution — copy and adjust for 512² / 2048². |
| `omni_qwen2_5_audio.yaml` | Qwen2.5-Omni audio-in/audio-out c=16 N=128 | Representative omni-audio cell. Shares the talker+vocoder split with Qwen3-TTS, so TTFP / audio_throughput are headline. |

## Adding a new cell

1. Copy an existing YAML to `<family>_<descriptor>.yaml`.
2. Fill in each of the 7-tuple fields. Be especially careful about:
   - **`deploy_yaml`** — does this cell need a non-default deploy YAML? For TTS,
     `qwen3_tts_high_concurrency.yaml` exercises code paths the default YAML
     does not. For diffusion, deploy YAMLs differ in stage partitioning.
   - **`model`** — TTS Base (`Qwen3-TTS-12Hz-1.7B-Base`) and CustomVoice
     (`Qwen3-TTS-12Hz-1.7B-CustomVoice`) are different checkpoints. Image
     families have base / instruct / refiner variants.
   - **`extra_body`** — TTS CustomVoice variants need `task_type`, `voice`,
     `language`. TTS `voice_clone` reads ref-audio from the dataset, no
     `extra_body` needed. Diffusion carries `width`, `height`,
     `num_inference_steps`, `guidance_scale`.
3. Set `num_warmups: 8` for bisect work (the kanban default of 0 is too noisy).
4. Set `num_prompts: 512` for a tight median (the kanban default of 128 has
   ~10% variance at c=64).
5. Optionally include `known_baselines:` for sanity-checking a fresh result.
   Source these from PR comments or the kanban; cite the source.

## Cell-definition discipline

Before benching, ALWAYS echo the cell to the user. The most common failure mode
is running the bisect against the wrong cell and falsely concluding "no
regression" — see `../../SKILL.md` for the load-bearing lesson and the
rationalization table.
