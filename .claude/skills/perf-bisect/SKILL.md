---
name: perf-bisect
description: >
  Use when a vllm-omni perf change — TTFP, RTF, audio throughput, step time,
  image latency — must be attributed to a specific commit, or when a PR's
  perf claim needs verification. Triggers include "find which PR regressed X",
  "bisect between commit A and B", "verify PR #N's perf claim",
  "高并发 TTFP 劣化". Applies across TTS, diffusion-image, and omni-audio
  model families. Not for trend reading (use `vllm-omni-kanban`) or writing
  new perf tests (use `tests/dfx/perf/`).
---

# Perf Regression Bisect

Attribute a vllm-omni perf change to a specific commit. Encodes the **cell-definition
discipline** that prevents the most common failure mode: benching the wrong workload
and falsely declaring "no regression" while one is live in a sibling cell.

The workflow was hardened against the TTS regression cluster traced through
PRs #3662 / #3681 / #3817 / #3839, but the harness, the discipline, and the
rationalization table generalise to diffusion-image and omni-audio. TTS appears
below as the worked example; family-specific knobs are called out where they bite.

## When to use

- "Which PR regressed `<metric>`?" — TTFP, RTF, audio throughput, image
  generation time, step latency, etc.
- "Bisect `<metric>` between commits A and B."
- "Verify PR #N's perf claim end-to-end against `main`."
- Any team-chat report of a delta requiring code-level attribution (vs.
  dataset, driver, kernel, or vllm-upstream changes).

## When NOT to use

- Read existing perf trends → start with [`vllm-omni-kanban`](https://github.com/hsliuustc0106/vllm-omni-kanban)
  (90-day H100 nightly series).
- Write a **new** perf test → `tests/dfx/perf/scripts/run_benchmark.py` plus a
  JSON spec under `tests/dfx/perf/tests/`.
- One-shot "does this PR regress anything?" → use the `tts-perf-check` skill
  for that. Use *this* skill for "which commit in a range caused the change".
- Cross-version bisect across the vllm 0.20 ↔ 0.21 rebase in one venv — needs
  two venvs (the dependency tree is incompatible).
- NPU / XPU / AMD — H100 / H20 / L20X only.

## Paired tools

- **`remote-gpu`** (Skill) — SSH patterns for H20, L20X, H200-hsliu, Duke lab
  hosts, DCC. Read `references/<class>.md` (e.g. `references/h20.md`) before
  any heavy work on a shared host: disk layout, China-network, voice-cloning
  base64 workaround, etc.
- **`tts-perf-check`** (Skill) — one-shot per-PR perf check.
- **`vllm-omni-kanban`** — daily H100 nightlies. **Always read this FIRST** —
  it narrows the suspect commit range from weeks of merges to one day's worth
  of PRs.

## Cell-definition discipline (read first)

The single most common failure: bench the wrong cell, conclude "no regression",
and ship the regression. A **cell** is the 7-tuple

```
(model, task, deploy_yaml, dataset, num_prompts, max_concurrency, num_warmups)
```

plus zero or more **family knobs** (`extra_body`, `stage_overrides`, sampler
options). Extract all 7 from the regression report — comment, screenshot, chat
message — **before** writing any script.

### Generic dimensions

| Dimension         | Source                                    | Example values |
|-------------------|-------------------------------------------|----------------|
| `model`           | bench `--model`                           | `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `…-CustomVoice`, `tencent/HunyuanImage-3.0`, `Qwen/Qwen2.5-Omni-7B` |
| `task`            | family-specific (see below)               | TTS: `voice_clone` / `default_voice` / `voice_design`; Diffusion: t2i vs edit; Omni: audio vs audio+video |
| `deploy_yaml`     | `vllm serve --deploy-config`              | `qwen3_tts.yaml` vs `qwen3_tts_high_concurrency.yaml` |
| `dataset`         | bench `--dataset-name` + `--dataset-path` | `seed-tts`, `seed-tts-text`, `seed-tts-design`, `geneval`, `audiocaps` |
| `num_prompts`     | bench `--num-prompts`                     | latency 20; throughput 80–128; stress 512 |
| `max_concurrency` | bench `--max-concurrency`                 | 1, 8, 16, 32, 64, 128, 256 |
| `num_warmups`     | bench `--num-warmups`                     | 0 (kanban); **8 (bisect standard)** |

### Family-specific knobs

Every family adds knobs the 7-tuple does not capture (`extra_body`,
`stage_overrides`, sampler options). Full table in
`references/family-knobs.md`. The TL;DR per family:

- **TTS** — `task_type`, `voice`, `language` in `extra_body`;
  `qwen3_tts_high_concurrency.yaml` enables `code_predictor_prefix_graphs`
  and gates a class of regressions invisible under the default YAML.
- **Diffusion-image** — `width`, `height`, `num_inference_steps`,
  `guidance_scale` in `extra_body`; `stage_overrides` partitions VAE / DiT
  / text-encoder across GPUs.
- **Omni-audio** — modality combinations live in dataset preparation, not
  in `extra_body`.

### Anti-pattern (the load-bearing lesson)

A bisect that measures Base `voice_clone` (default YAML, N=128) when the
report is CustomVoice `default_voice` (`qwen3_tts_high_concurrency.yaml`,
N=512) is measuring a **different cell**. Every dimension is wrong, the
bench completes with apples-vs-oranges numbers, and the verdict is a
confident "no regression" while a >100% TTFP regression is live. The
regression exists only when the high-concurrency YAML is loaded; the
default YAML's code path doesn't touch the regressed flag.

**Always echo the cell back to the user before any bench:**

> "Benching `<model>` on `<task>` with `<deploy_yaml>`, N=`<N>`,
> c=`<c>`, `<W>` warmups. Confirm?"

A correction at this step saves 30–60 minutes of wrong-cell GPU time.

## Workflow

### Step 1 — Kanban triage

If the regression is on a cell the kanban tracks (Base `voice_clone`,
CustomVoice `default_voice` / `voice_design`, HunyuanImage t2i), the kanban
likely has a daily series for it.

```bash
git clone --depth 1 https://github.com/hsliuustc0106/vllm-omni-kanban /tmp/kanban
python scripts/kanban_trend.py /tmp/kanban result_test_qwen3_tts_base_seed-tts_64_128
```

`scripts/kanban_trend.py` prints `(date, build, TTFP, Δ%, RTF, Δ%, throughput, Δ%)`
with `←REG` / `←IMP` markers at the 10% threshold.

**Map build → commit.** Kanban "build" directories are not strictly 1:1 with
main-branch nightly commits (some are PR-CI runs). Read the date from the
JSON filename suffix, then resolve to a main-branch commit window via
`git log --first-parent upstream/main --since=<date> --until=<date+1d>`.

**Caveat.** The kanban's nightly bench uses each test's default
`stage_overrides`. Regressions that only manifest under a non-default
`deploy_yaml` (e.g. `qwen3_tts_high_concurrency.yaml`) are **invisible** to
the kanban. Use kanban for the **onset date** when it sees the regression;
fall back to reporter-supplied SHAs when it doesn't.

### Step 2 — Pick the bisect span

From the kanban onset (or PR review trail), the bisect starts with a
`[good, bad]` commit pair. Path-filter the log to PRs touching the suspect
blast radius. The shared worker / scheduler / common-model paths matter for
every family; the family-specific paths below pick out the relevant
model/processor pair.

**Shared hot path** (always include):

```
vllm_omni/worker/gpu_ar_model_runner.py
vllm_omni/worker/gpu_generation_model_runner.py
vllm_omni/worker/gpu_model_runner.py
vllm_omni/core/sched/omni_ar_scheduler.py
vllm_omni/core/sched/omni_generation_scheduler.py
vllm_omni/core/sched/omni_scheduling_coordinator.py
vllm_omni/core/prefix_cache.py
vllm_omni/model_executor/models/common/
```

**Add per family** (model + stage-input-processor):

| Family            | Add to the path filter                                                                                  |
|-------------------|---------------------------------------------------------------------------------------------------------|
| TTS               | `vllm_omni/model_executor/models/qwen3_tts/` + `stage_input_processors/qwen3_tts.py` (and `cosyvoice3.py` / `glm_tts*.py` / `omnivoice.py` for siblings) |
| Diffusion-image   | `vllm_omni/model_executor/models/hunyuan_image3/` + `stage_input_processors/hunyuan_image3.py` + `vllm_omni/distributed/` (stage partitioning) |
| Omni-audio        | `vllm_omni/model_executor/models/qwen2_5_omni/` (or `qwen3_omni/`, `ming_flash_omni/`, `dynin_omni/`) + matching `stage_input_processors/*.py` |

Example, TTS:

```bash
git log --oneline --first-parent upstream/main <good>..<bad> -- \
  vllm_omni/model_executor/models/qwen3_tts/ \
  vllm_omni/model_executor/stage_input_processors/qwen3_tts.py \
  vllm_omni/worker/gpu_ar_model_runner.py \
  vllm_omni/worker/gpu_generation_model_runner.py \
  vllm_omni/worker/gpu_model_runner.py \
  vllm_omni/core/sched/omni_ar_scheduler.py \
  vllm_omni/core/sched/omni_generation_scheduler.py \
  vllm_omni/core/prefix_cache.py \
  vllm_omni/model_executor/models/common/
```

Typical result: 5–15 PRs. Linear if ≤8; binary bisect (midpoint first) if
more.

### Step 3 — Bench harness on the remote host

`scripts/run_bisect.sh` is the vendored bench loop. Edit the block at the
top. The example below is TTS; swap `MODEL` / `DEPLOY_YAML` / `DATASET_*` /
`EXTRA_BODY` for diffusion or omni — `scripts/cells/<family>_*.yaml` carries
ready-to-paste values for each.

```bash
COMMITS=("<sha1>" "<sha2>" "<sha3>")
LABELS=("good_baseline" "midpoint" "bad_main")

# TTS example
MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEPLOY_YAML="vllm_omni/deploy/qwen3_tts_high_concurrency.yaml"
DATASET_NAME="seed-tts-text"
EXTRA_BODY='{"voice":"Vivian","language":"English","task_type":"CustomVoice"}'
NUM_PROMPTS=512
MAX_CONCURRENCY=64
NUM_WARMUPS=8

# Diffusion-image example (HunyuanImage3 t2i @ 1024²)
# MODEL="tencent/HunyuanImage-3.0"
# DEPLOY_YAML="vllm_omni/deploy/hunyuan_image3.yaml"
# DATASET_NAME="random-image-prompts"
# EXTRA_BODY='{"width":1024,"height":1024,"num_inference_steps":28}'

# Omni-audio example (Qwen2.5-Omni)
# MODEL="Qwen/Qwen2.5-Omni-7B"
# DEPLOY_YAML="vllm_omni/deploy/qwen2_5_omni.yaml"
# DATASET_NAME="audiocaps"
# EXTRA_BODY='{}'
```

Mandatory env (see `remote-gpu`'s `references/h20.md` for full setup):

```bash
source <venv>/bin/activate
export PATH="<venv>/bin:$PATH"   # belt & suspenders for subprocess `ninja`
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/dataX/<your-dir>/hf_cache
export CUDA_VISIBLE_DEVICES=4,5  # pick free GPUs; nvidia-smi first
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
```

Per-commit loop (encoded in the script):

1. `git checkout <sha>` in a dedicated worktree (don't disturb the main checkout).
2. `pkill -9 -f "vllm.*serve|StageEngineCore|OmniServer"`. Verify port is free.
3. `vllm serve <model> --omni --deploy-config <yaml> --port 8092 &`.
4. Poll `/v1/models` until the model entry appears, then **sleep 30s** for
   CUDA-graph compile to settle.
5. `vllm bench serve --omni ... --save-result --result-filename result.json`.
6. Kill server, sleep 10s, next commit.

### Step 4 — Interpret

Pull the headline metric (`median_audio_ttfp_ms` / `median_audio_rtf` /
`audio_throughput` for TTS; `image_latency_ms` for diffusion) from each
commit's `result.json`. Build a table:

```
| commit     | label         | median TTFP | p99 TTFP | RTF   | tput  |
|------------|---------------|-------------|----------|-------|-------|
| <sha1>     | good_baseline | 712 ms      | 1340 ms  | 0.18  | 24.1  |
| <sha2>     | midpoint      | 1230 ms     | 1810 ms  | 0.27  | 19.0  |  ← jump
| <sha3>     | bad_main      | 1604 ms     | 1936 ms  | 0.31  | 17.4  |
```

A jump > 15% between consecutive labels is a candidate regression PR.
Cross-check the suspect's diff against the Step-2 path filter.

### Step 5 — Variance check

Single-run variance at c=64:

| Setup                                | Variance | Verdict                                                                 |
|--------------------------------------|----------|-------------------------------------------------------------------------|
| N=128, W=0 (kanban default)          | ±10%     | Unreliable for <20% deltas; needs 2-3 runs.                             |
| N=512, W=8 (bisect standard)         | ±3%      | Single run reasonable for >15% deltas; 2 runs if signal is 5-15%.       |

**Always run N=512 + 8 warmups for bisect work.** The kanban's N=128/W=0 is
fine for daily trend but too noisy to attribute a regression to a single PR.

## Cross-platform notes

Per-host SSH and setup details live in `remote-gpu`'s `references/` (e.g.
`references/h20.md`, `references/h200.md`). The invariant this skill relies
on: numbers must AGREE on **direction and rough magnitude** across
platforms. If H20 reports +100% and H100 reports +0%, suspect the deploy
YAML first — the kanban's H100 nightly may not be loading the regressed
YAML, so the H100 sees no signal even when one is live.

## Rationalization table

Excuses observed when running bisects without this skill, and the reality:

| Excuse                                                           | Reality                                                                                   |
|------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| "The script ran clean, the numbers look normal"                  | Apples-vs-oranges numbers also look normal. Echo the 7-tuple before benching.             |
| "I'll use the kanban's cell because it has a baseline"           | Kanban's cell may not exercise the regressed code path. Match the reporter's cell exactly. |
| "Pre-warmup is overkill, just run N=128"                         | N=128 c=64 W=0 is ±10% — <15% signals get lost. N=512 W=8.                                |
| "The YAML doesn't matter, the model is the same"                 | `qwen3_tts.yaml` and `qwen3_tts_high_concurrency.yaml` enable different feature flags.    |
| "`/v1/models` returns 200 means bench-ready"                     | The API server loads before CUDA graphs finish compiling. Add 30s settle.                 |
| "I'll skip kanban triage, the suspect PR is obvious"             | Kanban narrows weeks to one day. Skipping it costs 5-15 wrong checkouts on average.       |
| "Single run is fine if the delta looks large"                    | OK for >15% deltas at N=512 W=8. Else two runs.                                           |
| "Cross-version venvs are too much work, I'll just patch deps"    | The vllm 0.20 ↔ 0.21 boundary breaks plugin loading. Two venvs.                           |

## Red flags — STOP and re-check

- About to launch a bench without saying the 7-tuple out loud
- About to use `pytest -k <expr>` without first running
  `pytest --collect-only -q -k <expr>` to confirm it matches anything
- Server PID from the previous commit is still alive when starting the next
- About to checkout commits across the vllm 0.20 → 0.21 boundary in the same venv
- About to declare "no regression" from a single run on a noisy cell

## Common pitfalls (mechanical)

Six failure modes caught in real bisects: `pytest -k` zero-match, venv PATH
not inherited by subprocess, stale server PID, multi-tenant GPU contention,
`/v1/models` returning before CUDA-graph compile finishes, and cold model
download exceeding the server-ready timeout. Full diagnostics and copy-paste
remediation snippets in `references/pitfalls.md`.

## What this skill does NOT do

- Cross-version (vllm 0.20 ↔ 0.21) bisects in one venv.
- NPU / XPU / AMD bisects.
- Auto-attribute multi-commit interactions ("which combo of two PRs?") — manual review.
- Write a regression test that catches the bisected commit — that's
  `tests/dfx/perf/`'s job; this skill is investigation, not prevention.

## Files in this skill

```
.claude/skills/perf-bisect/
├── SKILL.md                                  # this file
├── references/
│   ├── family-knobs.md                       # extra_body / stage_overrides per family
│   └── pitfalls.md                           # mechanical failure modes + remediations
└── scripts/
    ├── run_bisect.sh                         # bench-loop template
    ├── kanban_trend.py                       # extract metric time series from the kanban repo
    └── cells/
        ├── README.md                         # how to define a new cell
        ├── tts_default_voice_high_c.yaml     # CustomVoice + high_concurrency.yaml + N=512 + c=64
        ├── tts_voice_clone_nightly.yaml      # Base + default YAML + N=128 + c=64 (kanban parity)
        ├── diffusion_hunyuan_t2i_1024.yaml   # HunyuanImage-3.0 t2i @ 1024² + c=4 + N=32
        └── omni_qwen2_5_audio.yaml           # Qwen2.5-Omni audio-only + c=16 + N=128
```

Cell files follow `<family>_<descriptor>.yaml`. The shipped set covers TTS,
diffusion-image, and omni-audio; copy the closest sibling when adding a new
cell — see `scripts/cells/README.md`.
