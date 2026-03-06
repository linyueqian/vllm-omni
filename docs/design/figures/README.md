# Figures for Qwen3-Omni blog

The images referenced in `qwen3_omni_blog.md` must be generated and placed here so they display correctly.

## Why images don't show

- **Relative paths**: The blog uses `figures/xxx.png`, which is resolved relative to the markdown file (`docs/design/qwen3_omni_blog.md`), i.e. `docs/design/figures/xxx.png`.
- If these PNG files are missing (not generated or not committed), the images will not display in GitHub, VS Code preview, or any static site.

## How to generate

### 1. vLLM-Omni vs HF Transformers (summary row)

From the **repository root** (or from `docs/design`):

```bash
python docs/design/plot_vllm_omni_vs_transformers.py
```

This creates in `docs/design/figures/`:

- `E2EL_s_vllm_omni_vs_transformers.png`
- `TTFP_s_vllm_omni_vs_transformers.png`
- `RTF_vllm_omni_vs_transformers.png`

### 2. Baseline vs Batch / Batch+CG / Async Chunk

Run your comparison bar-chart script (the one that uses the DataFrame with Baseline, Batch, Batch + CUDA Graph, Batch + CUDA Graph + Async Chunk). Then **copy or save** the generated PNGs into `docs/design/figures/` with these exact names (blog expects underscores, no spaces in metric names):

| Blog expects | Your script may output |
|--------------|-------------------------|
| `Mean_E2EL_ms_Baseline_vs_Batch.png` | Adjust script to save here with this name |
| `Mean_AUDIO_TTFP_ms_Baseline_vs_Batch.png` | same |
| `Mean_AUDIO_RTF_Baseline_vs_Batch.png` | same |
| `Mean_E2EL_ms_Batch_vs_Batch_CUDA_Graph.png` | same |
| `Mean_AUDIO_TTFP_ms_Batch_vs_Batch_CUDA_Graph.png` | same |
| `Mean_AUDIO_RTF_Batch_vs_Batch_CUDA_Graph.png` | same |
| `Mean_E2EL_ms_Batch_CUDA_Graph_vs_Async_Chunk.png` | same |
| `Mean_AUDIO_TTFP_ms_Batch_CUDA_Graph_vs_Async_Chunk.png` | same |
| `Mean_AUDIO_RTF_Batch_CUDA_Graph_vs_Async_Chunk.png` | same |

If your script writes files with different names (e.g. with spaces or `(ms)`), rename or change the script to save directly under `docs/design/figures/` with the names above.

## Check

After generating, ensure:

- `docs/design/figures/` contains all PNGs referenced in `qwen3_omni_blog.md`.
- Commit the PNG files (or add them to your doc build) so the blog renders with images.
