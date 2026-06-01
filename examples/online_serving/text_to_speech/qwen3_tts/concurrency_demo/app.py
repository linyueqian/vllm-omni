"""Gradio app: two tabs (N=8 per-stream, N=64 aggregate dashboard).

Launch:
    python -m examples.online_serving.text_to_speech.qwen3_tts.concurrency_demo.app \
        --api-base http://localhost:8000 --port 7860
"""

from __future__ import annotations

import argparse
import logging
import time

try:
    import gradio as gr
except ImportError:
    raise ImportError("gradio is required to run this demo. Install with: pip install 'vllm-omni[demo]'") from None

from .metrics import MetricsAggregator, MetricsSnapshot
from .orchestrator import Orchestrator
from .prompts import DEMO_PROMPT
from .runtime import REF_TEXT, WorkerLoop, load_ref_audio_b64
from .views import render_counters_html, render_grid_html, render_row_html

logger = logging.getLogger(__name__)

N_PAGE_A = 8
N_PAGE_B = 64


def _log_burst_exception(future) -> None:
    exc = future.exception()
    if exc is not None:
        logger.exception("Concurrency-demo burst failed", exc_info=exc)


def build_ui(api_base: str) -> gr.Blocks:
    worker = WorkerLoop()
    ref_b64 = load_ref_audio_b64()
    orchestrator = Orchestrator(api_base=api_base, ref_audio_b64=ref_b64, ref_text=REF_TEXT)
    agg_a = MetricsAggregator(n=N_PAGE_A)
    agg_b = MetricsAggregator(n=N_PAGE_B)

    # Mutable closure cells holding the sequence number of the last snapshot
    # rendered into the DOM. When the aggregator's seq has not changed since
    # the previous tick we return ``gr.skip()`` so Gradio does not push a
    # redundant update and the page stays visually idle between bursts.
    _last_seq_a = [-1]
    _last_seq_b = [-1]

    def _eta_label(snap: MetricsSnapshot) -> str:
        serial = snap.serial_eta_s
        parallel = snap.parallel_eta_s
        speedup = snap.speedup_x
        serial_str = f"{serial:.1f} s" if serial is not None else "—"
        parallel_str = f"{parallel:.1f} s" if parallel is not None else "—"
        speedup_str = f"{speedup:.1f}×" if speedup is not None else "—"
        chip = (
            "display:inline-flex;align-items:baseline;gap:6px;"
            "padding:4px 10px;border-radius:999px;"
            "background:#eef2f7;font-family:monospace;font-size:13px;"
        )
        label = "color:#666;font-size:11px;text-transform:uppercase;letter-spacing:0.04em"
        speedup_color = "#4A90D9" if speedup is not None else "#999"
        return (
            f'<div style="display:flex;gap:8px;align-items:center;margin:6px 0">'
            f'<span style="{chip}"><span style="{label}">Serial</span><b>{serial_str}</b></span>'
            f'<span style="color:#999">→</span>'
            f'<span style="{chip}"><span style="{label}">Parallel</span><b>{parallel_str}</b></span>'
            f'<span style="color:#999">→</span>'
            f'<span style="{chip}background:#fff;border:1px solid {speedup_color}">'
            f'<span style="{label}">Speedup</span>'
            f'<b style="color:{speedup_color};font-size:16px">{speedup_str}</b></span>'
            f"</div>"
        )

    def _on_start(which: str):
        agg = agg_a if which == "A" else agg_b
        n = N_PAGE_A if which == "A" else N_PAGE_B
        future = worker.submit(orchestrator.run_burst(n=n, prompt=DEMO_PROMPT, aggregator=agg))
        future.add_done_callback(_log_burst_exception)
        return gr.update(value=f"Started N={n}…")

    def _on_reset(which: str):
        agg = agg_a if which == "A" else agg_b
        agg.reset()
        return gr.update(value=f"Reset N={N_PAGE_A if which == 'A' else N_PAGE_B}")

    def _tick_a():
        seq = agg_a.seq
        if seq == _last_seq_a[0]:
            # Nothing changed since the last tick — skip the update so Gradio
            # doesn't repaint the row strip while the page sits idle.
            return gr.skip(), gr.skip()
        _last_seq_a[0] = seq
        now = time.perf_counter()
        snap = agg_a.snapshot(now=now)
        rows_html = "".join(render_row_html(snap, i) for i in range(N_PAGE_A))
        return rows_html, _eta_label(snap)

    def _tick_b():
        seq = agg_b.seq
        if seq == _last_seq_b[0]:
            return gr.skip(), gr.skip(), gr.skip()
        _last_seq_b[0] = seq
        now = time.perf_counter()
        snap = agg_b.snapshot(now=now)
        return render_counters_html(snap), render_grid_html(snap), _eta_label(snap)

    with gr.Blocks(title="Qwen3-TTS Concurrency Demo") as ui:
        gr.HTML(
            f"""
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
          <img src="https://raw.githubusercontent.com/vllm-project/vllm-omni/main/docs/source/logos/vllm-omni-logo.png"
               alt="vLLM-Omni" style="height:42px;">
          <div>
            <h1 style="margin:0; font-size:1.5em;">Qwen3-TTS Concurrency Demo</h1>
            <span style="font-size:0.85em; color:#666;">
              Served by <a href="https://github.com/vllm-project/vllm-omni" target="_blank"
              style="color:#4A90D9; text-decoration:none; font-weight:600;">vLLM-Omni</a>
              &nbsp;&middot;&nbsp; <code style="background:#eef2f7; padding:2px 6px; border-radius:4px; font-size:0.9em;">{api_base}</code>
              &nbsp;&middot;&nbsp; parallel-vs-serial throughput visualisation
            </span>
          </div>
        </div>
        """
        )
        with gr.Tabs():
            with gr.Tab("Page A — N=8"):
                status_a = gr.Markdown("Idle.")
                eta_a = gr.HTML()
                rows_a = gr.HTML()
                with gr.Row():
                    start_a = gr.Button("▶ Start 8-stream race", variant="primary", scale=3)
                    reset_a = gr.Button("Reset", variant="secondary", scale=1)
                timer_a = gr.Timer(0.1)
                start_a.click(fn=lambda: _on_start("A"), outputs=status_a)
                reset_a.click(fn=lambda: _on_reset("A"), outputs=status_a)
                timer_a.tick(fn=_tick_a, outputs=[rows_a, eta_a], queue=False)
            with gr.Tab("Page B — N=64"):
                status_b = gr.Markdown("Idle.")
                eta_b = gr.HTML()
                counters_b = gr.HTML()
                grid_b = gr.HTML()
                with gr.Row():
                    start_b = gr.Button("▶ Start 64-stream race", variant="primary", scale=3)
                    reset_b = gr.Button("Reset", variant="secondary", scale=1)
                timer_b = gr.Timer(0.2)
                start_b.click(fn=lambda: _on_start("B"), outputs=status_b)
                reset_b.click(fn=lambda: _on_reset("B"), outputs=status_b)
                timer_b.tick(fn=_tick_b, outputs=[counters_b, grid_b, eta_b], queue=False)

    return ui


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", default="http://localhost:8000")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    ui = build_ui(api_base=args.api_base)
    ui.queue().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
