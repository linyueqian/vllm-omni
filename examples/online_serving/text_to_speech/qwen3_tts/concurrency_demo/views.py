"""HTML rendering helpers for the concurrency demo.

All renderers are pure functions of a MetricsSnapshot — no Gradio state.
"""

from __future__ import annotations

from .metrics import MetricsSnapshot, StreamState


def _row_progress_pct(s: StreamState, est_total_audio_s: float = 7.0) -> float:
    if s.status == "done":
        return 100.0
    pct = (s.audio_seconds / est_total_audio_s) * 100.0
    return min(99.0, max(0.0, pct))


def render_row_html(snap: MetricsSnapshot, stream_id: int) -> str:
    s: StreamState = snap.per_stream[stream_id]
    label = f"#{stream_id + 1}"
    status = s.status
    pct = _row_progress_pct(s)
    ttfb_str = f"{int(s.ttfb_s * 1000)} ms" if s.ttfb_s else "—"
    rtf = s.final_rtf
    rtf_str = f"{rtf:.2f}" if rtf is not None else "—"
    color = {
        "pending": "#888",
        "streaming": "#1f77b4",
        "done": "#2ca02c",
        "error": "#d62728",
    }[status]
    return (
        f'<div class="ccd-row" data-stream="{stream_id}" '
        f'data-status="{status}" style="display:flex;align-items:center;gap:8px;font-family:monospace">'
        f'<span style="width:32px">{label}</span>'
        f'<div style="flex:1;height:14px;background:#eee;border-radius:7px;overflow:hidden">'
        f'<div style="width:{pct:.1f}%;height:100%;background:{color}"></div></div>'
        f'<span style="width:80px">TTFB {ttfb_str}</span>'
        f'<span style="width:60px">RTF {rtf_str}</span>'
        f'<span style="width:64px;color:{color}">{status}</span>'
        f"</div>"
    )


def render_grid_html(snap: MetricsSnapshot) -> str:
    cells = []
    for s in snap.per_stream:
        color = {
            "pending": "#cccccc",
            "streaming": "#1f77b4",
            "done": "#2ca02c",
            "error": "#d62728",
        }[s.status]
        cells.append(
            f'<span data-cell="{s.stream_id}" '
            f'style="display:inline-block;width:18px;height:18px;margin:2px;'
            f'border-radius:9px;background:{color}"></span>'
        )
    # Group every 8 cells per row so the 8x8 grid is visually obvious.
    rows = []
    for i in range(0, len(cells), 8):
        rows.append(f'<div style="line-height:0">{"".join(cells[i : i + 8])}</div>')
    return f'<div class="ccd-grid">{"".join(rows)}</div>'


def render_counters_html(snap: MetricsSnapshot) -> str:
    ttfb_str = f"{int(snap.ttfb_p99_ms)} ms" if snap.ttfb_p99_ms is not None else "— ms"
    rtf_str = f"{snap.rtf_p99:.2f}" if snap.rtf_p99 is not None else "—"
    return (
        f'<div class="ccd-counters" style="display:flex;gap:24px;font-family:monospace">'
        f"<div><b>Active</b><br/>{snap.active}</div>"
        f"<div><b>Done</b><br/>{snap.completed}</div>"
        f"<div><b>Throughput</b><br/>{snap.throughput_x:.1f}×</div>"
        f"<div><b>TTFB p99</b><br/>{ttfb_str}</div>"
        f"<div><b>RTF p99</b><br/>{rtf_str}</div>"
        f"</div>"
    )
