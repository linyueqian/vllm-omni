"""HTML rendering helpers for the concurrency demo.

All renderers are pure functions of a MetricsSnapshot — no Gradio state.
"""

from __future__ import annotations

from .metrics import MetricsSnapshot, StreamState

_ROW_COLORS = {
    "pending": "#9ca3af",
    "streaming": "#3b82f6",
    "done": "#22c55e",
    "error": "#ef4444",
}

_GRID_COLORS = {
    "pending": "#1f2937",
    "streaming": "#3b82f6",
    "done": "#22c55e",
    "error": "#ef4444",
}

# SVG viewbox dimensions for the per-stream waveform strip.
_WAVE_W = 320
_WAVE_H = 44
_WAVE_MID = _WAVE_H / 2
_WAVE_MARGIN = 3  # px of vertical breathing room top/bottom


def _row_progress_pct(s: StreamState, est_total_audio_s: float = 7.0) -> float:
    if s.status == "done":
        return 100.0
    pct = (s.audio_seconds / est_total_audio_s) * 100.0
    return min(99.0, max(0.0, pct))


def _render_waveform_svg(s: StreamState) -> str:
    """Render one stream's recent audio as a centered min/max-peak waveform.

    The waveform draws one vertical line per (min, max) pair stored in
    ``s.waveform_peaks``, scaled to fit ``_WAVE_W x _WAVE_H``. Pre-roll
    streams render as an empty axis line; error streams render in red.
    """
    color = _ROW_COLORS[s.status]
    peaks = s.waveform_peaks
    if not peaks:
        # Idle axis line so the row keeps its height even before audio arrives.
        return (
            f'<svg viewBox="0 0 {_WAVE_W} {_WAVE_H}" '
            f'preserveAspectRatio="none" '
            f'style="width:100%;height:{_WAVE_H}px;background:#0b1220;border-radius:6px">'
            f'<line x1="0" y1="{_WAVE_MID}" x2="{_WAVE_W}" y2="{_WAVE_MID}" '
            f'stroke="#1f2937" stroke-width="1"/></svg>'
        )

    n = len(peaks)
    half = _WAVE_MID - _WAVE_MARGIN
    lines = []
    for i, (lo, hi) in enumerate(peaks):
        # x positions evenly spaced; one vertical stroke per window.
        x = (i + 0.5) / n * _WAVE_W
        y_hi = _WAVE_MID - max(-1.0, min(1.0, hi)) * half
        y_lo = _WAVE_MID - max(-1.0, min(1.0, lo)) * half
        # Ensure visible 1px line even when |lo|==|hi|==0.
        if abs(y_hi - y_lo) < 1.0:
            y_lo += 0.5
            y_hi -= 0.5
        lines.append(
            f'<line x1="{x:.1f}" y1="{y_hi:.1f}" x2="{x:.1f}" y2="{y_lo:.1f}" '
            f'stroke="{color}" stroke-width="1.6" stroke-linecap="round"/>'
        )
    return (
        f'<svg viewBox="0 0 {_WAVE_W} {_WAVE_H}" '
        f'preserveAspectRatio="none" '
        f'style="width:100%;height:{_WAVE_H}px;background:#0b1220;border-radius:6px">'
        f'<line x1="0" y1="{_WAVE_MID}" x2="{_WAVE_W}" y2="{_WAVE_MID}" '
        f'stroke="#1f2937" stroke-width="1"/>'
        f"{''.join(lines)}</svg>"
    )


def render_row_html(snap: MetricsSnapshot, stream_id: int) -> str:
    s: StreamState = snap.per_stream[stream_id]
    label = f"#{stream_id + 1}"
    status = s.status
    ttfb_str = f"{int(s.ttfb_s * 1000)} ms" if s.ttfb_s is not None else "—"
    rtf = s.final_rtf
    rtf_str = f"{rtf:.2f}" if rtf is not None else "—"
    color = _ROW_COLORS[status]
    return (
        f'<div class="ccd-row" data-stream="{stream_id}" data-status="{status}" '
        f'style="display:flex;align-items:center;gap:10px;font-family:monospace;'
        f'padding:4px 0">'
        f'<span style="width:32px;color:#9ca3af">{label}</span>'
        f'<div style="flex:1;min-width:0">{_render_waveform_svg(s)}</div>'
        f'<span style="width:88px">TTFB {ttfb_str}</span>'
        f'<span style="width:72px">RTF {rtf_str}</span>'
        f'<span style="width:72px;color:{color}">{status}</span>'
        f"</div>"
    )


def render_grid_html(snap: MetricsSnapshot) -> str:
    """Render 64 streams as a horizontal skyline of vertical bars.

    Bar height encodes per-stream progress; bar color encodes status. A short
    CSS height transition makes the bars rise smoothly between snapshot ticks
    so the dashboard doesn't visibly snap on every refresh.
    """
    bars = []
    for s in snap.per_stream:
        pct = _row_progress_pct(s)
        color = _GRID_COLORS[s.status]
        bars.append(
            f'<div data-cell="{s.stream_id}" '
            f'style="flex:1;height:{pct:.1f}%;background:{color};'
            f"border-radius:2px 2px 0 0;"
            f'transition:height 0.25s cubic-bezier(0.22,1,0.36,1),background 0.2s linear"></div>'
        )
    return (
        f'<div class="ccd-skyline" '
        f'style="display:flex;align-items:flex-end;gap:2px;'
        f"height:180px;padding:6px 8px;background:#0b1220;border-radius:8px;"
        f'box-shadow:inset 0 0 0 1px #1f2937">{"".join(bars)}</div>'
    )


def render_counters_html(snap: MetricsSnapshot) -> str:
    ttfb_str = f"{int(snap.ttfb_p99_ms)} ms" if snap.ttfb_p99_ms is not None else "— ms"
    rtf_str = f"{snap.rtf_p99:.2f}" if snap.rtf_p99 is not None else "—"
    cell = "flex:1;padding:8px 12px;background:#0b1220;border-radius:8px;color:#e5e7eb;font-family:monospace"
    return (
        f'<div class="ccd-counters" style="display:flex;gap:10px">'
        f'<div style="{cell}"><div style="color:#9ca3af;font-size:11px">ACTIVE</div>'
        f'<div style="font-size:22px;font-weight:600">{snap.active}</div></div>'
        f'<div style="{cell}"><div style="color:#9ca3af;font-size:11px">DONE</div>'
        f'<div style="font-size:22px;font-weight:600">{snap.completed}</div></div>'
        f'<div style="{cell}"><div style="color:#9ca3af;font-size:11px">THROUGHPUT</div>'
        f'<div style="font-size:22px;font-weight:600">{snap.throughput_x:.1f}×</div></div>'
        f'<div style="{cell}"><div style="color:#9ca3af;font-size:11px">TTFB p99</div>'
        f'<div style="font-size:22px;font-weight:600">{ttfb_str}</div></div>'
        f'<div style="{cell}"><div style="color:#9ca3af;font-size:11px">RTF p99</div>'
        f'<div style="font-size:22px;font-weight:600">{rtf_str}</div></div>'
        f"</div>"
    )
