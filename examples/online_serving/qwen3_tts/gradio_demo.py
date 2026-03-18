"""Gradio demo for Qwen3-TTS with gapless streaming audio playback.

Uses a custom AudioWorklet-based player for gap-free streaming,
inspired by github.com/KoljaB/RealtimeVoiceChat. Audio is streamed
from the vLLM server through a same-origin proxy and played via the
Web Audio API's AudioWorklet, which maintains a FIFO buffer queue
and plays samples at the audio clock rate — eliminating inter-chunk
gaps inherent in Gradio's built-in streaming audio component.

Also supports non-streaming mode (full audio download) via gr.Audio.

Supports all 3 task types:
  - CustomVoice: Predefined speaker with optional style instructions
  - VoiceDesign: Natural language voice description
  - Base: Voice cloning from reference audio (upload or URL)

Usage:
    # Start the vLLM server first (see run_server.sh), then:
    python gradio_demo.py --api-base http://localhost:8000
"""

import argparse
import io
import json
import logging

import gradio as gr
import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from tts_common import (
    PCM_SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    TASK_TYPES,
    add_common_args,
    build_payload,
    fetch_voices,
    on_task_type_change,
)

logger = logging.getLogger(__name__)

# ── AudioWorklet processor (loaded in browser via Blob URL) ──────────
WORKLET_JS = r"""
class TTSPlaybackProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.queue = [];
        this.buf = null;
        this.pos = 0;
        this.playing = false;
        this.played = 0;
        this.port.onmessage = (e) => {
            if (e.data && e.data.type === 'clear') {
                this.queue = []; this.buf = null; this.pos = 0; this.played = 0;
                if (this.playing) { this.playing = false; this.port.postMessage({type:'stopped'}); }
                return;
            }
            this.queue.push(e.data);
        };
    }
    process(inputs, outputs) {
        const out = outputs[0][0];
        for (let i = 0; i < out.length; i++) {
            if (!this.buf || this.pos >= this.buf.length) {
                if (this.queue.length > 0) {
                    this.buf = this.queue.shift(); this.pos = 0;
                } else {
                    for (let j = i; j < out.length; j++) out[j] = 0;
                    if (this.playing) { this.playing = false; this.port.postMessage({type:'stopped', played:this.played}); }
                    return true;
                }
            }
            out[i] = this.buf[this.pos++] / 32768;
            this.played++;
        }
        if (!this.playing) { this.playing = true; this.port.postMessage({type:'started'}); }
        return true;
    }
}
registerProcessor('tts-playback-processor', TTSPlaybackProcessor);
"""

# ── Player HTML (container with metric cards) ────────────────────────
PLAYER_HTML = """
<div id="tts-player" style="padding:20px; border:1px solid #e0e0e0; border-radius:12px; background:linear-gradient(135deg,#f5f8ff,#edf2ff); min-height:80px;">
  <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
    <div id="tts-status-dot" style="width:10px;height:10px;border-radius:50%;background:#ccc;flex-shrink:0;"></div>
    <span id="tts-status" style="font-weight:600;font-size:1.05em;">Ready</span>
    <button id="tts-stop-btn" onclick="window.ttsStop()"
      style="display:none; margin-left:auto; padding:5px 16px; border-radius:6px; border:1px solid #EF5552;
             background:#fff; color:#EF5552; cursor:pointer; font-size:0.85em; transition:all 0.15s;">Stop</button>
  </div>
  <div id="tts-metrics" style="display:none; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:12px;">
    <div style="background:#fff;border-radius:8px;padding:10px 12px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
      <div style="font-size:0.7em;text-transform:uppercase;color:#888;letter-spacing:0.5px;margin-bottom:2px;">TTFP</div>
      <div id="tts-m-ttfp" style="font-size:1.3em;font-weight:700;color:#333;">—</div>
    </div>
    <div style="background:#fff;border-radius:8px;padding:10px 12px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
      <div style="font-size:0.7em;text-transform:uppercase;color:#888;letter-spacing:0.5px;margin-bottom:2px;">RTF</div>
      <div id="tts-m-rtf" style="font-size:1.3em;font-weight:700;color:#333;">—</div>
    </div>
    <div style="background:#fff;border-radius:8px;padding:10px 12px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
      <div style="font-size:0.7em;text-transform:uppercase;color:#888;letter-spacing:0.5px;margin-bottom:2px;">Audio</div>
      <div id="tts-m-dur" style="font-size:1.3em;font-weight:700;color:#333;">—</div>
    </div>
    <div style="background:#fff;border-radius:8px;padding:10px 12px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
      <div style="font-size:0.7em;text-transform:uppercase;color:#888;letter-spacing:0.5px;margin-bottom:2px;">Speed</div>
      <div id="tts-m-speed" style="font-size:1.3em;font-weight:700;color:#333;">—</div>
    </div>
  </div>
  <div id="tts-rtf-bar-wrap" style="display:none; background:#e8ecf1; border-radius:6px; height:22px; overflow:hidden; position:relative;">
    <div id="tts-rtf-bar" style="height:100%; border-radius:6px; transition:width 0.3s ease, background 0.3s ease; width:0%;"></div>
    <span id="tts-rtf-label" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:0.75em;font-weight:600;color:#444;"></span>
  </div>
  <div id="tts-elapsed" style="display:none; margin-top:8px; font-size:0.8em; color:#999; text-align:right;"></div>
</div>
"""


def _build_player_js(sample_rate: int) -> str:
    """Build the JavaScript that powers the AudioWorklet player."""
    return f"""
() => {{
    const SR = {sample_rate};
    const WC = {json.dumps(WORKLET_JS)};
    let ctx = null, node = null, abort = null, gen = false, st = {{}};

    async function init() {{
        if (ctx) return;
        ctx = new AudioContext({{ sampleRate: SR }});
        const b = new Blob([WC], {{ type: 'application/javascript' }});
        const u = URL.createObjectURL(b);
        await ctx.audioWorklet.addModule(u);
        URL.revokeObjectURL(u);
        node = new AudioWorkletNode(ctx, 'tts-playback-processor');
        node.connect(ctx.destination);
        node.port.onmessage = (e) => {{
            if (e.data.type === 'started') setStatus('Playing...', '#64dd17');
            else if (e.data.type === 'stopped' && !gen) {{
                setStatus('Done', '#64dd17'); showStats(true);
                const btn = document.getElementById('tts-stop-btn');
                if (btn) btn.style.display = 'none';
            }}
        }};
    }}

    function setStatus(text, color) {{
        const s = document.getElementById('tts-status');
        const d = document.getElementById('tts-status-dot');
        if (s) s.textContent = text;
        if (d) d.style.background = color || '#ccc';
    }}

    function showStats(fin) {{
        if (!st.t0) return;
        const elapsed = (fin && st.streamEnd ? (st.streamEnd - st.t0) : (performance.now() - st.t0)) / 1000;
        const dur = st.samples / SR;
        const mTtfp = document.getElementById('tts-m-ttfp');
        const mRtf = document.getElementById('tts-m-rtf');
        const mDur = document.getElementById('tts-m-dur');
        const mSpeed = document.getElementById('tts-m-speed');
        const bar = document.getElementById('tts-rtf-bar');
        const barLabel = document.getElementById('tts-rtf-label');
        const elapsedEl = document.getElementById('tts-elapsed');

        if (mTtfp && st.ttfp != null) mTtfp.textContent = st.ttfp.toFixed(0) + 'ms';
        if (mDur) mDur.textContent = dur.toFixed(1) + 's';

        if (dur > 0 && elapsed > 0) {{
            const rtf = elapsed / dur;
            const speed = 1 / rtf;
            if (mRtf) {{
                mRtf.textContent = rtf.toFixed(2) + 'x';
                mRtf.style.color = rtf < 1 ? '#64dd17' : rtf < 1.5 ? '#e8a317' : '#EF5552';
            }}
            if (mSpeed) {{
                mSpeed.textContent = speed.toFixed(1) + 'x';
                mSpeed.style.color = speed > 1 ? '#64dd17' : speed > 0.7 ? '#e8a317' : '#EF5552';
            }}
            if (bar) {{
                const pct = Math.min(speed / 3 * 100, 100);
                bar.style.width = pct + '%';
                bar.style.background = speed > 1 ? 'linear-gradient(90deg,#4A90D9,#64dd17)' : speed > 0.7 ? 'linear-gradient(90deg,#e8a317,#f0b866)' : 'linear-gradient(90deg,#EF5552,#f87171)';
            }}
            if (barLabel) barLabel.textContent = speed.toFixed(1) + 'x realtime';
        }}
        if (elapsedEl) {{
            elapsedEl.style.display = 'block';
            elapsedEl.textContent = fin ? 'Completed in ' + elapsed.toFixed(1) + 's  (' + st.chunks + ' chunks)' : elapsed.toFixed(1) + 's elapsed  (' + st.chunks + ' chunks)';
        }}
    }}

    window.ttsStop = function() {{
        if (abort) abort.abort();
        if (node) node.port.postMessage({{ type: 'clear' }});
        gen = false;
        setStatus('Stopped', '#999');
        const btn = document.getElementById('tts-stop-btn');
        if (btn) btn.style.display = 'none';
    }};

    window.ttsGenerate = async function(payload) {{
        try {{ await init(); if (ctx.state === 'suspended') await ctx.resume(); }}
        catch (e) {{ const s = document.getElementById('tts-status'); if (s) s.textContent = 'Audio init error: ' + e.message; return; }}

        node.port.postMessage({{ type: 'clear' }});
        gen = true;
        st = {{ t0: performance.now(), chunks: 0, samples: 0, ttfp: null }};
        setStatus('Connecting...', '#4A90D9');
        const bEl = document.getElementById('tts-stop-btn');
        if (bEl) bEl.style.display = 'inline-block';
        const mp = document.getElementById('tts-metrics');
        if (mp) {{ mp.style.display = 'grid'; ['tts-m-ttfp','tts-m-rtf','tts-m-dur','tts-m-speed'].forEach(id => {{ const e = document.getElementById(id); if(e) {{ e.textContent = '—'; e.style.color = '#333'; }} }}); }}
        const bw = document.getElementById('tts-rtf-bar-wrap');
        if (bw) bw.style.display = 'block';
        const bar = document.getElementById('tts-rtf-bar');
        if (bar) bar.style.width = '0%';
        const bl = document.getElementById('tts-rtf-label');
        if (bl) bl.textContent = '';
        const ee = document.getElementById('tts-elapsed');
        if (ee) {{ ee.style.display = 'none'; ee.textContent = ''; }}
        abort = new AbortController();

        try {{
            const r = await fetch('/proxy/v1/audio/speech', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(payload),
                signal: abort.signal,
            }});
            if (!r.ok) {{ const t = await r.text(); throw new Error('Server ' + r.status + ': ' + t.slice(0, 200)); }}
            setStatus('Streaming...', '#4A90D9');

            const reader = r.body.getReader();
            let left = new Uint8Array(0);
            while (true) {{
                const {{ done, value }} = await reader.read();
                if (done) break;
                let raw;
                if (left.length > 0) {{
                    raw = new Uint8Array(left.length + value.length);
                    raw.set(left); raw.set(value, left.length);
                }} else {{ raw = value; }}
                const usable = raw.length - (raw.length % 2);
                left = usable < raw.length ? raw.slice(usable) : new Uint8Array(0);
                if (usable > 0) {{
                    const ab = new ArrayBuffer(usable);
                    new Uint8Array(ab).set(raw.subarray(0, usable));
                    const pcm = new Int16Array(ab);
                    node.port.postMessage(pcm);
                    st.chunks++;
                    st.samples += pcm.length;
                    if (st.ttfp == null) st.ttfp = performance.now() - st.t0;
                    showStats(false);
                }}
            }}
        }} catch (e) {{
            if (e.name !== 'AbortError') {{
                setStatus('Error: ' + e.message, '#EF5552');
                console.error('TTS error:', e);
            }}
        }} finally {{
            // Freeze RTF at stream-end time (before playback finishes)
            st.streamEnd = performance.now();
            showStats(true);
            gen = false;
            if (st.samples > 0) setStatus('Finishing playback...', '#64dd17');
            else {{
                setStatus('No audio received', '#999');
                if (bEl) bEl.style.display = 'none';
            }}
        }}
    }};
}}
"""


def generate_speech(
    api_base: str,
    text: str,
    task_type: str,
    voice: str,
    language: str,
    instructions: str,
    ref_audio: tuple | None,
    ref_audio_url: str,
    ref_text: str,
    x_vector_only: bool,
    response_format: str,
    speed: float,
):
    """Non-streaming: call /v1/audio/speech and return complete audio."""
    payload = build_payload(
        text,
        task_type,
        voice,
        language,
        instructions,
        ref_audio,
        ref_audio_url,
        ref_text,
        x_vector_only,
        response_format,
        speed,
        stream=False,
    )
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{api_base}/v1/audio/speech",
                json=payload,
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            )
    except httpx.TimeoutException:
        raise gr.Error("Request timed out. The server may be busy.")
    except httpx.ConnectError:
        raise gr.Error(f"Cannot connect to server at {api_base}.")

    if resp.status_code != 200:
        raise gr.Error(f"Server error ({resp.status_code}): {resp.text}")

    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            raise gr.Error(f"Server error: {resp.json()}")
        except ValueError:
            pass

    try:
        if response_format == "pcm":
            audio_np = np.frombuffer(resp.content, dtype=np.int16).astype(np.float32) / 32767.0
            return (PCM_SAMPLE_RATE, audio_np)
        audio_np, sample_rate = sf.read(io.BytesIO(resp.content))
        if audio_np.ndim > 1:
            audio_np = audio_np[:, 0]
        return (sample_rate, audio_np.astype(np.float32))
    except Exception as e:
        raise gr.Error(f"Failed to decode audio: {e}")


def create_app(api_base: str):
    """Create the FastAPI app with streaming proxy + Gradio UI."""
    fastapi_app = FastAPI()

    # ── Streaming proxy (same-origin, no CORS issues) ────────────
    @fastapi_app.post("/proxy/v1/audio/speech")
    async def proxy_speech(request: Request):
        body = await request.json()
        client = httpx.AsyncClient(timeout=300)
        try:
            resp = await client.send(
                client.build_request(
                    "POST",
                    f"{api_base}/v1/audio/speech",
                    json=body,
                    headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
                ),
                stream=True,
            )
        except Exception as exc:
            await client.aclose()
            return Response(content=str(exc), status_code=502)

        if resp.status_code != 200:
            content = await resp.aread()
            await resp.aclose()
            await client.aclose()
            return Response(content=content, status_code=resp.status_code)

        async def relay():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(relay(), media_type="application/octet-stream")

    # ── Gradio UI ────────────────────────────────────────────────
    voices = fetch_voices(api_base)

    css = """
    #generate-btn button { width: 100%; }
    .task-info { padding: 8px 12px; border-radius: 6px; background: #f0f4ff; margin-bottom: 8px; }
    button.primary { background: #4A90D9 !important; border-color: #4A90D9 !important; }
    button.primary:hover { background: #3a7bc8 !important; border-color: #3a7bc8 !important; }
    .gradio-container .gr-button-primary { background: #4A90D9 !important; border-color: #4A90D9 !important; }
    .gradio-container .gr-button-primary:hover { background: #3a7bc8 !important; border-color: #3a7bc8 !important; }
    """

    theme = gr.themes.Default(
        primary_hue=gr.themes.Color(
            c50="#f0f5ff",
            c100="#dce6f9",
            c200="#b8cef3",
            c300="#8eb2eb",
            c400="#6496e0",
            c500="#4A90D9",
            c600="#3a7bc8",
            c700="#2d66b0",
            c800="#1f4f8f",
            c900="#163a6e",
            c950="#0e2650",
        ),
    )

    with gr.Blocks(
        css=css,
        title="Qwen3-TTS Demo",
        js=_build_player_js(PCM_SAMPLE_RATE),
        theme=theme,
    ) as demo:
        gr.Markdown("# Qwen3-TTS Online Serving Demo")
        gr.Markdown(f"**Server:** `{api_base}`")

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter text here, e.g., Hello, how are you?",
                    lines=4,
                )
                with gr.Row():
                    task_type = gr.Radio(
                        choices=TASK_TYPES,
                        value="CustomVoice",
                        label="Task Type",
                        scale=2,
                    )
                    language = gr.Dropdown(
                        choices=SUPPORTED_LANGUAGES,
                        value="Auto",
                        label="Language",
                        scale=1,
                    )
                voice = gr.Dropdown(
                    choices=voices,
                    value=voices[0] if voices else None,
                    label="Speaker",
                    visible=True,
                    allow_custom_value=True,
                )
                instructions = gr.Textbox(
                    label="Instructions",
                    placeholder="e.g., Speak with excitement / A warm, friendly female voice",
                    lines=2,
                    visible=True,
                    info="Optional style/emotion instructions",
                )
                ref_audio = gr.Audio(
                    label="Reference Audio (upload for voice cloning)",
                    type="numpy",
                    sources=["upload", "microphone"],
                    visible=False,
                )
                ref_audio_url = gr.Textbox(
                    label="Reference Audio URL",
                    placeholder="https://example.com/reference.wav",
                    lines=1,
                    visible=False,
                )
                ref_text = gr.Textbox(
                    label="Reference Audio Transcript",
                    placeholder="Transcript of reference audio (optional, improves quality)",
                    lines=2,
                    visible=False,
                )
                x_vector_only = gr.Checkbox(
                    label="Use x-vector only",
                    value=False,
                    visible=False,
                    info="Skip reference transcript, use speaker embedding only",
                )

                with gr.Row():
                    response_format = gr.Dropdown(
                        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
                        value="pcm",
                        label="Audio Format",
                        scale=1,
                        interactive=False,
                    )
                    speed = gr.Slider(
                        minimum=0.25,
                        maximum=4.0,
                        value=1.0,
                        step=0.05,
                        label="Speed",
                        scale=1,
                        interactive=False,
                    )
                    stream_checkbox = gr.Checkbox(
                        label="Stream (gapless)",
                        value=True,
                        info="AudioWorklet streaming (recommended)",
                        scale=1,
                    )

                generate_btn = gr.Button(
                    "Generate Speech",
                    variant="primary",
                    size="lg",
                    elem_id="generate-btn",
                )

            with gr.Column(scale=2):
                player_html = gr.HTML(value=PLAYER_HTML, visible=True)
                audio_output = gr.Audio(
                    label="Generated Audio",
                    interactive=False,
                    autoplay=True,
                    visible=False,
                )
                gr.Markdown(
                    "### Task Types\n"
                    "- **CustomVoice**: Predefined speaker with optional style\n"
                    "- **VoiceDesign**: Describe voice in natural language\n"
                    "- **Base**: Clone from reference audio (upload or URL)"
                )

        # Hidden textbox to pass payload from Python → JavaScript
        hidden_payload = gr.Textbox(visible=False, elem_id="tts-payload")

        # ── Event wiring ─────────────────────────────────────────
        task_type.change(
            fn=on_task_type_change,
            inputs=[task_type],
            outputs=[voice, instructions, ref_audio, ref_audio_url, ref_text, x_vector_only],
        )

        def on_stream_change(stream: bool):
            if stream:
                return (
                    gr.update(value="pcm", interactive=False),
                    gr.update(interactive=False),
                    gr.update(visible=True),  # player
                    gr.update(visible=False),  # audio
                )
            return (
                gr.update(value="wav", interactive=True),
                gr.update(interactive=True),
                gr.update(visible=False),
                gr.update(visible=True),
            )

        stream_checkbox.change(
            fn=on_stream_change,
            inputs=[stream_checkbox],
            outputs=[response_format, speed, player_html, audio_output],
        )

        all_inputs = [
            text_input,
            task_type,
            voice,
            language,
            instructions,
            ref_audio,
            ref_audio_url,
            ref_text,
            x_vector_only,
            response_format,
            speed,
        ]

        def on_generate(stream_enabled, *args):
            if stream_enabled:
                text, task_type_v, voice_v, lang_v, instr_v, ref_a, ref_url, ref_t, xvec, _fmt, _spd = args
                payload = build_payload(
                    text,
                    task_type_v,
                    voice_v,
                    lang_v,
                    instr_v,
                    ref_a,
                    ref_url,
                    ref_t,
                    xvec,
                    stream=True,
                )
                # Add timestamp so repeated identical requests still trigger change
                payload["_nonce"] = int(__import__("time").time() * 1000)
                return json.dumps(payload), gr.update()
            else:
                audio = generate_speech(api_base, *args)
                return "", audio

        generate_btn.click(
            fn=on_generate,
            inputs=[stream_checkbox] + all_inputs,
            outputs=[hidden_payload, audio_output],
        )

        # When hidden_payload changes, trigger JavaScript streaming
        hidden_payload.change(
            fn=None,
            inputs=[hidden_payload],
            js="(p) => { if (p && p.trim()) { const d = JSON.parse(p); delete d._nonce; window.ttsGenerate(d); } return p; }",
        )

        demo.queue()

    return gr.mount_gradio_app(fastapi_app, demo, path="/")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradio demo for Qwen3-TTS with gapless AudioWorklet streaming.",
    )
    add_common_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    print(f"Connecting to vLLM server at: {args.api_base}")

    app = create_app(args.api_base)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
