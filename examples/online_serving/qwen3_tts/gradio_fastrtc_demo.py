"""Gradio demo for Qwen3-TTS using FastRTC for gapless streaming audio.

Uses WebRTC via FastRTC to deliver audio chunks as a continuous real-time
stream, avoiding the inter-chunk gaps present in Gradio's built-in streaming.

Requires: pip install fastrtc

Supports all 3 task types:
  - CustomVoice: Predefined speaker with optional style instructions
  - VoiceDesign: Natural language voice description
  - Base: Voice cloning from reference audio (URL only in this demo)

Usage:
    # Start the server first (see run_server.sh), then:
    python gradio_fastrtc_demo.py --api-base http://localhost:8000

    # With public sharing:
    python gradio_fastrtc_demo.py --api-base http://localhost:8000 --share
"""

import argparse
import logging
import time
from collections.abc import Generator

import gradio as gr
import numpy as np
from fastrtc import WebRTC
from tts_common import (
    PCM_SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    TASK_TYPES,
    add_common_args,
    build_payload,
    build_ws_config,
    fetch_voices,
    stream_pcm_chunks,
    stream_pcm_chunks_ws,
)

logger = logging.getLogger(__name__)


def make_tts_generator(api_base: str, stats: dict):
    """Create a generator function for FastRTC's mode='receive'.

    FastRTC calls this with the additional_inputs values whenever the user
    clicks "Start Stream". It must be a synchronous generator yielding
    (sample_rate, int16_ndarray) tuples. Timing stats are written to the
    shared `stats` dict so the UI can poll them.
    """

    def tts_generate(
        text: str,
        task_type: str,
        voice: str,
        language: str,
        instructions: str,
        ref_audio_url: str,
        ref_text: str,
        x_vector_only: bool,
        transport: str,
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        if not text or not text.strip():
            return

        stats.update(
            {
                "transport": transport,
                "t_start": time.time(),
                "ttfp": None,
                "total_time": None,
                "chunks": 0,
                "total_samples": 0,
                "done": False,
                "error": None,
            }
        )

        t0 = time.time()
        chunk_count = 0
        total_samples = 0

        try:
            if transport == "WebSocket":
                ws_url = api_base.replace("http://", "ws://").replace("https://", "wss://")
                ws_url = f"{ws_url}/v1/audio/speech/stream"
                config = build_ws_config(
                    task_type,
                    voice,
                    language,
                    instructions,
                    ref_audio=None,
                    ref_audio_url=ref_audio_url,
                    ref_text=ref_text,
                    x_vector_only=x_vector_only,
                )
                pcm_iter = stream_pcm_chunks_ws(ws_url, text, config)
            else:
                payload = build_payload(
                    text,
                    task_type,
                    voice,
                    language,
                    instructions,
                    ref_audio=None,
                    ref_audio_url=ref_audio_url,
                    ref_text=ref_text,
                    x_vector_only=x_vector_only,
                    stream=True,
                )
                pcm_iter = stream_pcm_chunks(api_base, payload)

            for samples in pcm_iter:
                chunk_count += 1
                total_samples += len(samples)
                if chunk_count == 1:
                    stats["ttfp"] = time.time() - t0
                stats["chunks"] = chunk_count
                stats["total_samples"] = total_samples
                yield (PCM_SAMPLE_RATE, samples)

        except Exception as e:
            stats["error"] = str(e)
            logger.exception("TTS streaming error")
        finally:
            stats["total_time"] = time.time() - t0
            stats["done"] = True

    return tts_generate


def _format_stats(stats: dict) -> str:
    """Format the stats dict into a readable string."""
    if not stats or "t_start" not in stats:
        return "*No generation yet.*"
    transport = stats.get("transport", "?")
    ttfp = stats.get("ttfp")
    total_time = stats.get("total_time")
    chunks = stats.get("chunks", 0)
    total_samples = stats.get("total_samples", 0)
    done = stats.get("done", False)
    error = stats.get("error")

    audio_dur = total_samples / PCM_SAMPLE_RATE if total_samples else 0
    elapsed = total_time if done else time.time() - stats["t_start"]

    lines = [f"**Transport:** {transport}"]
    if ttfp is not None:
        lines.append(f"**TTFP:** {ttfp * 1000:.0f} ms")
    else:
        lines.append("**TTFP:** waiting...")
    lines.append(f"**Chunks:** {chunks}")
    lines.append(f"**Audio:** {audio_dur:.2f}s")
    if done:
        lines.append(f"**Total time:** {elapsed:.2f}s")
        if audio_dur > 0 and elapsed > 0:
            rtf = elapsed / audio_dur
            lines.append(f"**RTF:** {rtf:.2f}x")
            lines.append(f"**Throughput:** {1 / rtf:.2f}x realtime")
        if error:
            lines.append(f"**Error:** {error}")
    else:
        lines.append(f"**Elapsed:** {elapsed:.1f}s (generating...)")
    return "\n\n".join(lines)


def build_interface(api_base: str):
    """Build a custom Gradio Blocks UI matching gradio_demo.py layout."""
    voices = fetch_voices(api_base)
    stats: dict = {}
    handler = make_tts_generator(api_base, stats)

    css = """
    #generate-btn button { width: 100%; }
    .task-info { padding: 8px 12px; border-radius: 6px;
                 background: #f0f4ff; margin-bottom: 8px; }
    """

    with gr.Blocks(css=css, title="Qwen3-TTS (FastRTC)") as demo:
        gr.Markdown("# Qwen3-TTS Online Serving Demo (FastRTC)")
        gr.Markdown(f"**Server:** `{api_base}` | **Transport:** WebRTC (gapless streaming)")

        with gr.Row():
            # Left column: inputs
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
                    placeholder=("e.g., Speak with excitement / A warm, friendly female voice"),
                    lines=2,
                    visible=True,
                    info="Optional style/emotion instructions",
                )

                # Base (voice clone) controls
                # Note: file upload is not supported with WebRTC,
                # use a URL instead.
                ref_audio_url = gr.Textbox(
                    label="Reference Audio URL",
                    placeholder=("https://example.com/reference.wav (alternative to uploading)"),
                    lines=1,
                    visible=False,
                )
                ref_text = gr.Textbox(
                    label="Reference Audio Transcript",
                    placeholder=("Transcript of the reference audio (optional, improves quality)"),
                    lines=2,
                    visible=False,
                )
                x_vector_only = gr.Checkbox(
                    label="Use x-vector only",
                    value=False,
                    visible=False,
                    info=("Skip reference transcript, use speaker embedding only (lower quality)"),
                )

                transport = gr.Radio(
                    choices=["HTTP", "WebSocket"],
                    value="HTTP",
                    label="Streaming Transport",
                    info=(
                        "HTTP: continuous stream, best for complete text. "
                        "WebSocket: sentence-level via /v1/audio/speech/stream."
                    ),
                )

                generate_btn = gr.Button(
                    "Generate Speech",
                    variant="primary",
                    size="lg",
                    elem_id="generate-btn",
                )

            # Right column: output
            with gr.Column(scale=2):
                webrtc_output = WebRTC(
                    label="Generated Audio (WebRTC)",
                    mode="receive",
                    modality="audio",
                )
                stats_display = gr.Markdown(
                    value="*No generation yet.*",
                    label="Streaming Stats",
                )
                timer = gr.Timer(value=0.5, active=False)
                gr.Markdown(
                    "### Task Types\n"
                    "- **CustomVoice**: Use a predefined speaker "
                    "(Vivian, Ryan, etc.) with optional style instructions\n"
                    "- **VoiceDesign**: Describe the desired voice in natural "
                    "language (instructions required)\n"
                    "- **Base**: Clone a voice from reference audio "
                    "(provide a URL)"
                )

        # Dynamic UI updates for task type
        # FastRTC doesn't support file upload, so we only toggle
        # 5 components (no ref_audio)
        def on_task_type_change_fastrtc(task_type: str):
            if task_type == "CustomVoice":
                return (
                    gr.update(visible=True),  # voice
                    gr.update(visible=True, info="Optional style/emotion instructions"),
                    gr.update(visible=False),  # ref_audio_url
                    gr.update(visible=False),  # ref_text
                    gr.update(visible=False),  # x_vector_only
                )
            elif task_type == "VoiceDesign":
                return (
                    gr.update(visible=False),
                    gr.update(visible=True, info="Required: describe the voice style"),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
            elif task_type == "Base":
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                )
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        task_type.change(
            fn=on_task_type_change_fastrtc,
            inputs=[task_type],
            outputs=[
                voice,
                instructions,
                ref_audio_url,
                ref_text,
                x_vector_only,
            ],
        )

        # Stats polling: timer updates the display every 0.5s
        def poll_stats():
            text = _format_stats(stats)
            # Stop timer once generation is done
            if stats.get("done", False):
                return text, gr.Timer(active=False)
            return text, gr.Timer(active=True)

        timer.tick(fn=poll_stats, outputs=[stats_display, timer])

        # Start timer when generate is clicked
        generate_btn.click(
            fn=lambda: (gr.Timer(active=True), "*Starting...*"),
            outputs=[timer, stats_display],
        )

        # Wire up WebRTC streaming
        all_inputs = [
            text_input,
            task_type,
            voice,
            language,
            instructions,
            ref_audio_url,
            ref_text,
            x_vector_only,
            transport,
        ]

        webrtc_output.stream(
            fn=handler,
            inputs=all_inputs,
            outputs=[webrtc_output],
            trigger=generate_btn.click,
        )

        demo.queue()
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio FastRTC demo for Qwen3-TTS (gapless streaming).")
    add_common_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    print(f"Connecting to vLLM server at: {args.api_base}")

    demo = build_interface(args.api_base)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
