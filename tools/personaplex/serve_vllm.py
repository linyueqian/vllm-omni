# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal e2e PersonaPlex serving with the 7B temporal transformer on vLLM.

Productizes the verified Milestone-B loop into an HTTP server: the temporal
transformer runs on vLLM (exported stock-Llama checkpoint; raw last hidden via the
embed pooler), while Moshi's depformer + Mimi run in-process. One model load at
startup; `/generate` takes an input WAV and returns the agent's audio + inner
monologue. Functional serve (re-prefill per frame, O(n^2)) — not yet real-time;
that is the omni paged-KV-streaming pipeline (future).

    HF_TOKEN=... VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=1 \
        python tools/personaplex/serve_vllm.py --hf /home/yueqian/helium_hf --port 8100
"""

import argparse
import base64

import numpy as np
import torch
from pydantic import BaseModel

_STATE: dict = {}


class GenReq(BaseModel):
    input_wav_b64: str
    seconds: float = 6.0


def _load(hf_dir: str) -> None:
    import sentencepiece
    import sphn
    from huggingface_hub import hf_hub_download
    from moshi.models import LMGen, loaders
    from vllm import LLM

    dev = "cuda"
    mimi_w = hf_hub_download("nvidia/personaplex-7b-v1", loaders.MIMI_NAME)
    moshi_w = hf_hub_download("nvidia/personaplex-7b-v1", loaders.MOSHI_NAME)
    tok = sentencepiece.SentencePieceProcessor(hf_hub_download("nvidia/personaplex-7b-v1", loaders.TEXT_TOKENIZER_NAME))
    mimi = loaders.get_mimi(mimi_w, dev)
    lm = loaders.get_moshi_lm(moshi_w, device=dev).eval()
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=dev,
        frame_rate=mimi.frame_rate,
        use_sampling=False,
    )
    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    llm = LLM(
        model=hf_dir,
        runner="pooling",
        convert="embed",
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        max_model_len=3000,
        enable_prompt_embeds=True,
        pooler_config={"pooling_type": "LAST", "use_activation": False},
    )
    _STATE.update(
        dev=dev,
        mimi=mimi,
        lm=lm,
        lm_gen=lm_gen,
        llm=llm,
        sphn=sphn,
        tok=tok,
        frame=int(mimi.sample_rate / mimi.frame_rate),
        text_linear=lm.text_linear,
    )


def _generate(input_pcm: np.ndarray, seconds: float) -> tuple[np.ndarray, str, int]:
    from moshi.models.lm import _iterate_audio as iter_audio
    from moshi.models.lm import encode_from_sphn

    dev, mimi, lm, lm_gen = _STATE["dev"], _STATE["mimi"], _STATE["lm"], _STATE["lm_gen"]
    text_linear, frame = _STATE["text_linear"], _STATE["frame"]
    llm = _STATE["llm"]

    mimi.reset_streaming()
    lm_gen.reset_streaming()
    buf: list[torch.Tensor] = []

    def vllm_forward_codes(seq: torch.Tensor):
        emb = lm.embed_codes(seq)
        buf.append(emb.detach())
        full = torch.cat(buf, dim=1)
        out = llm.embed({"prompt_embeds": full.squeeze(0).to(torch.bfloat16).cpu()})
        h = torch.tensor(np.asarray(out[0].outputs.embedding), device=dev, dtype=emb.dtype).reshape(1, 1, -1)
        return h, text_linear(h)[:, None]

    lm_gen._streaming_state.graphed_main = vllm_forward_codes

    user = np.ascontiguousarray(input_pcm, dtype=np.float32)[: int(seconds * mimi.sample_rate)]
    user = user.reshape(1, -1)
    frames, texts = [], []
    for enc in encode_from_sphn(mimi, iter_audio(user, sample_interval_size=frame, pad=True), max_batch=1):
        for c in range(enc.shape[-1]):
            tokens = lm_gen.step(enc[:, :, c : c + 1])
            if tokens is None:
                continue
            frames.append(mimi.decode(tokens[:, 1:9]).detach().cpu().numpy()[0, 0].astype(np.float32))
            tid = int(tokens[0, 0, 0].item())
            if tid not in (0, 3):
                texts.append(_STATE["tok"].id_to_piece(tid).replace("▁", " "))
    audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
    return audio, "".join(texts), len(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()

    _load(args.hf)

    from fastapi import Body, FastAPI

    app = FastAPI()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "v": 2, "model": "personaplex-7b (temporal on vLLM)"}

    @app.post("/generate")
    def generate(req: GenReq = Body(...)) -> dict:
        import os
        import tempfile

        sphn = _STATE["sphn"]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(base64.b64decode(req.input_wav_b64))
            tmp = tf.name
        try:
            pcm, sr = sphn.read(tmp)
        finally:
            os.unlink(tmp)
        pcm = sphn.resample(pcm, src_sample_rate=sr, dst_sample_rate=_STATE["mimi"].sample_rate)
        audio, text, n = _generate(np.asarray(pcm[0], dtype=np.float32), req.seconds)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as otf:
            out_path = otf.name
        try:
            sphn.write_wav(out_path, audio, _STATE["mimi"].sample_rate)
            wav_bytes = open(out_path, "rb").read()
        finally:
            os.unlink(out_path)
        rms = float(np.sqrt((audio**2).mean())) if audio.size else 0.0
        return {
            "audio_wav_b64": base64.b64encode(wav_bytes).decode(),
            "text": text,
            "frames": n,
            "rms": rms,
            "duration_s": audio.shape[0] / _STATE["mimi"].sample_rate,
        }

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
