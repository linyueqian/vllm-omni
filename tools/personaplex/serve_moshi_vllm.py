# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the OFFICIAL PersonaPlex web UI / WS server, but with the temporal
transformer served by vLLM.

The official ``moshi.server`` provides the browser mic client, the WebSocket +
Opus transport, and the full-duplex streaming loop. We keep all of that and swap
ONLY the 7B temporal transformer's per-frame forward to vLLM's paged-attention
engine — by monkeypatching ``LMGen._init_streaming_state`` so every session's
``graphed_main`` re-prefills the growing embedding sequence through a vLLM embed
call (raw last hidden via ``use_activation=False``). Moshi's depformer + Mimi stay
in-process; the wire protocol and web client are 100% official.

Caveat: re-prefill per frame is O(n^2), so the live conversation drifts away from
real-time as it grows — functional, not yet real-time (that is the omni paged-KV
pipeline). Good enough to confirm the official client works against our backend.

    HF_TOKEN=... VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0 \
        PPLEX_HELIUM_DIR=/path/to/helium_hf PPLEX_SSL_DIR=/tmp/ssl PPLEX_PORT=8998 \
        python tools/personaplex/serve_moshi_vllm.py
"""

import sys

import numpy as np
import torch

_LLM = None


def _patch_temporal_with_vllm(llm) -> None:
    """Make every LMGen session run its temporal forward on vLLM."""
    import moshi.models.lm as lm_mod

    orig_init = lm_mod.LMGen._init_streaming_state

    def patched_init(self, batch_size: int):
        state = orig_init(self, batch_size)
        buf: list[torch.Tensor] = []

        def vllm_forward_codes(seq: torch.Tensor):
            emb = self.lm_model.embed_codes(seq)  # [1,1,dim]
            buf.append(emb.detach())
            full = torch.cat(buf, dim=1)
            out = llm.embed({"prompt_embeds": full.squeeze(0).to(torch.bfloat16).cpu()})
            h = torch.tensor(np.asarray(out[0].outputs.embedding), device=emb.device, dtype=emb.dtype).reshape(1, 1, -1)
            return h, self.lm_model.text_linear(h)[:, None]

        state.graphed_main = vllm_forward_codes
        return state

    lm_mod.LMGen._init_streaming_state = patched_init


def main() -> None:
    # Config via env vars (NOT argparse): we hand sys.argv to moshi.server.main(), so
    # our own CLI flags would collide with moshi.server's parser (and the vLLM spawn).
    import os

    helium_dir = os.environ["PPLEX_HELIUM_DIR"]
    ssl_dir = os.environ.get("PPLEX_SSL_DIR", "/home/yueqian/pplex_ssl")
    port = os.environ.get("PPLEX_PORT", "8998")
    repo = os.environ.get("PPLEX_REPO", "nvidia/personaplex-7b-v1")

    from vllm import LLM

    global _LLM
    _LLM = LLM(
        model=helium_dir,
        runner="pooling",
        convert="embed",
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.2,
        max_model_len=3000,
        enable_prompt_embeds=True,
        pooler_config={"pooling_type": "LAST", "use_activation": False},
    )
    _patch_temporal_with_vllm(_LLM)

    # Hand off to the official server's own CLI (loads moshi LM + Mimi, serves the web UI/WS).
    import moshi.server

    sys.argv = [
        "moshi.server",
        "--host",
        "localhost",
        "--port",
        str(port),
        "--ssl",
        ssl_dir,
        "--hf-repo",
        repo,
        "--device",
        "cuda",
    ]
    print("[serve_moshi_vllm] handing off to moshi.server argv:", sys.argv, flush=True)
    moshi.server.main()


if __name__ == "__main__":
    main()
