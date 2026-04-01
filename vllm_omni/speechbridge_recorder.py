"""SpeechBridge feature recorder for vllm-omni.

Collects hidden states and codec codes during TTS inference for bridge training.
Activated by setting SPEECHBRIDGE_COLLECT_DIR environment variable.

The recorder accumulates per-step data during inference and saves to disk when
a request finishes. Output format matches what BridgeDataset expects:
  - hidden_states: [T, H] tensor (fp16)
  - codebook_codes: [T, Q] tensor (int64)
  - text: str
  - n_steps: int
"""

import logging
import threading
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class SpeechBridgeRecorder:
    """Thread-safe feature recorder.

    Called from talker's postprocess() and talker_mtp() to accumulate
    hidden states and codec codes per decode step. Since vllm-omni
    processes one request at a time in the talker stage (max_batch_size=1),
    we use a simple single-request buffer.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._hidden_states: list[torch.Tensor] = []
        self._codec_codes: list[torch.Tensor] = []
        self._counter = 0
        self._total_steps = 0
        logger.info("[SpeechBridge] Recorder initialized -> %s", self.output_dir)

    def record_hidden_state(self, h: torch.Tensor):
        """Called from postprocess() for each decode step."""
        with self._lock:
            self._hidden_states.append(h.detach().cpu().half())

    def record_codec_codes(self, codes: torch.Tensor):
        """Called from talker_mtp() for each decode step."""
        with self._lock:
            self._codec_codes.append(codes.detach().cpu())

    def save_utterance(self, req_id: str, text: str = ""):
        """Save accumulated features for a finished request."""
        with self._lock:
            hs = list(self._hidden_states)
            cc = list(self._codec_codes)
            self._hidden_states.clear()
            self._codec_codes.clear()
            idx = self._counter
            self._counter += 1

        if not hs:
            logger.debug("[SpeechBridge] No hidden states for req %s, skipping", req_id)
            return

        n_steps = len(hs)
        save_data = {
            "hidden_states": torch.stack(hs),  # [T, H] fp16
            "text": text,
            "n_steps": n_steps,
        }
        if cc:
            save_data["codebook_codes"] = torch.stack(cc)  # [T, Q]

        out_path = self.output_dir / f"utterance_{idx:04d}.pt"
        torch.save(save_data, out_path)
        self._total_steps += n_steps

        logger.info(
            "[SpeechBridge] Saved utterance_%04d: %d steps | total: %d utt, %d steps | text: %.50s...",
            idx,
            n_steps,
            idx + 1,
            self._total_steps,
            text,
        )

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "utterances_saved": self._counter,
                "total_steps": self._total_steps,
                "pending_hidden_states": len(self._hidden_states),
            }
