"""SpeechBridge inference integration for vllm-omni.

Loads a trained bridge checkpoint and provides bridge predictions at skipped
decode steps. Activated by setting SPEECHBRIDGE_CHECKPOINT environment variable.

For DAgger collection, also set SPEECHBRIDGE_DAGGER_DIR to save oracle pairs.
"""

import logging
import os
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _import_bridge_model():
    """Import bridge model builder from speechbridge project."""
    speechbridge_src = Path(__file__).resolve().parent.parent.parent / "speechbridge" / "src"
    if not speechbridge_src.exists():
        speechbridge_src = Path(os.path.expanduser("~/proj/speechbridge/src"))
    if str(speechbridge_src) not in sys.path:
        sys.path.insert(0, str(speechbridge_src))
    from bridge.model import build_bridge

    return build_bridge


class SpeechBridgeInference:
    """Bridge inference state for a single talker instance.

    Tracks decode steps, manages bridge predictions at skipped steps,
    and optionally collects DAgger training pairs.
    """

    def __init__(self, checkpoint_path: str, skip_frequency: int = 3, device: str | None = None):
        self.skip_frequency = skip_frequency
        if device is None:
            device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._decode_step = 0
        self._last_anchor_hidden: torch.Tensor | None = None
        self._codec_history: list[torch.Tensor] = []

        # DAgger collection
        dagger_dir = os.environ.get("SPEECHBRIDGE_DAGGER_DIR")
        self._dagger_dir = Path(dagger_dir) if dagger_dir else None
        if self._dagger_dir:
            self._dagger_dir.mkdir(parents=True, exist_ok=True)
        self._dagger_pairs: list[dict] = []
        self._dagger_counter = 0

        # Load bridge model from checkpoint
        build_bridge = _import_bridge_model()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = ckpt.get("args", {})

        arch = args.get("arch", "mlp")
        bridge_kwargs = {}
        if arch == "mlp":
            bridge_kwargs = {
                "talker_hidden_size": args.get("talker_hidden_size", 2048),
                "hidden_size": args.get("bridge_hidden_size", 2048),
                "num_layers": args.get("bridge_num_layers", 4),
                "dropout": 0.0,  # no dropout at inference
                "max_skip": args.get("max_skip", 1),
                "step_embed_dim": args.get("step_embed_dim", 0),
            }
        elif arch == "transformer":
            bridge_kwargs = {
                "talker_hidden_size": args.get("talker_hidden_size", 2048),
                "bridge_hidden_size": args.get("bridge_hidden_size", 512),
                "num_layers": args.get("bridge_num_layers", 6),
                "num_heads": args.get("bridge_num_heads", 8),
                "num_codebooks": args.get("num_codebooks", 16),
                "codec_context_len": args.get("codec_context_len", 4),
                "dropout": 0.0,
            }

        self.bridge = build_bridge(arch, **bridge_kwargs)
        self.bridge.load_state_dict(ckpt["model"])
        self.bridge = self.bridge.float().to(device).eval()
        self.codec_context_len = args.get("codec_context_len", 4)

        # Whether to actually skip the forward pass (vs just replacing hidden in postprocess)
        self.skip_forward = os.environ.get("SPEECHBRIDGE_SKIP_FORWARD", "1") == "1"

        # Stats
        self.total_anchor = 0
        self.total_bridged = 0
        self._forward_was_skipped = False

        logger.info(
            "[SpeechBridge] Inference loaded from %s (arch=%s, freq=%d, epoch=%d, skip_forward=%s)",
            checkpoint_path,
            arch,
            skip_frequency,
            ckpt.get("epoch", -1),
            self.skip_forward,
        )

    @property
    def is_skip_step(self) -> bool:
        """Whether the current decode step should use bridge prediction."""
        if self._last_anchor_hidden is None:
            return False
        return (self._decode_step % self.skip_frequency) != 0

    @property
    def skip_distance(self) -> int:
        """Current distance from the last anchor step."""
        return self._decode_step % self.skip_frequency

    def on_new_request(self):
        """Reset state for a new request."""
        self._decode_step = 0
        self._last_anchor_hidden = None
        self._codec_history.clear()
        self._forward_was_skipped = False

    def get_bridge_prediction(self) -> torch.Tensor:
        """Compute bridge prediction for the current skipped step."""
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            h_prev = self._last_anchor_hidden.float().to(device=self.device).reshape(1, -1)

            # Codec context
            codec_ctx = self._get_codec_context()

            # Skip distance
            skip_dist = torch.tensor([self.skip_distance], device=self.device, dtype=torch.long)

            h_pred = self.bridge(h_prev, codec_ctx, skip_distance=skip_dist)
        return h_pred.squeeze(0).to(dtype=torch.bfloat16).cpu().contiguous()

    def on_postprocess(self, real_hidden: torch.Tensor) -> torch.Tensor:
        """Called from postprocess() with the hidden state from forward().

        Returns the hidden state to use for the code predictor:
        - At anchor steps: returns real hidden (and caches it)
        - At skipped steps (forward was skipped): passes through bridge hidden
        - At skipped steps (forward ran, e.g. DAgger mode): computes bridge prediction
        """
        if self._forward_was_skipped:
            # forward() already returned bridge hidden - just pass through
            self._forward_was_skipped = False
            self.total_bridged += 1
            self._decode_step += 1
            return real_hidden  # this IS the bridge hidden from forward()

        if self.is_skip_step:
            # Forward ran but we still want bridge prediction (DAgger collection mode)
            bridge_hidden = self.get_bridge_prediction()
            self.total_bridged += 1

            # DAgger: save (bridge_pred, oracle) pair
            if self._dagger_dir is not None:
                self._dagger_pairs.append(
                    {
                        "h_prev": self._last_anchor_hidden.clone(),
                        "h_bridge": bridge_hidden.clone(),
                        "h_oracle": real_hidden.clone(),
                        "skip_distance": self.skip_distance,
                        "decode_step": self._decode_step,
                    }
                )

            self._decode_step += 1
            return bridge_hidden
        else:
            # Anchor step: cache real hidden
            self._last_anchor_hidden = real_hidden.clone()
            self.total_anchor += 1
            self._decode_step += 1
            return real_hidden

    def on_codec_codes(self, codes: torch.Tensor):
        """Track codec codes for bridge conditioning."""
        self._codec_history.append(codes.detach().cpu())

    def save_dagger_data(self, req_id: str):
        """Save accumulated DAgger pairs for a finished request."""
        if not self._dagger_dir or not self._dagger_pairs:
            return

        pairs = list(self._dagger_pairs)
        self._dagger_pairs.clear()
        idx = self._dagger_counter
        self._dagger_counter += 1

        save_data = {
            "pairs": pairs,
            "n_pairs": len(pairs),
            "req_id": req_id,
        }
        out_path = self._dagger_dir / f"dagger_{idx:04d}.pt"
        torch.save(save_data, out_path)
        logger.info("[SpeechBridge] Saved %d DAgger pairs -> %s", len(pairs), out_path)

    def _get_codec_context(self) -> torch.Tensor | None:
        if not self._codec_history:
            return None
        K = min(self.codec_context_len, len(self._codec_history))
        ctx = torch.stack(self._codec_history[-K:])  # [K, num_codebooks]
        if ctx.shape[0] < self.codec_context_len:
            pad = torch.zeros(
                self.codec_context_len - ctx.shape[0],
                ctx.shape[1],
                dtype=ctx.dtype,
            )
            ctx = torch.cat([pad, ctx], dim=0)
        return ctx.unsqueeze(0).to(self.device)  # [1, K, num_codebooks]

    def log_stats(self):
        total = self.total_anchor + self.total_bridged
        if total == 0:
            return
        pct = self.total_bridged / total * 100
        logger.info(
            "[SpeechBridge] Stats: %d anchor + %d bridged = %d total (%.1f%% bridged)",
            self.total_anchor,
            self.total_bridged,
            total,
            pct,
        )
