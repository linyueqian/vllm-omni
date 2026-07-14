# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-1 Code2Wav model for PersonaPlex (Moshi finetune).

This is the ``LLM_GENERATION`` codec stage of the 2-stage PersonaPlex pipeline.
It consumes the per-frame audio codebooks produced by the talker (stage 0) and
the depformer (built by the lead), and turns them into 24 kHz PCM by calling the
external Mimi neural codec from the ``moshi`` package.

Layout contract (mirrors ``Qwen3TTSCode2Wav``):

* Per request, ``input_ids`` holds a flat codebook-major codec sequence
  ``[k * F]`` where ``k`` is the number of *active* codebooks Mimi decodes
  (``num_codebooks``, i.e. the ``cb 0..7`` slice) and ``F`` is the number of
  codec frames. The talker emits a 17-row token stack per frame; the input
  processor (built by the lead) keeps only rows ``1:9`` (the 8 PCM-bearing audio
  codebooks) and flattens them codebook-major before this stage.
* ``forward(...)`` returns an :class:`OmniOutput` whose ``multimodal_outputs``
  carries ``{"model_outputs": [wav_per_request], "sr": [sr_per_request]}`` — the
  exact shape ``Qwen3TTSCode2Wav`` returns, so the downstream audio packer is
  unchanged.

The Mimi decoder is the transformers ``MimiModel`` (kyutai/mimi weights), not
in vLLM's safetensors loader, so they are loaded eagerly in ``load_weights``
part of the vLLM weights iterator (the codec owns its own checkpoint; see the
experimental fullduplex package for the same pattern).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


class PersonaPlexCode2Wav(nn.Module):
    """Stage-1 Code2Wav model for PersonaPlex (GenerationModelRunner).

    Wraps the external Mimi decoder. The vLLM generation runner only calls a
    handful of methods on this module: ``embed_input_ids`` (dummy embeddings),
    ``compute_logits`` (none -- this stage never samples), ``forward`` (the
    actual codec->PCM decode), ``make_omni_output`` (output normalization), and
    ``load_weights`` (eager Mimi construction).
    """

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.config = vllm_config.model_config.hf_config

        # Runner-facing capability flags, matching Qwen3TTSCode2Wav so the
        # GenerationModelRunner drives this stage the same way.
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        mimi_cfg = getattr(self.config, "mimi_config", None)
        # Number of *active* codebooks Mimi decodes to PCM (cb 0..7).
        self._num_codebooks = int(getattr(mimi_cfg, "num_codebooks", 8))
        self._output_sample_rate = int(getattr(mimi_cfg, "sample_rate", 24000))
        self._samples_per_frame = int(getattr(mimi_cfg, "samples_per_frame", 1920))
        self._mimi_name = getattr(mimi_cfg, "mimi_name", None) or getattr(self.config, "mimi_name", None)

        # The Mimi module is constructed in load_weights() (it owns its own
        # weight format) and assigned here so vLLM's memory profiler can see it.
        self.mimi: nn.Module | None = None
        self._mimi_device: torch.device | None = None

    # ------------------------------------------------------------------
    # Runner-facing no-op / placeholder hooks (mirror Qwen3TTSCode2Wav).
    # ------------------------------------------------------------------
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # This stage ignores token embeddings; keep a stable dummy embedding so
        # the vLLM runner has a valid hidden-state placeholder.
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> None:
        # Code2Wav never samples; the runner short-circuits when logits are None.
        return None

    def _split_request_ids(
        self,
        ids: torch.Tensor,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """Split concatenated ``input_ids`` into per-request segments.

        Uses ``seq_token_counts`` (injected by the runner via model_kwargs) when
        available, falling back to forward-context ``ubatch_slices`` when
        micro-batching is active. Returns ``[ids]`` for single-request batches.
        Mirrors ``Qwen3TTSCode2Wav._split_request_ids``.
        """
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + s)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

    # ------------------------------------------------------------------
    # Decode.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode flat codebook-major codec ids into PCM via Mimi.

        ``input_ids`` per request is ``[k * F]`` (codebook-major), where ``k``
        is ``self._num_codebooks``. We reshape to ``[k, F]``, batch a leading
        dim for Mimi, and call ``mimi.decode`` to produce the waveform. Mimi's
        ``decode`` is one-shot: faithful to the seam
        ``tokens[:, 1:9] -> mimi.decode -> 1920 * F PCM`` documented for
        PersonaPlex.
        """
        sr_val = int(self._output_sample_rate)
        sr_tensor = torch.tensor(sr_val, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))
        num_req = len(request_ids_list)

        if self.mimi is None:
            raise RuntimeError("PersonaPlexCode2Wav.forward called before Mimi was loaded in load_weights().")

        k = int(self._num_codebooks)
        device = self._mimi_device or ids.device
        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        for i, req_ids in enumerate(request_ids_list):
            flat = req_ids
            n = int(flat.numel())
            if n == 0 or n % k != 0:
                if n > 0:
                    logger.warning(
                        "PersonaPlex Code2Wav input_ids length %d not divisible by "
                        "num_codebooks %d; skipping malformed request.",
                        n,
                        k,
                    )
                continue
            frames = n // k
            # [k * F] -> [k, F] -> [1, k, F] (Mimi expects [B, K, T]).
            codes_kf = flat.reshape(k, frames).to(device=device)
            codes_bkf = codes_kf.unsqueeze(0)
            wav = self.mimi.decode(codes_bkf, return_dict=True).audio_values
            wav = wav[..., : frames * self._samples_per_frame]
            wav = self._flatten_wav(wav)
            if wav.numel() > 0:
                audios[i] = (wav if wav.dtype == torch.float32 else wav.to(torch.float32)).reshape(-1)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    @staticmethod
    def _flatten_wav(wav: torch.Tensor) -> torch.Tensor:
        """Normalize Mimi's decode output to a 1D PCM tensor.

        Mimi may return ``[B, samples]`` or ``[B, C, samples]`` depending on the
        moshi version; PersonaPlex is mono (``C == 1``), so collapse to 1D.
        """
        if wav.dim() == 3:
            # [B, C, T] -> take batch 0, channel 0.
            return wav[0, 0]
        if wav.dim() == 2:
            # [B, T] -> batch 0.
            return wav[0]
        return wav.reshape(-1)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput | tuple, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)

        if not (isinstance(model_outputs, tuple) and len(model_outputs) == 2):
            raise TypeError(
                "PersonaPlexCode2Wav expected OmniOutput, OmniOutput tuple, "
                f"or (audio_tensor, sr) outputs, got {type(model_outputs)}"
            )

        audio_tensor, sr = model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": audio_tensor,
                "sr": sr,
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Construct and load the external Mimi decoder.

        The primary vLLM weights iterator carries no Code2Wav parameters (Mimi
        owns its own checkpoint format), so it is drained and the Mimi module is
        built from the moshi package's loader.
        """
        # Drain the primary iterator so callers don't hang on an unconsumed
        # generator; none of these weights belong to this stage.
        for _ in weights:
            pass

        from transformers import MimiModel

        device = self.vllm_config.device_config.device
        self.mimi = MimiModel.from_pretrained("kyutai/mimi").to(str(device)).eval()
        self._mimi_device = torch.device(str(device))
        reported_sr = getattr(self.mimi.config, "sampling_rate", None)
        if reported_sr is not None:
            self._output_sample_rate = int(reported_sr)
        logger.info(
            "PersonaPlex Code2Wav loaded Mimi (transformers): num_codebooks=%d sample_rate=%d samples_per_frame=%d",
            self._num_codebooks,
            self._output_sample_rate,
            self._samples_per_frame,
        )
        return {f"mimi.{name}" for name, _ in self.mimi.named_parameters()}
