# SPDX-License-Identifier: Apache-2.0
"""VoxCPM serving adapter (AR base-LM + diffusion side-computation)."""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class VoxCPMAdapter(ARTTSAdapter):
    """VoxCPM shares ``latent_generator`` with VoxCPM2; disambiguated by the
    presence of a ``vae`` stage (and/or ``model_arch``)."""

    stage_keys = frozenset({"latent_generator", "vae"})
    name = "voxcpm"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        # VoxCPM has no dedicated validator; the legacy dispatch falls through
        # to the Qwen3-TTS validator.
        return self.ctx.server._validate_qwen_tts_request(request)

    async def build(self, request: "OpenAICreateSpeechRequest", sampling_params_list: list) -> PreparedRequest:
        prompt, tts_params, warmup_key = await self.ctx.server._build_default_tts_request(request)
        return PreparedRequest(
            prompt=prompt,
            tts_params=tts_params,
            model_type="voxcpm",
            warmup_artifact_key=warmup_key,
        )
