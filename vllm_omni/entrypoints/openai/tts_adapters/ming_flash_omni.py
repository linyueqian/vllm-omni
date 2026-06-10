# SPDX-License-Identifier: Apache-2.0
"""Ming-flash-omni TTS serving adapter."""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class MingFlashOmniAdapter(ARTTSAdapter):
    stage_keys = frozenset({"ming_tts"})
    name = "ming_flash_omni_tts"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        return self.ctx.server._validate_ming_flash_omni_tts_request(request)

    async def build(self, request: "OpenAICreateSpeechRequest", sampling_params_list: list) -> PreparedRequest:
        prompt = self.ctx.server._build_ming_flash_omni_prompt(request)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="ming_flash_omni_tts")
