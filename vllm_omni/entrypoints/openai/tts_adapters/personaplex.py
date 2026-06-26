# SPDX-License-Identifier: Apache-2.0
"""PersonaPlex serving adapter.

Dispatch surface for PersonaPlex (a Moshi finetune served as a 2-stage TTS):
the serving orchestrator routes prompt/param building through ``build()``
instead of an inline branch in ``_prepare_speech_generation``.

Mirrors the Qwen3-TTS adapter (RFC #4327): it reuses the single-source helper
implementations on the serving instance through ``ctx.server`` rather than
copying them, so the streaming and batch paths stay behaviour-identical.

NOTE: ``_detect_tts_model_type`` in ``serving_speech.py`` must map the
PersonaPlex stage (``model_stage="personaplex"`` /
``model_arch="PersonaPlexTalkerForConditionalGeneration"``) to the ``name``
below, and the serving instance must expose ``_build_personaplex_request`` /
route ``_validate_tts_request`` for this model type. Both are part of the
talker<->serving wiring built by the lead.
"""

from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)


@register_tts_adapter
class PersonaPlexAdapter(ARTTSAdapter):
    """Adapter for PersonaPlex (AR ``engine_client`` backend)."""

    stage_keys = frozenset({"personaplex"})
    name = "personaplex"

    def normalize(self, request: "OpenAICreateSpeechRequest") -> None:
        """PersonaPlex normalization (task inference, voice lowercasing) is
        performed inside ``validate`` today, kept fused for a strict behaviour
        match with the other AR adapters."""

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        # Route through the shared dispatcher (which forwards to the
        # PersonaPlex-specific validator for this model type) rather than a leaf
        # validator directly, matching the Qwen3-TTS path.
        return self.ctx.server._validate_tts_request(request)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        # built by lead: server-side prompt/param builder for PersonaPlex.
        prompt, tts_params, warmup_key = await self.ctx.server._build_personaplex_request(request)
        return PreparedRequest(
            prompt=prompt,
            tts_params=tts_params,
            model_type=tts_params.get("task_type", ["unknown"])[0],
            warmup_artifact_key=warmup_key,
        )
