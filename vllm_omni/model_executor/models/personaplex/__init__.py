# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-omni integration for PersonaPlex (a Moshi finetune, two-stage TTS)."""

from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
    PersonaPlexConfig,
    PersonaPlexDepformerConfig,
    PersonaPlexMimiConfig,
)

__all__ = [
    "PersonaPlexConfig",
    "PersonaPlexDepformerConfig",
    "PersonaPlexMimiConfig",
]
