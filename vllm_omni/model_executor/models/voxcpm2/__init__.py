# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from .configuration_voxcpm2 import VoxCPM2Config
from .voxcpm2_model import VoxCPM2ForConditionalGeneration
from .voxcpm2_talker import VoxCPM2TalkerForConditionalGeneration

__all__ = [
    "VoxCPM2Config",
    "VoxCPM2ForConditionalGeneration",
    "VoxCPM2TalkerForConditionalGeneration",
]
