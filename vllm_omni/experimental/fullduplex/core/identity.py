# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DuplexFence:
    session_id: str
    epoch: int = 0
    turn_id: int = 0
    response_seq: int = 0
