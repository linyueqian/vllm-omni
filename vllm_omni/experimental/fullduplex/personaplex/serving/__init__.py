# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex full-duplex WebSocket serving (PCM in / PCM + text out)."""

from vllm_omni.experimental.fullduplex.personaplex.serving.server import (
    create_app,
    main,
)

__all__ = ["create_app", "main"]
