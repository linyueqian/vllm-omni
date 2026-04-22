# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared helper utilities for OpenAI-compatible image generation API.

This module provides common helper functions for the image generation endpoint.
All functions work with plain Python types to maintain separation from the
FastAPI HTTP layer.
"""

import base64
import io

from PIL import Image

SUPPORTED_LAYERED_RESOLUTIONS = (640, 1024)
SUPPORTED_LAYERED_LAYERS_RANGE = range(3, 11)


def parse_size(size_str: str) -> tuple[int, int]:
    """Parse size string to width and height tuple.

    Args:
        size_str: Size in format "WIDTHxHEIGHT" (e.g., "1024x1024")

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If size format is invalid
    """
    if not size_str or not isinstance(size_str, str):
        raise ValueError(
            f"Size must be a non-empty string in format 'WIDTHxHEIGHT' (e.g., '1024x1024'), got: {size_str}"
        )

    parts = size_str.split("x")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid size format: '{size_str}'. Expected format: 'WIDTHxHEIGHT' (e.g., '1024x1024'). "
            f"Did you mean to use 'x' as separator?"
        )

    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid size format: '{size_str}'. Width and height must be integers.")

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size: {width}x{height}. Width and height must be positive integers.")

    return width, height


def encode_image_base64_with_compression(
    image: Image.Image, format: str = "png", output_compression: int = 100
) -> str:
    """Encode PIL Image to a base64 image string.

    Args:
        image: PIL Image object
        format: Output image format (e.g., "PNG", "JPEG", "WEBP")
        output_compression: Compression level (0-100%), 100 for best quality
    Returns:
        Base64-encoded image as string
    """
    buffer = io.BytesIO()
    image = _prepare_image_for_output_format(image, format)
    save_kwargs = {}
    if format in ("jpg", "jpeg", "webp"):
        save_kwargs["quality"] = output_compression
    elif format == "png":
        save_kwargs["compress_level"] = max(0, min(9, 9 - output_compression // 11))  # Map 0-100 to 9-0

    image.save(buffer, format=format, **save_kwargs)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _prepare_image_for_output_format(image: Image.Image, format: str) -> Image.Image:
    fmt = format.lower()
    if fmt not in {"jpg", "jpeg"}:
        return image

    if image.mode == "RGB":
        return image

    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        alpha_image = image.convert("RGBA")
        flattened = Image.new("RGB", alpha_image.size, (255, 255, 255))
        flattened.paste(alpha_image, mask=alpha_image.getchannel("A"))
        return flattened

    return image.convert("RGB")


def choose_output_format(output_format: str | None, background: str | None) -> str:
    # Normalize and choose extension
    fmt = (output_format or "").lower()
    if fmt in {"jpg", "png", "webp", "jpeg"}:
        return fmt
    # If transparency requested, prefer png
    if (background or "auto").lower() == "transparent":
        return "png"
    # Default
    return "jpeg"


def get_vllm_image_params(vllm_xargs: dict | None):
    if not vllm_xargs:
        return "png", 100, "auto"

    image_format = vllm_xargs.get("image_format")
    image_compression = vllm_xargs.get("image_compression")
    image_background = vllm_xargs.get("image_background")

    # format：must not empty
    if not isinstance(image_format, str) or not image_format.strip():
        image_format = "png"

    # compression：must > 0 and < 100
    if not isinstance(image_compression, (int, float)) or image_compression <= 0:
        image_compression = 100

    # background：str（"white" / "#FFFFFF" / "transparent"）
    if not isinstance(image_background, str) or not image_background.strip():
        image_background = "auto"

    return image_format, image_compression, image_background


def validate_layered_layers(layers: int | None) -> int | None:
    """Validate the Qwen-Image-Layered ``layers`` parameter."""
    if layers is None:
        return None
    if layers not in SUPPORTED_LAYERED_LAYERS_RANGE:
        raise ValueError(
            f"Invalid layers value {layers}. layers must be between "
            f"{SUPPORTED_LAYERED_LAYERS_RANGE.start} and "
            f"{SUPPORTED_LAYERED_LAYERS_RANGE.stop - 1} inclusive."
        )
    return layers
