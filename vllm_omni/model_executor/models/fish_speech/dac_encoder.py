"""DAC codec encoder for Fish Speech S2 Pro voice cloning.

Encodes reference audio into VQ codes for use as prompt conditioning.
Runs on CPU in the API server process — loaded lazily on first use.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)

_DAC_SAMPLE_RATE = 44100
_codec_cache: dict[str, nn.Module] = {}


def _load_dac_codec(model_path: str) -> nn.Module:
    """Load the DAC codec model from codec.pth (cached per model_path)."""
    if model_path in _codec_cache:
        return _codec_cache[model_path]

    codec_path = os.path.join(model_path, "codec.pth")
    if not os.path.exists(codec_path):
        from transformers.utils.hub import cached_file

        cached = cached_file(model_path, "codec.pth")
        if cached is not None:
            codec_path = cached

    if not os.path.exists(codec_path):
        raise FileNotFoundError(
            f"codec.pth not found for {model_path}. Required for voice cloning with Fish Speech S2 Pro."
        )

    from fish_speech.models.dac.modded_dac import (
        DAC,
        ModelArgs,
        WindowLimitedTransformer,
    )
    from fish_speech.models.dac.rvq import DownsampleResidualVectorQuantize

    base_transformer_kwargs = dict(
        block_size=16384,
        n_local_heads=-1,
        head_dim=64,
        rope_base=10000,
        norm_eps=1e-5,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        channels_first=True,
    )

    def _make_transformer_config(**kw):
        return ModelArgs(**{**base_transformer_kwargs, **kw})

    quantizer_transformer_config = ModelArgs(
        block_size=4096,
        n_layer=8,
        n_head=16,
        dim=1024,
        intermediate_size=3072,
        n_local_heads=-1,
        head_dim=64,
        rope_base=10000,
        norm_eps=1e-5,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        channels_first=True,
    )
    post_module = WindowLimitedTransformer(
        causal=True,
        window_size=128,
        input_dim=1024,
        config=quantizer_transformer_config,
    )
    pre_module = WindowLimitedTransformer(
        causal=True,
        window_size=128,
        input_dim=1024,
        config=quantizer_transformer_config,
    )
    quantizer = DownsampleResidualVectorQuantize(
        input_dim=1024,
        n_codebooks=9,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=0.0,
        downsample_factor=[2, 2],
        post_module=post_module,
        pre_module=pre_module,
        semantic_codebook_size=4096,
    )
    codec = DAC(
        encoder_dim=64,
        encoder_rates=[2, 4, 8, 8],
        decoder_dim=1536,
        decoder_rates=[8, 8, 4, 2],
        quantizer=quantizer,
        sample_rate=44100,
        causal=True,
        encoder_transformer_layers=[0, 0, 0, 4],
        decoder_transformer_layers=[4, 0, 0, 0],
        transformer_general_config=_make_transformer_config,
    )

    state_dict = torch.load(codec_path, map_location="cpu", weights_only=True)
    if "generator" in state_dict:
        state_dict = state_dict["generator"]
    codec.load_state_dict(state_dict, strict=False)
    codec.eval()

    _codec_cache[model_path] = codec
    logger.info("Loaded DAC codec encoder from %s (CPU)", codec_path)
    return codec


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resample."""
    if sr == target_sr:
        return wav
    ratio = target_sr / sr
    n_out = int(len(wav) * ratio)
    indices = np.linspace(0, len(wav) - 1, n_out)
    return np.interp(indices, np.arange(len(wav)), wav).astype(np.float32)


@torch.no_grad()
def encode_reference_audio(
    model_path: str,
    wav_samples: list[float] | np.ndarray,
    sample_rate: int,
) -> list[int]:
    """Encode reference audio into semantic token IDs for prompt conditioning.

    Args:
        model_path: HuggingFace model path (for locating codec.pth).
        wav_samples: Audio waveform samples (mono, float).
        sample_rate: Sample rate of the input audio.

    Returns:
        List of semantic token IDs (151678 + code_value for each frame).
    """
    codec = _load_dac_codec(model_path)

    wav = np.asarray(wav_samples, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)

    # Resample to DAC sample rate (44100).
    wav = _resample(wav, sample_rate, _DAC_SAMPLE_RATE)

    # Encode: [1, 1, T] -> codes [1, num_codebooks, num_frames]
    wav_tensor = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float()
    feature_lengths = torch.tensor([wav_tensor.shape[-1]])
    codes, feature_lengths_out = codec.encode(wav_tensor, feature_lengths)

    # Extract semantic codebook (index 0) — shape [num_frames].
    semantic_codes = codes[0, 0, :].tolist()

    # Convert to semantic token IDs: <|semantic:{i}|> = 151678 + i
    SEMANTIC_TOKEN_OFFSET = 151678
    semantic_token_ids = [SEMANTIC_TOKEN_OFFSET + int(c) for c in semantic_codes]

    logger.info(
        "Encoded reference audio: %d samples @ %dHz -> %d semantic tokens",
        len(wav_samples),
        sample_rate,
        len(semantic_token_ids),
    )
    return semantic_token_ids
