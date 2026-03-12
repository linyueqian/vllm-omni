# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from torch.nn import functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav
from vllm_omni.model_executor.models.cosyvoice3.utils import make_pad_mask
from vllm_omni.model_executor.models.funaudiochat.common import resolve_funaudiochat_root
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)
_OFFICIAL_TOKEN_HOP_LEN = 25 * 30
_OFFICIAL_MIN_SEGMENT_TOKENS = 50


class FunAudioChatCosyVoice3Code2Wav(nn.Module):
    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        del prefix
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = self._resolve_model_path(vllm_config.model_config.model)
        self.have_multimodal_outputs = True
        self.enable_update_additional_information = False
        self.requires_raw_input_tokens = True
        self.hf_config_path = getattr(vllm_config.model_config, "hf_config_path", None)

        from transformers import AutoConfig

        from vllm_omni.model_executor.models.cosyvoice3.config import CosyVoice3Config

        try:
            AutoConfig.register(CosyVoice3Config.model_type, CosyVoice3Config)
        except ValueError:
            pass

        config_source = self.hf_config_path or self.model_path
        self.config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
        self.code2wav = CosyVoice3Code2Wav(self.config)
        # Keep FunAudioChat's stage-1 flow stack in float32 to match the
        # official runtime without changing global CosyVoice3 behavior.
        self.code2wav.flow_model = self.code2wav.flow_model.float()
        device = vllm_config.device_config.device
        self.code2wav.load_weights(self.model_path, device=device)
        estimator = getattr(self.code2wav.decoder, "estimator", None)
        if estimator is not None and hasattr(estimator, "static_chunk_size"):
            estimator.static_chunk_size = 2 * _OFFICIAL_TOKEN_HOP_LEN
        self._speaker_embedding = self._load_default_speaker_embedding()
        self._max_codec_token_id = int(self.config.flow["vocab_size"]) - 1
        self._max_supported_token_len = self._compute_max_supported_token_len()
        self._dummy_profile_token_len = min(32, self._max_supported_token_len)
        self._logged_dummy_profile_cap = False

    def _resolve_model_path(self, model_path: str) -> str:
        local_path = Path(model_path)
        if local_path.exists():
            return str(local_path)

        logger.info("Resolving FunAudioChat CosyVoice3 weights to a local snapshot: %s", model_path)
        return snapshot_download(model_path)

    def _load_default_speaker_embedding(self) -> torch.Tensor:
        env_path = os.environ.get("FUN_AUDIO_CHAT_SPK_INFO")
        if env_path:
            spk_path = Path(env_path).expanduser()
        else:
            spk_path = resolve_funaudiochat_root() / "utils" / "new_spk2info.pt"
        if not spk_path.exists():
            raise FileNotFoundError(
                f"Default speaker embedding not found: {spk_path}. "
                "Set FUN_AUDIO_CHAT_SPK_INFO or install Fun-Audio-Chat from source."
            )
        spk_info = torch.load(spk_path, map_location="cpu")
        return spk_info["中文女"]["embedding"].reshape(1, -1).float()

    def _compute_max_supported_token_len(self) -> int:
        max_audio_samples = 300 * int(self.config.hift["sampling_rate"])
        sine_waves = getattr(self.code2wav.hift.m_source, "sine_waves", None)
        if isinstance(sine_waves, torch.Tensor) and sine_waves.ndim >= 2:
            max_audio_samples = int(sine_waves.shape[1])
        samples_per_mel = int(self.config.hift["istft_params"]["hop_len"])
        for rate in self.config.hift["upsample_rates"]:
            samples_per_mel *= int(rate)
        samples_per_token = int(self.config.flow["token_mel_ratio"]) * samples_per_mel
        return max_audio_samples // samples_per_token

    @staticmethod
    def _get_prompt_token_id_batches(sampling_metadata: Any) -> list[torch.Tensor] | None:
        prompt_token_ids = getattr(sampling_metadata, "prompt_token_ids", None)
        if prompt_token_ids is None:
            return None

        if isinstance(prompt_token_ids, torch.Tensor):
            prompt_token_ids = prompt_token_ids.detach().to(torch.long)
            if prompt_token_ids.ndim <= 1:
                return [prompt_token_ids.view(-1)]
            return [row.reshape(-1) for row in prompt_token_ids]

        if isinstance(prompt_token_ids, list):
            if len(prompt_token_ids) == 0:
                return None
            if isinstance(prompt_token_ids[0], (list, tuple, torch.Tensor)):
                batches = [torch.as_tensor(item, dtype=torch.long).reshape(-1) for item in prompt_token_ids]
            else:
                batches = [torch.tensor(prompt_token_ids, dtype=torch.long)]
            return batches or None

        return None

    @staticmethod
    def _split_request_ids(ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        return [ids]

    def _build_decode_tokens(
        self,
        input_ids: torch.Tensor,
        sampling_metadata: Any,
        seq_token_counts: list[int] | None = None,
    ) -> tuple[list[torch.Tensor], bool]:
        prompt_token_id_batches = self._get_prompt_token_id_batches(sampling_metadata)
        if prompt_token_id_batches is not None:
            raw_id_batches = prompt_token_id_batches
        elif input_ids is not None:
            raw_id_batches = self._split_request_ids(input_ids.reshape(-1), seq_token_counts)
        else:
            raw_id_batches = [torch.empty((0,), dtype=torch.long)]

        token_batches = [
            raw_ids.reshape(1, -1)
            .to(dtype=torch.long, device=self.vllm_config.device_config.device)
            .clamp_(
                min=0,
                max=self._max_codec_token_id,
            )
            for raw_ids in raw_id_batches
        ]

        is_dummy_profile = bool(
            sampling_metadata is None
            and prompt_token_id_batches is None
            and len(token_batches) == 1
            and (token_batches[0].numel() == 0 or torch.count_nonzero(token_batches[0]).item() == 0)
        )
        if is_dummy_profile and token_batches[0].shape[1] > self._dummy_profile_token_len:
            if not self._logged_dummy_profile_cap:
                logger.info(
                    "FunAudioChat code2wav dummy/profile run detected. Capping decode length from %d to %d tokens.",
                    token_batches[0].shape[1],
                    self._dummy_profile_token_len,
                )
                self._logged_dummy_profile_cap = True
            token_batches[0] = token_batches[0][:, : self._dummy_profile_token_len]

        return token_batches, is_dummy_profile

    @staticmethod
    def _split_tokens_like_official(token: torch.Tensor) -> list[torch.Tensor]:
        flat = token.reshape(-1)
        if flat.numel() == 0:
            return [flat]

        segments: list[torch.Tensor] = []
        time_step = 0
        while time_step * 25 < flat.numel():
            start = time_step * 25
            end = min((time_step + 30) * 25, flat.numel())
            segments.append(flat[start:end])
            time_step += 30

        if len(segments) > 1 and segments[-1].numel() < _OFFICIAL_MIN_SEGMENT_TOKENS:
            merged = torch.cat([segments[-2], segments[-1]], dim=0)
            split_point = merged.numel() // 2
            segments = [*segments[:-2], merged[:split_point], merged[split_point:]]

        return segments

    @staticmethod
    def _fade_in_out(fade_in_tensor: torch.Tensor, fade_out_tensor: torch.Tensor, window: Any) -> torch.Tensor:
        if fade_in_tensor.numel() == 0 or fade_out_tensor.numel() == 0:
            return fade_in_tensor

        overlap = min(int(len(window) // 2), fade_in_tensor.shape[-1], fade_out_tensor.shape[-1])
        if overlap <= 0:
            return fade_in_tensor

        fade_window = torch.as_tensor(window, device=fade_in_tensor.device, dtype=fade_in_tensor.dtype)
        mixed = fade_in_tensor.clone()
        mixed[..., :overlap] = (
            mixed[..., :overlap] * fade_window[:overlap] + fade_out_tensor[..., -overlap:] * fade_window[-overlap:]
        )
        return mixed

    def _run_flow_like_official(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        *,
        finalize: bool,
        flow_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flow_model = self.code2wav.flow_model
        device = token.device
        token = token.to(device=device, dtype=torch.long)
        prompt_token = prompt_token.to(device=device, dtype=torch.long)
        prompt_feat = prompt_feat.to(device=device, dtype=torch.float32)
        embedding = embedding.to(device=device, dtype=torch.float32)
        embedding = F.normalize(embedding, dim=1)
        embedding = flow_model.spk_embed_affine_layer(embedding)

        token_len = torch.tensor([token.shape[1]], dtype=torch.int32, device=device)
        prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.int32, device=device)
        full_token = torch.cat([prompt_token, token], dim=1)
        full_token_len = prompt_token_len + token_len
        mask = (~make_pad_mask(full_token_len)).unsqueeze(-1).to(embedding)
        token_emb = flow_model.input_embedding(torch.clamp(full_token, min=0)) * mask

        if finalize:
            h = flow_model.pre_lookahead_layer(token_emb)
        else:
            h = flow_model.pre_lookahead_layer(
                token_emb[:, : -flow_model.pre_lookahead_len],
                context=token_emb[:, -flow_model.pre_lookahead_len :],
            )
        h = h.repeat_interleave(flow_model.token_mel_ratio, dim=1)

        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - mel_len1
        conds = torch.zeros(
            [1, mel_len1 + mel_len2, flow_model.output_size],
            device=device,
            dtype=h.dtype,
        )
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mel_mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2], device=device))).to(h)
        decoder_kwargs = {
            "mu": h.transpose(1, 2).contiguous(),
            "mask": mel_mask.unsqueeze(1),
            "spks": embedding,
            "cond": conds,
            "n_timesteps": 10,
        }
        try:
            decoder_out = flow_model.decoder(cache=flow_cache, **decoder_kwargs)
        except TypeError as exc:
            if "cache" not in str(exc):
                raise
            decoder_out = flow_model.decoder(**decoder_kwargs)

        if isinstance(decoder_out, tuple):
            feat, next_flow_cache = decoder_out
        else:
            feat, next_flow_cache = decoder_out, flow_cache
        feat = feat[:, :, mel_len1:]
        return feat.float(), next_flow_cache

    def _run_hift_like_official(
        self,
        speech_feat: torch.Tensor,
        *,
        finalize: bool,
        cache_source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hift = self.code2wav.hift
        speech_feat = speech_feat.to(dtype=torch.float32)
        hift.f0_predictor.to("cpu")
        f0 = hift.f0_predictor(speech_feat.cpu(), finalize=finalize).to(speech_feat)
        source = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        source, _, _ = hift.m_source(source)
        source = source.transpose(1, 2)
        if cache_source.shape[2] != 0:
            source[:, :, : cache_source.shape[2]] = cache_source

        if finalize:
            speech = hift.decode(x=speech_feat, s=source, finalize=True)
        else:
            padding = hift.f0_predictor.condnet[0].causal_padding
            speech = hift.decode(x=speech_feat[:, :, :-padding], s=source, finalize=False)
        return speech, source

    def _decode_segment_like_official(
        self,
        token_segment: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        if token_segment.numel() == 0:
            return torch.zeros((0,), device=embedding.device, dtype=torch.float32)

        device = token_segment.device
        flow_cache = torch.zeros((1, 80, 0, 2), device=device, dtype=torch.float32)
        mel_overlap = torch.zeros((1, self.code2wav.output_size, 0), device=device, dtype=torch.float32)
        hift_cache: dict[str, torch.Tensor] | None = None
        pre_lookahead_len = int(self.config.flow["pre_lookahead_len"])
        token_offset = 0
        speech_chunks: list[torch.Tensor] = []

        while token_offset < token_segment.numel():
            chunk_len = min(token_offset + _OFFICIAL_TOKEN_HOP_LEN + pre_lookahead_len, token_segment.numel())
            chunk = token_segment[:chunk_len].reshape(1, -1)
            finalize = chunk.shape[1] == token_segment.numel()
            tts_mel, flow_cache = self._run_flow_like_official(
                chunk,
                prompt_token,
                prompt_feat,
                embedding,
                finalize=finalize,
                flow_cache=flow_cache,
            )
            if mel_overlap.shape[2] != 0:
                tts_mel = self._fade_in_out(tts_mel, mel_overlap, self.code2wav.mel_window)

            if hift_cache is not None:
                cache_source = hift_cache["source"]
                tts_mel = torch.cat([hift_cache["mel"], tts_mel], dim=2)
            else:
                cache_source = torch.zeros((1, 1, 0), device=device, dtype=tts_mel.dtype)

            if not finalize:
                mel_overlap = tts_mel[:, :, -self.code2wav.mel_overlap_len :]
                tts_mel = tts_mel[:, :, : -self.code2wav.mel_overlap_len]
                if tts_mel.shape[2] == 0:
                    token_offset += _OFFICIAL_TOKEN_HOP_LEN
                    continue
                tts_speech, tts_source = self._run_hift_like_official(
                    tts_mel,
                    finalize=False,
                    cache_source=cache_source,
                )
                if hift_cache is not None:
                    tts_speech = self._fade_in_out(tts_speech, hift_cache["speech"], self.code2wav.speech_window)
                hift_cache = {
                    "mel": tts_mel[:, :, -self.code2wav.mel_cache_len :],
                    "source": tts_source[:, :, -self.code2wav.source_cache_len :],
                    "speech": tts_speech[:, -self.code2wav.source_cache_len :],
                }
                if tts_speech.shape[1] > self.code2wav.source_cache_len:
                    tts_speech = tts_speech[:, : -self.code2wav.source_cache_len]
                else:
                    tts_speech = tts_speech[:, :0]
            else:
                tts_speech, _ = self._run_hift_like_official(
                    tts_mel,
                    finalize=True,
                    cache_source=cache_source,
                )
                if hift_cache is not None:
                    tts_speech = self._fade_in_out(tts_speech, hift_cache["speech"], self.code2wav.speech_window)

            if tts_speech.numel() > 0:
                speech_chunks.append(tts_speech.reshape(-1))

            token_offset += _OFFICIAL_TOKEN_HOP_LEN

        if not speech_chunks:
            return torch.zeros((0,), device=device, dtype=torch.float32)
        return torch.cat(speech_chunks, dim=0)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids is None or input_ids.numel() == 0:
            return torch.empty((0, 1), dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        del hidden_states, sampling_metadata
        return None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds

        sampling_metadata = kwargs.get("sampling_metadata")
        token_batches, is_dummy_profile = self._build_decode_tokens(
            input_ids,
            sampling_metadata,
            kwargs.get("seq_token_counts"),
        )
        num_reqs = len(token_batches)
        empty = torch.zeros((0,), dtype=torch.float32)
        sr = torch.tensor(24000, dtype=torch.int32)
        if not token_batches or all(token.numel() == 0 for token in token_batches):
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": [empty] * max(num_reqs, 1), "sr": [sr] * max(num_reqs, 1)},
            )

        if is_dummy_profile:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": [empty.to(device=token_batches[0].device)], "sr": [sr]},
            )

        audios: list[torch.Tensor] = []
        srs: list[torch.Tensor] = []
        for token in token_batches:
            if token.numel() == 0:
                audios.append(empty)
                srs.append(sr)
                continue
            prompt_token = torch.zeros((1, 0), dtype=torch.long, device=token.device)
            prompt_feat = torch.zeros((1, 0, 80), dtype=torch.float32, device=token.device)
            embedding = self._speaker_embedding.to(device=token.device, dtype=torch.float32)
            audio_segments: list[torch.Tensor] = []
            for token_segment in self._split_tokens_like_official(token):
                if token_segment.numel() == 0:
                    continue
                segment_audio = self._decode_segment_like_official(
                    token_segment,
                    prompt_token,
                    prompt_feat,
                    embedding,
                )
                if segment_audio.numel() > 0:
                    audio_segments.append(segment_audio.reshape(-1))
            audio = torch.cat(audio_segments, dim=0) if audio_segments else torch.zeros((0,), device=token.device)
            audios.append(audio.reshape(-1).detach().cpu())
            srs.append(sr)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"audio": audios, "sr": srs},
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        # All parameters are loaded eagerly from the local snapshot in `__init__`.
        return {name for name, _ in self.named_parameters()}
