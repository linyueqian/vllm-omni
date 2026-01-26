import asyncio
from typing import Any

from fastapi import Request
from fastapi.responses import Response
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
from vllm_omni.entrypoints.openai.protocol.audio import (
    AudioResponse,
    CreateAudio,
    OpenAICreateSpeechRequest,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

# TTS model stage identifiers
_TTS_MODEL_STAGES: set[str] = {"qwen3_tts"}


class OmniOpenAIServingSpeech(OpenAIServing, AudioMixin):
    def _is_tts_model(self) -> bool:
        """Check if the current model is a TTS model (e.g., Qwen3-TTS)."""
        stage_list = getattr(self.engine_client, "stage_list", None)
        if stage_list:
            for stage in stage_list:
                model_stage = getattr(stage, "model_stage", None)
                if model_stage in _TTS_MODEL_STAGES:
                    return True
        return False

    def _build_tts_prompt(self, text: str) -> str:
        """Build TTS prompt in the format expected by Qwen3-TTS."""
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_tts_additional_info(self, request: OpenAICreateSpeechRequest) -> dict[str, Any]:
        """Build additional_information dict for Qwen3-TTS.

        Maps OpenAI speech API parameters to Qwen3-TTS parameters.
        All values are wrapped in lists as required by the TTS model.
        """
        params: dict[str, Any] = {}

        # Determine task type
        task_type = request.task_type or "CustomVoice"
        params["task_type"] = [task_type]

        # Language
        params["language"] = [request.language or "Auto"]

        # Text content
        params["text"] = [request.input]

        # Speaker (voice) - for CustomVoice task
        if request.voice:
            params["speaker"] = [request.voice]
        elif task_type == "CustomVoice":
            params["speaker"] = ["Vivian"]  # Default speaker

        # Instructions (instruct) - for emotional/style control
        params["instruct"] = [request.instructions or ""]

        # Voice clone parameters (Base task)
        if request.ref_audio is not None:
            params["ref_audio"] = [request.ref_audio]
        if request.ref_text is not None:
            params["ref_text"] = [request.ref_text]
        if request.x_vector_only_mode is not None:
            params["x_vector_only_mode"] = [request.x_vector_only_mode]

        # Generation parameters
        params["max_new_tokens"] = [request.max_new_tokens or 2048]

        return params

    async def create_speech(
        self,
        request: OpenAICreateSpeechRequest,
        raw_request: Request | None = None,
    ):
        """
        Create Speech API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/audio/createSpeech
        for the API specification. This API mimics the OpenAI
        Create Speech API.

        For Qwen3-TTS models, additional parameters are supported:
        - task_type: "CustomVoice", "VoiceDesign", or "Base"
        - language: Language code (e.g., "Chinese", "English", "Auto")
        - voice: Speaker name (e.g., "Vivian", "Ryan") for CustomVoice
        - instructions: Voice style/emotion instructions
        - ref_audio: Reference audio for voice cloning (Base task)
        - ref_text: Transcript of reference audio (Base task)
        - x_vector_only_mode: Use speaker embedding only (Base task)

        NOTE: Streaming audio generation is not currently supported.
        """

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        request_id = f"speech-{random_uuid()}"

        try:
            # Check if this is a Qwen3-TTS model
            if self._is_tts_model():
                # Build TTS-specific prompt and parameters
                prompt_text = self._build_tts_prompt(request.input)
                additional_info = self._build_tts_additional_info(request)

                prompt = {
                    "prompt": prompt_text,
                    "additional_information": additional_info,
                }

                logger.info(
                    "TTS speech request %s: text=%r, task_type=%s",
                    request_id,
                    request.input[:50] + "..." if len(request.input) > 50 else request.input,
                    additional_info.get("task_type", ["unknown"])[0],
                )
            else:
                # Fallback for other TTS models
                prompt = {"prompt": request.input}

            sampling_params_list = self.engine_client.default_sampling_params_list

            generator = self.engine_client.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
                output_modalities=["audio"],
            )

            final_output: OmniRequestOutput | None = None
            async for res in generator:
                final_output = res

            if final_output is None:
                return self.create_error_response("No output generated from the model.")

            # Extract audio from output
            # Audio can be in final_output.multimodal_output or final_output.request_output.multimodal_output
            audio_output = None
            if hasattr(final_output, "multimodal_output") and final_output.multimodal_output:
                audio_output = final_output.multimodal_output
            if (not audio_output or "audio" not in audio_output) and hasattr(final_output, "request_output"):
                if final_output.request_output and hasattr(final_output.request_output, "multimodal_output"):
                    audio_output = final_output.request_output.multimodal_output

            if not audio_output or "audio" not in audio_output:
                return self.create_error_response("TTS model did not produce audio output.")

            audio_tensor = audio_output["audio"]
            sample_rate = audio_output.get("sr", 24000)
            if hasattr(sample_rate, "item"):
                sample_rate = sample_rate.item()

            # Convert tensor to numpy
            if hasattr(audio_tensor, "float"):
                audio_tensor = audio_tensor.float().detach().cpu().numpy()

            # Squeeze batch dimension if present, but preserve channel dimension for stereo
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.squeeze()

            audio_obj = CreateAudio(
                audio_tensor=audio_tensor,
                sample_rate=int(sample_rate),
                response_format=request.response_format or "wav",
                speed=request.speed or 1.0,
                stream_format=request.stream_format,
                base64_encode=False,
            )

            audio_response: AudioResponse = self.create_audio(audio_obj)
            return Response(content=audio_response.audio_data, media_type=audio_response.media_type)

        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(str(e))
        except Exception as e:
            logger.exception("Speech generation failed: %s", e)
            return self.create_error_response(f"Speech generation failed: {e}")
