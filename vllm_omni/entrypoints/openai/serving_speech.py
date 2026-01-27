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

# Model stages that require Qwen3-TTS prompt format
_QWEN3_TTS_STAGES: set[str] = {"qwen3_tts"}

# Supported speakers for Qwen3-TTS CustomVoice task
_QWEN3_TTS_SPEAKERS: set[str] = {
    "Vivian",  # Bright, slightly edgy young female voice (Chinese)
    "Serena",  # Warm, gentle young female voice (Chinese)
    "Uncle_Fu",  # Seasoned male voice with a low, mellow timbre (Chinese)
    "Dylan",  # Youthful Beijing male voice (Chinese, Beijing Dialect)
    "Eric",  # Lively Chengdu male voice (Chinese, Sichuan Dialect)
    "Ryan",  # Dynamic male voice with strong rhythmic drive (English)
    "Aiden",  # Sunny American male voice (English)
    "One_Anna",  # Playful Japanese female voice (Japanese)
    "Sohee",  # Warm Korean female voice (Korean)
}

# Supported languages for Qwen3-TTS
_QWEN3_TTS_LANGUAGES: set[str] = {"Auto", "Chinese", "English", "Japanese", "Korean"}

# Constraints for Qwen3-TTS parameters
_MAX_INSTRUCTIONS_LENGTH = 500
_MAX_NEW_TOKENS_MIN = 1
_MAX_NEW_TOKENS_MAX = 4096


class OmniOpenAIServingSpeech(OpenAIServing, AudioMixin):
    def _requires_qwen3_tts_prompt(self) -> bool:
        """Check if the model requires Qwen3-TTS prompt format.

        Used only for determining prompt structure. Parameter handling
        is done separately based on what's present in the request.
        """
        stage_list = getattr(self.engine_client, "stage_list", None)
        if stage_list:
            for stage in stage_list:
                model_stage = getattr(stage, "model_stage", None)
                if model_stage in _QWEN3_TTS_STAGES:
                    return True
        return False

    def _validate_qwen3_tts_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate Qwen3-TTS specific parameters.

        Returns error message if validation fails, None if valid.
        """
        task_type = request.task_type or "CustomVoice"

        # Validate input is not empty
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        # Validate language
        if request.language is not None and request.language not in _QWEN3_TTS_LANGUAGES:
            return f"Invalid language '{request.language}'. Supported: {', '.join(sorted(_QWEN3_TTS_LANGUAGES))}"

        # Validate speaker for CustomVoice task
        if task_type == "CustomVoice" and request.voice is not None:
            if request.voice not in _QWEN3_TTS_SPEAKERS:
                return f"Invalid speaker '{request.voice}'. Supported: {', '.join(sorted(_QWEN3_TTS_SPEAKERS))}"

        # Validate Base task requirements
        if task_type == "Base":
            if request.ref_audio is None:
                return "Base task requires 'ref_audio' for voice cloning"
            # Validate ref_audio format
            if not (request.ref_audio.startswith(("http://", "https://")) or request.ref_audio.startswith("data:")):
                return "ref_audio must be a URL (http/https) or base64 data URL (data:...)"

        # Validate cross-parameter dependencies
        if task_type != "Base":
            if request.ref_text is not None:
                return "'ref_text' is only valid for Base task"
            if request.x_vector_only_mode is not None:
                return "'x_vector_only_mode' is only valid for Base task"

        # Validate VoiceDesign task requirements
        if task_type == "VoiceDesign" and not request.instructions:
            return "VoiceDesign task requires 'instructions' to describe the voice"

        # Validate instructions length
        if request.instructions and len(request.instructions) > _MAX_INSTRUCTIONS_LENGTH:
            return f"Instructions too long (max {_MAX_INSTRUCTIONS_LENGTH} characters)"

        # Validate max_new_tokens range
        if request.max_new_tokens is not None:
            if request.max_new_tokens < _MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_MAX_NEW_TOKENS_MIN}"
            if request.max_new_tokens > _MAX_NEW_TOKENS_MAX:
                return f"max_new_tokens cannot exceed {_MAX_NEW_TOKENS_MAX}"

        return None

    def _build_qwen3_tts_prompt(self, text: str) -> str:
        """Build prompt in Qwen3-TTS format."""
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_tts_params(self, request: OpenAICreateSpeechRequest) -> dict[str, Any]:
        """Build TTS parameters from request.

        Processes each parameter if present, skips if not.
        Values are wrapped in lists as required by the model.
        """
        params: dict[str, Any] = {}

        # Text content (always required)
        params["text"] = [request.input]

        # Task type
        if request.task_type is not None:
            params["task_type"] = [request.task_type]
        else:
            params["task_type"] = ["CustomVoice"]

        # Language
        if request.language is not None:
            params["language"] = [request.language]
        else:
            params["language"] = ["Auto"]

        # Speaker (voice)
        if request.voice is not None:
            params["speaker"] = [request.voice]
        elif params["task_type"][0] == "CustomVoice":
            params["speaker"] = ["Vivian"]  # Default for CustomVoice

        # Instructions for style/emotion control
        if request.instructions is not None:
            params["instruct"] = [request.instructions]
        else:
            params["instruct"] = [""]

        # Voice clone parameters (used with Base task)
        if request.ref_audio is not None:
            params["ref_audio"] = [request.ref_audio]
        if request.ref_text is not None:
            params["ref_text"] = [request.ref_text]
        if request.x_vector_only_mode is not None:
            params["x_vector_only_mode"] = [request.x_vector_only_mode]

        # Generation parameters
        if request.max_new_tokens is not None:
            params["max_new_tokens"] = [request.max_new_tokens]
        else:
            params["max_new_tokens"] = [2048]

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
            # Use model-specific prompt format
            if self._requires_qwen3_tts_prompt():
                # Validate Qwen3-TTS specific parameters
                validation_error = self._validate_qwen3_tts_request(request)
                if validation_error:
                    return self.create_error_response(validation_error)

                # Build TTS parameters from request
                tts_params = self._build_tts_params(request)
                prompt_text = self._build_qwen3_tts_prompt(request.input)
                prompt = {
                    "prompt": prompt_text,
                    "additional_information": tts_params,
                }
            else:
                # Fallback for other TTS models
                tts_params = {}
                prompt = {"prompt": request.input}

            logger.info(
                "TTS speech request %s: text=%r, task_type=%s",
                request_id,
                request.input[:50] + "..." if len(request.input) > 50 else request.input,
                tts_params.get("task_type", ["unknown"])[0],
            )

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
