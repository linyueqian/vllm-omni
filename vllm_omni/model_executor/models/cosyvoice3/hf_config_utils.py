from pathlib import Path

_COSYVOICE3_MODEL_ARCHES = {
    "CosyVoice3Model",
    "FunAudioChatCosyVoice3Code2Wav",
}


def resolve_bundled_hf_config_path(model_arch: str, hf_config_path: str | None) -> str | None:
    if hf_config_path is not None or model_arch not in _COSYVOICE3_MODEL_ARCHES:
        return hf_config_path

    bundled_hf_config_path = Path(__file__).resolve().parent / "hf_config"
    if not (bundled_hf_config_path / "config.json").is_file():
        return None

    return str(bundled_hf_config_path)
