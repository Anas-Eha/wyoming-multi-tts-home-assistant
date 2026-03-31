"""Engine registry creation."""

from __future__ import annotations

from .base import TtsEngine
from .http_engine import HttpEngine
from .isolated_engine import IsolatedEngine


def _isolated_engine(**kwargs) -> IsolatedEngine:
    return IsolatedEngine(**kwargs)


def _qwen_engine() -> HttpEngine:
    return HttpEngine(
        engine_id="qwen_tts_polish",
        display_name="Qwen3-TTS",
        module_path="app.engines.qwen_engine",
        class_name="QwenTtsEngine",
        python_env_var="QWEN_PYTHON",
        default_python="/app/.venv-qwen/bin/python",
        fallback_languages=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        port_env_var="QWEN_HTTP_RUNNER_PORT",
        default_port=18001,
    )


def build_registry() -> dict[str, TtsEngine]:
    engines = [
        _isolated_engine(
            engine_id="chatterbox",
            display_name="Chatterbox",
            module_path="app.engines.chatterbox_engine",
            class_name="ChatterboxEngine",
            python_env_var="CHATTERBOX_PYTHON",
            default_python="/app/.venv-chatterbox/bin/python",
            fallback_languages=[
                "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko",
                "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
            ],
        ),
        _isolated_engine(
            engine_id="whisperspeech",
            display_name="WhisperSpeech",
            module_path="app.engines.whisperspeech_engine",
            class_name="WhisperSpeechEngine",
            python_env_var="WHISPERSPEECH_PYTHON",
            default_python="/app/.venv-whisperspeech/bin/python",
            fallback_languages=["de", "en", "es", "fr", "it", "nl", "pl"],
        ),
        _isolated_engine(
            engine_id="xtts_v2",
            display_name="XTTS-v2",
            module_path="app.engines.xtts_engine",
            class_name="XttsEngine",
            python_env_var="XTTS_PYTHON",
            default_python="/app/.venv-xtts/bin/python",
            fallback_languages=[
                "ar", "cs", "de", "en", "es", "fr", "hi", "hu", "it", "ja", "ko", "nl", "pl", "pt", "ru", "tr", "zh-cn",
            ],
        ),
        _qwen_engine(),
        _isolated_engine(
            engine_id="mms_tts_pol",
            display_name="MMS TTS Polish",
            module_path="app.engines.mms_engine",
            class_name="MmsPolishEngine",
            python_env_var="MMS_PYTHON",
            default_python="/app/.venv-mms/bin/python",
            fallback_languages=["pl"],
        ),
        _isolated_engine(
            engine_id="fish_s2_pro",
            display_name="Fish Audio S2 Pro",
            module_path="app.engines.fish_engine",
            class_name="FishS2ProEngine",
            python_env_var="FISH_PYTHON",
            default_python="/app/.venv-fish/bin/python",
            fallback_languages=[
                "pl", "en", "zh", "ja", "ko", "es", "pt", "ar", "ru", "fr", "de", "sv", "it", "tr", "no", "nl", "fi", "cs",
            ],
        ),
    ]
    return {engine.engine_id(): engine for engine in engines}
