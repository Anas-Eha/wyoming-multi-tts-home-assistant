from __future__ import annotations

import pytest

from app.engines.base import EngineError
from app.engines.qwen_engine import QwenTtsEngine


class FakeQwenModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def generate_custom_voice(self, *, text: str, language: str, speaker: str, instruct: str):
        self.calls.append(
            {
                "text": text,
                "language": language,
                "speaker": speaker,
                "instruct": instruct,
            }
        )
        return [[0.0, 0.0, 0.0, 0.0]], 24000


def create_loaded_engine() -> tuple[QwenTtsEngine, FakeQwenModel]:
    engine = QwenTtsEngine()
    fake_model = FakeQwenModel()
    engine._model = fake_model
    engine._device = "cuda:0"
    engine._speakers = {
        "Ryan": {"native_language": "en", "description": "Dynamic male voice with rhythm."},
        "Vivian": {"native_language": "zh", "description": "Bright young female voice."},
    }
    engine._state = "ready"
    return engine, fake_model


def test_qwen_english_language_uses_supported_language_token():
    engine, fake_model = create_loaded_engine()

    result = engine.synthesize(
        "This is a test.",
        voice="Ryan",
        language="en",
        options=None,
    )

    assert result.language == "en"
    assert fake_model.calls[0]["language"] == "english"
    assert fake_model.calls[0]["instruct"] == ""


def test_qwen_defaults_to_speaker_native_language():
    engine, fake_model = create_loaded_engine()

    result = engine.synthesize(
        "This is a test.",
        voice="Ryan",
        language=None,
        options={"instruct": "Calm and warm."},
    )

    assert result.language == "en"
    assert fake_model.calls[0]["language"] == "english"
    assert fake_model.calls[0]["instruct"] == "Calm and warm."


def test_qwen_rejects_unsupported_polish_language():
    engine, _ = create_loaded_engine()

    with pytest.raises(EngineError):
        engine.synthesize(
            "To jest test.",
            voice="Ryan",
            language="pl",
            options=None,
        )
