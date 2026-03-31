from __future__ import annotations

from app.engines.base import EngineStatus, EngineVoice
from app.wyoming.server import MultiTtsEventHandler, build_info


class FakeEngine:
    def engine_id(self) -> str:
        return "qwen_tts_polish"

    def display_name(self) -> str:
        return "Qwen3-TTS"

    def status(self) -> EngineStatus:
        return EngineStatus(
            engine_id="qwen_tts_polish",
            display_name="Qwen3-TTS",
            state="ready",
            loaded=True,
            device="cuda:0",
            available_voices=[
                EngineVoice(
                    id="ryan",
                    label="Ryan",
                    languages=["en"],
                    default_language="en",
                    description="Official Qwen3-TTS speaker.",
                )
            ],
            extra={"wyoming_program_name": "qwen_tts"},
        )


class FakeManager:
    def __init__(self) -> None:
        self.active_engine = FakeEngine()


class FakeVoice:
    def __init__(self, *, name: str | None = None, language: str | None = None, speaker: str | None = None) -> None:
        self.name = name
        self.language = language
        self.speaker = speaker


class FakeSynthesize:
    def __init__(self, voice: FakeVoice | None) -> None:
        self.voice = voice


def test_build_info_uses_wyoming_program_name_override():
    info = build_info(FakeManager())

    assert info.tts[0].name == "qwen_tts"
    assert info.tts[0].voices[0].name == "ryan"


def test_resolve_voice_name_prefers_speaker():
    synth = FakeSynthesize(FakeVoice(name="default", speaker="speaker-a", language="pl"))

    assert MultiTtsEventHandler._resolve_voice_name(synth) == "speaker-a"


def test_resolve_voice_name_falls_back_to_name():
    synth = FakeSynthesize(FakeVoice(name="default", language="pl"))

    assert MultiTtsEventHandler._resolve_voice_name(synth) == "default"
