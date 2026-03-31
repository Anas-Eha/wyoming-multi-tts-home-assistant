from __future__ import annotations

from app.engines.base import EngineStatus, EngineVoice
from app.wyoming.server import MultiTtsEventHandler, build_info


class FakeEngine:
    def __init__(
        self,
        *,
        engine_id: str = "qwen_tts_polish",
        display_name: str = "Qwen3-TTS",
        supported_languages: list[str] | None = None,
        available_voices: list[EngineVoice] | None = None,
    ) -> None:
        self._engine_id = engine_id
        self._display_name = display_name
        self._supported_languages = supported_languages or ["en", "pl"]
        self._available_voices = available_voices or [
            EngineVoice(
                id="ryan",
                label="Ryan",
                languages=["en"],
                default_language="en",
                description="Official Qwen3-TTS speaker.",
            )
        ]

    def engine_id(self) -> str:
        return self._engine_id

    def display_name(self) -> str:
        return self._display_name

    def supported_languages(self) -> list[str]:
        return list(self._supported_languages)

    def status(self) -> EngineStatus:
        return EngineStatus(
            engine_id=self._engine_id,
            display_name=self._display_name,
            state="ready",
            loaded=True,
            device="cuda:0",
            available_voices=list(self._available_voices),
            extra={},
        )


class FakeManager:
    def __init__(self) -> None:
        self.active_engine = FakeEngine()
        self.registry = {
            "qwen_tts_polish": self.active_engine,
            "mms_tts_pol": FakeEngine(
                engine_id="mms_tts_pol",
                display_name="MMS TTS Polish",
                supported_languages=["pl"],
                available_voices=[
                    EngineVoice(
                        id="default",
                        label="Default",
                        languages=["pl"],
                        default_language="pl",
                    )
                ],
            ),
        }


class FakeVoice:
    def __init__(self, *, name: str | None = None, language: str | None = None, speaker: str | None = None) -> None:
        self.name = name
        self.language = language
        self.speaker = speaker


class FakeSynthesize:
    def __init__(self, voice: FakeVoice | None) -> None:
        self.voice = voice


def test_build_info_uses_stable_multi_tts_program_name():
    info = build_info(FakeManager())

    assert info.tts[0].name == "wyoming-multi-tts"
    assert info.tts[0].attribution.name == "wyoming-multi-tts"
    assert info.tts[0].voices[0].name == "default"
    assert info.tts[0].voices[0].languages == ["en", "pl"]


def test_build_info_exposes_single_stable_default_voice():
    info = build_info(FakeManager())

    assert len(info.tts[0].voices) == 1
    assert "active engine: Qwen3-TTS" in info.tts[0].voices[0].description


def test_build_info_aggregates_languages_from_all_registered_engines():
    info = build_info(FakeManager())

    assert "pl" in info.tts[0].voices[0].languages


def test_resolve_voice_name_prefers_speaker():
    synth = FakeSynthesize(FakeVoice(name="default", speaker="speaker-a", language="pl"))

    assert MultiTtsEventHandler._resolve_voice_name(synth) == "speaker-a"


def test_resolve_voice_name_falls_back_to_name():
    synth = FakeSynthesize(FakeVoice(name="default", language="pl"))

    assert MultiTtsEventHandler._resolve_voice_name(synth) == "default"
