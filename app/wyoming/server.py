"""Wyoming server implementation."""

from __future__ import annotations

import argparse
import asyncio
from functools import partial

from app import __version__
from app.engines.base import EngineError, EngineNotLoadedError
from app.engines.manager import EngineManager

from .protocol import (
    AsyncEventHandler,
    AsyncServer,
    Attribution,
    AudioChunk,
    AudioStart,
    AudioStop,
    Describe,
    Error,
    Event,
    Info,
    Synthesize,
    TtsProgram,
    TtsVoice,
    WYOMING_AVAILABLE,
)

WYOMING_PROGRAM_NAME = "wyoming-multi-tts"
WYOMING_VOICE_NAME = "default"


def _collect_supported_languages(manager: EngineManager) -> list[str]:
    languages: list[str] = []
    registry = getattr(manager, "registry", None)
    engines = registry.values() if isinstance(registry, dict) else [manager.active_engine]
    for engine in engines:
        status = engine.status()
        for voice in status.available_voices:
            for language in voice.languages:
                if language not in languages:
                    languages.append(language)
        for language in engine.supported_languages():
            if language not in languages:
                languages.append(language)
    return languages


def _build_voices(manager: EngineManager) -> list[TtsVoice]:
    active = manager.active_engine
    languages = _collect_supported_languages(manager)
    return [
        TtsVoice(
            name=WYOMING_VOICE_NAME,
            description=f"Default voice routed through the active engine: {active.display_name()}",
            attribution=Attribution(name=WYOMING_PROGRAM_NAME, url=None),
            installed=True,
            version=None,
            languages=languages,
        )
    ]


def build_info(manager: EngineManager) -> Info:
    active = manager.active_engine
    status = active.status()
    return Info(
        tts=[
            TtsProgram(
                name=WYOMING_PROGRAM_NAME,
                description=f"Multi-engine Wyoming TTS server (active engine: {active.display_name()})",
                attribution=Attribution(name=WYOMING_PROGRAM_NAME, url=None),
                installed=True,
                version=__version__,
                voices=_build_voices(manager),
                supports_synthesize_streaming=status.supports_streaming,
            )
        ]
    )


class MultiTtsEventHandler(AsyncEventHandler):
    def __init__(self, manager: EngineManager, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.manager = manager

    @staticmethod
    def _resolve_voice_name(synthesize: Synthesize) -> str | None:
        voice = getattr(synthesize, "voice", None)
        if voice is None:
            return None
        speaker = getattr(voice, "speaker", None)
        if speaker:
            return speaker
        return getattr(voice, "name", None)

    async def _write_synthesis_result(self, result) -> None:
        await self.write_event(
            AudioStart(
                rate=result.sample_rate,
                width=result.sample_width,
                channels=result.channels,
            ).event()
        )
        await self.write_event(
            AudioChunk(
                audio=result.pcm_audio,
                rate=result.sample_rate,
                width=result.sample_width,
                channels=result.channels,
            ).event()
        )
        await self.write_event(AudioStop().event())

    async def handle_event(self, event: Event) -> bool:
        try:
            if Describe.is_type(event.type):
                await self.write_event(build_info(self.manager).event())
                return True
            if not Synthesize.is_type(event.type):
                return True
            synthesize = Synthesize.from_event(event)
            result = await self.manager.synthesize(
                text=synthesize.text,
                voice=self._resolve_voice_name(synthesize),
                language=getattr(synthesize.voice, "language", None),
                options=getattr(synthesize, "context", None),
            )
            await self._write_synthesis_result(result)
            return True
        except (EngineNotLoadedError, EngineError) as err:
            await self.write_event(Error(text=str(err), code=err.__class__.__name__).event())
            return True


async def serve_wyoming(manager: EngineManager, uri: str) -> None:
    if not WYOMING_AVAILABLE:
        raise RuntimeError("The 'wyoming' package is not installed.")
    server = AsyncServer.from_uri(uri)
    await server.run(partial(MultiTtsEventHandler, manager))
