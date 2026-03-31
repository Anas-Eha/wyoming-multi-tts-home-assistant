"""Wyoming imports with test-friendly fallbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


try:
    from wyoming.audio import AudioChunk, AudioStart, AudioStop
    from wyoming.error import Error
    from wyoming.event import Event
    from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
    from wyoming.server import AsyncEventHandler, AsyncServer
    from wyoming.tts import Synthesize

    WYOMING_AVAILABLE = True
except ImportError:  # pragma: no cover
    WYOMING_AVAILABLE = False

    @dataclass
    class Event:
        type: str
        data: dict[str, Any] = field(default_factory=dict)

    class AsyncEventHandler:
        async def write_event(self, event: Event) -> None:
            raise NotImplementedError

    class AsyncServer:
        @classmethod
        def from_uri(cls, uri: str):
            raise RuntimeError("The 'wyoming' package is not installed.")

        async def run(self, factory):
            raise RuntimeError("The 'wyoming' package is not installed.")

    @dataclass
    class Attribution:
        name: str | None = None
        url: str | None = None

    @dataclass
    class TtsVoice:
        name: str
        description: str = ""
        attribution: Attribution | None = None
        installed: bool = True
        version: str | None = None
        languages: list[str] = field(default_factory=list)

    @dataclass
    class TtsProgram:
        name: str
        description: str = ""
        attribution: Attribution | None = None
        installed: bool = True
        version: str | None = None
        voices: list[TtsVoice] = field(default_factory=list)
        supports_synthesize_streaming: bool = False

    @dataclass
    class Info:
        tts: list[TtsProgram] = field(default_factory=list)

        def event(self) -> Event:
            return Event("describe", {"tts": self.tts})

    class Describe:
        @staticmethod
        def is_type(event_type: str) -> bool:
            return event_type == "describe"

    @dataclass
    class VoiceRef:
        name: str | None = None
        language: str | None = None

    @dataclass
    class Synthesize:
        text: str
        voice: VoiceRef | None = None
        context: dict[str, Any] | None = None

        @staticmethod
        def is_type(event_type: str) -> bool:
            return event_type == "synthesize"

        @classmethod
        def from_event(cls, event: Event) -> "Synthesize":
            voice = event.data.get("voice")
            return cls(
                text=event.data.get("text", ""),
                voice=VoiceRef(**voice) if isinstance(voice, dict) else voice,
                context=event.data.get("context"),
            )

    @dataclass
    class AudioStart:
        rate: int
        width: int
        channels: int

        def event(self) -> Event:
            return Event("audio-start", {"rate": self.rate, "width": self.width, "channels": self.channels})

    @dataclass
    class AudioChunk:
        audio: bytes
        rate: int
        width: int
        channels: int

        def event(self) -> Event:
            return Event("audio-chunk", {"audio": self.audio, "rate": self.rate, "width": self.width, "channels": self.channels})

    @dataclass
    class AudioStop:
        def event(self) -> Event:
            return Event("audio-stop", {})

    @dataclass
    class Error:
        text: str
        code: str

        def event(self) -> Event:
            return Event("error", {"text": self.text, "code": self.code})

