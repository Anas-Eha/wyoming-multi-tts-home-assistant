"""Common engine models and interfaces."""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EngineVoice:
    id: str
    label: str
    languages: list[str]
    default_language: str
    description: str = ""

    @classmethod
    def fromdict(cls, payload: dict[str, Any]) -> "EngineVoice":
        return cls(
            id=str(payload["id"]),
            label=str(payload["label"]),
            languages=[str(language) for language in payload.get("languages", [])],
            default_language=str(payload["default_language"]),
            description=str(payload.get("description", "")),
        )


@dataclass
class EngineStatus:
    engine_id: str
    display_name: str
    state: str = "not_loaded"
    loaded: bool = False
    loading: bool = False
    device: str | None = None
    load_time_ms: float | None = None
    last_loaded_at: float | None = None
    last_error: str | None = None
    supports_streaming: bool = False
    available_voices: list[EngineVoice] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["available_voices"] = [
            asdict(voice) if isinstance(voice, EngineVoice) else dict(voice)
            for voice in self.available_voices
        ]
        return payload

    @classmethod
    def fromdict(cls, payload: dict[str, Any]) -> "EngineStatus":
        return cls(
            engine_id=str(payload["engine_id"]),
            display_name=str(payload["display_name"]),
            state=str(payload.get("state", "not_loaded")),
            loaded=bool(payload.get("loaded", False)),
            loading=bool(payload.get("loading", False)),
            device=payload.get("device"),
            load_time_ms=payload.get("load_time_ms"),
            last_loaded_at=payload.get("last_loaded_at"),
            last_error=payload.get("last_error"),
            supports_streaming=bool(payload.get("supports_streaming", False)),
            available_voices=[
                EngineVoice.fromdict(voice)
                for voice in payload.get("available_voices", [])
            ],
            extra=dict(payload.get("extra", {})),
        )


@dataclass
class SynthesisMetrics:
    load_time_ms: float | None
    synthesis_time_ms: float
    end_to_end_time_ms: float
    audio_duration_ms: float
    real_time_factor: float
    cold_start: bool
    time_to_first_chunk_ms: float | None = None

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def fromdict(cls, payload: dict[str, Any]) -> "SynthesisMetrics":
        return cls(
            load_time_ms=payload.get("load_time_ms"),
            synthesis_time_ms=float(payload["synthesis_time_ms"]),
            end_to_end_time_ms=float(payload["end_to_end_time_ms"]),
            audio_duration_ms=float(payload["audio_duration_ms"]),
            real_time_factor=float(payload["real_time_factor"]),
            cold_start=bool(payload["cold_start"]),
            time_to_first_chunk_ms=payload.get("time_to_first_chunk_ms"),
        )


@dataclass
class EngineSynthesisResult:
    engine_id: str
    voice: str
    language: str
    device: str
    sample_rate: int
    channels: int
    sample_width: int
    wav_audio: bytes
    pcm_audio: bytes
    metrics: SynthesisMetrics
    backend: str

    def asdict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["wav_bytes"] = len(self.wav_audio)
        payload["pcm_bytes"] = len(self.pcm_audio)
        payload["metrics"] = self.metrics.asdict()
        del payload["wav_audio"]
        del payload["pcm_audio"]
        return payload

    def to_transport_dict(self) -> dict[str, Any]:
        payload = self.asdict()
        payload["wav_audio_base64"] = base64.b64encode(self.wav_audio).decode("ascii")
        payload["pcm_audio_base64"] = base64.b64encode(self.pcm_audio).decode("ascii")
        return payload

    def to_file_transport_dict(self, *, wav_path: str, pcm_path: str) -> dict[str, Any]:
        payload = self.asdict()
        payload["wav_path"] = wav_path
        payload["pcm_path"] = pcm_path
        return payload

    @classmethod
    def from_transport_dict(cls, payload: dict[str, Any]) -> "EngineSynthesisResult":
        wav_audio: bytes
        pcm_audio: bytes
        if "wav_path" in payload and "pcm_path" in payload:
            wav_audio = Path(str(payload["wav_path"])).read_bytes()
            pcm_audio = Path(str(payload["pcm_path"])).read_bytes()
        else:
            wav_audio = base64.b64decode(payload["wav_audio_base64"])
            pcm_audio = base64.b64decode(payload["pcm_audio_base64"])
        return cls(
            engine_id=str(payload["engine_id"]),
            voice=str(payload["voice"]),
            language=str(payload["language"]),
            device=str(payload["device"]),
            sample_rate=int(payload["sample_rate"]),
            channels=int(payload["channels"]),
            sample_width=int(payload["sample_width"]),
            wav_audio=wav_audio,
            pcm_audio=pcm_audio,
            metrics=SynthesisMetrics.fromdict(payload["metrics"]),
            backend=str(payload["backend"]),
        )


class EngineError(RuntimeError):
    """Base engine error."""


class EngineNotLoadedError(EngineError):
    """Raised when an engine is used before loading."""


class TtsEngine(ABC):
    """Common contract for all engines."""

    @abstractmethod
    def engine_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def display_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def supported_languages(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def list_voices(self) -> list[EngineVoice]:
        raise NotImplementedError

    @abstractmethod
    def load(self, device_preference: str | None = None) -> EngineStatus:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_loaded(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def status(self) -> EngineStatus:
        raise NotImplementedError

    @abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        voice: str | None,
        language: str | None,
        options: dict[str, Any] | None = None,
    ) -> EngineSynthesisResult:
        raise NotImplementedError

    @abstractmethod
    def health_payload(self) -> dict[str, Any]:
        raise NotImplementedError
