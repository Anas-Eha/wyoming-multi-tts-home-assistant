from __future__ import annotations

from app.engines.base import EngineNotLoadedError, EngineSynthesisResult, EngineStatus, EngineVoice, SynthesisMetrics, TtsEngine


class FakeIsolatedEngine(TtsEngine):
    def __init__(self) -> None:
        self._loaded = False

    def engine_id(self) -> str:
        return "fake_isolated"

    def display_name(self) -> str:
        return "Fake Isolated"

    def supported_languages(self) -> list[str]:
        return ["pl"]

    def list_voices(self) -> list[EngineVoice]:
        return [
            EngineVoice(
                id="default",
                label="Default",
                languages=["pl"],
                default_language="pl",
                description="Fake worker voice",
            )
        ]

    def load(self, device_preference: str | None = None) -> EngineStatus:
        self._loaded = True
        return self.status()

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def status(self) -> EngineStatus:
        return EngineStatus(
            engine_id=self.engine_id(),
            display_name=self.display_name(),
            state="ready" if self._loaded else "not_loaded",
            loaded=self._loaded,
            device="cpu",
            available_voices=self.list_voices(),
        )

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None,
        language: str | None,
        options: dict | None = None,
    ) -> EngineSynthesisResult:
        if not self._loaded:
            raise EngineNotLoadedError("Fake isolated backend is not loaded")
        return EngineSynthesisResult(
            engine_id=self.engine_id(),
            voice=voice or "default",
            language=language or "pl",
            device="cpu",
            sample_rate=24000,
            channels=1,
            sample_width=2,
            wav_audio=b"RIFFFAKE" + (b"\x00" * 1024 * 1024),
            pcm_audio=b"\x00\x00" * 512 * 1024,
            backend="fake-isolated",
            metrics=SynthesisMetrics(
                load_time_ms=3.0,
                synthesis_time_ms=4.0,
                end_to_end_time_ms=4.0,
                audio_duration_ms=8.0,
                real_time_factor=0.5,
                cold_start=False,
            ),
        )

    def health_payload(self) -> dict:
        return self.status().asdict()
