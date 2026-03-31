"""Facebook MMS Polish TTS adapter."""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    class _TorchShim:
        @staticmethod
        def inference_mode():
            return nullcontext()

    torch = _TorchShim()  # type: ignore[assignment]

from app.audio.audio_utils import float32_to_pcm16, pcm16_to_wav_bytes

from .base import EngineError, EngineNotLoadedError, EngineSynthesisResult, EngineVoice, TtsEngine
from .helpers import cleanup_torch, preferred_device, snapshot_download_local_first, status_from_values, synthesis_metrics, with_cpu_fallback


MMS_MODEL_ID = "facebook/mms-tts-pol"
MMS_REQUIRED_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.txt",
]


class MmsPolishEngine(TtsEngine):
    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.getenv("MMS_TTS_MODEL_ID", MMS_MODEL_ID)
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device: str | None = None
        self._state = "not_loaded"
        self._load_time_ms: float | None = None
        self._last_loaded_at: float | None = None
        self._last_error: str | None = None
        self._sample_rate = 16000

    def engine_id(self) -> str:
        return "mms_tts_pol"

    def display_name(self) -> str:
        return "MMS TTS Polish"

    def supported_languages(self) -> list[str]:
        return ["pl"]

    def list_voices(self) -> list[EngineVoice]:
        return [
            EngineVoice(
                id="default",
                label="default",
                languages=["pl"],
                default_language="pl",
                description="facebook/mms-tts-pol default Polish voice",
            )
        ]

    def _load_model(self, device: str):
        import torch
        from transformers import AutoTokenizer, VitsModel

        model_source = snapshot_download_local_first(
            self.model_id,
            allow_patterns=MMS_REQUIRED_PATTERNS,
            token=os.getenv("HF_TOKEN"),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        model = VitsModel.from_pretrained(model_source)
        model_device = device if device.startswith("cuda") else "cpu"
        model.to(model_device)
        model.eval()
        sample_rate = int(getattr(model.config, "sampling_rate", 16000))
        return model, tokenizer, sample_rate

    def load(self, device_preference: str | None = None):
        if self._model is not None and self._tokenizer is not None:
            return self.status()
        self._state = "loading"
        self._last_error = None
        try:
            payload, device, load_time_ms = with_cpu_fallback(
                lambda selected: self._load_model(device_preference or selected)
            )
            model, tokenizer, sample_rate = payload
            self._model = model
            self._tokenizer = tokenizer
            self._device = device
            self._load_time_ms = load_time_ms
            self._last_loaded_at = time.time()
            self._sample_rate = sample_rate
            self._state = "ready"
        except Exception as err:
            self._state = "error"
            self._last_error = str(err)
            raise
        return self.status()

    def unload(self) -> None:
        self._state = "unloading"
        self._model = None
        self._tokenizer = None
        cleanup_torch()
        self._state = "not_loaded"

    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def status(self):
        return status_from_values(
            engine_id=self.engine_id(),
            display_name=self.display_name(),
            state=self._state,
            loaded=self.is_loaded(),
            device=self._device or preferred_device(),
            load_time_ms=self._load_time_ms,
            last_loaded_at=self._last_loaded_at,
            last_error=self._last_error,
            supports_streaming=False,
            voices=self.list_voices(),
            extra={
                "model_id": self.model_id,
                "sample_rate": self._sample_rate,
            },
        )

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None,
        language: str | None,
        options: dict[str, Any] | None = None,
    ) -> EngineSynthesisResult:
        del options
        if self._model is None or self._tokenizer is None:
            raise EngineNotLoadedError("MMS Polish engine is not loaded")

        resolved_language = "pl"
        resolved_voice = "default"

        normalized_text = text.strip()
        if not normalized_text:
            raise EngineError("Text must not be empty")

        inputs = self._tokenizer(normalized_text, return_tensors="pt")
        model_device = self._device or "cpu"
        if hasattr(inputs, "to"):
            inputs = inputs.to(model_device)
        else:
            inputs = {
                key: value.to(model_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

        started = time.perf_counter()
        inference_mode = getattr(torch, "inference_mode", None)
        if callable(inference_mode):
            with inference_mode():
                output = self._model(**inputs)
        else:
            output = self._model(**inputs)
        synthesis_time_ms = round((time.perf_counter() - started) * 1000.0, 2)

        waveform = output.waveform[0]
        pcm_audio = float32_to_pcm16(waveform)
        wav_audio = pcm16_to_wav_bytes(pcm_audio, sample_rate=self._sample_rate)
        return EngineSynthesisResult(
            engine_id=self.engine_id(),
            voice=resolved_voice,
            language=resolved_language,
            device=self._device or "cpu",
            sample_rate=self._sample_rate,
            channels=1,
            sample_width=2,
            wav_audio=wav_audio,
            pcm_audio=pcm_audio,
            metrics=synthesis_metrics(
                load_time_ms=self._load_time_ms,
                synthesis_time_ms=synthesis_time_ms,
                end_to_end_time_ms=synthesis_time_ms,
                pcm_bytes=len(pcm_audio),
                sample_rate=self._sample_rate,
                cold_start=False,
            ),
            backend="transformers-vits",
        )

    def health_payload(self) -> dict[str, Any]:
        return self.status().asdict()
