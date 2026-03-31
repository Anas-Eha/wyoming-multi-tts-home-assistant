"""Chatterbox engine adapter."""

from __future__ import annotations

import os
import time
from importlib import import_module
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from app.audio.audio_utils import float32_to_pcm16, pcm16_to_wav_bytes

from .base import EngineError, EngineNotLoadedError, EngineSynthesisResult, EngineVoice, TtsEngine
from .helpers import cleanup_torch, preferred_device, snapshot_download_local_first, status_from_values, synthesis_metrics, with_cpu_fallback


SUPPORTED_LANGUAGES = [
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko",
    "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
]

CHATTERBOX_REPO_ID = "ResembleAI/chatterbox"
CHATTERBOX_REQUIRED_FILES = [
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
]


class ChatterboxEngine(TtsEngine):
    def __init__(self) -> None:
        self._model: Any | None = None
        self._device: str | None = None
        self._state = "not_loaded"
        self._load_time_ms: float | None = None
        self._last_loaded_at: float | None = None
        self._last_error: str | None = None
        self._voice_prompts: dict[str, str] = {}
        self._sample_rate = 24000

    def engine_id(self) -> str:
        return "chatterbox"

    def display_name(self) -> str:
        return "Chatterbox"

    def supported_languages(self) -> list[str]:
        return list(SUPPORTED_LANGUAGES)

    def list_voices(self) -> list[EngineVoice]:
        voices = [
            EngineVoice(
                id=f"default__{language}",
                label=f"Default ({language.upper()})",
                languages=[language],
                default_language=language,
                description=f"Chatterbox default voice for {language}",
            )
            for language in self.supported_languages()
        ]
        for voice_id, prompt_path in sorted(self._voice_prompts.items()):
            language = voice_id.rsplit("__", 1)[-1]
            voices.append(
                EngineVoice(
                    id=voice_id,
                    label=voice_id,
                    languages=[language],
                    default_language=language,
                    description=f"Reference prompt from {Path(prompt_path).name}",
                )
            )
        return voices

    def _load_model(self, device: str):
        module = import_module("chatterbox.mtl_tts")
        perth_module = getattr(module, "perth", None)
        watermarker = getattr(perth_module, "PerthImplicitWatermarker", None)
        if watermarker is None:
            class NullWatermarker:
                def apply_watermark(self, wav, sample_rate=None):
                    return wav

            if perth_module is not None:
                perth_module.PerthImplicitWatermarker = NullWatermarker
        model_class = getattr(module, "ChatterboxMultilingualTTS")
        ckpt_dir = self._resolve_checkpoint_dir()
        return model_class.from_local(ckpt_dir, device)

    def _missing_checkpoint_files(self, ckpt_dir: str | Path) -> list[str]:
        root = Path(ckpt_dir)
        return [name for name in CHATTERBOX_REQUIRED_FILES if not (root / name).exists()]

    def _download_checkpoint(self) -> str:
        return snapshot_download(
            repo_id=CHATTERBOX_REPO_ID,
            repo_type="model",
            allow_patterns=CHATTERBOX_REQUIRED_FILES,
            token=os.getenv("HF_TOKEN"),
        )

    def _resolve_checkpoint_dir(self) -> str:
        ckpt_dir = snapshot_download_local_first(
            CHATTERBOX_REPO_ID,
            allow_patterns=CHATTERBOX_REQUIRED_FILES,
            token=os.getenv("HF_TOKEN"),
        )
        missing_files = self._missing_checkpoint_files(ckpt_dir)
        if missing_files:
            ckpt_dir = self._download_checkpoint()
            missing_files = self._missing_checkpoint_files(ckpt_dir)
        if missing_files:
            missing = ", ".join(missing_files)
            raise EngineError(f"Chatterbox checkpoint is incomplete, missing: {missing}")
        return ckpt_dir

    def load(self, device_preference: str | None = None):
        if self._model is not None:
            return self.status()
        self._state = "loading"
        self._last_error = None
        try:
            model, device, load_time_ms = with_cpu_fallback(lambda selected: self._load_model(device_preference or selected))
            self._model = model
            self._device = device
            self._load_time_ms = load_time_ms
            self._last_loaded_at = time.time()
            self._sample_rate = int(getattr(model, "sr", 24000))
            self._state = "ready"
        except Exception as err:
            self._state = "error"
            self._last_error = str(err)
            raise
        return self.status()

    def unload(self) -> None:
        self._state = "unloading"
        self._model = None
        cleanup_torch()
        self._state = "not_loaded"

    def is_loaded(self) -> bool:
        return self._model is not None

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
            extra={"sample_rate": self._sample_rate},
        )

    def synthesize(self, text: str, *, voice: str | None, language: str | None, options: dict[str, Any] | None = None):
        if self._model is None:
            raise EngineNotLoadedError("Chatterbox engine is not loaded")
        resolved_voice = voice or "default__pl"
        resolved_language = (language or resolved_voice.rsplit("__", 1)[-1]).lower()
        generate_kwargs: dict[str, Any] = {"language_id": resolved_language}
        prompt_path = self._voice_prompts.get(resolved_voice)
        if prompt_path:
            generate_kwargs["audio_prompt_path"] = prompt_path
        started = time.perf_counter()
        audio = self._model.generate(text.strip(), **generate_kwargs)
        synthesis_time_ms = round((time.perf_counter() - started) * 1000.0, 2)
        pcm_audio = float32_to_pcm16(audio)
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
            backend="chatterbox-multilingual",
        )

    def health_payload(self) -> dict[str, Any]:
        return self.status().asdict()
