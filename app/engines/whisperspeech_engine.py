"""WhisperSpeech engine adapter."""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.audio.audio_utils import float32_to_pcm16, pcm16_to_wav_bytes
from app.audio.text_chunking import split_text_for_realtime_tts

from .base import EngineNotLoadedError, EngineSynthesisResult, EngineVoice, TtsEngine
from .helpers import cleanup_torch, env_path, preferred_device, status_from_values, synthesis_metrics
from .speaker_store import SpeakerStore


LOGGER = logging.getLogger(__name__)
WHISPERSPEECH_REPO_ID = "WhisperSpeech/WhisperSpeech"
DEFAULT_T2S_MODEL = "t2s-small-en+pl.model"
DEFAULT_S2A_MODEL = "s2a-q4-small-en+pl.model"
SUPPORTED_LANGUAGES = ["de", "en", "es", "fr", "it", "nl", "pl"]
LANGUAGE_ALIASES = {
    "de": "de",
    "de-de": "de",
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
    "es": "es",
    "es-es": "es",
    "fr": "fr",
    "fr-fr": "fr",
    "it": "it",
    "it-it": "it",
    "nl": "nl",
    "nl-nl": "nl",
    "pl": "pl",
    "pl-pl": "pl",
    "polish": "pl",
    "polski": "pl",
}
LANGUAGE_TO_LABEL = {
    "de": "DE",
    "en": "EN",
    "es": "ES",
    "fr": "FR",
    "it": "IT",
    "nl": "NL",
    "pl": "PL",
}


@dataclass(frozen=True)
class ModelPreset:
    name: str
    t2s_ref: str
    s2a_ref: str
    label: str
    languages: tuple[str, ...]


BUILTIN_MODEL_PRESETS = {
    "base-en+pl": ModelPreset(
        name="base-en+pl",
        t2s_ref="t2s-base-en+pl.model",
        s2a_ref="s2a-q4-base-en+pl.model",
        label="Base",
        languages=("en", "pl"),
    ),
    "small-en+pl": ModelPreset(
        name="small-en+pl",
        t2s_ref=DEFAULT_T2S_MODEL,
        s2a_ref=DEFAULT_S2A_MODEL,
        label="Small",
        languages=("en", "pl"),
    ),
    "tiny-en+pl": ModelPreset(
        name="tiny-en+pl",
        t2s_ref="t2s-tiny-en+pl.model",
        s2a_ref="s2a-q4-tiny-en+pl.model",
        label="Tiny",
        languages=("en", "pl"),
    ),
    "medium-7lang": ModelPreset(
        name="medium-7lang",
        t2s_ref="t2s-v1.95-medium-7lang.model",
        s2a_ref="s2a-v1.95-medium-7lang.model",
        label="Medium 7LANG",
        languages=("de", "en", "es", "fr", "it", "nl", "pl"),
    ),
    "fast-en+pl": ModelPreset(
        name="fast-en+pl",
        t2s_ref="t2s-hq-fast-en+pl.model",
        s2a_ref="s2a-q4-hq-fast-en+pl.model",
        label="HQ Fast",
        languages=("en", "pl"),
    ),
}


class WhisperSpeechEngine(TtsEngine):
    _original_hf_download: Callable[..., str] | None = None
    _patched_hf_download: Callable[..., str] | None = None
    _active_engine: "WhisperSpeechEngine | None" = None

    def __init__(self, speaker_dir: str | None = None, model_cache_dir: str | None = None) -> None:
        self.speaker_store = SpeakerStore(speaker_dir or env_path("SPEAKER_DIR", "/data/speakers"))
        self.model_cache_dir = model_cache_dir or os.getenv("WHISPERSPEECH_MODEL_CACHE_DIR") or os.getenv("HF_HOME")
        self.default_language = "pl"
        self.default_voice = "default"
        self.default_model_preset = "small-en+pl"
        self.optimize = True
        self.torch_compile = False
        self.cps = 15.0
        self.max_loaded_pipelines = 1
        self.max_chunk_chars = int(os.getenv("WHISPERSPEECH_MAX_CHUNK_CHARS", "120"))
        self.first_fragment_min_words = int(os.getenv("WHISPERSPEECH_FIRST_FRAGMENT_MIN_WORDS", "6"))
        self.first_fragment_max_words = int(os.getenv("WHISPERSPEECH_FIRST_FRAGMENT_MAX_WORDS", "12"))
        self.first_fragment_delimiters = os.getenv("WHISPERSPEECH_FIRST_FRAGMENT_DELIMITERS", ".?!;:,")
        self._pipelines: OrderedDict[str, Any] = OrderedDict()
        self._pipeline_devices: dict[str, str] = {}
        self._speaker_embeddings: dict[str, Any] = {}
        self._state = "not_loaded"
        self._device: str | None = None
        self._load_time_ms: float | None = None
        self._last_loaded_at: float | None = None
        self._last_error: str | None = None
        self._sample_rate = 24000
        self._model_presets = dict(BUILTIN_MODEL_PRESETS)

    def engine_id(self) -> str:
        return "whisperspeech"

    def display_name(self) -> str:
        return "WhisperSpeech"

    def supported_languages(self) -> list[str]:
        return list(SUPPORTED_LANGUAGES)

    def list_voices(self) -> list[EngineVoice]:
        base_names = ["default", *self.speaker_store.profile_names()]
        seen: set[str] = set()
        voices: list[EngineVoice] = []
        for base_name in base_names:
            for preset_name, preset in self._model_presets.items():
                for language in preset.languages:
                    voice_id = self._voice_variant_name(base_name, preset_name, language)
                    if voice_id in seen:
                        continue
                    seen.add(voice_id)
                    voices.append(
                        EngineVoice(
                            id=voice_id,
                            label=voice_id,
                            languages=list(preset.languages),
                            default_language=self.default_language,
                            description=f"WhisperSpeech voice ({preset.label})",
                        )
                    )
        return voices

    def load(self, device_preference: str | None = None):
        if self.is_loaded():
            return self.status()
        self._state = "loading"
        self._last_error = None
        started = time.perf_counter()
        try:
            self._patch_hf_download_local_first()
            pipeline, resolved_device = self._get_pipeline_for_request(
                self.default_model_preset,
                self.default_language,
                requested_device=device_preference or preferred_device(),
            )
            self._pipelines[self.default_model_preset] = pipeline
            self._pipeline_devices[self.default_model_preset] = resolved_device
            self._device = resolved_device
            self._load_time_ms = round((time.perf_counter() - started) * 1000.0, 2)
            self._last_loaded_at = time.time()
            self._state = "ready"
        except Exception as err:
            self._state = "error"
            self._last_error = str(err)
            raise
        return self.status()

    def unload(self) -> None:
        self._state = "unloading"
        self._pipelines.clear()
        self._pipeline_devices.clear()
        self._speaker_embeddings.clear()
        cleanup_torch()
        gc.collect()
        self._state = "not_loaded"

    def is_loaded(self) -> bool:
        return bool(self._pipelines)

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
            supports_streaming=True,
            voices=self.list_voices(),
            extra={"preset": self.default_model_preset, "loaded_pipeline_keys": list(self._pipelines.keys())},
        )

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None,
        language: str | None,
        options: dict[str, Any] | None = None,
    ):
        if not self.is_loaded():
            raise EngineNotLoadedError("WhisperSpeech engine is not loaded")
        resolved_language = self.resolve_language(language)
        resolved_voice = voice or self._voice_variant_name(
            self.default_voice,
            self.default_model_preset,
            resolved_language,
        )
        base_voice_name, preset_name, voice_language = self.resolve_voice_selection(resolved_voice)
        preset = self._model_presets.get(preset_name, self._model_presets[self.default_model_preset])
        selected_language = self._resolve_language_for_preset(voice_language or resolved_language, preset)
        cache_key = self._pipeline_cache_key(preset_name)
        pipeline = self._pipelines.get(cache_key)
        if pipeline is None:
            pipeline, resolved_device = self._get_pipeline_for_request(
                preset_name,
                selected_language,
                requested_device=self._device or preferred_device(),
            )
            self._pipelines[cache_key] = pipeline
            self._pipeline_devices[cache_key] = resolved_device
        self._mark_pipeline_recent(cache_key)
        chunks = split_text_for_realtime_tts(
            text.strip(),
            max_chars=self.max_chunk_chars,
            first_fragment_min_words=self.first_fragment_min_words,
            first_fragment_max_words=self.first_fragment_max_words,
            first_fragment_delimiters=self.first_fragment_delimiters,
        )
        started = time.perf_counter()
        audio = self._generate_audio_chunks(
            pipeline=pipeline,
            chunks=chunks,
            language=selected_language,
            voice_name=base_voice_name,
            cps=self.cps,
        )
        synthesis_time_ms = round((time.perf_counter() - started) * 1000.0, 2)
        pcm_audio = float32_to_pcm16(audio)
        wav_audio = pcm16_to_wav_bytes(pcm_audio, sample_rate=self._sample_rate)
        return EngineSynthesisResult(
            engine_id=self.engine_id(),
            voice=resolved_voice,
            language=selected_language,
            device=self._pipeline_devices.get(cache_key, self._device or "cpu"),
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
            backend="whisperspeech",
        )

    def health_payload(self) -> dict[str, Any]:
        return self.status().asdict()

    def resolve_language(self, language: str | None) -> str:
        if not language:
            return self.default_language
        normalized = language.strip().lower().replace("_", "-")
        if normalized in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[normalized]
        primary = normalized.split("-", 1)[0]
        return LANGUAGE_ALIASES.get(primary, primary)

    def resolve_voice_selection(self, voice_name: str) -> tuple[str, str, str | None]:
        stripped = voice_name.strip()
        for preset_name, preset in self._model_presets.items():
            for language in preset.languages:
                suffix = f"{preset.label} {LANGUAGE_TO_LABEL.get(language, language.upper())}"
                if stripped == suffix:
                    return self.default_voice, preset_name, language
                if stripped.endswith(f" [{suffix}]"):
                    return stripped[: -(len(suffix) + 3)], preset_name, language
        return stripped, self.default_model_preset, None

    def _voice_variant_name(self, base_voice_name: str, preset_name: str, language: str) -> str:
        preset = self._model_presets[preset_name]
        label = f"{preset.label} {LANGUAGE_TO_LABEL.get(language, language.upper())}"
        if base_voice_name == self.default_voice:
            return label
        return f"{base_voice_name} [{label}]"

    def _resolve_language_for_preset(self, requested_language: str, preset: ModelPreset) -> str:
        if requested_language in preset.languages:
            return requested_language
        if self.default_language in preset.languages:
            return self.default_language
        return preset.languages[0]

    def _pipeline_cache_key(self, preset_name: str) -> str:
        return preset_name.strip().lower() or self.default_model_preset

    def _get_pipeline_for_request(
        self,
        preset_name: str,
        language: str,
        *,
        requested_device: str,
    ) -> tuple[Any, str]:
        cache_key = self._pipeline_cache_key(preset_name)
        if cache_key in self._pipelines:
            return self._pipelines[cache_key], self._pipeline_devices.get(cache_key, requested_device)
        self._evict_pipelines_if_needed()
        resolved_t2s_ref, resolved_s2a_ref = self._resolve_pipeline_refs(preset_name, language)
        from whisperspeech.pipeline import Pipeline as WhisperSpeechPipeline

        pipeline, resolved_device = self._build_pipeline(
            WhisperSpeechPipeline,
            t2s_ref=resolved_t2s_ref,
            s2a_ref=resolved_s2a_ref,
            requested_device=requested_device,
        )
        return pipeline, resolved_device

    def _resolve_pipeline_refs(self, preset_name: str, language: str) -> tuple[str, str]:
        del language
        preset = self._model_presets.get(preset_name, self._model_presets[self.default_model_preset])
        return (
            self._resolve_model_ref(preset.t2s_ref, DEFAULT_T2S_MODEL),
            self._resolve_model_ref(preset.s2a_ref, DEFAULT_S2A_MODEL),
        )

    def _resolve_model_ref(self, explicit_ref: str | None, default_filename: str) -> str:
        ref = explicit_ref or default_filename
        if Path(ref).exists():
            return ref
        cached = self._find_cached_hf_file(WHISPERSPEECH_REPO_ID, ref)
        if cached:
            LOGGER.info("Using cached WhisperSpeech model for %s: %s", ref, cached)
            return cached
        return f"{WHISPERSPEECH_REPO_ID}:{ref}"

    def _find_cached_hf_file(self, repo_id: str, filename: str) -> str | None:
        roots = self._cache_roots()
        for root in roots:
            if not root.exists():
                continue
            repo_dir = root / f"models--{repo_id.replace('/', '--')}" / "snapshots"
            if not repo_dir.exists():
                continue
            for snapshot in sorted(repo_dir.iterdir(), reverse=True):
                candidate = snapshot / filename
                if candidate.exists():
                    return str(candidate)
        return None

    def _cache_roots(self) -> list[Path]:
        candidates = [
            self.model_cache_dir,
            os.getenv("HF_HOME"),
            os.getenv("HUGGINGFACE_HUB_CACHE"),
            os.path.expanduser("~/.cache/huggingface"),
            os.path.expanduser("~/.cache/huggingface/hub"),
            "/data/hf-cache",
            "/data/hf-cache/hub",
        ]
        roots: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            for root in (path, path / "hub"):
                if root in seen:
                    continue
                seen.add(root)
                roots.append(root)
        return roots

    def _patch_hf_download_local_first(self) -> None:
        try:
            import huggingface_hub
            import huggingface_hub.file_download as hf_file_download
        except ImportError:
            return
        WhisperSpeechEngine._active_engine = self
        if WhisperSpeechEngine._original_hf_download is None:
            WhisperSpeechEngine._original_hf_download = huggingface_hub.hf_hub_download
        if WhisperSpeechEngine._patched_hf_download is None:
            original_download = WhisperSpeechEngine._original_hf_download
            assert original_download is not None

            def _local_first_download(*args, **kwargs):
                repo_id = kwargs.get("repo_id") or (args[0] if args else None)
                filename = kwargs.get("filename") or (args[1] if len(args) > 1 else None)
                active = WhisperSpeechEngine._active_engine
                if active is not None and repo_id and filename:
                    cached = active._find_cached_hf_file(str(repo_id), str(filename))
                    if cached:
                        return cached
                return original_download(*args, **kwargs)

            WhisperSpeechEngine._patched_hf_download = _local_first_download
        patched_download = WhisperSpeechEngine._patched_hf_download
        assert patched_download is not None
        huggingface_hub.hf_hub_download = patched_download
        hf_file_download.hf_hub_download = patched_download
        self._rebind_module_hf_downloads(patched_download)

    def _rebind_module_hf_downloads(self, patched_download: Callable[..., str]) -> None:
        module_names = (
            "vocos.pretrained",
            "whisperspeech.inference",
            "whisperspeech.t2s_up_wds_mlang_enclm",
            "whisperspeech.s2a_delar_mup_wds_mlang",
            "whisperspeech.s2a_delar_mup_wds_mlang_cond",
            "whisperspeech.vq_stoks",
        )
        for module_name in module_names:
            module = sys.modules.get(module_name)
            if module is not None:
                setattr(module, "hf_hub_download", patched_download)

    def _build_pipeline(
        self,
        pipeline_cls: Any,
        *,
        t2s_ref: str,
        s2a_ref: str,
        requested_device: str,
    ) -> tuple[Any, str]:
        candidate_devices = [requested_device]
        if requested_device.startswith("cuda"):
            candidate_devices.append("cpu")
        for candidate_device in candidate_devices:
            try:
                kwargs: dict[str, Any] = {
                    "t2s_ref": t2s_ref,
                    "s2a_ref": s2a_ref,
                    "optimize": self.optimize,
                    "torch_compile": self.torch_compile,
                    "cache_dir": self.model_cache_dir,
                }
                if candidate_device:
                    kwargs["device"] = candidate_device
                pipeline = pipeline_cls(**kwargs)
                self._ensure_pipeline_dtype(pipeline, candidate_device or "cpu")
                return pipeline, candidate_device or "cpu"
            except Exception as err:
                if candidate_device == "cpu":
                    raise
                LOGGER.warning("Failed to initialize WhisperSpeech on %s, retrying on cpu: %s", candidate_device, err)
        raise RuntimeError("WhisperSpeech pipeline initialization failed")

    def _ensure_pipeline_dtype(self, pipeline: Any, device: str) -> None:
        try:
            import torch
        except ImportError:
            return
        default_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        for attr_name in ("t2s", "s2a"):
            component = getattr(pipeline, attr_name, None)
            if component is not None and not hasattr(component, "dtype"):
                component.dtype = default_dtype

    def _mark_pipeline_recent(self, cache_key: str) -> None:
        if cache_key in self._pipelines:
            self._pipelines.move_to_end(cache_key)

    def _evict_pipelines_if_needed(self) -> None:
        while len(self._pipelines) >= self.max_loaded_pipelines:
            cache_key, _ = self._pipelines.popitem(last=False)
            self._pipeline_devices.pop(cache_key, None)
            gc.collect()
            cleanup_torch()

    def _speaker_for_voice(self, voice_name: str) -> Any | None:
        if voice_name == self.default_voice:
            return None
        if voice_name in self._speaker_embeddings:
            return self._speaker_embeddings[voice_name]
        profile = self.speaker_store.get_profile(voice_name)
        if profile is None:
            return None
        default_pipeline = self._pipelines.get(self.default_model_preset) or next(iter(self._pipelines.values()), None)
        if default_pipeline is None or not hasattr(default_pipeline, "extract_spk_emb"):
            return None
        speaker_source = profile.wav_paths[0]
        embedding = default_pipeline.extract_spk_emb(str(speaker_source))
        self._speaker_embeddings[voice_name] = embedding
        return embedding

    def _generate_audio(
        self,
        *,
        pipeline: Any,
        text: str,
        language: str,
        voice_name: str,
        cps: float,
    ) -> list[float]:
        speaker = self._speaker_for_voice(voice_name)
        generated = pipeline.generate(text, speaker=speaker, lang=language, cps=cps)
        if hasattr(generated, "detach"):
            generated = generated.detach()
        if hasattr(generated, "cpu"):
            generated = generated.cpu()
        if hasattr(generated, "numpy"):
            generated = generated.numpy()
        if hasattr(generated, "tolist"):
            generated = generated.tolist()
        if generated and isinstance(generated[0], list):
            generated = generated[0]
        return [float(sample) for sample in generated]

    def _generate_audio_chunks(
        self,
        *,
        pipeline: Any,
        chunks: list[str],
        language: str,
        voice_name: str,
        cps: float,
    ) -> list[float]:
        if not chunks:
            return []
        if len(chunks) == 1:
            return self._generate_audio(
                pipeline=pipeline,
                text=chunks[0],
                language=language,
                voice_name=voice_name,
                cps=cps,
            )

        audio: list[float] = []
        for chunk in chunks:
            chunk_audio = self._generate_audio(
                pipeline=pipeline,
                text=chunk,
                language=language,
                voice_name=voice_name,
                cps=cps,
            )
            audio.extend(chunk_audio)
        return audio
