"""Shared speaker discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass(frozen=True)
class SpeakerProfile:
    name: str
    wav_paths: list[Path]


class SpeakerStore:
    def __init__(self, speaker_dir: str | Path) -> None:
        self.speaker_dir = Path(speaker_dir)

    def list_profiles(self) -> list[SpeakerProfile]:
        if not self.speaker_dir.exists():
            return []
        profiles: list[SpeakerProfile] = []
        for entry in sorted(self.speaker_dir.iterdir(), key=lambda item: item.name.lower()):
            if entry.is_file() and entry.suffix.lower() in SUPPORTED_SUFFIXES:
                profiles.append(SpeakerProfile(name=entry.stem, wav_paths=[entry]))
                continue
            if not entry.is_dir():
                continue
            wav_paths = [
                path
                for path in sorted(entry.iterdir(), key=lambda item: item.name.lower())
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            ]
            if wav_paths:
                profiles.append(SpeakerProfile(name=entry.name, wav_paths=wav_paths))
        return profiles

    def profile_names(self) -> list[str]:
        return [profile.name for profile in self.list_profiles()]

    def get_profile(self, name: str | None, default_name: str | None = None) -> SpeakerProfile | None:
        target = name or default_name
        if not target or target == "default":
            return None
        for profile in self.list_profiles():
            if profile.name == target:
                return profile
        target_path = Path(target)
        if target_path.exists() and target_path.is_file() and target_path.suffix.lower() in SUPPORTED_SUFFIXES:
            return SpeakerProfile(name=target_path.stem, wav_paths=[target_path])
        return None

    def ensure_exists(self) -> None:
        self.speaker_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def wav_paths(profile: SpeakerProfile | None) -> list[str] | None:
        if profile is None:
            return None
        return [str(path) for path in profile.wav_paths]

