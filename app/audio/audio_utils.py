"""Shared audio conversion helpers."""

from __future__ import annotations

import io
import math
import wave
from array import array
from typing import Any, Iterable

import numpy as np
import soundfile as sf


def _as_mono_float_array(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    elif hasattr(audio, "tolist"):
        audio = audio.tolist()

    array_audio = np.asarray(audio, dtype=np.float32)
    if array_audio.ndim == 0:
        return array_audio.reshape(1)
    return array_audio.reshape(-1)


def float32_to_pcm16(audio: Iterable[float] | Any) -> bytes:
    mono_audio = _as_mono_float_array(audio)
    pcm = array("h")
    for sample in mono_audio:
        clipped = max(-1.0, min(1.0, float(sample)))
        pcm.append(int(clipped * 32767.0))
    return pcm.tobytes()


def pcm16_to_wav_bytes(
    pcm_audio: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_audio)
    return buffer.getvalue()


def float32_to_wav_bytes(audio: Iterable[float], *, sample_rate: int, channels: int = 1) -> bytes:
    return pcm16_to_wav_bytes(
        float32_to_pcm16(audio),
        sample_rate=sample_rate,
        channels=channels,
        sample_width=2,
    )


def pcm16_bytes_from_audio_file(audio_bytes: bytes) -> bytes:
    samples, _sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    return float32_to_pcm16(samples)


def estimate_audio_duration_ms(
    *,
    pcm_bytes: int,
    sample_rate: int,
    channels: int = 1,
    sample_width: int = 2,
) -> float:
    bytes_per_second = sample_rate * channels * sample_width
    if bytes_per_second <= 0:
        return 0.0
    return round((pcm_bytes / bytes_per_second) * 1000.0, 2)


def real_time_factor(*, synthesis_time_ms: float, audio_duration_ms: float) -> float:
    if audio_duration_ms <= 0:
        return 0.0
    return round(synthesis_time_ms / audio_duration_ms, 4)


def silent_audio(duration_ms: int, sample_rate: int = 24000) -> list[float]:
    frames = max(0, math.ceil(sample_rate * (duration_ms / 1000.0)))
    return [0.0] * frames
