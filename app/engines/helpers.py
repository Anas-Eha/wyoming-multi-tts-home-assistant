"""Shared helpers for optional TTS engines."""

from __future__ import annotations

import gc
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

try:
    import torch
except ImportError:  # pragma: no cover
    class _CudaShim:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            return None

    class _TorchShim:
        cuda = _CudaShim()

    torch = _TorchShim()  # type: ignore[assignment]

from app.audio.audio_utils import estimate_audio_duration_ms, real_time_factor

from .base import EngineStatus, EngineVoice, SynthesisMetrics


def preferred_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if not torch.cuda.is_available():
        return "cpu"
    return best_cuda_device() or "cuda"


def best_cuda_device() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "cuda"

    devices: list[tuple[int, int]] = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            devices.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue

    if not devices:
        return "cuda"
    best_index = min(devices, key=lambda item: item[1])[0]
    return f"cuda:{best_index}"


def with_cpu_fallback(loader) -> tuple[Any, str, float]:
    last_error: Exception | None = None
    primary_device = preferred_device()
    devices = [primary_device] if primary_device == "cpu" else [primary_device, "cpu"]
    for device in devices:
        started = time.perf_counter()
        try:
            model = loader(device)
            load_time_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return model, device, load_time_ms
        except Exception as err:  # pragma: no cover - runtime dependent
            last_error = err
            if device == "cpu":
                break
    assert last_error is not None
    raise last_error


def cleanup_torch() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    ipc_collect = getattr(torch.cuda, "ipc_collect", None)
    if callable(ipc_collect):
        ipc_collect()


def status_from_values(
    *,
    engine_id: str,
    display_name: str,
    state: str,
    loaded: bool,
    device: str | None,
    load_time_ms: float | None,
    last_loaded_at: float | None,
    last_error: str | None,
    supports_streaming: bool,
    voices: list[EngineVoice],
    extra: dict[str, Any] | None = None,
) -> EngineStatus:
    return EngineStatus(
        engine_id=engine_id,
        display_name=display_name,
        state=state,
        loaded=loaded,
        loading=(state == "loading"),
        device=device,
        load_time_ms=load_time_ms,
        last_loaded_at=last_loaded_at,
        last_error=last_error,
        supports_streaming=supports_streaming,
        available_voices=voices,
        extra=extra or {},
    )


def synthesis_metrics(
    *,
    load_time_ms: float | None,
    synthesis_time_ms: float,
    end_to_end_time_ms: float,
    pcm_bytes: int,
    sample_rate: int,
    channels: int = 1,
    sample_width: int = 2,
    cold_start: bool = False,
    time_to_first_chunk_ms: float | None = None,
) -> SynthesisMetrics:
    audio_duration_ms = estimate_audio_duration_ms(
        pcm_bytes=pcm_bytes,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )
    return SynthesisMetrics(
        load_time_ms=load_time_ms,
        synthesis_time_ms=synthesis_time_ms,
        end_to_end_time_ms=end_to_end_time_ms,
        audio_duration_ms=audio_duration_ms,
        real_time_factor=real_time_factor(
            synthesis_time_ms=synthesis_time_ms,
            audio_duration_ms=audio_duration_ms,
        ),
        cold_start=cold_start,
        time_to_first_chunk_ms=time_to_first_chunk_ms,
    )


def env_path(name: str, default: str) -> str:
    return str(Path(os.getenv(name, default)))


def snapshot_download_local_first(
    repo_id: str,
    *,
    allow_patterns: list[str] | None = None,
    token: str | None = None,
) -> str:
    local_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "model",
        "local_files_only": True,
    }
    remote_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "model",
    }
    if allow_patterns:
        local_kwargs["allow_patterns"] = allow_patterns
        remote_kwargs["allow_patterns"] = allow_patterns
    if token:
        local_kwargs["token"] = token
        remote_kwargs["token"] = token

    try:
        return snapshot_download(**local_kwargs)
    except Exception:
        return snapshot_download(**remote_kwargs)
