"""Subprocess-backed engine wrapper for conflicting runtimes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from .base import EngineError, EngineNotLoadedError, EngineSynthesisResult, EngineStatus, EngineVoice, TtsEngine


class IsolatedEngine(TtsEngine):
    def __init__(
        self,
        *,
        engine_id: str,
        display_name: str,
        module_path: str,
        class_name: str,
        python_env_var: str,
        default_python: str,
        fallback_languages: list[str],
    ) -> None:
        self._engine_id = engine_id
        self._display_name = display_name
        self._module_path = module_path
        self._class_name = class_name
        self._python_env_var = python_env_var
        self._default_python = default_python
        self._fallback_languages = list(fallback_languages)
        self._rpc_timeout_seconds = float(
            os.getenv(f"{engine_id.upper()}_RPC_TIMEOUT_SECONDS", "300")
        )
        self._process: subprocess.Popen[str] | None = None
        self._log_path = Path(f"/tmp/{engine_id}-worker.log")
        self._cached_status = EngineStatus(
            engine_id=engine_id,
            display_name=display_name,
            available_voices=[],
            extra={
                "python_env_var": python_env_var,
                "rpc_timeout_seconds": self._rpc_timeout_seconds,
            },
        )

    def engine_id(self) -> str:
        return self._engine_id

    def display_name(self) -> str:
        return self._display_name

    def supported_languages(self) -> list[str]:
        return self._fallback_languages

    def list_voices(self) -> list[EngineVoice]:
        return list(self._cached_status.available_voices)

    def load(self, device_preference: str | None = None) -> EngineStatus:
        payload = self._rpc("load", {"device_preference": device_preference})
        self._cached_status = EngineStatus.fromdict(payload["status"])
        return self._cached_status

    def unload(self) -> None:
        if self._process is None:
            payload = self._cached_status.asdict()
            payload.update(
                {
                    "state": "not_loaded",
                    "loaded": False,
                    "loading": False,
                }
            )
            self._cached_status = EngineStatus.fromdict(payload)
            return
        try:
            payload = self._rpc("unload", {})
            self._cached_status = EngineStatus.fromdict(payload["status"])
        finally:
            self._stop_process()

    def is_loaded(self) -> bool:
        return self.status().loaded

    def status(self) -> EngineStatus:
        if self._process is None:
            return self._cached_status
        payload = self._rpc("status", {})
        self._cached_status = EngineStatus.fromdict(payload["status"])
        return self._cached_status

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None,
        language: str | None,
        options: dict[str, Any] | None = None,
    ) -> EngineSynthesisResult:
        payload = self._rpc(
            "synthesize",
            {
                "text": text,
                "voice": voice,
                "language": language,
                "options": options or {},
            },
        )
        result_payload = dict(payload["result"])
        try:
            return EngineSynthesisResult.from_transport_dict(result_payload)
        finally:
            self._cleanup_transport_files(result_payload)

    def health_payload(self) -> dict[str, Any]:
        return self.status().asdict()

    def _rpc(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        process = self._ensure_process()
        response_path = self._response_file_path()
        message = {
            "command": command,
            "payload": payload,
            "response_path": str(response_path),
        }
        assert process.stdin is not None
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()
        response = self._read_file_response(process, response_path)
        if response.get("ok"):
            return response
        error_type = response.get("error_type", "EngineError")
        error_text = str(response.get("error", "Unknown worker error"))
        if error_type == "EngineNotLoadedError":
            raise EngineNotLoadedError(error_text)
        raise EngineError(error_text)

    def _read_file_response(
        self,
        process: subprocess.Popen[str],
        response_path: Path,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self._rpc_timeout_seconds
        poll_interval = 0.05
        
        while True:
            if response_path.exists():
                try:
                    content = response_path.read_text(encoding="utf-8")
                    if content:  # Ensure file is not empty (partial write)
                        payload = json.loads(content)
                        response_path.unlink(missing_ok=True)
                        return payload
                except json.JSONDecodeError:
                    # File exists but not valid JSON yet, wait for write to complete
                    pass
                except OSError:
                    pass
                    
            remaining = max(0.0, deadline - time.monotonic())
            if remaining == 0.0:
                self._stop_process()
                raise EngineError(
                    f"{self._display_name} worker timed out after {self._rpc_timeout_seconds:.0f}s"
                )
                
            if process.poll() is not None:
                stderr = self._read_stderr()
                self._stop_process()
                raise EngineError(
                    f"{self._display_name} worker terminated unexpectedly: {stderr}"
                )
                
            time.sleep(min(poll_interval, remaining))

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        python_path = self._resolve_python()
        env = os.environ.copy()
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # Rotate log if it gets too large (>10MB)
        if self._log_path.exists() and self._log_path.stat().st_size > 10 * 1024 * 1024:
            backup = self._log_path.with_suffix(".log.old")
            if backup.exists():
                backup.unlink()
            self._log_path.rename(backup)
        log_handle = self._log_path.open("a+", encoding="utf-8")
        self._process = subprocess.Popen(
            [
                python_path,
                "-m",
                "app.engines.worker",
                "--module",
                self._module_path,
                "--class-name",
                self._class_name,
            ],
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            stdin=subprocess.PIPE,
            stdout=log_handle,
            stderr=log_handle,
            text=True,
            bufsize=1,
        )
        return self._process

    def _resolve_python(self) -> str:
        configured = os.getenv(self._python_env_var)
        if configured:
            return configured
        candidate = Path(self._default_python)
        if candidate.exists():
            return str(candidate)
        return sys.executable

    def _read_stderr(self) -> str:
        if not self._log_path.exists():
            return ""
        try:
            lines = self._log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return ""
        return " | ".join(lines[-20:])

    def _stop_process(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        self._process = None

    def _response_file_path(self) -> Path:
        temp_dir = Path(tempfile.gettempdir()) / f"{self._engine_id}-responses"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir / f"{time.time_ns()}.json"

    def _cleanup_transport_files(self, payload: dict[str, Any]) -> None:
        for key in ("wav_path", "pcm_path"):
            file_path = payload.get(key)
            if not file_path:
                continue
            try:
                Path(str(file_path)).unlink(missing_ok=True)
            except OSError:
                continue
