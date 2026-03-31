"""Worker process entrypoint for isolated engines."""

from __future__ import annotations

import argparse
import io
import json
import os
from contextlib import redirect_stdout
from importlib import import_module
from pathlib import Path
import sys
import tempfile
from typing import Any

from .base import EngineError, EngineNotLoadedError, EngineSynthesisResult, EngineStatus, TtsEngine


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="isolated engine worker")
    parser.add_argument("--module", required=True)
    parser.add_argument("--class-name", required=True)
    return parser.parse_args(argv)


def load_engine(module_path: str, class_name: str) -> TtsEngine:
    with redirect_stdout(io.StringIO()):
        module = import_module(module_path)
        engine_class = getattr(module, class_name)
        return engine_class()


def write_ok(payload: dict[str, Any]) -> None:
    write_message({"ok": True, **payload})


def write_error(err: Exception) -> None:
    write_message(
        {
            "ok": False,
            "error": str(err),
            "error_type": err.__class__.__name__,
        }
    )


def write_message(payload: dict[str, Any]) -> None:
    body = json.dumps(payload)
    sys.stdout.write(f"__JSON__ {len(body)}\n")
    sys.stdout.flush()
    sys.stdout.write(body)
    sys.stdout.write("\n")
    sys.stdout.flush()


def write_response_file(response_path: str, payload: dict[str, Any]) -> None:
    response_file = Path(response_path)
    response_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(response_file.parent),
        prefix=response_file.name + ".tmp-",
        delete=False,
    ) as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name
    os.replace(temp_name, response_file)


def run_command(engine: TtsEngine, command: str, payload: dict[str, Any]) -> dict[str, Any]:
    with redirect_stdout(io.StringIO()):
        if command == "status":
            return {"status": engine.status().asdict()}
        if command == "load":
            status = engine.load(payload.get("device_preference"))
            return {"status": status.asdict()}
        if command == "unload":
            engine.unload()
            return {"status": engine.status().asdict()}
        if command == "synthesize":
            result = engine.synthesize(
                payload["text"],
                voice=payload.get("voice"),
                language=payload.get("language"),
                options=payload.get("options"),
            )
            wav_file = tempfile.NamedTemporaryFile(
                prefix="wyoming-multi-tts-wav-",
                suffix=".wav",
                delete=False,
            )
            pcm_file = tempfile.NamedTemporaryFile(
                prefix="wyoming-multi-tts-pcm-",
                suffix=".bin",
                delete=False,
            )
            try:
                Path(wav_file.name).write_bytes(result.wav_audio)
                Path(pcm_file.name).write_bytes(result.pcm_audio)
            finally:
                wav_file.close()
                pcm_file.close()
            return {
                "result": result.to_file_transport_dict(
                    wav_path=wav_file.name,
                    pcm_path=pcm_file.name,
                )
            }
    raise EngineError(f"Unsupported command '{command}'")


def main() -> None:
    args = parse_args()
    engine = load_engine(args.module, args.class_name)
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        response_path = message.get("response_path")
        try:
            payload = run_command(
                engine,
                str(message["command"]),
                dict(message.get("payload", {})),
            )
            response_payload = {"ok": True, **payload}
        except (EngineError, EngineNotLoadedError, Exception) as err:
            response_payload = {
                "ok": False,
                "error": str(err),
                "error_type": err.__class__.__name__,
            }
        if response_path:
            write_response_file(str(response_path), response_payload)
        else:
            write_message(response_payload)


if __name__ == "__main__":
    main()
