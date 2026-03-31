from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


BASE_URL = "http://127.0.0.1:8280"


@dataclass
class EngineCase:
    engine_id: str
    text: str
    voice: str | None = None
    language: str | None = None
    load_timeout: int = 180
    synth_timeout: int = 180


def http_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: int = 30,
) -> tuple[int, dict[str, Any]]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"detail": body}
        return err.code, parsed


def short_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "status",
        "ready",
        "engine_id",
        "display_name",
        "state",
        "loaded",
        "device",
        "load_time_ms",
        "last_error",
        "backend",
    ]
    compact = {key: payload[key] for key in keys if key in payload}
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        compact["metrics"] = {
            "synthesis_time_ms": metrics.get("synthesis_time_ms"),
            "end_to_end_time_ms": metrics.get("end_to_end_time_ms"),
            "audio_duration_ms": metrics.get("audio_duration_ms"),
            "real_time_factor": metrics.get("real_time_factor"),
        }
    if "available_voices" in payload:
        voices = payload["available_voices"]
        compact["available_voice_count"] = len(voices) if isinstance(voices, list) else None
    if "detail" in payload:
        compact["detail"] = payload["detail"]
    return compact


def print_step(title: str, status: int, payload: dict[str, Any]) -> None:
    print(f"{title}: HTTP {status}")
    print(json.dumps(short_payload(payload), ensure_ascii=False, indent=2))


def run_case(case: EngineCase) -> bool:
    ok = True
    print(f"\n=== {case.engine_id} ===")

    status, payload = http_json("POST", "/api/engines/select", {"engine_id": case.engine_id}, timeout=30)
    print_step("select", status, payload)
    ok &= status == 200

    status, payload = http_json("POST", "/api/engines/load", {}, timeout=case.load_timeout)
    print_step("load", status, payload)
    ok &= status == 200 and payload.get("state") == "ready"

    status, payload = http_json("GET", "/api/voices", timeout=30)
    print_step("voices", status, payload)
    ok &= status == 200

    if ok:
        voices = payload.get("voices") or []
        if not case.voice and voices:
            case.voice = voices[0].get("id")
        if not case.language and voices:
            languages = voices[0].get("languages") or []
            case.language = languages[0] if languages else None

        synth_payload = {
            "text": case.text,
            "voice": case.voice,
            "language": case.language,
        }
        status, payload = http_json("POST", "/api/synthesize", synth_payload, timeout=case.synth_timeout)
        print_step("synthesize", status, payload)
        ok &= status == 200

    status, payload = http_json("POST", "/api/engines/unload", {}, timeout=30)
    print_step("unload", status, payload)
    ok &= status == 200
    return ok


def main() -> int:
    cases = [
        EngineCase("chatterbox", "To jest sekwencyjny smoke test silnika Chatterbox.", voice="default__pl", language="pl"),
        EngineCase("whisperspeech", "To jest sekwencyjny smoke test silnika WhisperSpeech.", language="pl"),
        EngineCase("xtts_v2", "To jest sekwencyjny smoke test silnika XTTS v2.", language="pl"),
        EngineCase("qwen_tts_polish", "This is a sequential smoke test for the Qwen3-TTS engine.", voice="Ryan", language="en", load_timeout=240, synth_timeout=240),
        EngineCase("fish_s2_pro", "To jest sekwencyjny smoke test silnika Fish S2 Pro.", language="pl", load_timeout=240, synth_timeout=240),
    ]

    status, payload = http_json("GET", "/health", timeout=30)
    print_step("health", status, payload)

    overall_ok = status == 200
    for case in cases:
        try:
            overall_ok &= run_case(case)
        except Exception as err:
            overall_ok = False
            print(f"{case.engine_id}: exception {err}")
        time.sleep(1)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
