from __future__ import annotations

from app.audio.audio_utils import pcm16_to_wav_bytes
from app.engines.fish_engine import FishS2ProEngine


class _RunningProcess:
    def poll(self):
        return None


def test_fish_engine_lists_default_and_reference_voices():
    engine = FishS2ProEngine()
    engine._reference_ids = ["speaker-a", "speaker-b"]

    voices = engine.list_voices()

    assert voices[0].id == "default"
    assert [voice.id for voice in voices[1:]] == ["speaker-a", "speaker-b"]


def test_fish_engine_synthesize_uses_reference_id(monkeypatch):
    engine = FishS2ProEngine()
    engine._server_process = _RunningProcess()
    engine._server_url = "http://127.0.0.1:9999"
    engine._device = "cuda:0"
    engine._sample_rate = 24000

    captured: dict[str, object] = {}

    def fake_request_bytes(path, payload, *, timeout, accept_json=False):
        captured["path"] = path
        captured["payload"] = payload
        captured["timeout"] = timeout
        captured["accept_json"] = accept_json
        return pcm16_to_wav_bytes(b"\x00\x00" * 240, sample_rate=24000)

    monkeypatch.setattr(engine, "_request_bytes", fake_request_bytes)

    result = engine.synthesize(
        "Test Fish",
        voice="speaker-a",
        language="pl",
        options={"latency": "balanced", "chunk_length": 220},
    )

    assert captured["path"] == "/v1/tts"
    payload = captured["payload"]
    assert payload["reference_id"] == "speaker-a"
    assert payload["chunk_length"] == 220
    assert result.backend == "fish-speech-api"
    assert result.device == "cuda:0"
