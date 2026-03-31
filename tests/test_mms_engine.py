from __future__ import annotations

from app.engines.mms_engine import MmsPolishEngine


class FakeInputs(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeTokenizer:
    def __call__(self, text: str, return_tensors: str):
        assert return_tensors == "pt"
        return FakeInputs({"text": text})


class FakeWaveform:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self._values


class FakeOutput:
    def __init__(self, values):
        self.waveform = FakeWaveform(values)


class FakeModel:
    def __call__(self, **inputs):
        assert "text" in inputs
        return FakeOutput([0.0, 0.1, -0.1, 0.0])


def test_mms_engine_lists_polish_only_voice():
    engine = MmsPolishEngine()

    voices = engine.list_voices()

    assert len(voices) == 1
    assert voices[0].id == "default"
    assert voices[0].languages == ["pl"]


def test_mms_engine_synthesize_returns_polish_audio():
    engine = MmsPolishEngine()
    engine._model = FakeModel()
    engine._tokenizer = FakeTokenizer()
    engine._device = "cuda:0"
    engine._sample_rate = 16000
    engine._state = "ready"

    result = engine.synthesize("To jest test", voice="default", language="pl")

    assert result.engine_id == "mms_tts_pol"
    assert result.voice == "default"
    assert result.language == "pl"
    assert result.backend == "transformers-vits"
    assert result.device == "cuda:0"
    assert len(result.wav_audio) > 0
