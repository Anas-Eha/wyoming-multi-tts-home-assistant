from __future__ import annotations

import sys

import pytest

from app.engines.base import EngineNotLoadedError
from app.engines.isolated_engine import IsolatedEngine


def build_fake_isolated_engine() -> IsolatedEngine:
    return IsolatedEngine(
        engine_id="fake_isolated",
        display_name="Fake Isolated",
        module_path="tests.fake_isolated_backend",
        class_name="FakeIsolatedEngine",
        python_env_var="TEST_ISOLATED_PYTHON",
        default_python=sys.executable,
        fallback_languages=["pl"],
    )


def test_isolated_engine_lifecycle(monkeypatch):
    monkeypatch.setenv("TEST_ISOLATED_PYTHON", sys.executable)
    engine = build_fake_isolated_engine()

    with pytest.raises(EngineNotLoadedError):
        engine.synthesize("hej", voice="default", language="pl")

    status = engine.load()
    assert status.loaded is True
    assert status.state == "ready"

    result = engine.synthesize("hej", voice="default", language="pl")
    assert result.engine_id == "fake_isolated"
    assert result.backend == "fake-isolated"
    assert result.voice == "default"

    engine.unload()
    assert engine.status().loaded is False
    payload = engine.status().asdict()
    assert payload["available_voices"][0]["id"] == "default"
