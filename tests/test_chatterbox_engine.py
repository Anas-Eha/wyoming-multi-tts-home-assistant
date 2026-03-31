from __future__ import annotations

from pathlib import Path

import pytest

from app.engines.base import EngineError
from app.engines.chatterbox_engine import CHATTERBOX_REQUIRED_FILES, ChatterboxEngine


def touch_required_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename in CHATTERBOX_REQUIRED_FILES:
        (directory / filename).write_bytes(b"ok")


def test_chatterbox_checkpoint_validation_redownloads_when_local_snapshot_is_stale(monkeypatch, tmp_path):
    engine = ChatterboxEngine()
    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()
    (stale_dir / "ve.pt").write_bytes(b"old")

    fresh_dir = tmp_path / "fresh"
    touch_required_files(fresh_dir)

    calls = {"downloads": 0}

    monkeypatch.setattr(
        "app.engines.chatterbox_engine.snapshot_download_local_first",
        lambda *args, **kwargs: str(stale_dir),
    )

    def fake_download(self):
        calls["downloads"] += 1
        return str(fresh_dir)

    monkeypatch.setattr(ChatterboxEngine, "_download_checkpoint", fake_download)

    resolved = engine._resolve_checkpoint_dir()

    assert resolved == str(fresh_dir)
    assert calls["downloads"] == 1


def test_chatterbox_checkpoint_validation_fails_when_snapshot_is_still_incomplete(monkeypatch, tmp_path):
    engine = ChatterboxEngine()
    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()

    monkeypatch.setattr(
        "app.engines.chatterbox_engine.snapshot_download_local_first",
        lambda *args, **kwargs: str(stale_dir),
    )
    monkeypatch.setattr(ChatterboxEngine, "_download_checkpoint", lambda self: str(stale_dir))

    with pytest.raises(EngineError):
        engine._resolve_checkpoint_dir()
