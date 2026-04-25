"""Session state management for persisting engine selection and options."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class SessionState:
    """Holds the persisted session state."""

    active_engine_id: str | None = None
    autoload_active_engine: bool = False
    active_engine_loaded: bool = False
    last_voice: str | None = None
    last_language: str | None = None
    engine_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    engine_selections: dict[str, tuple[str, str]] = field(default_factory=dict)

    def selection_for(self, engine_id: str) -> tuple[str | None, str | None]:
        """Get the (voice, language) selection for a given engine."""
        return self.engine_selections.get(engine_id, (None, None))

    def set_selection(self, engine_id: str, voice: str, language: str) -> None:
        """Set the (voice, language) selection for a given engine."""
        self.engine_selections[engine_id] = (voice, language)


class SessionStateStore:
    """Persists session state to a JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state: SessionState) -> None:
        """Save session state to disk."""
        data = asdict(state)
        # Convert tuple values to lists for JSON serialization
        data["engine_selections"] = {
            k: list(v) for k, v in data["engine_selections"].items()
        }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, default_engine_id: str | None = None) -> SessionState:
        """Load session state from disk, or return defaults if not found."""
        if not self._path.exists():
            return SessionState(active_engine_id=default_engine_id)

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return SessionState(active_engine_id=default_engine_id)

        # Convert list values back to tuples for engine_selections
        engine_selections = {
            k: tuple(v) for k, v in data.get("engine_selections", {}).items()
        }
        data["engine_selections"] = engine_selections

        # Ensure engine_options exists
        if "engine_options" not in data:
            data["engine_options"] = {}

        return SessionState(**data)
