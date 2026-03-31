#!/usr/bin/env python3
"""Patch upstream fish-speech for torchaudio backend detection on torchaudio 2.8/2.9."""

from __future__ import annotations

from pathlib import Path


TARGET = Path("/opt/fish-speech/fish_speech/inference_engine/reference_loader.py")

OLD_IMPORT = "import io\nimport re\nfrom hashlib import sha256\nfrom pathlib import Path\nfrom typing import Callable, Literal, Tuple\n\nimport torch\nimport torchaudio\nfrom loguru import logger\n"
NEW_IMPORT = "import importlib\nimport io\nimport re\nfrom hashlib import sha256\nfrom pathlib import Path\nfrom typing import Callable, Literal, Tuple\n\nimport torch\nimport torchaudio\nfrom loguru import logger\n"

OLD_BLOCK = """        try:\n            backends = torchaudio.list_audio_backends()\n            if \"ffmpeg\" in backends:\n                self.backend = \"ffmpeg\"\n            else:\n                self.backend = \"soundfile\"\n        except AttributeError:\n            # torchaudio 2.9+ removed list_audio_backends()\n            # Try ffmpeg first, fallback to soundfile\n            try:\n                import torchaudio.io._load_audio_fileobj  # noqa: F401\n\n                self.backend = \"ffmpeg\"\n            except (ImportError, ModuleNotFoundError):\n                self.backend = \"soundfile\"\n"""
NEW_BLOCK = """        try:\n            backends = torchaudio.list_audio_backends()\n            if \"ffmpeg\" in backends:\n                self.backend = \"ffmpeg\"\n            else:\n                self.backend = \"soundfile\"\n        except AttributeError:\n            # torchaudio 2.9+ removed list_audio_backends()\n            # Try ffmpeg first, fallback to soundfile\n            try:\n                importlib.import_module(\"torchaudio.io._load_audio_fileobj\")\n                self.backend = \"ffmpeg\"\n            except (ImportError, ModuleNotFoundError):\n                self.backend = \"soundfile\"\n"""


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")
    if "importlib.import_module(\"torchaudio.io._load_audio_fileobj\")" in source:
        return
    if OLD_IMPORT not in source:
        raise RuntimeError("Unexpected fish-speech reference_loader import block")
    if OLD_BLOCK not in source:
        raise RuntimeError("Unexpected fish-speech torchaudio backend block")
    source = source.replace(OLD_IMPORT, NEW_IMPORT, 1)
    source = source.replace(OLD_BLOCK, NEW_BLOCK, 1)
    TARGET.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
