"""TTS backend registry.

To add a new backend:
  1. Drop a new file `app/backends/<name>.py` with a class implementing the
     `TTSBackend` Protocol from `base.py` and the extension metadata used by
     the CLI / downloader (`weights_dir`, `weights_repos`, `pip_install`).
  2. Import it here and append to `ALL_BACKENDS`.

That's it — `app.registry`, `app.cli`, and `scripts.download_model` discover
it from this list automatically.
"""
from __future__ import annotations

from .base import TTSBackend
from .chatterbox import ChatterboxBackend
from .chatterbox_multilingual import ChatterboxMultilingualBackend
from .kokoro import KokoroBackend
from .qwen3_tts import Qwen3TTSBackend

ALL_BACKENDS: tuple[type[TTSBackend], ...] = (
    KokoroBackend,
    ChatterboxBackend,
    ChatterboxMultilingualBackend,
    Qwen3TTSBackend,
)

# id -> backend class
BACKENDS_BY_ID: dict[str, type[TTSBackend]] = {b.id: b for b in ALL_BACKENDS}

__all__ = [
    "ALL_BACKENDS",
    "BACKENDS_BY_ID",
    "TTSBackend",
    "ChatterboxBackend",
    "ChatterboxMultilingualBackend",
    "KokoroBackend",
    "Qwen3TTSBackend",
]
