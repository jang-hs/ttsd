"""Model registry + single-job concurrency guard.

Only models whose weights are present in ./models are registered.
Run scripts/download_model.py to fetch them.

One generation/preload runs at a time; concurrent requests get HTTP 429.
"""
from __future__ import annotations

import asyncio
import logging

from . import config

logger = logging.getLogger("local-tts.registry")

_BACKENDS: dict = {}


def _try_register(make, available: bool) -> None:
    if not available:
        return
    try:
        backend = make()
        _BACKENDS[backend.id] = backend
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping backend (%s): %s", make, exc)


def _init() -> None:
    from .backends.omnivoice import OmniVoiceBackend
    from .backends.kokoro import KokoroBackend
    from .backends.chatterbox import ChatterboxBackend
    from .backends.qwen3_tts import Qwen3TTSBackend
    from .backends.chatterbox_multilingual import ChatterboxMultilingualBackend

    _try_register(OmniVoiceBackend, config.has_local_model("OmniVoice-bf16"))
    _try_register(KokoroBackend, config.has_local_model("Kokoro-82M-bf16"))
    _try_register(ChatterboxBackend, config.has_local_model("chatterbox-turbo-fp16"))
    _try_register(Qwen3TTSBackend, config.has_local_model("Qwen3-TTS-CustomVoice-8bit"))
    _try_register(ChatterboxMultilingualBackend, config.has_local_model("Chatterbox-Multilingual-Q8"))


_init()

# Only one generation or preload job may run at a time.
JOB_LOCK = asyncio.Lock()


def get(model_id: str):
    return _BACKENDS.get(model_id)


def all_backends():
    return list(_BACKENDS.values())
