"""Model registry + single-job concurrency guard.

Only models whose weights are present in ./models are registered.
Run scripts/download_model.py to fetch them.

One generation/preload runs at a time; concurrent requests get HTTP 429.
"""
from __future__ import annotations

import asyncio
import logging
import os

from . import config
from .backends import ALL_BACKENDS

logger = logging.getLogger("local-tts.registry")

_BACKENDS: dict = {}


def _try_register(backend_cls, available: bool) -> None:
    if not available:
        return
    try:
        backend = backend_cls()
        _BACKENDS[backend.id] = backend
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping backend %s: %s", backend_cls.__name__, exc)


def _allowed_ids() -> set[str] | None:
    """Optional runtime filter: TTSD_ONLY=qwen3-tts-medium,kokoro restricts
    which backends register even when weights are present. Returns None for
    no filter (the default)."""
    raw = os.environ.get("TTSD_ONLY", "").strip()
    if not raw:
        return None
    return {tok.strip() for tok in raw.split(",") if tok.strip()}


def _init() -> None:
    allow = _allowed_ids()
    for cls in ALL_BACKENDS:
        if allow is not None and cls.id not in allow:
            continue
        _try_register(cls, config.has_local_model(cls.weights_dir))


_init()

# Only one generation or preload job may run at a time.
JOB_LOCK = asyncio.Lock()


def get(model_id: str):
    return _BACKENDS.get(model_id)


def all_backends():
    return list(_BACKENDS.values())
