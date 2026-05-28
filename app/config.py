"""Paths and constants for the local TTS server."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOICES_MANIFEST = PROJECT_ROOT / "voices.manifest.jsonl"

# Model weights live here after running scripts/download_model.py.
# Each backend declares its own subdirectory via the `weights_dir` class
# attribute (see app/backends/*.py).
LOCAL_MODELS = PROJECT_ROOT / "models"


def has_local_model(dir_name: str) -> bool:
    """A backend is considered installed if its directory holds any
    .safetensors or .pt weight file."""
    d = LOCAL_MODELS / dir_name
    if not d.is_dir():
        return False
    for ext in ("*.safetensors", "*.pt"):
        if any(d.rglob(ext)):
            return True
    return False


# All backends emit 24 kHz mono PCM.
SAMPLE_RATE = 24000

# Static creation timestamp reported in /v1/models (cosmetic, matches OpenVox shape)
MODEL_CREATED_TS = 1779859956
