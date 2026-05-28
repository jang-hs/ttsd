"""Audio encoding helper (float32 -> WAV bytes)."""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf


def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
