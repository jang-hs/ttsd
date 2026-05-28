"""Backend adapter interface. Each TTS model implements this."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..voices import Voice


class TTSBackend(Protocol):
    id: str
    display_name: str
    model_key: str
    supports_streaming: bool
    sample_rate: int

    def load(self) -> None:
        """Warm the model so the first request is fast."""

    def is_loaded(self) -> bool: ...

    def synth(self, text: str, language: str, voice: Voice) -> np.ndarray:
        """Return a mono float32 waveform at self.sample_rate."""
