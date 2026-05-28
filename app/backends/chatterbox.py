"""Chatterbox Turbo backend — English voice cloning with native streaming.

Reuses the English reference clips from the OmniVoice voice catalog as preset
voices (Chatterbox clones from a reference clip; no transcript needed).
"""
from __future__ import annotations

import threading

import numpy as np

from .. import config, voices
from ..voices import Catalog, Voice


def _english_catalog() -> Catalog:
    base = voices.load_omnivoice_catalog()
    eng = [v for v in base.all() if v.language_code == "en"]
    # Re-key ids under a chatterbox- namespace so they don't collide with OmniVoice.
    out = [
        Voice(
            voice_id=f"cb-{v.voice_id}",
            name=v.name,
            language="English",
            language_code="en",
            gender=v.gender,
            age=v.age,
            tags=v.tags,
            audio_path=v.audio_path,
        )
        for v in eng
    ]
    return Catalog(out)


class ChatterboxBackend:
    # We bundle the fp16 weights, which OpenVox calls the "Large" variant.
    id = "chatterbox-turbo-large"
    display_name = "Chatterbox Turbo (Large)"
    model_key = "chatterbox"
    voice_model_label = "chatterbox"
    supports_streaming = True
    sample_rate = config.SAMPLE_RATE

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._model_dir = config.chatterbox_model_path()
        self.catalog = _english_catalog()

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from mlx_audio.tts.utils import load_model

            self._model = load_model(str(self._model_dir))

    def synth(self, text: str, language: str, voice: Voice) -> np.ndarray:
        if self._model is None:
            self.load()
        kwargs = {"text": text}
        if voice.audio_path:
            kwargs["ref_audio"] = voice.audio_path
        result = next(self._model.generate(**kwargs))
        audio = np.asarray(result.audio, dtype=np.float32)
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        return audio
