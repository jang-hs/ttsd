"""Chatterbox Multilingual backend — 25-language cloning TTS.

Uses the mlx-audio `chatterbox` module (the multilingual variant, distinct from
`chatterbox_turbo`). Reuses the OmniVoice reference clips, filtered to the
25 languages Chatterbox Multilingual actually supports.
"""
from __future__ import annotations

import threading

import numpy as np

from .. import config, voices
from ..voices import Catalog, Voice

SUPPORTED_LANGS = {
    "ar", "cs", "da", "de", "el", "en", "es", "fi", "fr", "he",
    "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru",
    "sv", "sw", "th", "tr", "zh",
}


def _catalog() -> Catalog:
    base = voices.load_omnivoice_catalog()
    out: list[Voice] = []
    for v in base.all():
        if v.language_code not in SUPPORTED_LANGS:
            continue
        out.append(Voice(
            voice_id=v.voice_id,
            name=v.name,
            language=v.language,
            language_code=v.language_code,
            gender=v.gender,
            age=v.age,
            tags=v.tags,
            audio_path=v.audio_path,
            transcript=v.transcript,
        ))
    return Catalog(out)


class ChatterboxMultilingualBackend:
    id = "chatterbox-multilingual-medium"
    display_name = "Chatterbox Multilingual (Medium)"
    model_key = "chatterbox_multilingual_q8"
    voice_model_label = "chatterbox_multilingual_q8"
    supports_streaming = True
    sample_rate = config.SAMPLE_RATE

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._model_dir = config.chatterbox_multilingual_model_path()
        self.catalog = _catalog()

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
        result = next(self._model.generate(
            text=text,
            ref_audio=voice.audio_path,
            lang_code=language or "en",
        ))
        audio = np.asarray(result.audio, dtype=np.float32)
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        return audio
