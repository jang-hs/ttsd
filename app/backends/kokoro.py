"""Kokoro backend — fast, lightweight, fixed-voice TTS (no cloning).

Voices ship as files in the model's voices/ dir, named like ``af_bella``:
the first letter is the language/accent, the second the gender.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .. import config
from ..voices import Catalog, Voice

# Kokoro voice-prefix -> (pipeline lang token, api language code, display name)
_LANG = {
    "a": ("a", "en", "English"),          # American English
    "b": ("b", "en-gb", "English (British)"),
    "e": ("e", "es", "Spanish"),
    "f": ("f", "fr", "French"),
    "h": ("h", "hi", "Hindi"),
    "i": ("i", "it", "Italian"),
    "j": ("j", "ja", "Japanese"),
    "p": ("p", "pt", "Portuguese"),
    "z": ("z", "zh", "Chinese"),
}
_GENDER = {"f": "Female", "m": "Male"}


def _build_catalog(model_dir: Path) -> Catalog:
    voices: list[Voice] = []
    vdir = model_dir / "voices"
    for pt in sorted(vdir.glob("*.pt")):
        vid = pt.stem                       # e.g. "af_bella"
        prefix, _, rest = vid.partition("_")
        lang_letter = prefix[0]
        gender_letter = prefix[1] if len(prefix) > 1 else "f"
        lang_tok, code, lang_name = _LANG.get(lang_letter, ("a", "en", "English"))
        voices.append(
            Voice(
                voice_id=vid,
                name=rest.replace("_", " ").title() or vid,
                language=lang_name,
                language_code=code,
                gender=_GENDER.get(gender_letter, "Unknown"),
                lang_code=lang_tok,
            )
        )
    return Catalog(voices)


class KokoroBackend:
    id = "kokoro"
    display_name = "Kokoro"
    model_key = "kokoro"
    voice_model_label = "kokoro"
    supports_streaming = True
    sample_rate = config.SAMPLE_RATE

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._model_dir = config.kokoro_model_path()
        self.catalog = _build_catalog(self._model_dir)

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
        lang_tok = voice.lang_code or "a"
        result = next(
            self._model.generate(text=text, voice=voice.voice_id, lang_code=lang_tok)
        )
        audio = np.asarray(result.audio, dtype=np.float32)
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        return audio
