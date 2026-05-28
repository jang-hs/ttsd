"""Qwen3-TTS backend — Qwen3-1.7B based TTS with voice cloning.

Mirrors OpenVox's `qwen3-tts-medium` model (8-bit). Voice ids are namespaced
with a ``qwen-`` prefix to avoid collisions with the OmniVoice catalog (the
two backends draw from the same set of reference clips).
"""
from __future__ import annotations

import threading

import numpy as np

from .. import config, voices
from ..voices import Catalog, Voice

# Qwen3-TTS-Medium exposes 10 languages in OpenVox.
SUPPORTED_LANGS = {"de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "zh"}
EXTRA_TAGS = ("Library",)  # OpenVox marks the curated set as "Library"

# Qwen3-TTS CustomVoice (8-bit) ships 9 built-in speakers. The full OpenVox-
# style voice catalog (one entry per reference clip) is exposed for parity, but
# at synth time we deterministically map (language, gender) -> a built-in
# speaker that actually exists in the checkpoint.
_SPEAKER_BY_LANG_GENDER: dict[tuple[str, str], str] = {
    ("zh", "Male"):   "uncle_fu",
    ("ja", "Female"): "ono_anna",
    ("ko", "Female"): "sohee",
    ("en", "Female"): "serena",
    ("en", "Male"):   "ryan",
}
_FEMALE_FALLBACK = "vivian"
_MALE_FALLBACK = "dylan"

# BCP-47 -> Qwen3-TTS language name (the model accepts long names or "auto").
_LANG_NAME = {
    "de": "german",   "en": "english",  "es": "spanish",
    "fr": "french",   "it": "italian",  "ja": "japanese",
    "ko": "korean",   "pt": "portuguese", "ru": "russian",
    "zh": "chinese",
}


def _pick_speaker(language_code: str, gender: str) -> str:
    if (language_code, gender) in _SPEAKER_BY_LANG_GENDER:
        return _SPEAKER_BY_LANG_GENDER[(language_code, gender)]
    return _FEMALE_FALLBACK if gender == "Female" else _MALE_FALLBACK


def _catalog() -> Catalog:
    base = voices.load_omnivoice_catalog()
    out: list[Voice] = []
    for v in base.all():
        if v.language_code not in SUPPORTED_LANGS:
            continue
        out.append(Voice(
            voice_id=f"qwen-{v.voice_id}",
            name=v.name,
            language=v.language,
            language_code=v.language_code,
            gender=v.gender,
            age=v.age,
            tags=tuple(v.tags) + EXTRA_TAGS,
            audio_path=v.audio_path,
            transcript=v.transcript,
        ))
    return Catalog(out)


class Qwen3TTSBackend:
    id = "qwen3-tts-medium"
    display_name = "Qwen3 TTS (Medium)"
    model_key = "qwen3_tts_8bit"
    voice_model_label = "qwen3_tts"
    supports_streaming = True
    sample_rate = config.SAMPLE_RATE

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._model_dir = config.qwen3_tts_model_path()
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
        speaker = _pick_speaker(voice.language_code, voice.gender)
        result = next(self._model.generate(
            text=text,
            voice=speaker,
            lang_code=_LANG_NAME.get(language, "auto"),
        ))
        audio = np.asarray(result.audio, dtype=np.float32)
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        return audio
