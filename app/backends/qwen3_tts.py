"""Qwen3-TTS backend — Qwen3-1.7B based TTS with built-in speakers.

Uses Alibaba Qwen's official `qwen-tts` PyTorch package and the
`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` weights. Voice ids are namespaced with
a ``qwen-`` prefix to avoid collisions with other backends that draw from the
same reference-clip pool.
"""
from __future__ import annotations

import threading

import numpy as np

from .. import config, voices
from .._device import pick_device, pick_dtype
from ..voices import Catalog, Voice
from ._common import resample_if_needed, to_mono_float32

# Qwen3-TTS-CustomVoice exposes 10 languages.
SUPPORTED_LANGS = {"de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "zh"}
EXTRA_TAGS = ("Library",)  # OpenVox marks the curated set as "Library"

# Qwen3-TTS CustomVoice ships 9 built-in speakers (Vivian, Serena, Uncle_Fu,
# Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee). At synth time we deterministically
# map (language, gender) -> one that actually exists in the checkpoint, matching
# the prior MLX-era selection so client voice routing keeps the same outcome.
_SPEAKER_BY_LANG_GENDER: dict[tuple[str, str], str] = {
    ("zh", "Male"):   "Uncle_Fu",
    ("ja", "Female"): "Ono_Anna",
    ("ko", "Female"): "Sohee",
    ("en", "Female"): "Serena",
    ("en", "Male"):   "Ryan",
}
_FEMALE_FALLBACK = "Vivian"
_MALE_FALLBACK = "Dylan"

# BCP-47 -> Qwen3-TTS language name. The model accepts capitalised names or
# "Auto" for automatic detection.
_LANG_NAME = {
    "de": "German",   "en": "English",   "es": "Spanish",
    "fr": "French",   "it": "Italian",   "ja": "Japanese",
    "ko": "Korean",   "pt": "Portuguese", "ru": "Russian",
    "zh": "Chinese",
}


def _pick_speaker(language_code: str, gender: str) -> str:
    if (language_code, gender) in _SPEAKER_BY_LANG_GENDER:
        return _SPEAKER_BY_LANG_GENDER[(language_code, gender)]
    return _FEMALE_FALLBACK if gender == "Female" else _MALE_FALLBACK


def _catalog() -> Catalog:
    base = voices.load_voice_catalog()
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
    model_key = "qwen3_tts"
    voice_model_label = "qwen3_tts"
    supports_streaming = True
    sample_rate = config.SAMPLE_RATE

    # ---- Extension metadata
    weights_dir = "qwen3-tts-pt"
    weights_repos: tuple = (
        ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "qwen3-tts-pt", None),
    )
    pip_install: tuple = (
        ("install", "qwen-tts>=0.1.1"),
    )

    def __init__(self) -> None:
        self._model = None
        self._sr: int | None = None
        self._lock = threading.Lock()
        self._model_dir = config.LOCAL_MODELS / self.weights_dir
        self.catalog = _catalog()

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from qwen_tts import Qwen3TTSModel

            device = pick_device()
            dtype = pick_dtype()
            # Prefer the locally downloaded weights when present; fall back to
            # the canonical HF repo id so callers can run without
            # pre-downloading.
            source = (
                str(self._model_dir)
                if self._model_dir.is_dir()
                else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            )
            self._model = Qwen3TTSModel.from_pretrained(
                source,
                device_map=str(device),
                dtype=dtype,
            )

    def synth(self, text: str, language: str, voice: Voice) -> np.ndarray:
        if self._model is None:
            self.load()
        speaker = _pick_speaker(voice.language_code, voice.gender)
        wavs, sr = self._model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=_LANG_NAME.get(language, "Auto"),
        )
        # generate_custom_voice returns (List[np.ndarray], int).
        first = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        audio = to_mono_float32(first)
        return resample_if_needed(audio, int(sr), self.sample_rate)
