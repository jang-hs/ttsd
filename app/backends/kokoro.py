"""Kokoro backend — fast, lightweight, fixed-voice TTS (no cloning).

Uses the upstream `kokoro` PyTorch package (https://github.com/hexgrad/kokoro)
and the official `hexgrad/Kokoro-82M` weights. Voices ship as ``.pt`` files in
``<model_dir>/voices/`` and are named like ``af_bella``: the first letter is
the language/accent, the second the gender.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .. import config
from .._device import pick_device
from ..voices import Catalog, Voice
from ._common import resample_if_needed, to_mono_float32

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

# Kokoro samples at 24 kHz (matches config.SAMPLE_RATE).
_NATIVE_SR = 24000


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
                audio_path=str(pt),
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

    # ---- Extension metadata (discovered by app/cli.py + download_model.py)
    weights_dir = "kokoro-82m-pt"
    weights_repos: tuple = (
        ("hexgrad/Kokoro-82M", "kokoro-82m-pt", None),
    )
    pip_install: tuple = (
        ("install", "kokoro>=0.9.2"),
    )

    def __init__(self) -> None:
        self._pipelines: dict[str, object] = {}    # one KPipeline per lang token
        self._lock = threading.Lock()
        self._model_dir = config.LOCAL_MODELS / self.weights_dir
        self.catalog = _build_catalog(self._model_dir)

    def is_loaded(self) -> bool:
        return bool(self._pipelines)

    def _pipeline_for(self, lang_tok: str):
        if lang_tok in self._pipelines:
            return self._pipelines[lang_tok]
        with self._lock:
            if lang_tok in self._pipelines:
                return self._pipelines[lang_tok]
            from kokoro import KPipeline

            device = pick_device()
            pipe = KPipeline(lang_code=lang_tok, device=str(device))
            self._pipelines[lang_tok] = pipe
            return pipe

    def load(self) -> None:
        # Warm the default (American English) pipeline; per-language pipelines
        # are loaded lazily on first synth call for that language.
        self._pipeline_for("a")

    def synth(self, text: str, language: str, voice: Voice) -> np.ndarray:
        lang_tok = voice.lang_code or "a"
        pipe = self._pipeline_for(lang_tok)
        # If the catalog cached a local path to the voice .pt, pass it through
        # so KPipeline doesn't try to re-download it from HF.
        voice_ref = voice.audio_path or voice.voice_id
        result = next(pipe(text, voice=voice_ref))
        audio = to_mono_float32(result.audio)
        return resample_if_needed(audio, _NATIVE_SR, self.sample_rate)
