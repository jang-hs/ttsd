"""Chatterbox Multilingual backend — 25-language cloning TTS.

Uses the official `chatterbox-tts` package's `ChatterboxMultilingualTTS` class
and weights from the upstream `ResembleAI/chatterbox` repo (the multilingual
variant shares the repo with the English variant; only the t3_mtl* weights
and a different tokenizer are loaded). Reuses the shared reference-clip pool,
filtered to the languages Chatterbox Multilingual actually supports.
"""
from __future__ import annotations

import threading

import numpy as np

from .. import config, voices
from .._device import pick_device
from ..voices import Catalog, Voice
from ._common import resample_if_needed, to_mono_float32

SUPPORTED_LANGS = {
    "ar", "cs", "da", "de", "el", "en", "es", "fi", "fr", "he",
    "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru",
    "sv", "sw", "th", "tr", "zh",
}

_NATIVE_SR = 24000


def _catalog() -> Catalog:
    base = voices.load_voice_catalog()
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
    model_key = "chatterbox_multilingual"
    voice_model_label = "chatterbox_multilingual"
    supports_streaming = True
    sample_rate = config.SAMPLE_RATE

    # ---- Extension metadata. Shares the chatterbox-tts package with the
    # English ChatterboxBackend, so pip_install is empty (pulled in by that
    # backend); the multilingual variant uses a different file set from the
    # same ResembleAI/chatterbox repo.
    weights_dir = "chatterbox-multilingual-pt"
    weights_repos: tuple = (
        ("ResembleAI/chatterbox", "chatterbox-multilingual-pt",
         ("ve.pt", "s3gen.pt",
          "t3_mtl23ls_v2.safetensors", "t3_mtl23ls_v3.safetensors",
          "mtl_tokenizer.json", "grapheme_mtl_merged_expanded_v1.json",
          "conds.pt", "Cangjie5_TC.json")),
    )
    pip_install: tuple = ()

    def __init__(self) -> None:
        self._model = None
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
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            device = pick_device()
            loader = getattr(ChatterboxMultilingualTTS, "from_local", None)
            if loader is not None and self._model_dir.is_dir():
                self._model = loader(str(self._model_dir), device=str(device))
            else:
                self._model = ChatterboxMultilingualTTS.from_pretrained(device=str(device))

    def synth(self, text: str, language: str, voice: Voice) -> np.ndarray:
        if self._model is None:
            self.load()
        kwargs: dict[str, object] = {"language_id": language or "en"}
        if voice.audio_path:
            kwargs["audio_prompt_path"] = voice.audio_path
        wav = self._model.generate(text, **kwargs)
        audio = to_mono_float32(wav)
        src_sr = int(getattr(self._model, "sr", _NATIVE_SR))
        return resample_if_needed(audio, src_sr, self.sample_rate)
