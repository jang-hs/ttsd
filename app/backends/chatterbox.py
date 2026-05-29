"""Chatterbox backend — English voice cloning via Resemble AI's PyTorch model.

Uses the official `chatterbox-tts` package and the `ResembleAI/chatterbox`
weights. Reuses the English reference clips from the shared voice catalog as
preset voices (Chatterbox clones from a reference clip; no transcript
needed).
"""
from __future__ import annotations

import threading

import numpy as np

from .. import config, voices
from .._device import pick_device
from ..voices import Catalog, Voice
from ._common import resample_if_needed, to_mono_float32

# ResembleAI's S3GEN decoder is hard-coded to 24 kHz; matches config.SAMPLE_RATE.
_NATIVE_SR = 24000


def _english_catalog() -> Catalog:
    base = voices.load_voice_catalog()
    eng = [v for v in base.all() if v.language_code == "en"]
    # Re-key ids under a chatterbox- namespace so they don't collide with
    # other backends that draw from the same reference-clip pool.
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
    # We bundle the upstream weights, which OpenVox calls the "Large" variant.
    id = "chatterbox-turbo-large"
    display_name = "Chatterbox Turbo (Large)"
    model_key = "chatterbox"
    voice_model_label = "chatterbox"
    supports_streaming = True
    sample_rate = config.SAMPLE_RATE

    # ---- Extension metadata
    weights_dir = "chatterbox-pt"
    # The current chatterbox-tts package loads via safetensors files (ve.safetensors,
    # s3gen.safetensors, t3_cfg.safetensors). The HF repo also ships .pt variants, but
    # ChatterboxTTS.from_local() hard-codes the safetensors filenames so we pull those.
    weights_repos: tuple = (
        ("ResembleAI/chatterbox", "chatterbox-pt",
         ("ve.safetensors", "s3gen.safetensors", "t3_cfg.safetensors",
          "tokenizer.json", "conds.pt")),
    )
    # chatterbox-tts pins transformers to a different version than qwen-tts;
    # install with --no-deps and bring the runtime deps in by hand.
    pip_install: tuple = (
        ("install", "librosa>=0.10"),
        ("install", "diffusers>=0.29"),
        ("install", "safetensors"),
        ("install", "omegaconf"),
        ("install", "conformer>=0.3.2"),
        ("install", "resemble-perth"),
        ("install", "s3tokenizer"),
        ("install", "pyloudnorm"),
        ("install", "spacy-pkuseg"),
        ("install", "pykakasi>=2.2"),
        ("install", "onnx"),
        ("install", "onnxruntime"),
        ("install-no-deps", "chatterbox-tts>=0.1.6"),
    )

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._model_dir = config.LOCAL_MODELS / self.weights_dir
        self.catalog = _english_catalog()

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from chatterbox.tts import ChatterboxTTS

            device = pick_device()
            # `from_local` loads weights from disk; fall back to `from_pretrained`
            # (which downloads from HF) if the directory wasn't populated.
            loader = getattr(ChatterboxTTS, "from_local", None)
            if loader is not None and self._model_dir.is_dir():
                self._model = loader(str(self._model_dir), device=str(device))
            else:
                self._model = ChatterboxTTS.from_pretrained(device=str(device))

    def synth(self, text: str, language: str, voice: Voice) -> np.ndarray:
        if self._model is None:
            self.load()
        kwargs: dict[str, object] = {}
        if voice.audio_path:
            kwargs["audio_prompt_path"] = voice.audio_path
        wav = self._model.generate(text, **kwargs)
        audio = to_mono_float32(wav)
        src_sr = int(getattr(self._model, "sr", _NATIVE_SR))
        return resample_if_needed(audio, src_sr, self.sample_rate)
