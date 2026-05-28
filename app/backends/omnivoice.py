"""OmniVoice backend — runs the local MLX checkpoint via mlx-audio.

Zero-shot voice cloning: each preset voice is a reference clip + transcript
(from voices.manifest.jsonl). One full HiggsAudio tokenizer handles both the
encode (reference -> tokens) and decode (tokens -> 24kHz waveform) paths.
"""
from __future__ import annotations

import json
import threading

import numpy as np

from .. import config, voices
from ..voices import Voice


class OmniVoiceBackend:
    id = "omnivoice"
    display_name = "OmniVoice"
    model_key = "omnivoice"
    voice_model_label = "omnivoice"
    supports_streaming = True
    sample_rate = config.SAMPLE_RATE

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._text_tokenizer = None
        self._lock = threading.Lock()  # guards one-time load
        self.catalog = voices.load_omnivoice_catalog()

    # --- lifecycle ---
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import mlx.core as mx
            from transformers import AutoTokenizer

            from mlx_audio.codec.models.higgs_audio.higgs_audio import (
                HiggsAudioTokenizer,
            )
            from mlx_audio.tts.models.omnivoice.omnivoice import (
                Model,
                OmniVoiceConfig,
            )

            model_path = config.omnivoice_model_path()
            full_tok_path = config.omnivoice_full_tokenizer_path()

            with open(model_path / "config.json") as f:
                cfg = OmniVoiceConfig.from_dict(json.load(f))
            model = Model(cfg)
            weights = model.sanitize(dict(mx.load(str(model_path / "model.safetensors"))))
            model.load_weights(list(weights.items()))
            mx.eval(model.parameters())

            # Full (encode + decode) tokenizer — required for cloning.
            tokenizer = HiggsAudioTokenizer.from_pretrained(str(full_tok_path))
            text_tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            self._model = model
            self._tokenizer = tokenizer
            self._text_tokenizer = text_tokenizer

    # --- synthesis ---
    def synth(self, text: str, language: str, voice: Voice) -> np.ndarray:
        if self._model is None:
            self.load()
        result = next(
            self._model.generate(
                text=text,
                language=language or "None",
                duration_s=None,  # auto-estimate from text
                ref_audio=voice.audio_path,
                ref_text=voice.transcript or None,
                num_steps=config.NUM_STEPS,
                guidance_scale=config.GUIDANCE_SCALE,
                tokenizer=self._tokenizer,
                text_tokenizer=self._text_tokenizer,
            )
        )
        audio = np.asarray(result.audio, dtype=np.float32)
        # Guard against clipping (cloned refs can push peaks > 1.0).
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        return audio
