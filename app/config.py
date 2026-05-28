"""Paths and constants for the local TTS server."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOICES_MANIFEST = PROJECT_ROOT / "voices.manifest.jsonl"

# Model weights live here after running scripts/download_model.py.
LOCAL_MODELS = PROJECT_ROOT / "models"


def omnivoice_model_path() -> Path:
    """bf16 OmniVoice LLM checkpoint (config.json + model.safetensors + tokenizer)."""
    return LOCAL_MODELS / "OmniVoice-bf16"


def omnivoice_full_tokenizer_path() -> Path:
    """Full HiggsAudio tokenizer (encode + decode) required for voice cloning."""
    return LOCAL_MODELS / "OmniVoice"


def kokoro_model_path() -> Path:
    """Kokoro checkpoint (config.json + kokoro-*.safetensors + voices/)."""
    return LOCAL_MODELS / "Kokoro-82M-bf16"


def chatterbox_model_path() -> Path:
    """Chatterbox Turbo checkpoint."""
    return LOCAL_MODELS / "chatterbox-turbo-fp16"


def qwen3_tts_model_path() -> Path:
    """Qwen3-TTS CustomVoice (8-bit)."""
    return LOCAL_MODELS / "Qwen3-TTS-CustomVoice-8bit"


def chatterbox_multilingual_model_path() -> Path:
    """Chatterbox Multilingual MLX v2 (Q8)."""
    return LOCAL_MODELS / "Chatterbox-Multilingual-Q8"


def has_local_model(dir_name: str) -> bool:
    d = LOCAL_MODELS / dir_name
    return d.is_dir() and any(d.rglob("*.safetensors"))


# Synthesis defaults
NUM_STEPS = int(os.environ.get("OMNIVOICE_NUM_STEPS", "16"))
GUIDANCE_SCALE = float(os.environ.get("OMNIVOICE_GUIDANCE_SCALE", "2.0"))
SAMPLE_RATE = 24000

# Static creation timestamp reported in /v1/models (cosmetic, matches OpenVox shape)
MODEL_CREATED_TS = 1779859956
