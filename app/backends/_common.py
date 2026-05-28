"""Shared helpers for backend adapters."""
from __future__ import annotations

from typing import Any

import numpy as np


def to_mono_float32(audio: Any) -> np.ndarray:
    """Coerce a backend's raw output into a 1-D float32 numpy waveform.

    Accepts torch tensors, numpy arrays, or nested lists. Reduces channels to
    mono by averaging if stereo, and divides by the peak when clipping
    (matches the previous MLX backends' normalization behavior)."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        torch = None  # type: ignore[assignment]

    if torch is not None and isinstance(audio, torch.Tensor):
        audio = audio.detach().to(torch.float32).cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        # (channels, samples) or (samples, channels); pick the smaller axis as channels.
        ch_axis = 0 if arr.shape[0] < arr.shape[1] else 1
        arr = arr.mean(axis=ch_axis)
    elif arr.ndim > 2:
        arr = arr.reshape(-1)
    arr = np.ascontiguousarray(arr.reshape(-1), dtype=np.float32)
    if arr.size:
        peak = float(np.abs(arr).max())
        if peak > 1.0:
            arr = arr / peak
    return arr


def resample_if_needed(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear resample to dst_sr when the model returned a different rate.

    Kept dependency-free (no librosa/soxr) — these backends only resample
    occasionally (Qwen3-TTS codec) so the linear approximation is fine for the
    /v1 contract's PCM_16 WAV output."""
    if src_sr == dst_sr or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    ratio = dst_sr / float(src_sr)
    new_len = int(round(audio.size * ratio))
    if new_len <= 1:
        return audio.astype(np.float32, copy=False)
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False, dtype=np.float64)
    return np.interp(x_new, x_old, audio).astype(np.float32, copy=False)
