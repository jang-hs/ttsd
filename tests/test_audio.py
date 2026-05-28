"""Unit tests for WAV encoding."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from app.audio import to_wav_bytes


def _parse_wav(data: bytes) -> dict:
    """Parse minimal WAV header fields."""
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
    total_size = struct.unpack_from("<I", data, 4)[0]
    # find fmt and data chunks
    pos = 12
    chunks = {}
    while pos + 8 <= len(data):
        tag = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        chunks[tag] = data[pos + 8: pos + 8 + size]
        pos += 8 + size
    return {"total_size": total_size, "chunks": chunks}


def test_output_is_valid_wav_riff_header() -> None:
    data = to_wav_bytes(np.zeros(100, dtype=np.float32), 24000)
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"


def test_fmt_chunk_is_pcm16() -> None:
    data = to_wav_bytes(np.zeros(100, dtype=np.float32), 24000)
    info = _parse_wav(data)
    fmt = info["chunks"][b"fmt "]
    audio_format = struct.unpack_from("<H", fmt, 0)[0]
    assert audio_format == 1  # PCM


def test_sample_rate_embedded_in_header() -> None:
    for rate in (16000, 24000, 44100):
        data = to_wav_bytes(np.zeros(100, dtype=np.float32), rate)
        info = _parse_wav(data)
        fmt = info["chunks"][b"fmt "]
        embedded_rate = struct.unpack_from("<I", fmt, 4)[0]
        assert embedded_rate == rate


def test_longer_audio_produces_larger_file() -> None:
    short = to_wav_bytes(np.zeros(1000, dtype=np.float32), 24000)
    long_ = to_wav_bytes(np.zeros(48000, dtype=np.float32), 24000)
    assert len(long_) > len(short)


def test_returns_bytes() -> None:
    result = to_wav_bytes(np.zeros(10, dtype=np.float32), 24000)
    assert isinstance(result, bytes)


def test_silent_audio_encodes_without_error() -> None:
    to_wav_bytes(np.zeros(24000, dtype=np.float32), 24000)


def test_clipped_audio_encodes_without_error() -> None:
    audio = np.full(24000, 2.0, dtype=np.float32)
    data = to_wav_bytes(audio, 24000)
    assert data[:4] == b"RIFF"


@pytest.mark.parametrize("n_samples", [0, 1, 100, 48000])
def test_various_lengths(n_samples: int) -> None:
    data = to_wav_bytes(np.zeros(n_samples, dtype=np.float32), 24000)
    assert data[:4] == b"RIFF"
