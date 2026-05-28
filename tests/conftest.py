"""Shared fixtures for all tests.

Backends are mocked so no model weights are needed to run the test suite.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app import registry
from app.main import app
from app.voices import Catalog, Voice


@pytest.fixture
def en_voice() -> Voice:
    return Voice(
        voice_id="English-Female-Alice",
        name="Alice",
        language="English",
        language_code="en",
        gender="Female",
        age="Young",
        tags=("Library",),
    )


@pytest.fixture
def mock_backend(en_voice: Voice) -> MagicMock:
    catalog = Catalog([en_voice])
    backend = MagicMock()
    backend.id = "qwen3-tts-medium"
    backend.display_name = "Qwen3 TTS (Medium)"
    backend.model_key = "qwen3_tts"
    backend.voice_model_label = "qwen3_tts"
    backend.supports_streaming = True
    backend.sample_rate = 24000
    backend.catalog = catalog
    backend.synth.return_value = np.zeros(24000, dtype=np.float32)
    return backend


@pytest.fixture(autouse=True)
def patch_registry(mock_backend: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_BACKENDS", {mock_backend.id: mock_backend})


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
