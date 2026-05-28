"""Integration tests for /v1 API routes.

All ML backends are mocked via conftest.py — no model weights needed.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app import registry


# ------------------------------------------------------------------ health
def test_health(client: TestClient) -> None:
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json() == {"object": "health", "status": "ok"}


# ------------------------------------------------------------------ models
def test_list_models(client: TestClient) -> None:
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    m = body["data"][0]
    assert m["id"] == "qwen3-tts-medium"
    assert m["supports_streaming"] is True
    assert m["voice_count"] == 1


def test_model_not_found_returns_500(client: TestClient) -> None:
    r = client.get("/v1/models/nonexistent/languages")
    assert r.status_code == 500
    err = r.json()["error"]
    assert "Unsupported model" in err["message"]
    assert err["type"] == "invalid_request_error"


# ------------------------------------------------------------------ languages
def test_list_languages(client: TestClient) -> None:
    r = client.get("/v1/models/qwen3-tts-medium/languages")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "qwen3-tts-medium"
    langs = body["data"]
    assert len(langs) == 1
    assert langs[0]["id"] == "en"
    assert langs[0]["code"] == "en"
    assert langs[0]["voice_count"] == 1


# ------------------------------------------------------------------ voices
def test_list_voices(client: TestClient) -> None:
    r = client.get("/v1/models/qwen3-tts-medium/voices?language=en")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["language"] == "en"
    voices = body["data"]
    assert len(voices) == 1
    v = voices[0]
    assert v["id"] == "English-Female-Alice"
    assert v["name"] == "Alice"
    assert v["gender"] == "Female"
    assert isinstance(v["tags"], list)


def test_list_voices_unknown_language_returns_empty(client: TestClient) -> None:
    r = client.get("/v1/models/qwen3-tts-medium/voices?language=xx")
    assert r.status_code == 200
    assert r.json()["data"] == []


def test_list_voices_missing_language_param(client: TestClient) -> None:
    r = client.get("/v1/models/qwen3-tts-medium/voices")
    assert r.status_code == 422  # FastAPI validation error (Query required)


# ------------------------------------------------------------------ load
def test_load_model(client: TestClient) -> None:
    r = client.post("/v1/models/qwen3-tts-medium/load")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["model"] == "qwen3-tts-medium"


def test_load_model_not_found(client: TestClient) -> None:
    r = client.post("/v1/models/nonexistent/load")
    assert r.status_code == 500


# ------------------------------------------------------------------ speech
def test_speech_returns_wav(client: TestClient) -> None:
    r = client.post("/v1/audio/speech", json={
        "model": "qwen3-tts-medium",
        "input": "hello",
        "language": "en",
        "voice": "English-Female-Alice",
    })
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.headers["x-openvox-model"] == "qwen3-tts-medium"
    assert r.headers["x-openvox-voice"] == "English-Female-Alice"
    assert r.content[:4] == b"RIFF"
    assert r.content[8:12] == b"WAVE"


def test_speech_default_voice(client: TestClient) -> None:
    r = client.post("/v1/audio/speech", json={
        "model": "qwen3-tts-medium",
        "input": "hello",
        "language": "en",
    })
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"


def test_speech_voice_not_found_returns_400(client: TestClient) -> None:
    r = client.post("/v1/audio/speech", json={
        "model": "qwen3-tts-medium",
        "input": "hello",
        "language": "en",
        "voice": "nonexistent-voice",
    })
    assert r.status_code == 400
    err = r.json()["error"]
    assert "Voice not found" in err["message"]
    assert "nonexistent-voice" in err["message"]


def test_speech_model_not_found(client: TestClient) -> None:
    r = client.post("/v1/audio/speech", json={
        "model": "nonexistent",
        "input": "hello",
        "language": "en",
    })
    assert r.status_code == 500


def test_speech_busy_returns_429(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(registry.JOB_LOCK, "locked", lambda: True)
    r = client.post("/v1/audio/speech", json={
        "model": "qwen3-tts-medium",
        "input": "hello",
        "language": "en",
    })
    assert r.status_code == 429
    assert "generation" in r.json()["error"]["message"].lower()


# ------------------------------------------------------------------ streaming
def test_speech_stream_sse_events(client: TestClient) -> None:
    r = client.post("/v1/audio/speech", json={
        "model": "qwen3-tts-medium",
        "input": "hello",
        "language": "en",
        "voice": "English-Female-Alice",
        "stream": True,
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]

    events = {}
    for line in r.text.splitlines():
        if line.startswith("event: "):
            current = line[len("event: "):]
        elif line.startswith("data: ") and current:
            events[current] = json.loads(line[len("data: "):])
            current = None

    assert "response.created" in events
    assert "audio.chunk" in events
    assert "response.completed" in events

    chunk = events["audio.chunk"]
    assert chunk["format"] == "wav"
    assert chunk["sample_rate"] == 24000
    assert isinstance(chunk["audio"], str)  # base64

    assert events["response.completed"]["done"] is True


def test_speech_stream_audio_is_valid_wav(client: TestClient) -> None:
    import base64

    r = client.post("/v1/audio/speech", json={
        "model": "qwen3-tts-medium",
        "input": "hello",
        "language": "en",
        "stream": True,
    })
    for line in r.text.splitlines():
        if line.startswith("data: "):
            data = json.loads(line[len("data: "):])
            if "audio" in data:
                wav = base64.b64decode(data["audio"])
                assert wav[:4] == b"RIFF"
                break
