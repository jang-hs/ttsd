"""Self-hosted local voice API — drop-in for the OpenVox Local API contract.

Exposes the same routes under /v1 so existing agents work unchanged by pointing
their base URL here. No daily limits.
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import audio, config, registry, voices

app = FastAPI(title="Local Voice API", version="1.0.0")

# Allow browser-based clients (Obsidian, Electron apps, web pages) to call the
# API. These send a CORS preflight (OPTIONS) that must be answered with
# Access-Control-Allow-* headers. Local-only server, so any origin is fine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-OpenVox-Model", "X-OpenVox-Voice"],
)


# ---------------------------------------------------------------- helpers
_BUSY_MSG = {
    "generation": "A generation request is already in progress",
    "preload": "Another generation or preload request is already in progress",
}


async def _run_job(fn, *args, op: str = "generation"):
    """Run a blocking job under the single-job lock; 429 if busy.

    `op` selects the OpenVox-compatible 429 message wording.
    """
    if registry.JOB_LOCK.locked():
        raise HTTPException(status_code=429, detail=_BUSY_MSG[op])
    async with registry.JOB_LOCK:
        return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


def _require_model(model_id: str):
    backend = registry.get(model_id)
    if backend is None:
        # Match OpenVox exactly: 500 + "Unsupported model: <id>"
        raise HTTPException(status_code=500, detail=f"Unsupported model: {model_id}")
    return backend


# ---------------------------------------------------------------- schemas
class SpeechRequest(BaseModel):
    model: str
    input: str
    language: str = "en"
    voice: Optional[str] = None
    response_format: str = "wav"
    stream: bool = False


# ---------------------------------------------------------------- routes
@app.get("/v1/health")
async def health():
    return {"object": "health", "status": "ok"}


@app.get("/v1/models")
async def list_models():
    data = []
    for b in registry.all_backends():
        data.append(
            {
                "id": b.id,
                "object": "model",
                "display_name": b.display_name,
                "model_key": b.model_key,
                "owned_by": "openvox",
                "created": config.MODEL_CREATED_TS,
                "supports_streaming": b.supports_streaming,
                "voice_count": b.catalog.count(),
            }
        )
    return {"object": "list", "data": data}


@app.post("/v1/models/{model}/load")
async def load_model(model: str):
    backend = _require_model(model)
    await _run_job(backend.load, op="preload")
    return {
        "object": "model_load",
        "model": model,
        "status": "ready",
        "message": "Model warmed and ready for the next request.",
    }


@app.get("/v1/models/{model}/languages")
async def list_languages(model: str):
    backend = _require_model(model)
    data = [
        {"id": l.code, "object": "language", "code": l.code, "name": l.name, "voice_count": l.voice_count}
        for l in backend.catalog.languages()
    ]
    return {"object": "list", "model": model, "data": data}


@app.get("/v1/models/{model}/voices")
async def list_voices(model: str, language: str = Query(...)):
    backend = _require_model(model)
    data = [
        {
            "id": v.voice_id,
            "object": "voice",
            "name": v.name,
            "language": v.language,
            "gender": v.gender,
            "age": v.age,
            "tags": list(v.tags),
            "has_preview": True,
            "model": backend.voice_model_label,
        }
        for v in backend.catalog.voices_for_language(language)
    ]
    return {"object": "list", "model": model, "language": language, "data": data}


def _resolve_voice(backend, req_voice: Optional[str], language: str) -> voices.Voice:
    if req_voice:
        v = backend.catalog.find_voice(req_voice)
        if v is None:
            raise HTTPException(status_code=400, detail=f"Voice not found: {req_voice}")
        return v
    v = backend.catalog.default_for_language(language)
    if v is None:
        raise HTTPException(status_code=500, detail=f"Voice not found for language: {language}")
    return v


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest, request: Request):
    backend = _require_model(req.model)
    voice = _resolve_voice(backend, req.voice, req.language)

    if req.stream:
        return await _speech_stream(backend, req, voice)

    audio_arr = await _run_job(backend.synth, req.input, req.language, voice)
    wav = audio.to_wav_bytes(audio_arr, backend.sample_rate)
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "X-OpenVox-Model": backend.id,
            "X-OpenVox-Voice": voice.voice_id,
        },
    )


async def _speech_stream(backend, req: SpeechRequest, voice: voices.Voice):
    # Match OpenVox's streaming shape exactly: one audio.chunk per response with
    # the full WAV payload, is_final=false, then response.completed{done:true}.
    audio_arr = await _run_job(backend.synth, req.input, req.language, voice)
    wav = audio.to_wav_bytes(audio_arr, backend.sample_rate)

    def sse(event: str, data: dict) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    async def gen():
        yield sse(
            "response.created",
            {"model": backend.id, "response_format": "wav", "voice": voice.voice_id},
        )
        yield sse(
            "audio.chunk",
            {
                "index": 0,
                "sample_rate": backend.sample_rate,
                "audio": base64.b64encode(wav).decode(),
                "format": "wav",
                "is_final": False,
            },
        )
        yield sse("response.completed", {"done": True})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.exception_handler(HTTPException)
async def _http_exc(request: Request, exc: HTTPException):
    # OpenAI-style envelope, matching OpenVox: {"error":{"message":..,"type":..}}
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
                "type": "invalid_request_error",
            }
        },
    )
