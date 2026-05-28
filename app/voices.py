"""Voice + language catalog primitives, shared across backends.

Each backend builds a Catalog: OmniVoice from voices.manifest.jsonl, Kokoro from
its bundled voice files. The API layer queries the catalog uniformly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

from . import config


@dataclass(frozen=True)
class Voice:
    voice_id: str
    name: str
    language: str          # full name, e.g. "English"
    language_code: str     # API-facing code, e.g. "en", "arb"
    gender: str
    age: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    # Synthesis inputs (backend-specific; empty when unused)
    audio_path: str = ""   # reference clip for cloning backends
    transcript: str = ""   # reference transcript for cloning backends
    lang_code: str = ""    # pipeline language token (e.g. Kokoro's "a")


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    voice_count: int


class Catalog:
    """An in-memory voice catalog with the lookups the API needs."""

    def __init__(self, voices: list[Voice]) -> None:
        self._voices = voices
        self._by_id = {v.voice_id: v for v in voices}

    def all(self) -> list[Voice]:
        return self._voices

    def count(self) -> int:
        return len(self._voices)

    def languages(self) -> list[Language]:
        counts: dict[str, int] = {}
        label: dict[str, str] = {}
        for v in self._voices:
            counts[v.language_code] = counts.get(v.language_code, 0) + 1
            label.setdefault(v.language_code, v.language)
        out = [Language(code=c, name=label.get(c, c), voice_count=n) for c, n in counts.items()]
        out.sort(key=lambda x: x.name.lower())
        return out

    def voices_for_language(self, code: str) -> list[Voice]:
        return [v for v in self._voices if v.language_code == code]

    def find_voice(self, voice_id: str) -> Voice | None:
        return self._by_id.get(voice_id)

    def default_for_language(self, code: str) -> Voice | None:
        scoped = self.voices_for_language(code)
        return scoped[0] if scoped else (self._voices[0] if self._voices else None)


@lru_cache(maxsize=1)
def load_omnivoice_catalog() -> Catalog:
    """Catalog from voices.manifest.jsonl (OpenVox-mirrored ids + tags + ref clips)."""
    voices: list[Voice] = []
    with open(config.VOICES_MANIFEST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not d.get("language_code"):
                continue
            voices.append(
                Voice(
                    voice_id=d["voice_id"],
                    name=d.get("name") or d["voice_id"],
                    language=d.get("language") or "",
                    language_code=d["language_code"],
                    gender=d.get("gender") or "Unknown",
                    age=d.get("age") or "",
                    tags=tuple(d.get("tags") or ()),
                    audio_path=d.get("audio_path") or "",
                    transcript=d.get("transcript") or "",
                )
            )
    return Catalog(voices)
