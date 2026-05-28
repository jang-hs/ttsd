"""Unit tests for Catalog and Voice operations."""
from __future__ import annotations

import pytest

from app.voices import Catalog, Language, Voice


def _voice(vid: str, lang: str, code: str, gender: str = "Female") -> Voice:
    return Voice(voice_id=vid, name=vid, language=lang, language_code=code, gender=gender)


# ------------------------------------------------------------------ Catalog
def test_count() -> None:
    assert Catalog([]).count() == 0
    assert Catalog([_voice("v1", "English", "en")]).count() == 1


def test_all() -> None:
    v = _voice("v1", "English", "en")
    assert Catalog([v]).all() == [v]


def test_languages_aggregation() -> None:
    voices = [
        _voice("en-f", "English", "en", "Female"),
        _voice("en-m", "English", "en", "Male"),
        _voice("ko-f", "Korean", "ko", "Female"),
    ]
    langs = Catalog(voices).languages()
    assert len(langs) == 2
    by_code = {l.code: l for l in langs}
    assert by_code["en"].voice_count == 2
    assert by_code["ko"].voice_count == 1


def test_languages_sorted_alphabetically() -> None:
    voices = [_voice("z", "Zulu", "zu"), _voice("a", "Afrikaans", "af")]
    codes = [l.code for l in Catalog(voices).languages()]
    assert codes == sorted(codes, key=str.lower)


def test_voices_for_language_filters_correctly() -> None:
    en = _voice("en", "English", "en")
    ko = _voice("ko", "Korean", "ko")
    result = Catalog([en, ko]).voices_for_language("en")
    assert result == [en]


def test_voices_for_language_unknown_returns_empty() -> None:
    assert Catalog([_voice("en", "English", "en")]).voices_for_language("xx") == []


def test_find_voice_by_id() -> None:
    v = _voice("my-id", "English", "en")
    assert Catalog([v]).find_voice("my-id") is v


def test_find_voice_missing_returns_none() -> None:
    assert Catalog([]).find_voice("nope") is None


def test_default_for_language_returns_first_match() -> None:
    en1 = _voice("en-1", "English", "en")
    en2 = _voice("en-2", "English", "en")
    assert Catalog([en1, en2]).default_for_language("en") is en1


def test_default_for_language_unknown_falls_back_to_first_voice() -> None:
    v = _voice("en", "English", "en")
    assert Catalog([v]).default_for_language("xx") is v


def test_default_for_language_empty_catalog_returns_none() -> None:
    assert Catalog([]).default_for_language("en") is None


# ------------------------------------------------------------------ Voice
def test_voice_is_frozen() -> None:
    v = _voice("v", "English", "en")
    with pytest.raises((AttributeError, TypeError)):
        v.name = "changed"  # type: ignore[misc]


def test_voice_tags_default_to_empty_tuple() -> None:
    v = Voice(voice_id="v", name="v", language="English", language_code="en", gender="Female")
    assert v.tags == ()
