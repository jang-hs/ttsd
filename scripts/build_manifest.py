#!/usr/bin/env python3
"""Build voices.manifest.jsonl from voice_catalog.json + voices/ directory.

The catalog lists voice metadata; the voices/ dir holds the actual audio +
transcript. This script joins them into the runtime manifest the server reads.

Audio discovery (per catalog entry):
  voices/<Language>/<Gender>/<Name (...)>.{mp3,wav,flac,ogg}
  -- the part before " (" must equal the catalog `name`.

Transcript discovery:
  <audio>.qwen.txt  OR  <audio>.txt  OR  same stem .txt next to audio.

Usage:
  python build_manifest.py                          # default paths
  python build_manifest.py --catalog X --voices Y --out Z
  python build_manifest.py --strict                 # fail if any entry lacks audio
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")


def _stem_clean(stem: str) -> str:
    """'Abigail (M,C)' -> 'Abigail' (strip trailing parenthetical tag groups)."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", stem).strip()


def _index_audio(voices_root: Path) -> dict[tuple[str, str, str], Path]:
    """Index audio by (language_folder, gender_folder, clean_name)."""
    index: dict[tuple[str, str, str], Path] = {}
    for ext in AUDIO_EXTS:
        for f in voices_root.rglob(f"*{ext}"):
            parts = f.relative_to(voices_root).parts
            if len(parts) < 3:
                continue
            lang, gender = parts[0], parts[1]
            name = _stem_clean(f.stem)
            index.setdefault((lang, gender, name), f)
    return index


def _read_transcript(audio: Path) -> str:
    for cand in (audio.with_suffix(audio.suffix + ".qwen.txt"),
                 audio.with_suffix(audio.suffix + ".txt"),
                 audio.with_suffix(".txt")):
        if cand.exists():
            return cand.read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def build(catalog_path: Path, voices_root: Path, out_path: Path, strict: bool) -> int:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    audio_index = _index_audio(voices_root)
    rows: list[dict] = []
    missing: list[str] = []
    for v in catalog:
        key = (v["language"], v["gender"], v["name"])
        audio = audio_index.get(key)
        if audio is None:
            missing.append(v["id"])
            continue
        rows.append({
            "voice_id": v["id"],
            "name": v["name"],
            "gender": v["gender"],
            "age": v.get("age", ""),
            "tags": v.get("tags", []),
            "language": v["language"],
            "language_code": v["language_code"],
            "audio_path": str(audio.resolve()),
            "transcript": _read_transcript(audio),
        })
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_tx = sum(1 for r in rows if r["transcript"])
    print(f"manifest: {len(rows)} voices ({n_tx} with transcript) -> {out_path}")
    if missing:
        print(f"  unmatched: {len(missing)} catalog entries had no audio file")
        if strict:
            print("  (strict mode) example missing:", missing[:5])
            return 1
    return 0


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", type=Path, default=here / "voice_catalog.json")
    ap.add_argument("--voices",  type=Path, default=here / "voices")
    ap.add_argument("--out",     type=Path, default=here / "voices.manifest.jsonl")
    ap.add_argument("--strict",  action="store_true",
                    help="fail if any catalog entry lacks an audio file")
    args = ap.parse_args()
    if not args.catalog.exists():
        print(f"catalog not found: {args.catalog}"); return 2
    if not args.voices.is_dir():
        print(f"voices dir not found: {args.voices}"); return 2
    return build(args.catalog, args.voices, args.out, args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
