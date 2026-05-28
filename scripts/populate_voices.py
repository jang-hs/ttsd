#!/usr/bin/env python
"""Populate voices/ and voice_catalog.json from a public dataset or a local dir.

Sources:

  fleurs          Google FLEURS via Hugging Face (102 languages, CC-BY-4.0).
                  NOT gated. Includes per-clip gender labels. Needs
                  `pip install datasets`. Recommended default.

  common-voice    Mozilla Common Voice via Hugging Face. Currently BROKEN on
                  HF: Mozilla's repo still uses a legacy loading script that
                  newer `datasets` versions refuse to run. Use FLEURS or a
                  Common Voice mirror that publishes Parquet directly.

  dir             Ingest a local directory already organised as
                  <root>/<Language>/<Gender>/<voice>.{mp3,wav}
                  Optional sibling <voice>.{txt|qwen.txt} for transcripts.

After ingest, voices.manifest.jsonl is rebuilt automatically.

Usage:
  python populate_voices.py fleurs --languages en,ko,ja,zh --per-gender 2
  python populate_voices.py dir --path /path/to/clips
  python populate_voices.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

# ISO 639-1/2 lang code -> human-readable language name (for catalog display).
# We keep the API-facing language_code identical to the source code (so OmniVoice
# gets the BCP-47 tag it expects).
LANG_NAME = {
    "en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese",
    "zh-CN": "Chinese", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
    "arb": "Arabic", "tr": "Turkish", "nl": "Dutch", "pl": "Polish",
    "hi": "Hindi", "vi": "Vietnamese", "id": "Indonesian", "th": "Thai",
    "uk": "Ukrainian", "cs": "Czech", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "no": "Norwegian", "el": "Greek", "he": "Hebrew",
    "ro": "Romanian", "hu": "Hungarian", "ca": "Catalan", "fa": "Persian",
    "bg": "Bulgarian", "hr": "Croatian", "sr": "Serbian", "sk": "Slovak",
    "sl": "Slovenian", "lt": "Lithuanian", "lv": "Latvian", "et": "Estonian",
}

CV_GENDER = {
    "male_masculine": "Male", "female_feminine": "Female",
    "male": "Male", "female": "Female",
}

# Short ISO -> default FLEURS subset code (FLEURS uses BCP-47 with region).
FLEURS_MAP = {
    "en": "en_us",   "ko": "ko_kr",   "ja": "ja_jp",   "zh": "cmn_hans_cn",
    "es": "es_419",  "fr": "fr_fr",   "de": "de_de",   "ar": "ar_eg",
    "it": "it_it",   "pt": "pt_br",   "ru": "ru_ru",   "hi": "hi_in",
    "tr": "tr_tr",   "nl": "nl_nl",   "pl": "pl_pl",   "vi": "vi_vn",
    "id": "id_id",   "th": "th_th",   "uk": "uk_ua",   "el": "el_gr",
    "cs": "cs_cz",   "sv": "sv_se",   "da": "da_dk",   "fi": "fi_fi",
    "no": "nb_no",   "ro": "ro_ro",   "hu": "hu_hu",   "he": "he_il",
    "fa": "fa_ir",   "bg": "bg_bg",   "hr": "hr_hr",   "sk": "sk_sk",
    "sl": "sl_si",   "ca": "ca_es",
}
FLEURS_GENDER = {0: "Male", 1: "Female"}

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


# ---------------------------------------------------------------- catalog I/O
def _load_catalog(path: Path) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def _save_catalog(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")


def _upsert(catalog: list[dict], entry: dict) -> bool:
    """Insert entry, skipping if voice_id already present. Returns True if added."""
    if any(v["id"] == entry["id"] for v in catalog):
        return False
    catalog.append(entry)
    return True


def _rebuild_manifest(project: Path) -> None:
    """Invoke build_manifest in-process to refresh voices.manifest.jsonl."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_manifest import build
    build(project / "voice_catalog.json", project / "voices",
          project / "voices.manifest.jsonl", strict=False)


# ---------------------------------------------------------------- fleurs
def populate_fleurs(
    project: Path, languages: list[str], per_gender: int, split: str
) -> int:
    try:
        from datasets import Audio, load_dataset
    except ImportError:
        print("Missing dep: pip install datasets", file=sys.stderr)
        return 2

    import io
    import soundfile as sf

    voices_root = project / "voices"
    catalog_path = project / "voice_catalog.json"
    catalog = _load_catalog(catalog_path)
    voices_root.mkdir(exist_ok=True)
    total_added = 0

    for lang in languages:
        fleurs_lang = FLEURS_MAP.get(lang, lang)            # accept either short or full code
        short = lang if lang in FLEURS_MAP else fleurs_lang.split("_")[0]
        lang_name = LANG_NAME.get(short, short.title())
        print(f"==> FLEURS [{fleurs_lang}] {lang_name}")
        try:
            ds = load_dataset("google/fleurs", fleurs_lang, split=split, streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))  # decode ourselves via soundfile
        except Exception as exc:  # noqa: BLE001
            print(f"  could not open google/fleurs/{fleurs_lang}: {exc}", file=sys.stderr)
            continue

        wanted = {"Male": per_gender, "Female": per_gender}
        added_this = 0
        scanned = 0
        max_scan = 3000

        for ex in ds:
            scanned += 1
            if scanned > max_scan or all(v == 0 for v in wanted.values()):
                break
            gender = FLEURS_GENDER.get(ex.get("gender"))
            if not gender or wanted.get(gender, 0) <= 0:
                continue
            transcript = (ex.get("transcription") or "").strip()
            if not (8 <= len(transcript) <= 250):
                continue
            try:
                arr, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]))
            except Exception:
                continue
            dur = len(arr) / sr
            if not (3.0 <= dur <= 15.0):
                continue

            name = f"fl_{int(ex.get('id', scanned)):05d}"
            voice_id = f"{lang_name}-{gender}-{name}"
            if any(v["id"] == voice_id for v in catalog):
                continue

            dst_dir = voices_root / lang_name / gender
            dst_dir.mkdir(parents=True, exist_ok=True)
            wav_path = dst_dir / f"{name}.wav"
            sf.write(str(wav_path), arr, sr, subtype="PCM_16")
            (dst_dir / f"{name}.wav.qwen.txt").write_text(transcript, encoding="utf-8")

            entry = {
                "id": voice_id, "name": name, "language": lang_name,
                "language_code": short, "gender": gender, "age": "",
                "tags": ["FLEURS"],
            }
            if _upsert(catalog, entry):
                wanted[gender] -= 1
                added_this += 1
                total_added += 1

        print(f"  added {added_this} clips (scanned {scanned})")

    _save_catalog(catalog_path, catalog)
    _rebuild_manifest(project)
    print(f"total added: {total_added}")
    return 0


# ---------------------------------------------------------------- common-voice
def _safe_token(s: str, n: int = 12) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", s)[:n] or "spk"


def populate_common_voice(
    project: Path, languages: list[str], per_gender: int, dataset_version: str
) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Missing dep: pip install datasets", file=sys.stderr)
        return 2

    import soundfile as sf

    voices_root = project / "voices"
    catalog_path = project / "voice_catalog.json"
    catalog = _load_catalog(catalog_path)
    voices_root.mkdir(exist_ok=True)

    repo = f"mozilla-foundation/common_voice_{dataset_version}"
    total_added = 0

    for lang in languages:
        lang_name = LANG_NAME.get(lang, lang.upper())
        print(f"==> Common Voice [{lang}] {lang_name}")
        try:
            ds = load_dataset(repo, lang, split="train", streaming=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  could not open {repo}/{lang}: {exc}", file=sys.stderr)
            print(f"  NOTE: Common Voice on HF currently fails because the repo"
                  f" uses a legacy loading script. Use FLEURS instead:"
                  f"  python populate_voices.py fleurs --languages {lang} --per-gender {per_gender}",
                  file=sys.stderr)
            continue

        wanted = {"Male": per_gender, "Female": per_gender}
        seen_clients: set[str] = set()
        added_this_lang = 0
        scanned = 0
        max_scan = 5000  # cap streaming work for rare-gender languages

        for ex in ds:
            scanned += 1
            if scanned > max_scan or all(v == 0 for v in wanted.values()):
                break
            gender = CV_GENDER.get(ex.get("gender") or "")
            if not gender or wanted.get(gender, 0) <= 0:
                continue
            sentence = (ex.get("sentence") or "").strip()
            if len(sentence) < 8 or len(sentence) > 200:
                continue
            audio = ex.get("audio") or {}
            arr = audio.get("array")
            sr = audio.get("sampling_rate")
            if arr is None or sr is None:
                continue
            dur = len(arr) / sr
            if not (3.0 <= dur <= 10.0):
                continue
            client = ex.get("client_id") or ""
            if client and client in seen_clients:
                continue  # one clip per speaker
            seen_clients.add(client)

            speaker = _safe_token(client or f"cv{scanned:05d}")
            name = f"cv_{speaker}"
            voice_id = f"{lang_name}-{gender}-{name}"
            if any(v["id"] == voice_id for v in catalog):
                continue

            dst_dir = voices_root / lang_name / gender
            dst_dir.mkdir(parents=True, exist_ok=True)
            wav_path = dst_dir / f"{name}.wav"
            sf.write(str(wav_path), arr, sr, subtype="PCM_16")
            (dst_dir / f"{name}.wav.qwen.txt").write_text(sentence, encoding="utf-8")

            entry = {
                "id": voice_id, "name": name, "language": lang_name,
                "language_code": lang, "gender": gender, "age": "",
                "tags": ["Common Voice"],
            }
            if _upsert(catalog, entry):
                wanted[gender] -= 1
                added_this_lang += 1
                total_added += 1

        print(f"  added {added_this_lang} clips (scanned {scanned})")

    _save_catalog(catalog_path, catalog)
    _rebuild_manifest(project)
    print(f"total added: {total_added}")
    return 0


# ---------------------------------------------------------------- dir source
def populate_from_dir(project: Path, src: Path) -> int:
    """Ingest <src>/<Language>/<Gender>/<voice>.{mp3,wav,...}"""
    voices_root = project / "voices"
    catalog_path = project / "voice_catalog.json"
    catalog = _load_catalog(catalog_path)
    voices_root.mkdir(exist_ok=True)

    added = 0
    for f in sorted(src.rglob("*")):
        if f.suffix.lower() not in AUDIO_EXTS:
            continue
        rel = f.relative_to(src).parts
        if len(rel) < 3:
            continue
        language, gender = rel[0], rel[1]
        name = re.sub(r"\s*\([^)]*\)\s*$", "", f.stem).strip()
        voice_id = f"{language}-{gender}-{name}"
        if any(v["id"] == voice_id for v in catalog):
            continue
        dst_dir = voices_root / language / gender
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f.name
        shutil.copy2(f, dst)
        # carry over a sibling transcript if present
        for cand in (f.with_suffix(f.suffix + ".qwen.txt"),
                     f.with_suffix(f.suffix + ".txt"),
                     f.with_suffix(".txt")):
            if cand.exists():
                shutil.copy2(cand, dst_dir / cand.name)
                break

        lang_code = next((c for c, n in LANG_NAME.items() if n.lower() == language.lower()),
                          language.lower())
        catalog.append({
            "id": voice_id, "name": name, "language": language,
            "language_code": lang_code, "gender": gender, "age": "", "tags": [],
        })
        added += 1

    _save_catalog(catalog_path, catalog)
    _rebuild_manifest(project)
    print(f"ingested {added} voices from {src}")
    return 0


# ---------------------------------------------------------------- CLI
def main() -> int:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)

    fl = sub.add_parser("fleurs", help="download from Google FLEURS (HF, no gating)")
    fl.add_argument("--languages", required=True,
                    help="comma-separated ISO codes (en,ko,ja,zh) or full FLEURS codes (en_us, ko_kr)")
    fl.add_argument("--per-gender", type=int, default=2,
                    help="how many male and how many female clips per language")
    fl.add_argument("--split", default="test", choices=("train", "validation", "test"),
                    help="FLEURS split to draw from (default: test, smallest+cleanest)")

    cv = sub.add_parser("common-voice", help="(broken upstream — use 'fleurs' instead)")
    cv.add_argument("--languages", required=True,
                    help="comma-separated ISO codes, e.g. en,ko,ja,zh")
    cv.add_argument("--per-gender", type=int, default=2,
                    help="how many male and how many female clips per language")
    cv.add_argument("--version", default="17_0",
                    help="Common Voice dataset version (default 17_0)")

    d = sub.add_parser("dir", help="ingest from <root>/<Language>/<Gender>/*.{mp3,wav}")
    d.add_argument("--path", required=True, type=Path)

    ap.add_argument("--project", type=Path, default=here)
    args = ap.parse_args()

    if args.source == "fleurs":
        return populate_fleurs(
            args.project, [s.strip() for s in args.languages.split(",") if s.strip()],
            args.per_gender, args.split,
        )
    if args.source == "common-voice":
        return populate_common_voice(
            args.project, [s.strip() for s in args.languages.split(",") if s.strip()],
            args.per_gender, args.version,
        )
    if args.source == "dir":
        if not args.path.is_dir():
            print(f"not a directory: {args.path}", file=sys.stderr); return 2
        return populate_from_dir(args.project, args.path)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
