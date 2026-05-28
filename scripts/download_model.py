#!/usr/bin/env python
"""Download TTS model weights from Hugging Face into ./models.

Usage:
  ./scripts/download_model.py                 # download the default set
  ./scripts/download_model.py kokoro          # one model
  ./scripts/download_model.py omnivoice kokoro chatterbox
  ./scripts/download_model.py --list
  ./scripts/download_model.py --force kokoro  # re-download even if present
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# model key -> list of (hf_repo_id, local_dir_name)
REGISTRY: dict[str, list[tuple[str, str]]] = {
    "omnivoice": [
        ("theoracleguy/OmniVoice-bf16", "OmniVoice-bf16"),
        ("theoracleguy/OmniVoice", "OmniVoice"),
    ],
    "kokoro": [
        ("theoracleguy/Kokoro-82M-bf16", "Kokoro-82M-bf16"),
    ],
    "chatterbox": [
        ("theoracleguy/chatterbox-turbo-fp16", "chatterbox-turbo-fp16"),
    ],
    "qwen3_tts": [
        ("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
         "Qwen3-TTS-CustomVoice-8bit"),
    ],
    "chatterbox_multilingual": [
        ("theoracleguy/Chatterbox-Multilingual-MLX-v2-Q8",
         "Chatterbox-Multilingual-Q8"),
    ],
}

DEFAULT_SET = list(REGISTRY.keys())


def _is_populated(dest: Path) -> bool:
    if not dest.is_dir():
        return False
    return any(f.stat().st_size > 50 * 1024 * 1024 for f in dest.rglob("*.safetensors"))


def fetch_repo(repo_id: str, dir_name: str, force: bool) -> None:
    dest = MODELS_DIR / dir_name
    if _is_populated(dest) and not force:
        print(f"  [skip] {dir_name} already present")
        return

    print(f"  [hf] downloading {repo_id} from Hugging Face ...")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        ignore_patterns=["*.md", ".gitattributes", "*.onnx"],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="model keys (default: all)")
    ap.add_argument("--list", action="store_true", help="list known models and exit")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if weights already present")
    args = ap.parse_args()

    if args.list:
        for k, repos in REGISTRY.items():
            print(f"{k:12s} -> {', '.join(r for r, _ in repos)}")
        return 0

    keys = args.models or DEFAULT_SET
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        print(f"Unknown model(s): {unknown}\nKnown: {list(REGISTRY)}", file=sys.stderr)
        return 2

    MODELS_DIR.mkdir(exist_ok=True)
    for key in keys:
        print(f"==> {key}")
        for repo_id, dir_name in REGISTRY[key]:
            fetch_repo(repo_id, dir_name, args.force)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
