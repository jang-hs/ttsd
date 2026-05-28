#!/usr/bin/env python3
"""Download TTS model weights from Hugging Face into ./models.

Weight repos are discovered from each backend class's `weights_repos`
attribute in `app/backends/`. To register a new model, add it there — no
edits here required.

Cross-platform: uses huggingface_hub.snapshot_download, no shell tricks.

Usage:
  python -m scripts.download_model               # download the default set
  python -m scripts.download_model kokoro        # one model
  python -m scripts.download_model kokoro chatterbox-turbo-large
  python -m scripts.download_model --list
  python -m scripts.download_model --force kokoro
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def _registry() -> dict[str, list[tuple[str, str, tuple[str, ...] | None]]]:
    """backend id -> list of (hf_repo_id, local_dir_name, allow_patterns_or_None).

    Discovered from `app.backends.ALL_BACKENDS`."""
    from app.backends import ALL_BACKENDS

    out: dict[str, list[tuple[str, str, tuple[str, ...] | None]]] = {}
    for cls in ALL_BACKENDS:
        out[cls.id] = list(cls.weights_repos)
    return out


def _is_populated(dest: Path) -> bool:
    if not dest.is_dir():
        return False
    for ext in ("*.safetensors", "*.pt"):
        if any(f.stat().st_size > 10 * 1024 * 1024 for f in dest.rglob(ext)):
            return True
    return False


def fetch_repo(
    repo_id: str,
    dir_name: str,
    allow_patterns: tuple[str, ...] | None,
    force: bool,
) -> None:
    dest = MODELS_DIR / dir_name
    if _is_populated(dest) and not force:
        print(f"  [skip] {dir_name} already present")
        return

    print(f"  [hf] downloading {repo_id} -> {dest.name}")
    from huggingface_hub import snapshot_download

    kwargs: dict = {
        "repo_id": repo_id,
        "local_dir": str(dest),
        "ignore_patterns": ["*.md", ".gitattributes", "*.onnx"],
    }
    if allow_patterns:
        kwargs["allow_patterns"] = list(allow_patterns)
    snapshot_download(**kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("models", nargs="*", help="backend ids (default: all)")
    ap.add_argument("--list", action="store_true",
                    help="list known backends and their HF repos")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if weights already present")
    args = ap.parse_args()

    registry = _registry()

    if args.list:
        for k, repos in registry.items():
            print(f"{k:34s} -> {', '.join(r for r, _, _ in repos)}")
        return 0

    keys = args.models or list(registry)
    unknown = [k for k in keys if k not in registry]
    if unknown:
        print(f"Unknown backend(s): {unknown}\nKnown: {list(registry)}",
              file=sys.stderr)
        return 2

    MODELS_DIR.mkdir(exist_ok=True)
    for key in keys:
        print(f"==> {key}")
        for repo_id, dir_name, allow_patterns in registry[key]:
            fetch_repo(repo_id, dir_name, allow_patterns, args.force)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
