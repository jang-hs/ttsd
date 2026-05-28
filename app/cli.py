"""Cross-platform CLI for ttsd.

Works on macOS, Linux, and Windows using only the standard library — no
shell, no bash, no platform-specific path tricks.

  python -m app.cli setup [--with <ids|all>] [--no-uv] [--yes]
  python -m app.cli setup-backends [name ...] [--no-uv]
  python -m app.cli run [--host H --port P] [--only id,id]
  python -m app.cli download [model ...]
  python -m app.cli populate <fleurs|dir> ...

Backend install recipes and HF weight repos are discovered from each
backend class's `pip_install` and `weights_repos` metadata. To add a new
model, drop a file in `app/backends/` and append the class to `ALL_BACKENDS`
in `app/backends/__init__.py` — no CLI edits required.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .backends import ALL_BACKENDS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BACKENDS = ("qwen3-tts-medium",)


def _backend_recipes() -> dict[str, tuple]:
    return {b.id: tuple(b.pip_install) for b in ALL_BACKENDS}


def _run(cmd: list[str]) -> int:
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def _pip_install(args: list[str], use_uv: bool) -> int:
    py = sys.executable
    uv = shutil.which("uv")
    if use_uv and uv:
        return _run([uv, "pip", "install", "--python", py, *args])
    return _run([py, "-m", "pip", "install", *args])


# ---------------------------------------------------------------- interactive
def _parse_with(raw: str, ids: list[str]) -> list[str]:
    """Accept 'all', a comma-separated mix of numbers or ids, or empty."""
    raw = (raw or "").strip().lower()
    if not raw:
        return []
    if raw == "all":
        return list(ids)
    out: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(ids):
                out.append(ids[idx])
                continue
            print(f"  warn: number out of range: {tok}", file=sys.stderr)
            continue
        if tok in ids:
            out.append(tok)
        else:
            print(f"  warn: unknown backend id: {tok}", file=sys.stderr)
    # de-dup while preserving order
    seen, dedup = set(), []
    for x in out:
        if x not in seen:
            seen.add(x); dedup.append(x)
    return dedup


def _prompt_backends(ids: list[str], default: list[str]) -> list[str]:
    print("\n==> ttsd setup")
    print("\nChoose which TTS backends to install:")
    print()
    for i, bid in enumerate(ids, 1):
        cls = next(c for c in ALL_BACKENDS if c.id == bid)
        tag = "  (default)" if bid in default else ""
        print(f"  {i}) {bid:<32s} {cls.display_name}{tag}")
    print()
    print("Enter comma-separated numbers or ids, 'all', or just Enter for default.")
    try:
        raw = input("> ")
    except (EOFError, KeyboardInterrupt):
        print("\naborted")
        return []
    selected = _parse_with(raw, ids) or default
    print(f"\n  Will install: {', '.join(selected) or '(none)'}")
    return selected


def _select_backends(args: argparse.Namespace) -> list[str]:
    """Resolve which backends to install from --with / env / TTY prompt."""
    recipes = _backend_recipes()
    ids = list(recipes)
    default = [b for b in DEFAULT_BACKENDS if b in recipes]

    # explicit flag wins
    if args.with_:
        chosen = _parse_with(args.with_, ids)
        if not chosen:
            print("ERROR: --with parsed to empty selection", file=sys.stderr)
            return []
        return chosen

    # env var
    env = os.environ.get("TTSD_BACKENDS", "").strip()
    if env:
        chosen = _parse_with(env, ids)
        if chosen:
            return chosen

    # interactive only when stdin is a TTY
    if sys.stdin.isatty() and not args.yes:
        return _prompt_backends(ids, default)
    return default


def _confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes or not sys.stdin.isatty():
        return True
    try:
        ans = input(f"{question} [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans in ("", "y", "yes")


# ---------------------------------------------------------------- commands
def cmd_setup(args: argparse.Namespace) -> int:
    """Install the core HTTP server + PyTorch backbone, then per-backend extras."""
    req = PROJECT_ROOT / "requirements.txt"
    use_uv = not args.no_uv

    if not use_uv:
        rc = _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        if rc != 0:
            return rc

    selected = _select_backends(args)
    if not selected:
        print("No backends selected; install nothing.", file=sys.stderr)
        return 1

    if not _confirm(f"Install core + backends [{', '.join(selected)}]?", args.yes):
        print("aborted")
        return 1

    print("\n==> Installing core dependencies")
    rc = _pip_install(["-r", str(req)], use_uv=use_uv)
    if rc != 0:
        return rc

    rc = _install_backends(selected, use_uv=use_uv)
    if rc != 0:
        return rc

    print("\n==> Done. Next steps:")
    print(f"      python -m app.cli download {' '.join(selected)}")
    print("      python -m scripts.populate_voices fleurs --languages en,ko --per-gender 2")
    print("      python -m app.cli run --port 8000")
    return 0


def _install_backends(names: list[str], use_uv: bool) -> int:
    recipes = _backend_recipes()
    unknown = [n for n in names if n not in recipes]
    if unknown:
        print(f"Unknown backend(s): {unknown}", file=sys.stderr)
        print(f"Known: {list(recipes)}", file=sys.stderr)
        return 2
    for name in names:
        recipe = recipes[name]
        if not recipe:
            print(f"==> {name}: no extra packages required")
            continue
        print(f"==> {name}: installing {len(recipe)} package(s)")
        for mode, pkg in recipe:
            extra = ["--no-deps"] if mode == "install-no-deps" else []
            rc = _pip_install([*extra, pkg], use_uv=use_uv)
            if rc != 0:
                return rc
    return 0


def cmd_setup_backends(args: argparse.Namespace) -> int:
    names = args.backends or list(_backend_recipes())
    return _install_backends(names, use_uv=not args.no_uv)


def cmd_run(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    if args.only:
        env["TTSD_ONLY"] = args.only
    cmd = [
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", args.host, "--port", str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, env=env)


def cmd_download(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "-m", "scripts.download_model", *args.passthrough]
    return _run(cmd)


def cmd_populate(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "-m", "scripts.populate_voices", *args.passthrough]
    return _run(cmd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ttsd")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ids = list(_backend_recipes())
    default_str = ",".join(DEFAULT_BACKENDS)

    sp = sub.add_parser("setup",
                        help="install core deps + selected per-backend packages")
    sp.add_argument("--with", dest="with_", default=None,
                    help=f"comma-separated backend ids or 'all' "
                         f"(known: {', '.join(ids)}; default: {default_str})")
    sp.add_argument("--no-uv", action="store_true",
                    help="use pip even if uv is on PATH")
    sp.add_argument("--yes", "-y", action="store_true",
                    help="skip confirmation; pick defaults when non-TTY")
    sp.set_defaults(func=cmd_setup)

    bp = sub.add_parser("setup-backends",
                        help="install per-backend Python packages only")
    bp.add_argument("backends", nargs="*",
                    help=f"backend ids (default: all of {ids})")
    bp.add_argument("--no-uv", action="store_true")
    bp.set_defaults(func=cmd_setup_backends)

    rp = sub.add_parser("run", help="start the TTS server")
    rp.add_argument("--host", default="127.0.0.1")
    rp.add_argument("--port", type=int, default=8000)
    rp.add_argument("--reload", action="store_true")
    rp.add_argument("--only", default=None,
                    help="comma-separated backend ids to expose (filters /v1/models)")
    rp.set_defaults(func=cmd_run)

    dp = sub.add_parser("download", help="download model weights")
    dp.add_argument("passthrough", nargs=argparse.REMAINDER,
                    help="forwarded to scripts.download_model")
    dp.set_defaults(func=cmd_download)

    pp = sub.add_parser("populate", help="populate voices/")
    pp.add_argument("passthrough", nargs=argparse.REMAINDER,
                    help="forwarded to scripts.populate_voices")
    pp.set_defaults(func=cmd_populate)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
