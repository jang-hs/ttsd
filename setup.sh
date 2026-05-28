#!/usr/bin/env sh
# Thin POSIX wrapper: creates a venv, then delegates to app.cli for all
# OS-variable work. Works on macOS, Linux, and Windows (via Git Bash / WSL).
# Windows users without a POSIX shell can run the underlying commands
# directly: `python -m venv .venv && .venv\Scripts\python.exe -m app.cli setup`.
set -e
cd "$(dirname "$0")"

PY_BOOT="${PYTHON:-python3}"
if ! command -v "$PY_BOOT" >/dev/null 2>&1; then
  PY_BOOT=python
fi

if [ ! -d .venv ]; then
  echo "==> Creating venv (.venv)"
  "$PY_BOOT" -m venv .venv
fi

# venv layout: POSIX uses .venv/bin/python; Windows uses .venv/Scripts/python.exe.
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe
else
  echo "ERROR: could not find python inside .venv" >&2
  exit 1
fi

"$PY" -m app.cli setup "$@"
