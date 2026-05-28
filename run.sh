#!/usr/bin/env sh
# Launch the TTS server. Works on macOS, Linux, and Windows (via Git Bash / WSL).
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe
else
  echo "ERROR: could not find python inside .venv — run ./setup.sh first" >&2
  exit 1
fi

exec "$PY" -m app.cli run "$@"
