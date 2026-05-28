#!/usr/bin/env bash
# Launch the local voice API on the same address/port the OpenVox agent expects.
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 "$@"
