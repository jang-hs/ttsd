#!/usr/bin/env bash
# One-shot setup for the self-hosted local voice API.
#
# Creates the venv, installs deps, and overlays mlx_audio from the OpenVox app
# bundle. PyPI's mlx-audio 0.2.10 is the same version but ships only a subset
# of TTS modules; the bundle adds omnivoice, qwen3_tts, higgs_audio codec, and
# newer base/utils. We re-target the bundled `_librosa_lite` to the standard
# `librosa` so nothing depends on OpenVox-private compiled shims.
#
# All overlaid code is Apache-2.0 mlx-audio source (github.com/Blaizzy/mlx-audio).
set -euo pipefail
cd "$(dirname "$0")"

APP_SP="/Applications/OpenVox – Local Voice AI.app/Contents/Resources/site-packages/mlx_audio"
PY=.venv/bin/python

echo "==> Creating venv"
python3 -m venv .venv
$PY -m pip install --quiet --upgrade pip

echo "==> Installing dependencies (uv preferred; falls back to pip)"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PY" -r requirements.txt
else
  $PY -m pip install -r requirements.txt
fi

echo "==> Overlaying full mlx_audio from the OpenVox bundle"
DST="$($PY -c 'import mlx_audio, os; print(os.path.dirname(mlx_audio.__file__))')"
if [ ! -d "$APP_SP" ]; then
  echo "ERROR: OpenVox app bundle not found at:"
  echo "  $APP_SP"
  echo "Install OpenVox first (just for the bundled mlx-audio modules) or set"
  echo "APP_SP to a checkout of mlx-audio that includes omnivoice/qwen3_tts/higgs_audio."
  exit 1
fi
rm -rf "$DST"
cp -R "$APP_SP" "$DST"
find "$DST" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "==> Patching _librosa_lite -> librosa (so we don't need OpenVox's private shim)"
grep -rl "import _librosa_lite as librosa" "$DST" | while read -r f; do
  sed -i '' 's/import _librosa_lite as librosa/import librosa/' "$f"
done

echo "==> Verifying imports for all 5 TTS modules + Higgs codec"
$PY - <<'PY'
from mlx_audio.tts.models.omnivoice import Model as OmniVoice
from mlx_audio.tts.models.kokoro import Model as Kokoro
from mlx_audio.tts.models.chatterbox import Model as ChatterboxML
from mlx_audio.tts.models.chatterbox_turbo import Model as ChatterboxTurbo
from mlx_audio.tts.models.qwen3_tts import Model as Qwen3TTS
from mlx_audio.codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer
print("OK: 5 TTS models + Higgs codec import cleanly")
PY

echo "==> Done."
echo "Next:"
echo "  ./scripts/download_model.py             # fetch model weights (~5 GB for the default set)"
echo "  ./scripts/populate_voices.py fleurs --languages en,ko --per-gender 2   # or migrate / dir"
echo "  ./run.sh                               # serve on 127.0.0.1:8000"
