# Local Voice API (self-hosted)

Self-hosted multilingual TTS server that exposes the same `/v1` HTTP contract
as the **OpenVox Local API** — with no daily character limit / "Daily Power"
quota. All models and runtimes are open source.

Cross-platform: runs on **macOS** (CPU/MPS), **Linux** (CPU/CUDA), and
**Windows** (CPU/CUDA). PyTorch is the inference backbone; the device is
selected automatically and can be forced with the `TTSD_DEVICE` env var.

> ## 🤖 One-shot install via an LLM
>
> Paste the following prompt into Claude Code / Codex / Cursor / any coding
> agent. Works on macOS, Linux, and Windows (Git Bash / WSL).
>
> <details><summary><b>copy prompt</b></summary>
>
> ````
> You are a coding agent. Install and run a self-hosted
> local TTS server with no daily-quota limits. The repo lives at ./ttsd
> (assume current directory).
>
> Requirements:
>   • Python 3.10–3.13 with `python3` on PATH. `uv` is optional but
>     recommended; the CLI uses it automatically when present.
>   • For GPU acceleration on Linux/Windows, install the matching PyTorch
>     CUDA wheel from https://pytorch.org BEFORE the steps below. macOS
>     Apple Silicon MPS is auto-detected.
>   • Hugging Face access for the Chatterbox weights: accept the model
>     license on https://huggingface.co/ResembleAI/chatterbox and run
>     `huggingface-cli login` (only needed if you install a chatterbox
>     backend). Skip this if you only want kokoro or qwen3-tts-medium.
>
> Steps:
>   1. cd ttsd
>   2. ./setup.sh --with qwen3-tts-medium --yes      # core + Qwen3-TTS (default)
>      # other choices: --with kokoro,qwen3-tts-medium  /  --with all
>   3. python -m app.cli download                    # weights for installed backends only
>   4. python -m scripts.populate_voices fleurs \
>        --languages en,ko,ja,zh --per-gender 2      # ~8 voices per language (CC-BY-4.0)
>   5. ./run.sh --port 8001 &                        # http://127.0.0.1:8001
>   6. Verify with:
>        curl -s http://127.0.0.1:8001/v1/health
>        curl -s http://127.0.0.1:8001/v1/models | jq
>        MODEL=qwen3-tts-medium
>        VID=$(curl -s "http://127.0.0.1:8001/v1/models/$MODEL/voices?language=en" \
>              | jq -r '.data[0].id')
>        curl -s http://127.0.0.1:8001/v1/audio/speech \
>          -H 'Content-Type: application/json' \
>          -d "{\"model\":\"$MODEL\",\"input\":\"hello from my local server\",
>               \"language\":\"en\",\"voice\":\"$VID\"}" --output /tmp/hi.wav
>
> Diagnose and fix any failure inline. Report the final /v1/models output
> and the size of /tmp/hi.wav so I can confirm it worked.
> ````
>
> </details>
>
> Adjust `--with` (which backends), `--languages` and `--per-gender` to taste.
> Windows users without a POSIX shell: swap `./setup.sh` for
> `python -m app.cli setup` and `./run.sh` for `python -m app.cli run`.

## What it serves

Same routes, same JSON shapes as OpenVox. Only the models whose weights are
present locally are registered, so `/v1/models` reflects reality.

| Model id | Upstream weights | Languages | Voice handling |
|---|---|---|---|
| `kokoro` | `hexgrad/Kokoro-82M` | 9 | fixed voices (ship with the model) |
| `chatterbox-turbo-large` | `ResembleAI/chatterbox` (English) | 1 (en) | zero-shot cloning |
| `chatterbox-multilingual-medium` | `ResembleAI/chatterbox` (multilingual files) | 25 | zero-shot cloning |
| `qwen3-tts-medium` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 10 | 9 built-in speakers, auto-picked by (lang, gender) |

Voice counts per model depend on which voices you've populated — see
"Providing voices" below.

Routes:

| Method | Path | Notes |
|--------|------|-------|
| GET  | `/v1/health` | `{"object":"health","status":"ok"}` |
| GET  | `/v1/models` | one entry per registered backend |
| POST | `/v1/models/{model}/load` | warm the weights |
| GET  | `/v1/models/{model}/languages` | backend-specific |
| GET  | `/v1/models/{model}/voices?language=` | backend-specific |
| POST | `/v1/audio/speech` | full WAV bytes (`audio/wav`) |
| POST | `/v1/audio/speech` + `"stream":true` | SSE: `response.created` → `audio.chunk` (base64 WAV) → `response.completed` |

Only one generation/preload runs at a time; concurrent requests get **HTTP 429**
(matching OpenVox).

## Requirements

- Python 3.10 – 3.13 (3.13 recommended).
- ~3–10 GB of disk per model.
- For GPU acceleration on Linux/Windows, install the matching PyTorch CUDA
  wheel from https://pytorch.org *before* `pip install -r requirements.txt`.
  On Apple Silicon, MPS is detected automatically.
- A Hugging Face account is needed for the Chatterbox weights (the
  `ResembleAI/chatterbox` repo is license-gated). Run `huggingface-cli login`
  once after accepting the model license on the repo page.
- [`uv`](https://github.com/astral-sh/uv) is optional but recommended — the
  CLI uses it automatically when present, otherwise falls back to `pip`.

## Setup

### macOS / Linux / Windows (Git Bash or WSL)

`setup.sh` / `run.sh` detect the venv layout automatically (`.venv/bin/` on
POSIX, `.venv/Scripts/` on Windows), so the same scripts work on all three
OSes as long as you have a POSIX shell. Git Bash ships with Git for Windows.

```bash
./setup.sh              # interactive — pick which backends to install
./setup.sh --with kokoro,qwen3-tts-medium --yes   # non-interactive
./setup.sh --with all --yes                       # install every backend
```

The interactive prompt looks like:

```
==> ttsd setup

Choose which TTS backends to install:

  1) kokoro                          Kokoro
  2) chatterbox-turbo-large          Chatterbox Turbo (Large)
  3) chatterbox-multilingual-medium  Chatterbox Multilingual (Medium)
  4) qwen3-tts-medium                Qwen3 TTS (Medium)  (default)

Enter comma-separated numbers or ids, 'all', or just Enter for default.
>
```

Default is `qwen3-tts-medium` (covers 10 languages with built-in speakers).
You can also preselect via the `TTSD_BACKENDS=qwen3-tts-medium,kokoro` env
var, which is useful in Docker / Make / CI contexts.

After setup, download weights + voices + run the server:

```bash
python -m app.cli download                                              # downloads weights for installed backends only
python -m scripts.populate_voices fleurs --languages en,ko --per-gender 2
./run.sh                                                                # http://127.0.0.1:8000
```

### Windows (cmd.exe / PowerShell, no POSIX shell)

If you don't want to install Git Bash / WSL, run the same steps directly
with the Windows-native commands:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m app.cli setup
.\.venv\Scripts\python.exe -m app.cli download
.\.venv\Scripts\python.exe -m scripts.populate_voices fleurs --languages en,ko --per-gender 2
.\.venv\Scripts\python.exe -m app.cli run --port 8000
```

### Manual

The shell scripts only create a venv and call `python -m app.cli` — you can
do the same by hand:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m app.cli setup-backends qwen3-tts-medium
python -m app.cli download qwen3-tts-medium
python -m scripts.populate_voices fleurs --languages en,ko --per-gender 2
python -m app.cli run --port 8000
```

The CLI commands:

| Command | Purpose |
|---|---|
| `python -m app.cli setup [--with ids] [--yes]` | install core + selected backend packages |
| `python -m app.cli setup-backends [ids ...]` | install per-backend packages only |
| `python -m app.cli download [ids ...]` | download HF weights into `models/` |
| `python -m app.cli populate <fleurs|dir> …` | run `scripts.populate_voices` |
| `python -m app.cli run [--port N] [--only ids]` | start uvicorn |
| `python -m scripts.download_model --list` | list known backends + their HF repos |

## Providing voices

The cloning backends (Chatterbox + Chatterbox Multilingual) need a `voices/`
directory + `voice_catalog.json`. Two ways to fill them:

### A. Download from a public dataset (Google FLEURS)

FLEURS (Google, CC-BY-4.0, 102 languages, per-clip gender labels) is **not
gated** — no Hugging Face login required.

```bash
python -m scripts.populate_voices fleurs --languages en,ko,ja,zh --per-gender 2
python -m scripts.populate_voices fleurs --languages en --per-gender 5 --split test
```

For each language you get `per-gender × 2` voices (male + female), each a
3–15 s clip with the transcript saved as `.qwen.txt`. Re-run with more
languages anytime — existing entries are skipped. Pass either short ISO
codes (`en`, `ko`, `ja`) or full FLEURS codes (`en_us`, `ko_kr`,
`cmn_hans_cn`).

> **Common Voice is currently broken on Hugging Face**: Mozilla's repo uses
> a legacy loading script that newer `datasets` versions refuse to run. Use
> FLEURS instead until that's resolved.

### B. Bring your own clips

Organise a directory like the catalog expects, then ingest it:

```
my_clips/
├── English/
│   ├── Female/
│   │   ├── Alice.wav
│   │   └── Alice.wav.qwen.txt       # optional transcript
│   └── Male/
│       └── Bob.mp3
└── Korean/
    └── Female/
        └── Sora.wav
```

```bash
python -m scripts.populate_voices dir --path ./my_clips
```

Clips become voices keyed `<Language>-<Gender>-<Name>`. Transcripts are
optional but improve cloning quality on backends that accept them.

### Manual procedure (no scripts)

1. Drop audio files into `voices/<Language>/<Gender>/<Name>.<ext>` (mp3,
   wav, flac, ogg, m4a all work).
2. (Optional) Drop a transcript next to the audio as `<file>.qwen.txt`
   (or `<file>.txt`).
3. Write a catalog row in `voice_catalog.json`:
   ```json
   {
     "id": "English-Female-Alice",
     "name": "Alice",
     "language": "English",
     "language_code": "en",
     "gender": "Female",
     "age": "",
     "tags": []
   }
   ```
4. Rebuild the runtime manifest:
   ```bash
   python -m scripts.build_manifest          # voice_catalog.json + voices/ -> voices.manifest.jsonl
   python -m scripts.build_manifest --strict # fail when any catalog entry has no audio
   ```
5. Restart the server.

## Adding a new TTS model

The backend registry is **self-describing**. To add a new model:

1. Create `app/backends/<my_model>.py` with a class that implements the
   `TTSBackend` Protocol (see `app/backends/base.py`) and the extension
   metadata used by the CLI + downloader:

   ```python
   from .. import config
   from .._device import pick_device
   from ..voices import Catalog, Voice
   from ._common import resample_if_needed, to_mono_float32

   class MyModelBackend:
       id = "my-model"
       display_name = "My Model"
       model_key = "my_model"
       voice_model_label = "my_model"
       supports_streaming = True
       sample_rate = config.SAMPLE_RATE

       # ---- Extension metadata
       weights_dir = "my-model-pt"
       weights_repos = (
           ("some-org/My-Model", "my-model-pt", None),
       )
       pip_install = (
           ("install", "my-model-tts>=0.1"),
       )

       def __init__(self): ...
       def is_loaded(self) -> bool: ...
       def load(self) -> None: ...
       def synth(self, text, language, voice) -> np.ndarray: ...
   ```

2. Append it to `ALL_BACKENDS` in `app/backends/__init__.py`:

   ```python
   from .my_model import MyModelBackend
   ALL_BACKENDS = (..., MyModelBackend)
   ```

Done. `app.registry` registers it automatically when its weights are present;
`app.cli` exposes it in the interactive prompt and `--with`; the downloader
discovers its repo via `weights_repos`; no edits required in `app/main.py`,
`scripts/download_model.py`, or anywhere else.

## Running

```bash
./run.sh                                   # macOS/Linux, binds 127.0.0.1:8000
.\run.sh                                   # Windows (Git Bash)
./run.sh --port 8001                       # different port (OpenVox uses 8000 by default)
python -m app.cli run --port 8001 --reload # dev autoreload
python -m app.cli run --only kokoro        # restrict /v1/models to a subset
TTSD_ONLY=kokoro,qwen3-tts-medium ./run.sh # same, via env
```

Example request (replace `<voice-id>` with one you actually have — see
`GET /v1/models/<model>/voices?language=en`):

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-medium","input":"Hello from my own server.",
       "language":"en","voice":"<voice-id>","response_format":"wav"}' \
  --output speech.wav
```

## Browser / Obsidian clients (CORS)

CORS is enabled for all origins, so browser-based callers (Obsidian's
`app://obsidian.md`, Electron apps, web pages) can call the API directly —
including the preflight `OPTIONS` request. The exposed response headers
include `X-OpenVox-Model` and `X-OpenVox-Voice` so client code can read which
backend / voice handled the request.

## How it works

- **Weights**: live in `models/`. `python -m app.cli download` fetches them
  from the upstream originator orgs on Hugging Face (hexgrad, Resemble AI,
  Qwen). `models/` is git-ignored.
- **Voices**: `voices.manifest.jsonl` is the runtime index, built from
  `voice_catalog.json` (metadata) joined with
  `voices/<Language>/<Gender>/<Name>.<ext>` (reference clip) and
  `<file>.qwen.txt` (transcript). For cloning backends, synthesis is
  zero-shot from the reference clip.
- **Lazy load**: every backend stays on disk until its first request. The
  HTTP layer only registers backends whose weights are present; even
  registered backends don't touch RAM/VRAM until you POST to
  `/v1/audio/speech` for them.
- **Concurrency**: one generation or preload at a time; concurrent requests
  get HTTP 429.
- **Device**: PyTorch device selection is automatic (CUDA → MPS → CPU);
  override via `TTSD_DEVICE`.

## Tuning (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `TTSD_DEVICE` | auto | Force a PyTorch device (`cpu`, `cuda`, `cuda:0`, `mps`) |
| `TTSD_BACKENDS` | (none) | Setup-time preselection; comma-separated backend ids |
| `TTSD_ONLY` | (none) | Runtime filter; only listed backends register / appear in `/v1/models` |

## Licensing

Server code: open source. Models and voices follow their originators' licenses
— check each Hugging Face model card / dataset card before redistributing.

`.gitignore` excludes `voices/`, `voice_catalog.json`, and
`voices.manifest.jsonl`.
