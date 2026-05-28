# Local Voice API (self-hosted, OpenVox-compatible)

Self-hosted multilingual TTS server that exposes the same `/v1` HTTP contract
as the **OpenVox Local API** — **with no daily character limit / "Daily Power"
quota**. All models and runtimes are open source.

> ## 🤖 One-shot install via an LLM
>
> Paste the following prompt into Claude Code / Codex / Cursor / any coding
> agent running on your Mac to install and start the server end-to-end.
>
> <details><summary><b>copy prompt</b></summary>
>
> ````
> You are a coding agent on an Apple Silicon Mac. Install and run a
> self-hosted, OpenVox-compatible local TTS server, with no daily-quota
> limits. The repo lives at ./local-tts-server (assume current directory).
>
> Requirements:
>   • Apple Silicon (MLX), Python 3.13, `uv` available on PATH.
>   • Use `uv pip` for installs (plain pip backtracks badly on 3.13).
>   • OpenVox app installed at the default path (setup.sh overlays its
>     bundled mlx-audio modules; the app itself is never launched).
>   • Pull model weights from Hugging Face (no OpenVox quota consumed).
>   • Use the public Google FLEURS dataset for reference voices (CC-BY-4.0,
>     no HF login required).
>
> Steps:
>   1. cd local-tts-server
>   2. ./setup.sh                                         # venv + deps + overlay mlx_audio
>   3. ./scripts/download_model.py                        # all 5 models from HF (~9 GB total)
>   4. ./scripts/populate_voices.py fleurs \
>        --languages en,ko,ja,zh --per-gender 2           # ~8 voices per language, with transcripts
>   5. ./run.sh --port 8001 &                             # avoids OpenVox on :8000
>   6. Verify with:
>        curl -s http://127.0.0.1:8001/v1/health
>        curl -s http://127.0.0.1:8001/v1/models | jq
>        VID=$(curl -s 'http://127.0.0.1:8001/v1/models/omnivoice/voices?language=en' \
>              | jq -r '.data[0].id')
>        curl -s http://127.0.0.1:8001/v1/audio/speech \
>          -H 'Content-Type: application/json' \
>          -d "{\"model\":\"omnivoice\",\"input\":\"hello from my local server\",
>               \"language\":\"en\",\"voice\":\"$VID\"}" --output /tmp/hi.wav
>
> Diagnose and fix any failure inline. Report the final /v1/models output
> and the size of /tmp/hi.wav so I can confirm it worked.
> ````
>
> </details>
>
> Adjust `--languages` and `--per-gender` to taste.

## Why this exists

OpenVox runs the same open models locally but gates usage behind a per-day
quota that its Local API decrements on every request. The underlying weights
(OmniVoice, Kokoro, Qwen3-TTS, Chatterbox) and the inference library
([`mlx-audio`](https://github.com/Blaizzy/mlx-audio)) are all open source
(Apache-2.0 / MIT), so we serve them ourselves and skip the quota.

## What it serves

Same routes, same JSON shapes as OpenVox. Only the models whose weights are
present locally are registered, so `/v1/models` reflects reality.

| Model id | Backend | Languages | Voice handling |
|---|---|---|---|
| `omnivoice` | Qwen3-0.6B + HiggsAudio (masked diffusion) | 49 | zero-shot cloning from reference clip |
| `kokoro` | fast small (82M) | 9 | fixed voices (ships with model) |
| `qwen3-tts-medium` | Qwen3-1.7B CustomVoice (8-bit) | 10 | 9 built-in speakers, auto-picked by (lang, gender) |
| `chatterbox-turbo-large` | autoregressive English (fp16) | 1 (en) | zero-shot cloning |
| `chatterbox-multilingual-medium` | autoregressive multilingual (Q8) | 25 | zero-shot cloning |

Voice counts per model depend on which voices you've populated — see
"Providing voices" below. For FLEURS at `--per-gender 2` you get roughly
`4 × (#languages-the-model-supports)` voices.

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

## Setup

Requires Apple Silicon (MLX), Python 3.13, and a local **OpenVox** install.
[`uv`](https://github.com/astral-sh/uv) is strongly recommended — plain pip
backtracks badly on this dependency set under 3.13.

> **Why OpenVox?** PyPI's `mlx-audio 0.2.10` ships only a subset of TTS
> modules. `setup.sh` overlays the full package (omnivoice, qwen3_tts,
> chatterbox, higgs_audio codec) from the OpenVox app bundle — the same
> Apache-2.0 mlx-audio source, just not yet published to PyPI. The app itself
> is not launched; only the bundled Python modules are used.
>
> `download_model.py` pulls weights from Hugging Face directly and reuses the
> OpenVox HF cache as a fast-path if present, so no quota is consumed during
> setup.

### First-time setup (4 steps)

```bash
./setup.sh                                                              # 1. venv + deps
./scripts/download_model.py                                             # 2. download model weights (~5 GB)

# 3. provide voices — pick ONE:
./scripts/populate_voices.py fleurs --languages en,ko --per-gender 2   #    A. public dataset (FLEURS, no login needed)
./scripts/populate_voices.py dir --path ./my_clips                      #    B. your own clips

./run.sh                                                                # 4. start server on :8000
```

Steps 2 and 3 each create files the next step needs:

| Step | What it creates |
|------|------|
| `scripts/download_model.py` | `models/OmniVoice-bf16/`, `models/OmniVoice/`, `models/Kokoro-82M-bf16/`, `models/chatterbox-turbo-fp16/`, `models/Qwen3-TTS-CustomVoice-8bit/`, `models/Chatterbox-Multilingual-Q8/` |
| voice step (A or B) | `voices/`, `voice_catalog.json`, `voices.manifest.jsonl` |

`scripts/download_model.py` pulls weights from Hugging Face into `models/`. Use
`./scripts/download_model.py --list` to see known model keys, or `kokoro` /
`omnivoice` / `chatterbox` to fetch one at a time.

> **You can run the server before step 3** — it just won't have cloning
> backends. Kokoro works immediately (its voices ship with the model);
> OmniVoice and Chatterbox auto-register once you've populated voices and
> restarted.

## Providing voices

OmniVoice and Chatterbox clone from a reference clip per voice, so the server
needs a `voices/` directory + `voice_catalog.json`. Two ways to fill them:

### A. Download from a public dataset (Google FLEURS)

FLEURS (Google, CC-BY-4.0, 102 languages, per-clip gender labels) is **not
gated** — no Hugging Face login required.

```bash
./scripts/populate_voices.py fleurs --languages en,ko,ja,zh --per-gender 2
./scripts/populate_voices.py fleurs --languages en --per-gender 5 --split test
```

For each language you get `per-gender × 2` voices (male + female), each a
3–15 s clip with the transcript saved as `.qwen.txt`. Re-run with more
languages anytime — existing entries are skipped. Pass either short ISO
codes (`en`, `ko`, `ja`) or full FLEURS codes (`en_us`, `ko_kr`,
`cmn_hans_cn`).

> **Common Voice is currently broken on Hugging Face**: Mozilla's repo uses
> a legacy loading script that newer `datasets` versions refuse to run. The
> `common-voice` subcommand is kept for when that's resolved, but FLEURS is
> the working path today.

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
./scripts/populate_voices.py dir --path ./my_clips
```

Clips become voices keyed `<Language>-<Gender>-<Name>`. Transcript is optional
but improves OmniVoice cloning quality. Languages are matched against an ISO
table (`English`→`en`, `Korean`→`ko`, …); unknown names fall back to a
lowercased folder name as the code.

### Manual procedure (no scripts)

If you'd rather wire things up by hand:

1. **Drop audio files** into `voices/<Language>/<Gender>/<Name>.<ext>` (mp3,
   wav, flac, ogg, m4a all work). Example:
   `voices/English/Female/Alice.wav`.
2. **(Optional) Drop a transcript** next to the audio as
   `<file>.qwen.txt` (or `<file>.txt`, or same stem `.txt`). Transcripts are
   only needed for OmniVoice quality; Chatterbox ignores them.
3. **Write a catalog row** in `voice_catalog.json` (a JSON array). One object
   per voice:
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
   The `id` must follow `Language-Gender-Name`; `language_code` is the BCP-47
   tag the backend feeds the model (e.g. `en`, `arb`, `zh`, `ja`).
4. **Rebuild the runtime manifest**:
   ```bash
   ./scripts/build_manifest.py          # joins voice_catalog.json + voices/ → voices.manifest.jsonl
   ./scripts/build_manifest.py --strict # exit non-zero if any catalog entry has no audio
   ```
5. **Restart the server.** It picks up the new manifest at startup; new voices
   appear immediately in `/v1/models/{model}/voices`.

## Adding more compatible models

mlx-audio (the underlying library) supports many other TTS architectures. To
add one:

1. Pick a model. The installed `mlx-audio` ships these modules:
   `kokoro`, `omnivoice`, `chatterbox`, `chatterbox_turbo`, `qwen3`, `qwen3_tts`,
   `bark`, `dia`, `indextts`, `outetts`, `sesame`, `soprano`, `spark`,
   `vibevoice`, `voxcpm`, `llama`. Find a matching MLX checkpoint on Hugging
   Face (e.g. `mlx-community/*`, `theoracleguy/*`).
2. Register it in `scripts/download_model.py` — add an entry to `REGISTRY`:
   ```python
   "my_model": [("mlx-community/Some-TTS-mlx-fp16", "Some-TTS-mlx-fp16")],
   ```
3. `./scripts/download_model.py my_model` to fetch it.
4. Write a small backend in `app/backends/`, modeled after `kokoro.py` (fixed
   voices) or `omnivoice.py` (cloning). Three things to provide:
   - `id / display_name / model_key / supports_streaming / sample_rate`
   - a `Catalog` of voices the API will expose
   - a `synth(text, language, voice) -> np.ndarray` method
5. Register it in `app/registry.py` next to the existing `_try_register(...)`
   lines, gated on `config.has_local_model("<dir-name>")`.

The same JSON shape is reused for every backend, so existing clients keep
working — the only thing that changes is which `id` they request.

## Running

```bash
./run.sh                  # binds 127.0.0.1:8000
./run.sh --port 8001      # OpenVox is on 8000 by default; use another port to coexist
```

Example request (replace `<voice-id>` with one you actually have — see
`GET /v1/models/omnivoice/voices?language=en`):

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"omnivoice","input":"Hello from my own server.",
       "language":"en","voice":"<voice-id>","response_format":"wav"}' \
  --output speech.wav
```

## Browser / Obsidian clients (CORS)

CORS is enabled for all origins, so browser-based callers (Obsidian's
`app://obsidian.md`, Electron apps, web pages) work — including the preflight
`OPTIONS` request. Note that **OpenVox's own server on :8000 does not send CORS
headers**, so for browser clients you must run *this* server (quit OpenVox and
use `./run.sh` on :8000, or point the client at the port this server uses).

## How it works

- **Weights**: live in `models/` (`OmniVoice-bf16` LLM + `OmniVoice` full
  encode+decode HiggsAudio tokenizer, plus optional Kokoro / Chatterbox).
  `scripts/download_model.py` fetches them from Hugging Face. `models/` is
  git-ignored (~5 GB+).
- **Voices**: `voices.manifest.jsonl` is the runtime index, built from
  `voice_catalog.json` (metadata) joined with `voices/<Language>/<Gender>/<Name>.<ext>`
  (reference clip) and `<file>.qwen.txt` (transcript). For OmniVoice and
  Chatterbox, synthesis is zero-shot cloning from the reference clip.
- **Duration**: auto-estimated from the input text (OmniVoice's rule
  estimator) — clients don't need to specify it.
- **Concurrency**: one generation or preload at a time; concurrent requests
  get HTTP 429 (matching OpenVox).

## Tuning (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `OMNIVOICE_NUM_STEPS` | `16` | diffusion steps; higher = better/slower |
| `OMNIVOICE_GUIDANCE_SCALE` | `2.0` | CFG strength |

## Licensing

The server code in this repo, the underlying models (OmniVoice, Kokoro,
Chatterbox), and the inference library (`mlx-audio`) are all open source
(Apache-2.0 / MIT). The repo does **not** include OpenVox's proprietary
application code.

Reference voice clips depend on which path you populated voices with:

| Path | Source | License |
|------|--------|---------|
| A. FLEURS | Google FLEURS via Hugging Face | CC-BY-4.0 (attribution required for redistribution) |
| B. Bring your own | whatever you supplied | your own |

`.gitignore` excludes `voices/`, `voice_catalog.json`, and
`voices.manifest.jsonl` so the repo stays clean to share regardless of path.
