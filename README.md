# hl-stt-services

**Status: POC + benchmark harness.** A self-hosted, fully offline speech-to-text service
(English + Japanese) for a CPU-only homelab (16 GB RAM, no GPU). This repo is the **proof of
concept and benchmark harness** used to pick the model/engine before the "true services" are
built later.

## Why

- Run ASR completely offline (no cloud, no API calls).
- Benchmark candidate models on real-time factor (RTF), words/sec, streaming latency, and WER.
- Record audio cases **live in the UI**, run them through each model, and store every run in
  SQLite so results can be analyzed later.

## Stack

- **Backend:** Python 3.12 + FastAPI + uvicorn + sherpa-onnx (SenseVoice, Qwen3-ASR, Whisper).
- **Frontend:** Vite + React + TypeScript, built to static files.
- **Persistence:** SQLite (stdlib `sqlite3`), no external DB service.
- **Deploy:** ONE Docker container — uvicorn serves both the API and the built SPA — behind
  Caddy at `stt.home.arpa` (internal TLS).

## Layout

```
backend/            FastAPI app + sherpa-onnx loaders + SQLite
  app/
    config.py         env-only settings
    db.py             thread-safe sqlite3, `runs` table
    audio.py          WAV -> float32 mono (stdlib, no numpy)
    metrics.py        words / chars / text_hash / RTF
    models.py         sherpa-onnx registry: availability + lazy load
    schemas.py        pydantic response models
    static.py         serves the built SPA as a 404 fallback
    dataset.py        archives every incoming clip as training data
    routes/           models · transcribe · asr · runs · stream (WS)
    main.py           app wiring + uvicorn runner
  scripts/          one-off maintenance (pronunciation ground-truth migration)
  tests/            pytest (decode tests skip without model weights)
frontend/           Vite + React + TS UI (record/send, stream, reports + WER)
  src/api/            TS mirrors of the backend schemas
  src/audio/          WAV encoder, mic recorder, live PCM capture
  src/lib/wer.ts      word error rate
  src/tabs/           Record & Send · Stream · Reports
Dockerfile          multi-stage: node builds the SPA, python runs everything
docker-compose.yml  single service, bind-mounted models + db, healthcheck
Caddyfile           template for the homelab reverse proxy (tls internal)
storage/            host volumes, git-ignored — models/, db/, dataset/
```

## API

| endpoint | what it is for |
|---|---|
| `GET /models` | which recognizers this host can run |
| `POST /transcribe?model=` | benchmark a WAV clip from the UI; full metric surface |
| `POST /asr?model=` | one-shot transcription for the Hermes voice-note gateway |
| `GET /runs` | recent runs (`mode` = `batch` / `stream` / `asr`) |
| `PATCH /runs/{id}` | correct a run's ground truth |
| `WS /stream?model=` | live capture with partials |

`POST /asr` is the service endpoint: Hermes posts a voice note, gets text back.
It accepts what messaging platforms actually send — **OGG/Opus** (Telegram),
**WebM/Opus** (browsers), WAV, FLAC, MP3, MP4/M4A — defaults to `sensevoice`,
and returns `{ text, model, audio_ms, processing_ms, rtf, run_id }`. See
[backend/README.md](backend/README.md#post-asr--the-voice-note-endpoint).

## Run it

### Docker (how it actually deploys)

```bash
docker compose up --build -d
curl -fsS http://127.0.0.1:8000/health
```

Then put the extracted sherpa-onnx model directories in `./storage/models/`
(`sensevoice/`, `qwen3/`, `whisper/` — see [backend/README.md](backend/README.md)
for the exact file names) and either restart or hit `GET /models?rescan=true`.
The models are downloaded **once** to that host directory; nothing is fetched at
runtime and the image contains no weights.

Port 8000 is published on loopback only — Caddy is the front door. The container
runs non-root as uid 1000 to match `./storage`; if your host user is not 1000:

```bash
APP_UID=$(id -u) APP_GID=$(id -g) docker compose build
```

### Behind Caddy

`Caddyfile` is a template — copy the `stt.home.arpa` block into your homelab
Caddyfile, or `import` it. It sets `tls internal`, raises the upload cap, and
gives the WebSocket a long timeout (a CPU-only decode can hold the connection
open for minutes).

> **Microphone access needs a secure context.** Over plain HTTP the UI loads but
> recording silently fails on anything but `localhost`. Use the TLS hostname and
> trust Caddy's local CA on the client machines.

### Local development

Two processes, with the Vite dev server proxying the API:

```bash
# backend
cd backend && .venv/bin/python -m app.main

# frontend
cd frontend && npm install && npm run dev
```

See [backend/README.md](backend/README.md) and
[frontend/README.md](frontend/README.md) for setup, env vars and model layout.

## Tests

```bash
cd backend && .venv/bin/python -m pytest -q
cd frontend && npm run build          # tsc -b && vite build
```

The decode tests skip cleanly when no model weights are present, so the suite is
green both before and after the multi-GB downloads.

## Security

- Public repository. No credentials, API keys, private keys, or `.env` values are committed.
  Real config lives in a git-ignored `.env`; `.env.example` holds placeholders only.
- Large artifacts (models, audio recordings, SQLite DBs) are git-ignored and live on the host.
- **LAN-only by design: the API has no authentication.** Caddy's `tls internal` provides
  transport security on the homelab network, not access control. Put an auth layer in front
  before exposing this anywhere else.

## Notes

- Realtime audio test cases are recorded in the UI (with an "expected text" field) so each run
  can also report WER. Ground truth is editable per row from the Reports tab, so a recording
  becomes a labelled test case after the fact.
- **Every clip is kept.** `/transcribe` and `/asr` both archive the original upload to
  `storage/dataset/<run_id>.<ext>` and record it on the run (`audio_path`). Paired with
  `expected_text` that is a labelled corpus for fine-tuning, growing from ordinary use.
- **WER is pronunciation-aware.** The speaker's name is written "Dhira" but pronounced "Dira",
  which cost 12.5% WER on an otherwise perfect transcript. An explicit `PRONUNCIATION_EQUIV`
  map (mirrored in the backend and frontend) equates the two. It is a fixed word list, not
  fuzzy matching — genuine mis-hears like "Dila" still count as errors.
- Models are downloaded once to a host volume and kept resident in memory.
- `mode=stream` rows sum **every** decode, because sherpa-onnx has no streaming recognizer for
  these three models and the backend re-decodes the buffer on an interval. Compare **RTF** on
  `mode=batch` rows and **latency** on `mode=stream` rows — see
  [backend/README.md](backend/README.md#streaming-limitation-read-before-benchmarking-latency).
