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

- **Backend:** Python 3.12 + FastAPI + uvicorn + sherpa-onnx (Qwen3-ASR, SenseVoice, Whisper).
- **Frontend:** Vite + React + TypeScript.
- **Persistence:** SQLite (stdlib `sqlite3`), no external DB service.
- **Deploy:** ONE Docker container; Caddy reverse proxy at `stt.home.arpa` (internal TLS).

## Layout

```
backend/    FastAPI app + sherpa-onnx loaders + SQLite
frontend/   Vite + React + TS UI (model picker, record/send, stream, reports)
compose.yaml  single-container deployment
Dockerfile
```

## Security

- Public repository. No credentials, API keys, private keys, or `.env` values are committed.
  Real config lives in a git-ignored `.env`; `.env.example` holds placeholders only.
- Large artifacts (models, audio recordings, SQLite DBs) are git-ignored and live on the host.

## Notes

- Realtime audio test cases are recorded in the UI (with an "expected text" field) so each run
  can also report WER.
- Models are downloaded once to a host volume and kept resident in memory.
