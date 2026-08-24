# hl-stt-services — frontend

Vite + React + TypeScript UI for the offline STT benchmark harness. Dark, three
tabs, no UI framework and no runtime dependencies beyond React.

## Run

```bash
npm install
npm run dev          # http://localhost:5173, proxies the API to :8000
npm run build        # tsc -b && vite build  ->  dist/
npm run typecheck    # tsc only
```

The backend must be running separately:

```bash
cd backend && STT_DB_PATH=/data/stt-runs.db .venv/bin/python -m app.main
```

## Config

All via `import.meta.env` — see `.env.example`, copy to `.env.local`
(git-ignored). Nothing here is a secret; this is a LAN-only POC.

| var | default | meaning |
|---|---|---|
| `VITE_API_BASE` | `` (same origin) | API origin. Leave empty behind Caddy and in dev |
| `VITE_DEV_BACKEND` | `http://localhost:8000` | dev-proxy target, read only by `vite.config.ts` |
| `VITE_STREAM_SAMPLE_RATE` | `16000` | requested mic capture rate for the Stream tab |
| `VITE_RUNS_LIMIT` | `100` | default row count on Reports |

In dev, `vite.config.ts` proxies `/health`, `/models`, `/transcribe`, `/runs`
and `/stream` (with `ws: true`) to the backend, so the app only ever talks to
relative paths — the same code works unchanged behind Caddy in production.

## Tabs

**Record & Send** — pick a model, record from the mic, transcribe, see
RTF/words/chars/WER. Optional *expected text* is posted alongside the audio and
stored on the run, which is what turns a recording into a reusable test case.

**Stream** — live capture over `WS /stream`, showing partials as they arrive and
the final with first-partial and final latency. Expected text rides along on the
`eof` frame.

**Reports** — the `runs` table with a per-model roll-up (mean RTF, mean WER),
model/mode filters, a *test cases only* toggle, and an empty state.

## Audio pipeline

The backend accepts **WAV only** (stdlib `wave`, no ffmpeg), but browsers record
webm/opus. Rather than shipping ffmpeg.wasm, `src/audio/wav.ts` uses what the
browser already has:

```
MediaRecorder (webm/opus)
  -> AudioContext.decodeAudioData      decode the container/codec
  -> OfflineAudioContext(1, …, 16000)  downmix to mono + resample
  -> encodeWav()                       write a 16-bit PCM RIFF header
  -> POST /transcribe (multipart `audio`)
```

The Stream tab skips the container entirely: an **AudioWorklet** (inlined and
loaded from a blob URL, so there is no extra asset to serve) taps the mic on the
audio thread, converts each render quantum to little-endian int16, and pushes it
straight down the websocket. `ScriptProcessorNode` is the fallback where
AudioWorklet is unavailable.

The rate the browser actually grants is read back from `AudioContext.sampleRate`
and passed as `?sample_rate=`, so the backend resamples from the truth rather
than from what we asked for.

> Microphone access needs a secure context: `localhost` in dev, HTTPS in
> production (Caddy's internal TLS at `stt.home.arpa` covers this).

## WER

Computed client-side in `src/lib/wer.ts` — Levenshtein over tokens with
backtracking for the substitution/deletion/insertion split. It is *not* stored:
the ground truth lives on the run, so the tokenizer can change without a
migration.

Tokenisation mirrors `count_words` in `backend/app/metrics.py` — each CJK/kana
character is its own token, everything else splits on whitespace — so the
`words` column and the WER denominator count the same units. Text is
NFKC-normalised, lowercased and stripped of punctuation before comparison.
WER can exceed 100% when the hypothesis is longer than the reference; that is
correct, not a display bug.

## Reading the numbers

`mode=stream` rows sum **every** decode, partials included, because the backend
re-decodes the whole buffer on an interval (sherpa-onnx has no streaming
recognizer for these three models — see `backend/app/routes/stream.py`). So:

- compare **RTF** across `mode=batch` rows;
- compare **latency** across `mode=stream` rows;
- filter to one mode before reading the per-model roll-up.

The Reports summary says the same thing inline.
