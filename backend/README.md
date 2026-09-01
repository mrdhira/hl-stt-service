# hl-stt-services — backend

FastAPI + sherpa-onnx. Three offline recognizers (SenseVoice, Qwen3-ASR, Whisper),
one SQLite table of benchmark runs. CPU-only, fully offline at request time.

## Layout

```
backend/
  app/
    config.py     env-only settings (stdlib os)
    db.py         thread-safe sqlite3, `runs` table
    audio.py      WAV -> float32 mono, stdlib only (no numpy)
    metrics.py    words / chars / text_hash / RTF
    models.py     sherpa-onnx registry: availability + lazy load
    schemas.py    pydantic response models
    routes/       models.py, transcribe.py, runs.py, stream.py
    main.py       app wiring + uvicorn runner
  tests/          pytest; decode tests skip when no weights are present
```

## Setup

Needs Python 3.12.

```bash
python3.12 -m venv backend/.venv          # or: uv venv backend/.venv --python 3.12
backend/.venv/bin/pip install -r backend/requirements-dev.txt
```

Config comes from the environment (see `backend/.env.example`; copy to
`backend/.env`, which is git-ignored). Nothing here is a secret — they are
tuning knobs.

| var | default | meaning |
|---|---|---|
| `STT_HOST` | `0.0.0.0` | bind address |
| `STT_PORT` | `8000` | bind port |
| `STT_MODELS_DIR` | `/data/models` | root of the model directories |
| `STT_NUM_THREADS` | `4` | onnxruntime threads per recognizer |
| `STT_DB_PATH` | `/data/stt-runs.db` | SQLite file holding `runs` (compose sets `/data/db/stt-runs.db`) |
| `STT_STATIC_DIR` | `<repo>/frontend/dist` | built SPA to serve; unset/absent = API only |
| `STT_DATASET_DIR` | `/data/dataset` | where incoming clips are archived as training data |
| `STT_PROVIDER` | `cpu` | onnxruntime provider |
| `STT_SENSEVOICE_LANGUAGE` | `` (auto) | `zh`/`en`/`ja`/`ko`/`yue`/empty |
| `STT_SENSEVOICE_USE_ITN` | `1` | inverse text normalisation + punctuation |
| `STT_WHISPER_LANGUAGE` | `` (auto) | e.g. `en`, `ja` |
| `STT_WHISPER_TASK` | `transcribe` | `transcribe` or `translate` |
| `STT_STREAM_PARTIAL_MS` | `1000` | audio accumulated between WS partials |
| `STT_STREAM_SAMPLE_RATE` | `16000` | assumed rate for raw-PCM WS frames |
| `STT_BACKEND_SENSEVOICE` / `STT_BACKEND_QWEN` / `STT_BACKEND_WHISPER` | model key | override the directory name under `STT_MODELS_DIR` |

Run:

```bash
STT_MODELS_DIR=/data/models STT_DB_PATH=/data/stt-runs.db \
  backend/.venv/bin/python -m app.main      # from inside backend/
```

Test:

```bash
cd backend && .venv/bin/python -m pytest -q
```

## Run with Docker

The whole POC is one container: uvicorn serves the API *and* the built SPA, so
there is no separate web server and no CORS hop. Build and run from the repo
root:

```bash
docker compose up --build -d
docker compose logs -f stt
curl -fsS http://172.18.0.1:8000/health   # or just: docker compose ps
```

The image is multi-stage — `node:22-alpine` builds `frontend/dist`, then
`python:3.12-slim` installs `backend/requirements.txt` and copies in the backend
plus the built SPA. sherpa-onnx ships manylinux wheels, so the runtime stage
needs no compiler and no extra shared libraries (the only apt package is `curl`,
for the compose healthcheck).

Inside the image the layout is `/app/backend` (working dir) and
`/app/frontend/dist`, and these are baked in as defaults:

| var | value in the image |
|---|---|
| `STT_MODELS_DIR` | `/data/models` |
| `STT_DB_PATH` | `/data/db/stt-runs.db` |
| `STT_STATIC_DIR` | `/app/frontend/dist` |
| `STT_DATASET_DIR` | `/data/dataset` |
| `STT_NUM_THREADS` | `4` |

`docker-compose.yml` bind-mounts `./storage/models`, `./storage/db` and
`./storage/dataset` onto `/data/models`, `/data/db` and `/data/dataset`. **Put the extracted model directories in
`./storage/models/`** — same layout as below — and the SQLite history survives
`docker compose down`.

The container runs as non-root (uid **1000** by default, matching a typical homelab
host user; override with `APP_UID=$(id -u)` at build time). Bind-mounted host directories
keep their host ownership, so if the container cannot write the database:

```bash
sudo chown -R 1000:1000 storage/
```

Port 8000 is bound to the **docker gateway** (`172.18.0.1`), not to the LAN and
not to `0.0.0.0`: Caddy reaches it over the proxy network, but raw port 8000 is
closed on the host's LAN address. `stt.home.arpa` via Caddy is the only front
door. See `../Caddyfile`.

> Microphone capture needs a secure context. Over plain HTTP the UI loads but
> **recording silently fails** on anything other than `localhost` — use Caddy's
> `tls internal` and trust its CA on the client machines.

## Where the models live

Each model gets its own directory under `STT_MODELS_DIR`. The directory name
defaults to the model key (`sensevoice`, `qwen3`, `whisper`) and can be pointed
elsewhere with the `STT_BACKEND_*` vars — so the released tarball name works
as-is, e.g. `STT_BACKEND_SENSEVOICE=sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`.

**Nothing is downloaded by the app.** Fetch the tarballs once to the host volume
(`k2-fsa/sherpa-onnx` releases on GitHub / the sherpa-onnx docs), extract, and
restart — or call `GET /models?rescan=true` to re-stat without a restart.
A model whose files are absent is reported `available: false` and its endpoints
return 503; the server still starts.

```
$STT_MODELS_DIR/
  sensevoice/
    model.int8.onnx          # or model.onnx; int8 is preferred (228 MB vs 894 MB)
    tokens.txt
  qwen3/
    conv_frontend.onnx
    encoder.int8.onnx        # or encoder.onnx
    decoder.int8.onnx        # or decoder.onnx
    tokenizer/               # directory: vocab.json + merges.txt + tokenizer_config.json
  whisper/
    <size>-encoder.int8.onnx # e.g. base-encoder.int8.onnx; globbed, prefix-agnostic
    <size>-decoder.int8.onnx
    <size>-tokens.txt
```

File resolution is glob-based and prefers `int8` where both exist, so an
extracted release directory generally works unchanged.

## sherpa-onnx constructors

Resolved against **sherpa-onnx 1.13.6** by introspecting the installed package
(`inspect.signature`), not from memory:

```python
OfflineRecognizer.from_sense_voice(model, tokens, num_threads=1, sample_rate=16000,
    feature_dim=80, decoding_method='greedy_search', debug=False, provider='cpu',
    language='', use_itn=False, ...)

OfflineRecognizer.from_qwen3_asr(conv_frontend, encoder, decoder, tokenizer,
    num_threads=1, sample_rate=16000, feature_dim=128, decoding_method='greedy_search',
    debug=False, provider='cpu', max_total_len=512, max_new_tokens=128, ...)

OfflineRecognizer.from_whisper(encoder, decoder, tokens, language='en',
    task='transcribe', num_threads=1, decoding_method='greedy_search', debug=False,
    provider='cpu', ...)
```

Decoding is `recognizer.create_stream()` → `stream.accept_waveform(sample_rate, samples)`
→ `recognizer.decode_stream(stream)` → `stream.result.text`. sherpa resamples
internally, so the API accepts any input rate.

## API

| endpoint | notes |
|---|---|
| `GET /health` | liveness + which models loaded |
| `GET /models` | availability of all three; `?rescan=true` re-stats the dir |
| `POST /transcribe?model=<key>` | multipart field `audio` (WAV) + optional field `expected_text`. Returns text + `audio_ms`, `processing_ms`, `rtf`, `words`, `chars`, `text_hash`; persists a `runs` row (`mode=batch`) |
| `POST /asr?model=<key>` | one-shot transcription for the Hermes voice-note gateway; accepts compressed audio; persists a `runs` row (`mode=asr`) |
| `GET /runs?limit=&model=&mode=` | recent rows, newest first (`mode` is `batch`, `stream` or `asr`) |
| `PATCH /runs/{id}` | correct a run's `expected_text` after the fact |
| `WS /stream?model=<key>&sample_rate=` | binary raw PCM16 frames (or a WAV frame), text `{"type":"eof", "expected_text": "..."}` to finish. Emits `partial`s then a `final`; persists a `runs` row (`mode=stream`) |

Anything that is *not* one of those paths is handled by the SPA fallback: a real
file under `STT_STATIC_DIR` is served verbatim, and any other GET returns
`index.html` so a browser refresh on a client-side route works. It runs as a
404 fallback rather than a mount at `/`, which keeps API routing precedence
intact — `GET /transcribe` is still a 405, not a 404 — and unknown paths under
an API prefix still 404 as JSON instead of returning the HTML shell. With no
built frontend present the app boots API-only and `/` reports that.

`text_hash` is the sha256 of NFKC-normalised, whitespace-collapsed text, so the
same transcript from two models collapses to one hash.

`words` counts whitespace-separated tokens **plus** each CJK/kana character —
Japanese has no spaces, so plain splitting would report ~1 word per utterance.
The frontend's WER tokenizer mirrors this exactly, so the `words` column and the
WER denominator count the same units.

`expected_text` is optional ground truth for a clip. Supplying it turns a
recording into a reusable **test case**: it is stored on the run (nullable
column) and the Reports view scores WER against it. WER itself is *not* stored —
it is computed from `text` + `expected_text` on read, so the tokenizer can change
without a migration. Databases created before this column gain it automatically
via an `ALTER TABLE` on connect.

## `POST /asr` — the voice-note endpoint

`/transcribe` is the benchmark harness: driven by the UI, WAV only, returns the
full metric surface. `/asr` is the service endpoint that Hermes (the agent
gateway) posts voice notes to. Same recognizers, different contract.

```
POST /asr?model=sensevoice
Content-Type: multipart/form-data

  audio          the clip (required)
  expected_text  optional ground truth, if the caller already knows it
```

`model` defaults to **`sensevoice`** — best quality of the three here, and it
covers English + Japanese in one model with no language hint. Override with
`?model=qwen3` or `?model=whisper`.

```json
{
  "text": "hello, my name is Dira. how are you?",
  "model": "sensevoice",
  "audio_ms": 5288.0,
  "processing_ms": 2310.4,
  "rtf": 0.437,
  "run_id": 42
}
```

```bash
curl -sS -X POST 'http://stt.home.arpa/asr' -F 'audio=@voice.ogg'
```

Errors match the rest of the API: `400` unknown model or empty upload, `415`
undecodable audio (the message names the formats that work), `503` when the
model has no files on disk.

### Accepted audio

Unlike `/transcribe`, which takes WAV only, `/asr` accepts what messaging
platforms actually send:

| format | decoder | notes |
|---|---|---|
| WAV | stdlib `wave` | no dependencies, no subprocess |
| OGG/Opus | `soundfile` | **Telegram voice notes** |
| OGG/Vorbis, FLAC, MP3 | `soundfile` | |
| WebM/Opus | `ffmpeg` | **browser `MediaRecorder`** |
| MP4 / M4A | `ffmpeg` | |

The container has both. Outside it they are optional and degrade in a defined
order: `soundfile` is tried first (in-process), `ffmpeg` covers the rest, and
WAV always works with neither installed. libsndfile has no Matroska demuxer, so
**webm needs ffmpeg** specifically. The format is sniffed from magic bytes — a
client-supplied `Content-Type` or filename is not trusted.

Every `/asr` call lands a `runs` row tagged `mode='asr'`, so production traffic
keeps growing the corpus. Filter it out of model comparisons with
`GET /runs?mode=batch`.

## Pronunciation-aware scoring

The speaker's name is written **"Dhira"** but pronounced **"Dira"** — the h is
silent. The models transcribe what they hear and emit "Dira", so ground truth
spelled "Dhira" charged one substitution on every utterance containing the name.
Run #17 scored 12.5% WER from that single word on an otherwise perfect
transcript. That measures spelling, not recognition.

`PRONUNCIATION_EQUIV` in `app/metrics.py` maps orthographic spellings to their
spoken form and is applied to **both** sides before comparison:

```python
PRONUNCIATION_EQUIV: dict[str, str] = {
    "dhira": "dira",
}
```

`frontend/src/lib/wer.ts` holds an identical map — **change both together**, and
the tokenizer-parity check keeps them honest.

Deliberately **explicit, not fuzzy**: no edit-distance threshold, no phonetic
algorithm. Those would hide real recognition errors. Only exact whole words in
the map are equated, so "Dila" (a genuine mis-hear, run #13) still scores as an
error, and unrelated words containing the key — "dhirama", "sandhira" — are
untouched. Adding an entry is a conscious decision.

Historical rows were corrected with a one-off script, which is idempotent and
takes a timestamped backup first:

```bash
cd backend && .venv/bin/python scripts/migrate_pronunciation.py --dry-run
cd backend && .venv/bin/python scripts/migrate_pronunciation.py
```

## Captured dataset

Every clip that produces a run — from `/transcribe` **and** `/asr` — is archived
to `STT_DATASET_DIR/<run_id>.<ext>` as the **original bytes**, not a re-encode.
The extension comes from sniffing the content, so a Telegram note lands as
`.ogg` and a browser recording as `.webm`. The path is recorded on the run in
the `audio_path` column.

Paired with `expected_text` — editable per row from the Reports tab — that is a
labelled corpus for fine-tuning later, accumulating for free from ordinary use:

```
storage/dataset/41.ogg   <- audio
runs.id = 41             -> expected_text = "what was actually said"
```

Writing is **best-effort by design**: a full disk or read-only mount logs a
warning and leaves `audio_path` NULL rather than failing a good transcription.
`storage/dataset/` is git-ignored; it holds recordings of real people.

## Streaming limitation (read before benchmarking latency)

`sherpa_onnx.OnlineRecognizer` in 1.13.6 offers only `from_nemo_ctc`,
`from_paraformer`, `from_t_one_ctc`, `from_transducer`, `from_wenet_ctc` and
`from_zipformer2_ctc`. **None of the three POC models has a streaming
constructor** — SenseVoice, Qwen3-ASR and Whisper are all attention
encoder-decoder / non-causal offline models.

`WS /stream` is therefore **pseudo-streaming**: it buffers audio and re-decodes
the *whole buffer* every `STT_STREAM_PARTIAL_MS` of accumulated audio. That
means:

- partial text can change non-monotonically between emissions;
- cost grows with utterance length — each partial decodes everything so far;
- `processing_ms` / `rtf` on a `mode=stream` row is the **sum of all decodes**,
  not a single pass, so stream RTF is not comparable to batch RTF.
  Use `mode=batch` rows for the RTF comparison, and `mode=stream` rows for the
  `latency_partial_ms` / `latency_final_ms` comparison.

Real low-latency streaming later needs either a streaming-capable model added as
a fourth benchmark entry (a Zipformer transducer via `OnlineRecognizer.from_transducer`),
or `sherpa_onnx.VoiceActivityDetector` (needs `silero_vad.onnx`) front-ending these
offline models so each speech segment is decoded once at its endpoint. Both are
out of scope for Phase 1.
