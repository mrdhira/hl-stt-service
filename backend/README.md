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
| `STT_DB_PATH` | `/data/stt-runs.db` | SQLite file holding `runs` |
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
| `GET /runs?limit=&model=&mode=` | recent rows, newest first |
| `WS /stream?model=<key>&sample_rate=` | binary raw PCM16 frames (or a WAV frame), text `{"type":"eof", "expected_text": "..."}` to finish. Emits `partial`s then a `final`; persists a `runs` row (`mode=stream`) |

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
