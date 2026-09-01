"""POST /asr — one-shot transcription for the Hermes voice-note gateway.

Separate from `/transcribe` on purpose. `/transcribe` is the benchmark harness:
it is driven by the UI, takes WAV that the browser has already normalised, and
returns the full metric surface. `/asr` is a service endpoint: an agent gateway
POSTs whatever the messaging platform handed it (Telegram sends OGG/Opus),
picks no model, and wants the text back.

Both land a `runs` row, so ordinary use keeps growing the benchmark corpus —
`/asr` rows are tagged `mode='asr'` and can be filtered out of model comparisons.
"""

from __future__ import annotations

import time

from anyio import to_thread
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from ..audio import (
    SUPPORTED_UPLOAD_NOTE,
    AudioDecodeError,
    AudioTooLongError,
    EmptyAudioError,
    decode_audio,
    duration_ms,
)
from ..config import MODEL_KEYS
from ..dataset import save_clip
from ._limits import reject_oversized
from ..db import db
from ..metrics import count_chars, count_words, rtf, text_hash
from ..models import registry, transcribe_samples
from ..schemas import MAX_EXPECTED_TEXT, AsrResult

router = APIRouter(tags=["asr"])

MODE = "asr"

#: SenseVoice is the default: best quality of the three on this corpus, and it
#: handles English + Japanese in one model without a language hint.
DEFAULT_MODEL = "sensevoice"


@router.post("/asr", response_model=AsrResult)
async def asr(
    model: str = Query(
        DEFAULT_MODEL, description=f"sensevoice | qwen3 | whisper (default {DEFAULT_MODEL})"
    ),
    audio: UploadFile = File(..., description=f"Audio upload ({SUPPORTED_UPLOAD_NOTE})"),
    expected_text: str | None = Form(
        None,
        max_length=MAX_EXPECTED_TEXT,
        description="Optional ground truth, if the caller already knows it",
    ),
) -> AsrResult:
    if model not in MODEL_KEYS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown model '{model}'; expected one of {list(MODEL_KEYS)}",
        )
    if not registry.is_available(model):
        info = next(i for i in registry.availability() if i["key"] == model)
        raise HTTPException(
            status_code=503,
            detail={
                "message": f"model '{model}' is not available on this host",
                "directory": info["directory"],
                "missing_files": info["missing_files"],
                "error": info["error"],
            },
        )

    raw = await audio.read()
    reject_oversized(raw)

    try:
        # Unlike /transcribe this accepts compressed containers — the caller is
        # a bot relaying a voice note, not a browser that can re-encode first.
        # decode_audio rejects an empty body itself, so there is no separate
        # emptiness check here.
        samples, sample_rate = await to_thread.run_sync(decode_audio, raw)
    except EmptyAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AudioTooLongError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except AudioDecodeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    if not samples:
        raise HTTPException(status_code=400, detail="audio contains no samples")

    audio_len_ms = duration_ms(len(samples), sample_rate)

    started = time.perf_counter()
    try:
        text = await to_thread.run_sync(transcribe_samples, model, samples, sample_rate)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    processing_ms = (time.perf_counter() - started) * 1000.0

    expected = (expected_text or "").strip() or None
    run_id = db.insert_run(
        model=model,
        mode=MODE,
        audio_ms=audio_len_ms,
        processing_ms=processing_ms,
        rtf=rtf(processing_ms, audio_len_ms),
        words=count_words(text),
        chars=count_chars(text),
        text=text,
        text_hash=text_hash(text),
        expected_text=expected,
    )

    # Archive the original bytes as training data. Best-effort: a failed write
    # leaves audio_path NULL rather than failing a good transcription.
    stored = await to_thread.run_sync(save_clip, run_id, raw)
    if stored:
        db.set_audio_path(run_id, stored)

    return AsrResult(
        text=text,
        model=model,
        audio_ms=audio_len_ms,
        processing_ms=processing_ms,
        rtf=rtf(processing_ms, audio_len_ms),
        run_id=run_id,
    )
