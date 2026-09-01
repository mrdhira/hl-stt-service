"""POST /transcribe — one-shot decode of an uploaded clip, persisted to `runs`."""

from __future__ import annotations

import time

from anyio import to_thread
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from ..audio import (
    SUPPORTED_NOTE,
    AudioDecodeError,
    AudioTooLongError,
    decode_wav,
    duration_ms,
)
from ..config import MODEL_KEYS
from ..dataset import save_clip
from ._limits import reject_oversized
from ..db import db
from ..metrics import count_chars, count_words, rtf, text_hash
from ..models import registry, transcribe_samples
from ..schemas import MAX_EXPECTED_TEXT, TranscribeResult

router = APIRouter(tags=["transcribe"])

MODE = "batch"


@router.post("/transcribe", response_model=TranscribeResult)
async def transcribe(
    model: str = Query(..., description="sensevoice | qwen3 | whisper"),
    audio: UploadFile = File(..., description=f"WAV upload ({SUPPORTED_NOTE})"),
    expected_text: str | None = Form(
        None,
        max_length=MAX_EXPECTED_TEXT,
        description="Optional ground truth for this clip; enables WER in Reports",
    ),
) -> TranscribeResult:
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

    expected = _clean_expected(expected_text)

    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    reject_oversized(raw)

    try:
        samples, sample_rate = decode_wav(raw)
    except AudioTooLongError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except AudioDecodeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    if not samples:
        raise HTTPException(status_code=400, detail="audio contains no samples")

    audio_len_ms = duration_ms(len(samples), sample_rate)

    started = time.perf_counter()
    try:
        # Decoding is CPU-bound C++; keep it off the event loop.
        text = await to_thread.run_sync(transcribe_samples, model, samples, sample_rate)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    processing_ms = (time.perf_counter() - started) * 1000.0

    result = TranscribeResult(
        model=model,
        mode=MODE,
        text=text,
        audio_ms=audio_len_ms,
        processing_ms=processing_ms,
        rtf=rtf(processing_ms, audio_len_ms),
        words=count_words(text),
        chars=count_chars(text),
        text_hash=text_hash(text),
        sample_rate=sample_rate,
        expected_text=expected,
    )
    result.run_id = db.insert_run(
        model=model,
        mode=MODE,
        audio_ms=result.audio_ms,
        processing_ms=result.processing_ms,
        rtf=result.rtf,
        words=result.words,
        chars=result.chars,
        text=result.text,
        text_hash=result.text_hash,
        expected_text=result.expected_text,
    )

    # Archive the original bytes as training data. Best-effort: a failed write
    # leaves audio_path NULL rather than failing a good transcription.
    stored = await to_thread.run_sync(save_clip, result.run_id, raw)
    if stored:
        db.set_audio_path(result.run_id, stored)
        result.audio_path = stored

    return result


def _clean_expected(value: str | None) -> str | None:
    """Blank form fields arrive as "" — store NULL, not an empty string."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
