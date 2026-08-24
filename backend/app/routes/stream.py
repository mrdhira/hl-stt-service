"""WS /stream — chunked transcription with partials.

LIMITATION (sherpa-onnx 1.13.6, verified by introspection)
-----------------------------------------------------------
``sherpa_onnx.OnlineRecognizer`` exposes only these factories::

    from_nemo_ctc  from_paraformer  from_t_one_ctc
    from_transducer  from_wenet_ctc  from_zipformer2_ctc

None of the three POC models has one — SenseVoice, Qwen3-ASR and Whisper are
all ``OfflineRecognizer`` (attention-encoder-decoder / non-causal) models, so
true frame-synchronous streaming is **not** available for them.

What this endpoint does instead is *pseudo-streaming*: it buffers the incoming
audio and re-decodes the whole buffer from scratch every
``STT_STREAM_PARTIAL_MS`` (default 1000 ms) *of accumulated audio*, emitting
each result as a partial, then emits one final decode when the client signals
end-of-audio. Partial text
can therefore change non-monotonically, and cost grows with utterance length
(each partial decodes the full buffer, not just the new tail).

Getting real low-latency streaming later means one of:
  * adding a streaming-capable model (e.g. a Zipformer transducer) as a fourth
    benchmark entry, decoded with ``OnlineRecognizer.from_transducer``; or
  * front-ending these offline models with ``sherpa_onnx.VoiceActivityDetector``
    (needs a ``silero_vad.onnx`` in the models dir) so each speech segment is
    decoded once at its endpoint rather than the buffer being re-decoded.

Both are deliberately out of scope for Phase 1.

Wire protocol
-------------
Connect to ``/stream?model=<key>&sample_rate=16000``.

  server -> ``{"type": "ready", ...}``
  client -> binary frames: raw mono PCM (16-bit LE) at ``sample_rate``.
            A frame carrying a full ``RIFF/WAVE`` header is parsed as WAV
            instead, and its own sample rate wins.
  server -> ``{"type": "partial", "text": ..., "elapsed_ms": ..., "audio_ms": ...}``
  client -> ``{"type": "eof"}`` (text frame) to request the final
  server -> ``{"type": "final", ...}`` — same fields as POST /transcribe
  server -> ``{"type": "error", "detail": ...}`` on any failure
"""

from __future__ import annotations

import json
import time

from anyio import to_thread
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from ..audio import AudioDecodeError, decode_wav, duration_ms, looks_like_wav, pcm16_to_float
from ..config import MODEL_KEYS, settings
from ..db import db
from ..metrics import count_chars, count_words, rtf, text_hash
from ..models import registry, transcribe_samples

router = APIRouter(tags=["stream"])

MODE = "stream"

STREAMING_NOTE = (
    "pseudo-streaming: sherpa-onnx 1.13.6 has no OnlineRecognizer for "
    "sensevoice/qwen3/whisper, so the buffer is re-decoded on an interval "
    "instead of decoded frame-synchronously"
)


@router.websocket("/stream")
async def stream(
    websocket: WebSocket,
    model: str = Query(..., description="sensevoice | qwen3 | whisper"),
    sample_rate: int = Query(0, description="Sample rate of raw PCM frames"),
) -> None:
    await websocket.accept()

    if model not in MODEL_KEYS:
        await _fail(websocket, f"unknown model '{model}'; expected one of {list(MODEL_KEYS)}")
        return
    if not registry.is_available(model):
        info = next(i for i in registry.availability() if i["key"] == model)
        await _fail(
            websocket,
            f"model '{model}' is not available on this host "
            f"(dir={info['directory']}, missing={info['missing_files']})",
        )
        return

    rate = sample_rate or settings.stream_sample_rate
    partial_interval_ms = float(settings.stream_partial_ms)

    await websocket.send_json(
        {
            "type": "ready",
            "model": model,
            "mode": MODE,
            "supports_true_streaming": False,
            "partial_interval_ms": settings.stream_partial_ms,
            "sample_rate": rate,
            "note": STREAMING_NOTE,
        }
    )

    samples: list[float] = []
    first_audio_at: float | None = None
    first_partial_at: float | None = None
    decode_ms_total = 0.0
    # Partials are gated on how much *audio* has arrived since the last one, not
    # on wall-clock: a client that bursts a whole clip in one go must still get
    # partials, and a slow box must not silently emit fewer of them.
    samples_at_last_partial = 0
    n_partials = 0

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                return

            chunk = message.get("bytes")
            if chunk:
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                try:
                    new_samples, rate = _chunk_to_samples(chunk, rate)
                except AudioDecodeError as exc:
                    await _fail(websocket, str(exc))
                    return
                samples.extend(new_samples)

                pending_ms = duration_ms(len(samples) - samples_at_last_partial, rate)
                if pending_ms >= partial_interval_ms:
                    text, took_ms = await _decode(model, samples, rate)
                    if text is None:
                        await _fail(websocket, took_ms)  # took_ms carries the message
                        return
                    decode_ms_total += took_ms
                    samples_at_last_partial = len(samples)
                    emitted_at = time.perf_counter()
                    n_partials += 1
                    if first_partial_at is None:
                        first_partial_at = emitted_at
                    await websocket.send_json(
                        {
                            "type": "partial",
                            "text": text,
                            "audio_ms": duration_ms(len(samples), rate),
                            "processing_ms": took_ms,
                            "elapsed_ms": (emitted_at - first_audio_at) * 1000.0,
                        }
                    )
                continue

            text_frame = message.get("text")
            if text_frame is None:
                continue

            control = _parse_control(text_frame)
            kind = control.get("type")

            if kind in ("eof", "final", "close"):
                eof_at = time.perf_counter()
                if not samples:
                    await _fail(websocket, "no audio received before eof")
                    return

                text, took_ms = await _decode(model, samples, rate)
                if text is None:
                    await _fail(websocket, took_ms)
                    return
                decode_ms_total += took_ms
                final_at = time.perf_counter()

                audio_len_ms = duration_ms(len(samples), rate)
                latency_partial_ms = (
                    (first_partial_at - first_audio_at) * 1000.0
                    if first_partial_at is not None and first_audio_at is not None
                    else None
                )
                latency_final_ms = (final_at - eof_at) * 1000.0

                run_id = db.insert_run(
                    model=model,
                    mode=MODE,
                    audio_ms=audio_len_ms,
                    processing_ms=decode_ms_total,
                    rtf=rtf(decode_ms_total, audio_len_ms),
                    words=count_words(text),
                    chars=count_chars(text),
                    latency_partial_ms=latency_partial_ms,
                    latency_final_ms=latency_final_ms,
                    text=text,
                    text_hash=text_hash(text),
                )
                await websocket.send_json(
                    {
                        "type": "final",
                        "model": model,
                        "mode": MODE,
                        "text": text,
                        "audio_ms": audio_len_ms,
                        # Sum of every decode (partials included) — this is what
                        # the stream actually cost, not just the last pass.
                        "processing_ms": decode_ms_total,
                        "final_decode_ms": took_ms,
                        "rtf": rtf(decode_ms_total, audio_len_ms),
                        "words": count_words(text),
                        "chars": count_chars(text),
                        "text_hash": text_hash(text),
                        "sample_rate": rate,
                        "partials_emitted": n_partials,
                        "latency_partial_ms": latency_partial_ms,
                        "latency_final_ms": latency_final_ms,
                        "run_id": run_id,
                    }
                )
                await websocket.close()
                return

            if kind == "reset":
                samples = []
                first_audio_at = first_partial_at = None
                decode_ms_total = 0.0
                samples_at_last_partial = 0
                n_partials = 0
                await websocket.send_json({"type": "reset"})
                continue

            await _fail(websocket, f"unknown control message: {text_frame[:200]!r}")
            return

    except WebSocketDisconnect:
        return


# --- helpers ----------------------------------------------------------------
def _parse_control(frame: str) -> dict:
    try:
        parsed = json.loads(frame)
    except json.JSONDecodeError:
        # Tolerate a bare "eof" sentinel from simple clients.
        return {"type": frame.strip().lower()}
    return parsed if isinstance(parsed, dict) else {"type": str(parsed)}


def _chunk_to_samples(chunk: bytes, rate: int) -> tuple[list[float], int]:
    """Raw PCM16 by default; a self-describing WAV frame carries its own rate."""
    if looks_like_wav(chunk):
        decoded, wav_rate = decode_wav(chunk)
        return decoded, wav_rate
    if len(chunk) % 2:
        chunk = chunk[:-1]  # drop a straddling byte rather than corrupting the frame
    return pcm16_to_float(chunk), rate


async def _decode(model: str, samples: list[float], rate: int):
    """Decode in a worker thread. Returns ``(text, ms)`` or ``(None, message)``."""
    started = time.perf_counter()
    try:
        # Copy: the buffer keeps growing while the worker thread reads it.
        text = await to_thread.run_sync(transcribe_samples, model, list(samples), rate)
    except RuntimeError as exc:
        return None, str(exc)
    return text, (time.perf_counter() - started) * 1000.0


async def _fail(websocket: WebSocket, detail: str) -> None:
    try:
        await websocket.send_json({"type": "error", "detail": detail})
        await websocket.close(code=1011)
    except (RuntimeError, WebSocketDisconnect):  # pragma: no cover - client already gone
        pass
