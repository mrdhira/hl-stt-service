"""Pydantic response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Availability of one recognizer."""

    key: str = Field(description="sensevoice | qwen3 | whisper")
    display_name: str
    available: bool = Field(description="Files present and sherpa-onnx importable")
    loaded: bool = Field(description="Recognizer already constructed in memory")
    supports_streaming: bool = Field(
        description="True only if sherpa-onnx exposes an OnlineRecognizer for it"
    )
    factory: str = Field(description="sherpa-onnx constructor used for this model")
    directory: str
    missing_files: list[str] = Field(default_factory=list)
    error: str | None = None


class ModelsResponse(BaseModel):
    sherpa_onnx_available: bool
    sherpa_onnx_version: str | None = None
    models_dir: str
    num_threads: int
    models: list[ModelInfo]


class TranscribeResult(BaseModel):
    """Final text plus the benchmark metrics for one decode."""

    model: str
    mode: str = Field(description="batch | stream")
    text: str
    audio_ms: float = Field(description="Duration of the input audio")
    processing_ms: float = Field(description="Wall-clock spent decoding")
    rtf: float = Field(description="processing_ms / audio_ms; < 1.0 is faster than realtime")
    words: int
    chars: int
    text_hash: str = Field(description="sha256 of the normalised text")
    sample_rate: int
    latency_partial_ms: float | None = Field(
        default=None, description="Time to the first partial (stream mode only)"
    )
    latency_final_ms: float | None = Field(
        default=None, description="Time from end-of-audio to the final (stream mode only)"
    )
    expected_text: str | None = Field(
        default=None, description="Ground truth supplied by the caller, for WER"
    )
    audio_path: str | None = Field(
        default=None, description="Where the raw upload was archived, if it was"
    )
    run_id: int | None = Field(default=None, description="rowid of the persisted runs row")


class RunRow(BaseModel):
    """One persisted row of the `runs` table."""

    id: int
    ts: float
    model: str
    mode: str
    audio_ms: float | None = None
    processing_ms: float | None = None
    rtf: float | None = None
    words: int | None = None
    chars: int | None = None
    latency_partial_ms: float | None = None
    latency_final_ms: float | None = None
    text: str | None = None
    text_hash: str | None = None
    expected_text: str | None = None
    audio_path: str | None = None


class RunsResponse(BaseModel):
    count: int
    runs: list[RunRow]


class ExpectedTextUpdate(BaseModel):
    """Body of PATCH /runs/{id} — correcting a run's ground truth by hand."""

    expected_text: str | None = Field(
        default=None,
        description="New ground truth. Empty or omitted clears it (stored NULL).",
    )


class AsrResult(BaseModel):
    """POST /asr — one-shot transcription for the Hermes voice-note gateway.

    Deliberately smaller than `TranscribeResult`: callers want the text and
    enough timing to spot a slow model, not the full benchmark surface.
    """

    text: str
    model: str
    audio_ms: float
    processing_ms: float
    rtf: float
    run_id: int | None = None
