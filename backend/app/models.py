"""sherpa-onnx model registry.

Three offline (non-streaming) recognizers, resolved against sherpa-onnx 1.13.6:

===========  ================================================  ===========================
key          ``sherpa_onnx.OfflineRecognizer`` factory          required files in model dir
===========  ================================================  ===========================
sensevoice   ``from_sense_voice(model, tokens, ...)``           ``model.int8.onnx`` (or
                                                                ``model.onnx``), ``tokens.txt``
qwen3        ``from_qwen3_asr(conv_frontend, encoder,``         ``conv_frontend.onnx``,
             ``decoder, tokenizer, ...)``                       ``encoder.int8.onnx``,
                                                                ``decoder.int8.onnx``,
                                                                ``tokenizer/``
whisper      ``from_whisper(encoder, decoder, tokens, ...)``    ``*-encoder*.onnx``,
                                                                ``*-decoder*.onnx``,
                                                                ``*tokens.txt``
===========  ================================================  ===========================

Nothing is loaded unless the files are actually present under
``STT_MODELS_DIR/<dir>/``; a model with missing files is reported as
unavailable rather than raising. ``sherpa_onnx`` itself is imported behind a
guard so the API (and the test suite) still runs on a box without it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import MODEL_KEYS, settings

# --- import guard -----------------------------------------------------------
try:  # pragma: no cover - depends on the host
    import sherpa_onnx  # type: ignore

    SHERPA_AVAILABLE = True
    SHERPA_IMPORT_ERROR: str | None = None
    SHERPA_VERSION: str | None = getattr(sherpa_onnx, "__version__", None)
except Exception as exc:  # pragma: no cover - only on a box without sherpa-onnx
    sherpa_onnx = None  # type: ignore[assignment]
    SHERPA_AVAILABLE = False
    SHERPA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    SHERPA_VERSION = None


def _first(directory: Path, *patterns: str) -> Path | None:
    """First file in ``directory`` matching any glob, in pattern order."""
    for pattern in patterns:
        matches = sorted(p for p in directory.glob(pattern) if p.is_file())
        if matches:
            return matches[0]
    return None


@dataclass
class ResolvedFiles:
    """What we found on disk for one model."""

    ok: bool
    paths: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


# --- per-model file resolution ----------------------------------------------
def _resolve_sensevoice(d: Path) -> ResolvedFiles:
    # int8 first: 228 MB vs 894 MB, and the size we actually want on a CPU box.
    model = _first(d, "model.int8.onnx", "model.onnx", "*.int8.onnx", "*.onnx")
    tokens = _first(d, "tokens.txt", "*tokens*.txt")
    missing = [n for n, p in (("model.onnx", model), ("tokens.txt", tokens)) if p is None]
    if missing:
        return ResolvedFiles(ok=False, missing=missing)
    return ResolvedFiles(ok=True, paths={"model": str(model), "tokens": str(tokens)})


def _resolve_qwen3(d: Path) -> ResolvedFiles:
    conv = _first(d, "conv_frontend.onnx", "*conv_frontend*.onnx")
    encoder = _first(d, "encoder.int8.onnx", "encoder.onnx", "*encoder*.onnx")
    decoder = _first(d, "decoder.int8.onnx", "decoder.onnx", "*decoder*.onnx")
    # from_qwen3_asr takes the tokenizer *directory* (vocab.json + merges.txt).
    tokenizer_dir = d / "tokenizer"
    tokenizer = tokenizer_dir if tokenizer_dir.is_dir() else None

    missing = [
        name
        for name, p in (
            ("conv_frontend.onnx", conv),
            ("encoder.onnx", encoder),
            ("decoder.onnx", decoder),
            ("tokenizer/", tokenizer),
        )
        if p is None
    ]
    if missing:
        return ResolvedFiles(ok=False, missing=missing)
    return ResolvedFiles(
        ok=True,
        paths={
            "conv_frontend": str(conv),
            "encoder": str(encoder),
            "decoder": str(decoder),
            "tokenizer": str(tokenizer),
        },
    )


def _resolve_whisper(d: Path) -> ResolvedFiles:
    # Released dirs use e.g. base-encoder.int8.onnx / base-decoder.int8.onnx /
    # base-tokens.txt, so glob rather than hard-code the size prefix.
    encoder = _first(d, "*encoder.int8.onnx", "*encoder.onnx", "*encoder*.onnx")
    decoder = _first(d, "*decoder.int8.onnx", "*decoder.onnx", "*decoder*.onnx")
    tokens = _first(d, "*tokens.txt", "*tokens*.txt")
    missing = [
        name
        for name, p in (
            ("*-encoder.onnx", encoder),
            ("*-decoder.onnx", decoder),
            ("*-tokens.txt", tokens),
        )
        if p is None
    ]
    if missing:
        return ResolvedFiles(ok=False, missing=missing)
    return ResolvedFiles(
        ok=True,
        paths={"encoder": str(encoder), "decoder": str(decoder), "tokens": str(tokens)},
    )


# --- per-model construction --------------------------------------------------
def _build_sensevoice(paths: dict[str, str]) -> Any:
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=paths["model"],
        tokens=paths["tokens"],
        num_threads=settings.num_threads,
        provider=settings.provider,
        language=settings.sensevoice_language,
        use_itn=settings.sensevoice_use_itn,
        debug=False,
    )


def _build_qwen3(paths: dict[str, str]) -> Any:
    return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=paths["conv_frontend"],
        encoder=paths["encoder"],
        decoder=paths["decoder"],
        tokenizer=paths["tokenizer"],
        num_threads=settings.num_threads,
        provider=settings.provider,
        debug=False,
    )


def _build_whisper(paths: dict[str, str]) -> Any:
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=paths["encoder"],
        decoder=paths["decoder"],
        tokens=paths["tokens"],
        language=settings.whisper_language,
        task=settings.whisper_task,
        num_threads=settings.num_threads,
        provider=settings.provider,
        debug=False,
    )


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    factory: str
    resolve: Callable[[Path], ResolvedFiles]
    build: Callable[[dict[str, str]], Any]
    # sherpa-onnx 1.13.6 has no OnlineRecognizer factory for any of these.
    supports_streaming: bool = False


SPECS: dict[str, ModelSpec] = {
    "sensevoice": ModelSpec(
        key="sensevoice",
        display_name="SenseVoice (zh/en/ja/ko/yue)",
        factory="OfflineRecognizer.from_sense_voice",
        resolve=_resolve_sensevoice,
        build=_build_sensevoice,
    ),
    "qwen3": ModelSpec(
        key="qwen3",
        display_name="Qwen3-ASR",
        factory="OfflineRecognizer.from_qwen3_asr",
        resolve=_resolve_qwen3,
        build=_build_qwen3,
    ),
    "whisper": ModelSpec(
        key="whisper",
        display_name="Whisper",
        factory="OfflineRecognizer.from_whisper",
        resolve=_resolve_whisper,
        build=_build_whisper,
    ),
}


@dataclass
class ModelState:
    spec: ModelSpec
    directory: Path
    files_present: bool
    missing: list[str]
    paths: dict[str, str]
    loaded: bool = False
    error: str | None = None
    recognizer: Any = None


class ModelRegistry:
    """Availability + lazily-constructed recognizers, safe across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, ModelState] = {}
        self.scan()

    # -- discovery ----------------------------------------------------------
    def scan(self) -> None:
        """(Re-)inspect the models dir. Cheap: stats only, loads nothing."""
        with self._lock:
            for key in MODEL_KEYS:
                spec = SPECS[key]
                directory = settings.model_dir(key)
                if directory.is_dir():
                    resolved = spec.resolve(directory)
                else:
                    resolved = ResolvedFiles(ok=False, missing=["<model directory>"])
                previous = self._states.get(key)
                # Keep an already-loaded recognizer if its files are still there.
                if previous and previous.loaded and resolved.ok:
                    previous.paths = resolved.paths
                    previous.directory = directory
                    continue
                self._states[key] = ModelState(
                    spec=spec,
                    directory=directory,
                    files_present=resolved.ok,
                    missing=resolved.missing,
                    paths=resolved.paths,
                )

    # -- availability -------------------------------------------------------
    def availability(self) -> list[dict[str, Any]]:
        """One dict per model, suitable for :class:`schemas.ModelInfo`."""
        out: list[dict[str, Any]] = []
        with self._lock:
            for key in MODEL_KEYS:
                st = self._states[key]
                available = st.files_present and SHERPA_AVAILABLE and st.error is None
                out.append(
                    {
                        "key": key,
                        "display_name": st.spec.display_name,
                        "available": available,
                        "loaded": st.loaded,
                        "supports_streaming": st.spec.supports_streaming,
                        "factory": st.spec.factory,
                        "directory": str(st.directory),
                        "missing_files": list(st.missing),
                        "error": st.error or (SHERPA_IMPORT_ERROR if not SHERPA_AVAILABLE else None),
                    }
                )
        return out

    def is_available(self, key: str) -> bool:
        if key not in SPECS:
            return False
        with self._lock:
            st = self._states[key]
            return st.files_present and SHERPA_AVAILABLE and st.error is None

    def available_keys(self) -> list[str]:
        return [k for k in MODEL_KEYS if self.is_available(k)]

    # -- loading ------------------------------------------------------------
    def get(self, key: str) -> Any:
        """Return a loaded recognizer, constructing it on first use.

        Raises ``KeyError`` for an unknown key and ``RuntimeError`` when the
        model is unavailable or failed to construct.
        """
        if key not in SPECS:
            raise KeyError(key)
        if not SHERPA_AVAILABLE:
            raise RuntimeError(f"sherpa-onnx is not installed ({SHERPA_IMPORT_ERROR})")

        with self._lock:
            st = self._states[key]
            if st.recognizer is not None:
                return st.recognizer
            if st.error:
                raise RuntimeError(f"model '{key}' failed to load: {st.error}")
            if not st.files_present:
                raise RuntimeError(
                    f"model '{key}' is unavailable; missing {st.missing} in {st.directory}"
                )
            paths = dict(st.paths)

        # Build outside the lock: constructing a recognizer takes seconds and
        # must not block /models or another model's load.
        try:
            recognizer = SPECS[key].build(paths)
        except Exception as exc:  # pragma: no cover - needs real model files
            with self._lock:
                self._states[key].error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"model '{key}' failed to load: {exc}") from exc

        with self._lock:
            st = self._states[key]
            if st.recognizer is None:
                st.recognizer = recognizer
                st.loaded = True
            return st.recognizer

    def preload_available(self) -> dict[str, str]:
        """Load every model whose files exist. Returns {key: status}."""
        report: dict[str, str] = {}
        for key in MODEL_KEYS:
            if not self.is_available(key):
                report[key] = "unavailable"
                continue
            try:
                self.get(key)
                report[key] = "loaded"
            except Exception as exc:  # pragma: no cover - needs real model files
                report[key] = f"error: {exc}"
        return report


registry = ModelRegistry()


# --- decoding ---------------------------------------------------------------
def transcribe_samples(key: str, samples: list[float], sample_rate: int) -> str:
    """Run one offline decode. Blocking — call it from a worker thread."""
    recognizer = registry.get(key)
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return (stream.result.text or "").strip()
