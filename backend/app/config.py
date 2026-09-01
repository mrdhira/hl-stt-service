"""Runtime configuration, read straight from the environment.

Deliberately dependency-free (stdlib ``os`` only) so config can be imported by
scripts, tests and the benchmark harness without pulling in FastAPI.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- keys used everywhere else in the app -----------------------------------
MODEL_KEYS: tuple[str, ...] = ("sensevoice", "qwen3", "whisper")

# Env var that overrides the on-disk directory name for each model key. These
# names match backend/.env.example.
_DIR_ENV_VAR: dict[str, str] = {
    "sensevoice": "STT_BACKEND_SENSEVOICE",
    "qwen3": "STT_BACKEND_QWEN",
    "whisper": "STT_BACKEND_WHISPER",
}


def _list_env(name: str, default: list[str]) -> list[str]:
    """Comma-separated env var to a list. Empty string means an empty list."""
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class Settings:
    """Snapshot of the environment.

    Instantiated once as :data:`settings`. Tests that need to point the app at a
    different models dir or database build their own instance.
    """

    def __init__(self) -> None:
        self.host: str = os.environ.get("STT_HOST", "0.0.0.0")
        self.port: int = _int_env("STT_PORT", 8000)
        self.models_dir: Path = Path(
            os.environ.get("STT_MODELS_DIR", "/data/models")
        ).expanduser()
        self.num_threads: int = _int_env("STT_NUM_THREADS", 4)

        # Where the benchmark runs are stored. Git-ignored (*.db).
        self.db_path: Path = Path(
            os.environ.get("STT_DB_PATH", "/data/stt-runs.db")
        ).expanduser()

        # Every clip that hits /transcribe or /asr is archived here as training
        # and benchmark data, named by run id. Git-ignored; bind-mounted in the
        # container so the corpus outlives the image.
        self.dataset_dir: Path = Path(
            os.environ.get("STT_DATASET_DIR", "/data/dataset")
        ).expanduser()

        # Built frontend (Vite `dist/`). One container serves the SPA and the
        # API from the same origin, so there is no CORS hop in production.
        # Defaults to <repo>/frontend/dist, which is also the layout inside the
        # image (/app/backend/app/config.py -> /app/frontend/dist).
        self.static_dir: Path = Path(
            os.environ.get("STT_STATIC_DIR", "")
            or Path(__file__).resolve().parents[2] / "frontend" / "dist"
        ).expanduser()

        # Decoding knobs. "" / "auto" lets SenseVoice detect the language; the
        # POC is English + Japanese so Whisper defaults to auto-detect too.
        self.sensevoice_language: str = os.environ.get("STT_SENSEVOICE_LANGUAGE", "")
        self.sensevoice_use_itn: bool = _int_env("STT_SENSEVOICE_USE_ITN", 1) == 1
        self.whisper_language: str = os.environ.get("STT_WHISPER_LANGUAGE", "")
        self.whisper_task: str = os.environ.get("STT_WHISPER_TASK", "transcribe")

        # WS /stream: how often to re-decode the buffer and emit a partial.
        self.stream_partial_ms: int = _int_env("STT_STREAM_PARTIAL_MS", 1000)
        # Sample rate assumed for raw-PCM websocket frames.
        self.stream_sample_rate: int = _int_env("STT_STREAM_SAMPLE_RATE", 16000)

        self.provider: str = os.environ.get("STT_PROVIDER", "cpu")

        # --- upload limits -------------------------------------------------
        # Decoding amplifies hugely: a 221 KB Opus file expands to ~19 MB of
        # PCM, and turning that into a Python float list costs ~307 MB — about
        # 1400x the upload. Unauthenticated callers therefore need two ceilings,
        # one on the bytes accepted and one on the *decoded* duration, checked
        # before any float conversion happens.
        #
        # 25 MB is far more than a voice note (minutes of Opus) but small enough
        # that even a pathological compression ratio cannot exhaust the 4 GB the
        # container is limited to.
        self.max_upload_bytes: int = _int_env("STT_MAX_UPLOAD_BYTES", 25 * 1024 * 1024)
        # 10 minutes of audio. At 16 kHz mono that is ~19 MB of PCM and ~300 MB
        # as floats — the practical worst case we are willing to allocate.
        self.max_audio_seconds: float = float(_int_env("STT_MAX_AUDIO_SECONDS", 600))

        # --- CORS ----------------------------------------------------------
        # One container serves the SPA and the API from the same origin, so the
        # browser makes no cross-origin request in production and this list can
        # stay empty there. It exists for `npm run dev`, where Vite serves the
        # UI from another port.
        #
        # Deliberately NOT "*": /asr and PATCH /runs/{id} mutate state and there
        # is no auth, so a wildcard would let any page a LAN user happens to
        # visit rewrite their ground truth or drive transcription. Set
        # STT_CORS_ORIGINS to a comma-separated list to override.
        self.cors_origins: list[str] = _list_env(
            "STT_CORS_ORIGINS",
            ["http://localhost:5173", "http://127.0.0.1:5173"],
        )

    def model_dir(self, key: str) -> Path:
        """Directory holding the files for ``key``.

        Defaults to ``<models_dir>/<key>`` and can be overridden per model with
        ``STT_BACKEND_SENSEVOICE`` / ``STT_BACKEND_QWEN`` / ``STT_BACKEND_WHISPER``.
        """
        override = os.environ.get(_DIR_ENV_VAR[key], "").strip()
        name = override or key
        candidate = Path(name).expanduser()
        if candidate.is_absolute():
            return candidate
        return self.models_dir / name


settings = Settings()
