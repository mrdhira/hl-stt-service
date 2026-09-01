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
