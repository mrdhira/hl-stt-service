"""Test fixtures.

Every test runs against a throwaway SQLite file and a throwaway models dir so a
developer box with real models in ``/data/models`` is never touched, and a box
with no models at all still passes.
"""

from __future__ import annotations

import io
import math
import os
import struct
import sys
import wave
from pathlib import Path

import pytest

# backend/ on the path so `import app` works without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_wav(
    duration_s: float = 0.5,
    sample_rate: int = 16000,
    freq: float = 440.0,
    channels: int = 1,
) -> bytes:
    """A tiny in-memory 16-bit PCM WAV — a quiet sine, no fixture file on disk."""
    n = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n):
            value = int(8000 * math.sin(2 * math.pi * freq * i / sample_rate))
            for _ in range(channels):
                frames += struct.pack("<h", value)
        wf.writeframes(bytes(frames))
    return buf.getvalue()


@pytest.fixture
def tiny_wav() -> bytes:
    return make_wav()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient wired to a temp DB + temp models dir.

    Env is patched *before* ``app`` is imported, because config/db/registry are
    module-level singletons built at import time.
    """
    monkeypatch.setenv("STT_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("STT_MODELS_DIR", str(tmp_path / "models"))
    # Pin the static dir at a path that does not exist: otherwise the suite
    # would pass or fail depending on whether frontend/dist has been built.
    monkeypatch.setenv("STT_STATIC_DIR", str(tmp_path / "no-frontend"))
    for var in ("STT_BACKEND_SENSEVOICE", "STT_BACKEND_QWEN", "STT_BACKEND_WHISPER"):
        monkeypatch.delenv(var, raising=False)

    for module in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        del sys.modules[module]

    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def real_models_client(tmp_path, monkeypatch):
    """TestClient pointed at the host's real ``STT_MODELS_DIR``.

    Used only by the tests that need an actual recognizer; they skip when no
    model files are present. The DB is still redirected to a temp file — the
    default ``/data/stt-runs.db`` is a container path and tests must not write
    real benchmark rows.
    """
    monkeypatch.setenv("STT_DB_PATH", str(tmp_path / "runs.db"))

    for module in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        del sys.modules[module]

    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def first_available_model(test_client) -> str | None:
    """Key of a model whose files are on disk, or ``None``."""
    payload = test_client.get("/models").json()
    for info in payload["models"]:
        if info["available"]:
            return info["key"]
    return None


@pytest.fixture
def stub_model_client(client, monkeypatch):
    """`client`, but with 'sensevoice' faked as available and decoding stubbed.

    Lets the persistence path and the websocket protocol be tested end-to-end on
    a host with no model weights. The stub stands in for
    ``models.transcribe_samples`` (the only place sherpa-onnx is actually
    called), so everything around it is the real code path.
    """
    import app.models as models_mod
    import app.routes.stream as stream_mod
    import app.routes.transcribe as transcribe_mod

    key = "sensevoice"
    calls: list[tuple[str, int, int]] = []

    def fake_transcribe(model_key: str, samples, sample_rate: int) -> str:
        calls.append((model_key, len(samples), sample_rate))
        return "hello world"

    monkeypatch.setattr(
        models_mod.registry, "is_available", lambda k: k == key, raising=True
    )
    monkeypatch.setattr(transcribe_mod, "transcribe_samples", fake_transcribe)
    monkeypatch.setattr(stream_mod, "transcribe_samples", fake_transcribe)

    client.stub_model = key
    client.stub_calls = calls
    return client


@pytest.fixture
def built_frontend(tmp_path, monkeypatch):
    """A stand-in `frontend/dist` so the SPA mount can be tested.

    Real `npm run build` output is not needed — the mount only cares that
    `index.html` exists and that sibling files are served verbatim.
    """
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>hl-stt</title><div id=root>")
    (dist / "assets" / "app.js").write_text("console.log('bundle')")
    (dist / "favicon.svg").write_text("<svg/>")

    monkeypatch.setenv("STT_STATIC_DIR", str(dist))
    monkeypatch.setenv("STT_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("STT_MODELS_DIR", str(tmp_path / "models"))

    for module in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        del sys.modules[module]

    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
