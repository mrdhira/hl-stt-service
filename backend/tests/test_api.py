"""API tests.

Split in two: everything that does not need model weights runs always; the
decode tests skip cleanly on a host with an empty ``STT_MODELS_DIR``, so the
suite is green both before and after the multi-GB downloads.
"""

from __future__ import annotations

import json

import pytest

from .conftest import first_available_model, make_wav


# --- always-on ---------------------------------------------------------------
def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert isinstance(body["models_available"], list)


def test_models_lists_all_three(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()

    assert [m["key"] for m in body["models"]] == ["sensevoice", "qwen3", "whisper"]
    assert body["sherpa_onnx_available"] in (True, False)
    assert body["num_threads"] >= 1

    for info in body["models"]:
        assert set(info) >= {
            "key",
            "display_name",
            "available",
            "loaded",
            "supports_streaming",
            "factory",
            "directory",
            "missing_files",
        }
        # No OnlineRecognizer factory exists for any of the three (sherpa 1.13.6).
        assert info["supports_streaming"] is False


def test_models_with_empty_models_dir_are_unavailable(client):
    # `client` points at an empty tmp models dir, so nothing can be available.
    for info in client.get("/models").json()["models"]:
        assert info["available"] is False
        assert info["missing_files"]


def test_transcribe_rejects_unknown_model(client, tiny_wav):
    resp = client.post(
        "/transcribe",
        params={"model": "nope"},
        files={"audio": ("a.wav", tiny_wav, "audio/wav")},
    )
    assert resp.status_code == 400


def test_transcribe_unavailable_model_returns_503(client, tiny_wav):
    resp = client.post(
        "/transcribe",
        params={"model": "sensevoice"},
        files={"audio": ("a.wav", tiny_wav, "audio/wav")},
    )
    assert resp.status_code == 503
    assert "missing_files" in resp.json()["detail"]


def test_transcribe_rejects_non_wav(client):
    resp = client.post(
        "/transcribe",
        params={"model": "sensevoice"},
        files={"audio": ("a.txt", b"this is not audio at all", "audio/wav")},
    )
    # 415 (bad audio) is checked before availability only if the model exists;
    # on an empty models dir the availability check fires first.
    assert resp.status_code in (415, 503)


def test_runs_empty_then_shape(client):
    body = client.get("/runs", params={"limit": 5}).json()
    assert body == {"count": 0, "runs": []}


def test_runs_limit_is_validated(client):
    assert client.get("/runs", params={"limit": 0}).status_code == 422
    assert client.get("/runs", params={"limit": 10_000}).status_code == 422


def test_runs_row_roundtrip(client):
    """Insert through the same Database the routes use, read back via the API."""
    from app.db import db

    db.insert_run(
        model="sensevoice",
        mode="batch",
        audio_ms=500.0,
        processing_ms=250.0,
        rtf=0.5,
        words=3,
        chars=11,
        text="hello world",
        text_hash="deadbeef",
    )
    body = client.get("/runs").json()
    assert body["count"] == 1
    row = body["runs"][0]
    assert row["model"] == "sensevoice"
    assert row["mode"] == "batch"
    assert row["rtf"] == pytest.approx(0.5)
    assert row["text"] == "hello world"

    assert client.get("/runs", params={"model": "qwen3"}).json()["count"] == 0
    assert client.get("/runs", params={"model": "sensevoice"}).json()["count"] == 1


def test_stream_rejects_unknown_model(client):
    with client.websocket_connect("/stream?model=nope") as ws:
        assert ws.receive_json()["type"] == "error"


def test_stream_unavailable_model_reports_error(client):
    with client.websocket_connect("/stream?model=whisper") as ws:
        message = ws.receive_json()
        assert message["type"] == "error"
        assert "not available" in message["detail"]


# --- audio + metrics helpers -------------------------------------------------
def test_decode_wav_roundtrip():
    from app.audio import decode_wav, duration_ms

    samples, rate = decode_wav(make_wav(duration_s=0.25, sample_rate=16000))
    assert rate == 16000
    assert len(samples) == 4000
    assert duration_ms(len(samples), rate) == pytest.approx(250.0)
    assert all(-1.0 <= s <= 1.0 for s in samples[:100])


def test_decode_wav_downmixes_stereo():
    from app.audio import decode_wav

    mono, _ = decode_wav(make_wav(duration_s=0.25, channels=1))
    stereo, _ = decode_wav(make_wav(duration_s=0.25, channels=2))
    assert len(stereo) == len(mono)


def test_decode_wav_rejects_garbage():
    from app.audio import AudioDecodeError, decode_wav

    with pytest.raises(AudioDecodeError):
        decode_wav(b"definitely not a wav")


def test_word_count_handles_japanese():
    from app.metrics import count_words

    assert count_words("hello world") == 2
    assert count_words("こんにちは") == 5  # no spaces: count kana/kanji
    assert count_words("") == 0


# --- decode tests: skipped unless model files are actually present -----------
def _require_model(test_client) -> str:
    key = first_available_model(test_client)
    if key is None:
        pytest.skip("no sherpa-onnx model files under STT_MODELS_DIR — skipping decode")
    return key


def test_transcribe_with_real_model(real_models_client, tiny_wav):
    model = _require_model(real_models_client)

    resp = real_models_client.post(
        "/transcribe",
        params={"model": model},
        files={"audio": ("tiny.wav", tiny_wav, "audio/wav")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["model"] == model
    assert body["mode"] == "batch"
    assert isinstance(body["text"], str)  # a 440 Hz sine may well decode to ""
    assert body["audio_ms"] == pytest.approx(500.0, abs=1.0)
    assert body["processing_ms"] > 0
    assert body["rtf"] == pytest.approx(body["processing_ms"] / body["audio_ms"])
    assert body["words"] >= 0
    assert len(body["text_hash"]) == 64
    assert body["run_id"] is not None

    # ...and the run was persisted.
    runs = real_models_client.get("/runs", params={"limit": 1, "model": model}).json()
    assert runs["count"] == 1
    assert runs["runs"][0]["id"] == body["run_id"]
    assert runs["runs"][0]["text_hash"] == body["text_hash"]


def test_stream_with_real_model(real_models_client):
    model = _require_model(real_models_client)

    # 2 s of audio pushed as 100 ms raw-PCM16 frames, so partials must fire.
    wav = make_wav(duration_s=2.0)
    pcm = wav[44:]  # strip the canonical 44-byte header; frames are raw PCM
    frame_bytes = 16000 * 2 // 10

    with real_models_client.websocket_connect(
        f"/stream?model={model}&sample_rate=16000"
    ) as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["supports_true_streaming"] is False

        for offset in range(0, len(pcm), frame_bytes):
            ws.send_bytes(pcm[offset : offset + frame_bytes])
        ws.send_text(json.dumps({"type": "eof"}))

        message = ws.receive_json()
        while message["type"] == "partial":
            assert isinstance(message["text"], str)
            assert message["elapsed_ms"] >= 0
            message = ws.receive_json()

        assert message["type"] == "final", message
        assert message["audio_ms"] == pytest.approx(2000.0, abs=5.0)
        assert message["latency_final_ms"] >= 0
        assert message["run_id"] is not None

    runs = real_models_client.get("/runs", params={"limit": 5, "mode": "stream"}).json()
    assert runs["count"] >= 1


# --- end-to-end with a stubbed recognizer (no model weights needed) ----------
def test_transcribe_persists_a_run_with_stub(stub_model_client, tiny_wav):
    client = stub_model_client
    resp = client.post(
        "/transcribe",
        params={"model": client.stub_model},
        files={"audio": ("tiny.wav", tiny_wav, "audio/wav")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["text"] == "hello world"
    assert body["mode"] == "batch"
    assert body["words"] == 2
    assert body["chars"] == 11
    assert body["sample_rate"] == 16000
    assert body["audio_ms"] == pytest.approx(500.0)
    assert body["rtf"] == pytest.approx(body["processing_ms"] / 500.0)
    assert len(body["text_hash"]) == 64
    assert body["run_id"] is not None

    # The recognizer saw the decoded samples at the WAV's own rate.
    assert client.stub_calls == [(client.stub_model, 8000, 16000)]

    runs = client.get("/runs").json()
    assert runs["count"] == 1
    row = runs["runs"][0]
    assert row["id"] == body["run_id"]
    assert row["model"] == client.stub_model
    assert row["mode"] == "batch"
    assert row["text"] == "hello world"
    assert row["text_hash"] == body["text_hash"]
    assert row["rtf"] == pytest.approx(body["rtf"])
    assert row["latency_partial_ms"] is None


def test_transcribe_rejects_bad_audio_for_available_model(stub_model_client):
    resp = stub_model_client.post(
        "/transcribe",
        params={"model": stub_model_client.stub_model},
        files={"audio": ("a.txt", b"not audio", "audio/wav")},
    )
    assert resp.status_code == 415


def test_stream_emits_partials_then_final_with_stub(stub_model_client):
    client = stub_model_client
    pcm = make_wav(duration_s=2.0)[44:]
    frame_bytes = 16000 * 2 // 10  # 100 ms per frame

    with client.websocket_connect(
        f"/stream?model={client.stub_model}&sample_rate=16000"
    ) as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["supports_true_streaming"] is False
        assert ready["partial_interval_ms"] > 0

        for offset in range(0, len(pcm), frame_bytes):
            ws.send_bytes(pcm[offset : offset + frame_bytes])
        ws.send_text(json.dumps({"type": "eof"}))

        partials = []
        message = ws.receive_json()
        while message["type"] == "partial":
            partials.append(message)
            message = ws.receive_json()

        assert partials, "expected at least one partial for 2 s of audio"
        assert all(p["text"] == "hello world" for p in partials)
        assert partials[0]["audio_ms"] > 0

        final = message
        assert final["type"] == "final"
        assert final["text"] == "hello world"
        assert final["mode"] == "stream"
        assert final["audio_ms"] == pytest.approx(2000.0, abs=5.0)
        assert final["partials_emitted"] == len(partials)
        assert final["latency_partial_ms"] >= 0
        assert final["latency_final_ms"] >= 0
        assert final["run_id"] is not None

    row = client.get("/runs", params={"mode": "stream"}).json()["runs"][0]
    assert row["id"] == final["run_id"]
    assert row["mode"] == "stream"
    assert row["words"] == 2
    assert row["latency_final_ms"] is not None


def test_stream_eof_without_audio_errors(stub_model_client):
    client = stub_model_client
    with client.websocket_connect(f"/stream?model={client.stub_model}") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_text(json.dumps({"type": "eof"}))
        message = ws.receive_json()
        assert message["type"] == "error"
        assert "no audio" in message["detail"]


def test_stream_accepts_wav_frames_too(stub_model_client):
    """A frame with a RIFF header carries its own sample rate."""
    client = stub_model_client
    with client.websocket_connect(
        f"/stream?model={client.stub_model}&sample_rate=16000"
    ) as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(make_wav(duration_s=1.5, sample_rate=8000))
        ws.send_text(json.dumps({"type": "eof"}))

        message = ws.receive_json()
        while message["type"] == "partial":
            message = ws.receive_json()
        assert message["type"] == "final"
        assert message["sample_rate"] == 8000
        assert message["audio_ms"] == pytest.approx(1500.0, abs=5.0)


# --- expected_text / test-case tagging ---------------------------------------
def test_transcribe_persists_expected_text(stub_model_client, tiny_wav):
    client = stub_model_client
    resp = client.post(
        "/transcribe",
        params={"model": client.stub_model},
        files={"audio": ("tiny.wav", tiny_wav, "audio/wav")},
        data={"expected_text": "  hello there world  "},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["expected_text"] == "hello there world"

    row = client.get("/runs").json()["runs"][0]
    assert row["expected_text"] == "hello there world"


def test_transcribe_blank_expected_text_stores_null(stub_model_client, tiny_wav):
    client = stub_model_client
    resp = client.post(
        "/transcribe",
        params={"model": client.stub_model},
        files={"audio": ("tiny.wav", tiny_wav, "audio/wav")},
        data={"expected_text": "   "},
    )
    assert resp.status_code == 200
    assert resp.json()["expected_text"] is None
    assert client.get("/runs").json()["runs"][0]["expected_text"] is None


def test_transcribe_without_expected_text_still_works(stub_model_client, tiny_wav):
    client = stub_model_client
    resp = client.post(
        "/transcribe",
        params={"model": client.stub_model},
        files={"audio": ("tiny.wav", tiny_wav, "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.json()["expected_text"] is None


def test_stream_persists_expected_text(stub_model_client):
    client = stub_model_client
    pcm = make_wav(duration_s=1.5)[44:]

    with client.websocket_connect(
        f"/stream?model={client.stub_model}&sample_rate=16000"
    ) as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(pcm)
        ws.send_text(json.dumps({"type": "eof", "expected_text": "  hello world  "}))

        message = ws.receive_json()
        while message["type"] == "partial":
            message = ws.receive_json()
        assert message["type"] == "final"
        assert message["expected_text"] == "hello world"

    row = client.get("/runs", params={"mode": "stream"}).json()["runs"][0]
    assert row["expected_text"] == "hello world"


def test_db_migrates_an_existing_runs_table(tmp_path):
    """A pre-expected_text database gains the column instead of blowing up."""
    import sqlite3

    from app.db import Database

    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(path)
    legacy.execute(
        "CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, "
        "model TEXT NOT NULL, mode TEXT NOT NULL, audio_ms REAL, processing_ms REAL, "
        "rtf REAL, words INTEGER, chars INTEGER, latency_partial_ms REAL, "
        "latency_final_ms REAL, text TEXT, text_hash TEXT)"
    )
    legacy.execute(
        "INSERT INTO runs (ts, model, mode, text) VALUES (1.0, 'whisper', 'batch', 'old row')"
    )
    legacy.commit()
    legacy.close()

    db = Database(path)
    db.connect()
    try:
        rows = db.recent_runs(limit=10)
        assert len(rows) == 1
        assert rows[0]["expected_text"] is None  # column added, old row back-filled NULL

        db.insert_run(model="qwen3", mode="batch", text="new", expected_text="truth")
        assert db.recent_runs(limit=1)[0]["expected_text"] == "truth"
    finally:
        db.close()


# --- SPA mount ---------------------------------------------------------------
def test_spa_serves_index_at_root(built_frontend):
    resp = built_frontend.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "hl-stt" in resp.text


def test_spa_serves_static_assets(built_frontend):
    resp = built_frontend.get("/assets/app.js")
    assert resp.status_code == 200
    assert resp.text == "console.log('bundle')"
    assert built_frontend.get("/favicon.svg").status_code == 200


def test_spa_deep_link_falls_back_to_index(built_frontend):
    """A refresh on a client-side route must return the shell, not a 404."""
    resp = built_frontend.get("/reports")
    assert resp.status_code == 200
    assert "hl-stt" in resp.text
    assert "<div id=root>" in resp.text


def test_api_routes_win_over_the_spa_mount(built_frontend):
    """The mount is registered at "/" — the API must still match first."""
    health = built_frontend.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    models = built_frontend.get("/models")
    assert models.status_code == 200
    assert [m["key"] for m in models.json()["models"]] == ["sensevoice", "qwen3", "whisper"]

    runs = built_frontend.get("/runs")
    assert runs.status_code == 200
    assert runs.json() == {"count": 0, "runs": []}

    # /transcribe is POST-only: GET must be 405 from the router, not the SPA shell.
    assert built_frontend.get("/transcribe").status_code == 405


def test_websocket_still_routes_under_the_spa_mount(built_frontend):
    with built_frontend.websocket_connect("/stream?model=nope") as ws:
        assert ws.receive_json()["type"] == "error"


def test_unknown_api_path_404s_instead_of_returning_html(built_frontend):
    """A typo'd endpoint must not come back as a 200 HTML page."""
    for path in ("/models/typo", "/runs/999", "/health/x", "/openapi.json/x"):
        resp = built_frontend.get(path)
        assert resp.status_code == 404, f"{path} returned {resp.status_code}"
        assert "hl-stt" not in resp.text


def test_non_get_on_unknown_path_is_not_the_spa(built_frontend):
    resp = built_frontend.post("/some/client/route")
    assert resp.status_code in (404, 405)
    assert "<div id=root>" not in resp.text


def test_api_only_when_frontend_is_not_built(client):
    """`client` has no STT_STATIC_DIR, so the app must still boot."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "api-only"
    assert client.get("/health").json()["status"] == "ok"
