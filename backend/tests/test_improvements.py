"""Tests for the feature/stt-improvements work.

Covers the four behavioural sections: pronunciation-aware scoring, POST /asr,
dataset capture, and editing ground truth.
"""

from __future__ import annotations

import subprocess
import wave

import pytest

from .conftest import make_wav


# --- 1. pronunciation-aware tokenization -------------------------------------
def test_pronunciation_map_is_whole_word_only():
    from app.metrics import apply_pronunciation_equiv as apply

    assert apply("Dhira") == "dira"
    assert apply("DHIRA") == "dira"
    assert apply("my name is Dhira.") == "my name is dira."
    # A word that merely contains the key must not be rewritten.
    assert apply("dhirama") == "dhirama"
    assert apply("sandhira") == "sandhira"
    # Hyphens are word boundaries.
    assert apply("Dhira-san") == "dira-san"
    assert apply("") == ""


def test_pronunciation_map_does_not_change_word_counts():
    """The map must not shift the `words` column for existing rows."""
    from app.metrics import count_words

    assert count_words("Hello my name is Dhira. How are you?") == 8
    assert count_words("Hello my name is Dira. How are you?") == 8
    assert count_words("こんにちは世界です") == 9


def test_pronunciation_map_is_not_fuzzy():
    """Only listed words are equated — near-misses stay errors."""
    from app.metrics import PRONUNCIATION_EQUIV, apply_pronunciation_equiv as apply

    assert PRONUNCIATION_EQUIV == {"dhira": "dira"}
    # "Dila" is a genuine recognition error and must survive normalisation.
    assert apply("Dila") == "Dila"
    assert apply("Dira") == "Dira"


# --- 2 + 3. POST /asr and dataset capture ------------------------------------
def _has_ffmpeg() -> bool:
    from app.audio import ffmpeg_path

    return ffmpeg_path() is not None


def _encode(wav_bytes: bytes, args: list[str], suffix: str, tmp_path) -> bytes:
    """Transcode a WAV with ffmpeg, for the compressed-upload tests."""
    source = tmp_path / "in.wav"
    target = tmp_path / f"out{suffix}"
    source.write_bytes(wav_bytes)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
        + args
        + [str(target)],
        check=True,
        capture_output=True,
    )
    return target.read_bytes()


def test_asr_transcribes_wav_and_persists_an_asr_run(stub_model_client, tiny_wav):
    client = stub_model_client
    resp = client.post("/asr", files={"audio": ("note.wav", tiny_wav, "audio/wav")})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Exactly the documented response shape — no benchmark surface.
    assert set(body) == {"text", "model", "audio_ms", "processing_ms", "rtf", "run_id"}
    assert body["text"] == "hello world"
    assert body["model"] == "sensevoice"  # the default
    assert body["audio_ms"] == pytest.approx(500.0)
    assert body["rtf"] == pytest.approx(body["processing_ms"] / 500.0)

    row = client.get("/runs", params={"mode": "asr"}).json()["runs"][0]
    assert row["id"] == body["run_id"]
    assert row["mode"] == "asr"
    assert row["text"] == "hello world"


def test_asr_defaults_to_sensevoice_and_accepts_override(stub_model_client, tiny_wav):
    client = stub_model_client
    assert (
        client.post("/asr", files={"audio": ("a.wav", tiny_wav, "audio/wav")}).json()["model"]
        == "sensevoice"
    )
    # The stub only makes sensevoice available, so an override to an
    # unavailable model must surface as 503 rather than silently falling back.
    resp = client.post(
        "/asr", params={"model": "whisper"}, files={"audio": ("a.wav", tiny_wav, "audio/wav")}
    )
    assert resp.status_code == 503
    assert client.post(
        "/asr", params={"model": "nope"}, files={"audio": ("a.wav", tiny_wav, "audio/wav")}
    ).status_code == 400


def test_asr_archives_the_clip_to_the_dataset(stub_model_client, tiny_wav, tmp_path):
    client = stub_model_client
    body = client.post("/asr", files={"audio": ("note.wav", tiny_wav, "audio/wav")}).json()

    dataset = tmp_path / "dataset"
    archived = dataset / f"{body['run_id']}.wav"
    assert archived.is_file(), list(dataset.iterdir()) if dataset.exists() else "no dir"
    # The *original* bytes, not a re-encode.
    assert archived.read_bytes() == tiny_wav

    row = client.get("/runs", params={"mode": "asr"}).json()["runs"][0]
    assert row["audio_path"] == str(archived)


def test_transcribe_also_archives_the_clip(stub_model_client, tiny_wav, tmp_path):
    client = stub_model_client
    body = client.post(
        "/transcribe",
        params={"model": client.stub_model},
        files={"audio": ("clip.wav", tiny_wav, "audio/wav")},
        data={"expected_text": "hello world"},
    ).json()

    archived = tmp_path / "dataset" / f"{body['run_id']}.wav"
    assert archived.is_file()
    assert body["audio_path"] == str(archived)
    row = client.get("/runs").json()["runs"][0]
    assert row["audio_path"] == str(archived)
    assert row["expected_text"] == "hello world"


def test_dataset_extension_is_sniffed_not_trusted(stub_model_client, tiny_wav, tmp_path):
    """A lying filename must not decide the archive extension."""
    client = stub_model_client
    body = client.post(
        "/asr", files={"audio": ("actually-a-wav.ogg", tiny_wav, "audio/ogg")}
    ).json()
    assert (tmp_path / "dataset" / f"{body['run_id']}.wav").is_file()


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not installed")
@pytest.mark.parametrize(
    "args,suffix,expected_ext",
    [
        (["-c:a", "libopus", "-f", "ogg"], ".ogg", "ogg"),   # Telegram voice note
        (["-c:a", "libopus", "-f", "webm"], ".webm", "webm"),  # browser recording
    ],
)
def test_asr_accepts_compressed_audio(
    stub_model_client, tiny_wav, tmp_path, args, suffix, expected_ext
):
    client = stub_model_client
    encoded = _encode(make_wav(duration_s=1.0), args, suffix, tmp_path)

    resp = client.post("/asr", files={"audio": (f"voice{suffix}", encoded, "application/octet-stream")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["text"] == "hello world"
    assert body["audio_ms"] == pytest.approx(1000.0, abs=60.0)

    # Archived under the sniffed container, byte-identical to the upload.
    archived = tmp_path / "dataset" / f"{body['run_id']}.{expected_ext}"
    assert archived.is_file()
    assert archived.read_bytes() == encoded


def test_asr_rejects_undecodable_bytes(stub_model_client):
    resp = stub_model_client.post(
        "/asr", files={"audio": ("junk.bin", b"not audio at all, really", "audio/ogg")}
    )
    assert resp.status_code == 415


def test_asr_rejects_empty_upload(stub_model_client):
    resp = stub_model_client.post("/asr", files={"audio": ("empty.wav", b"", "audio/wav")})
    assert resp.status_code == 400


def test_decode_audio_handles_wav_without_ffmpeg_or_soundfile():
    """The WAV path must stay dependency-free."""
    from app.audio import decode_audio

    samples, rate = decode_audio(make_wav(duration_s=0.25))
    assert rate == 16000
    assert len(samples) == 4000


# --- 4. editing ground truth --------------------------------------------------
def test_patch_run_sets_expected_text(stub_model_client, tiny_wav):
    client = stub_model_client
    run_id = client.post(
        "/transcribe",
        params={"model": client.stub_model},
        files={"audio": ("a.wav", tiny_wav, "audio/wav")},
    ).json()["run_id"]

    resp = client.patch(f"/runs/{run_id}", json={"expected_text": "  hello world  "})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == run_id
    assert body["expected_text"] == "hello world"  # trimmed
    # The measured numbers must come back untouched.
    assert body["text"] == "hello world"
    assert body["mode"] == "batch"

    assert client.get("/runs").json()["runs"][0]["expected_text"] == "hello world"


def test_patch_run_clears_expected_text(stub_model_client, tiny_wav):
    client = stub_model_client
    run_id = client.post(
        "/transcribe",
        params={"model": client.stub_model},
        files={"audio": ("a.wav", tiny_wav, "audio/wav")},
        data={"expected_text": "original"},
    ).json()["run_id"]

    # Blank clears to NULL rather than storing "", so "unlabelled" has one form.
    assert client.patch(f"/runs/{run_id}", json={"expected_text": "   "}).json()[
        "expected_text"
    ] is None
    assert client.patch(f"/runs/{run_id}", json={}).json()["expected_text"] is None


def test_patch_unknown_run_is_404(stub_model_client):
    assert stub_model_client.patch("/runs/99999", json={"expected_text": "x"}).status_code == 404


def test_patch_rejects_a_bad_run_id(stub_model_client):
    assert stub_model_client.patch("/runs/0", json={"expected_text": "x"}).status_code == 422


def test_runs_can_be_filtered_by_asr_mode(stub_model_client, tiny_wav):
    client = stub_model_client
    client.post("/asr", files={"audio": ("a.wav", tiny_wav, "audio/wav")})
    client.post(
        "/transcribe",
        params={"model": client.stub_model},
        files={"audio": ("b.wav", tiny_wav, "audio/wav")},
    )
    assert client.get("/runs", params={"mode": "asr"}).json()["count"] == 1
    assert client.get("/runs", params={"mode": "batch"}).json()["count"] == 1
    assert client.get("/runs").json()["count"] == 2


# --- migrations ---------------------------------------------------------------
def test_db_migrates_audio_path_onto_an_existing_table(tmp_path):
    """A database from before this branch gains audio_path, keeping its rows."""
    import sqlite3

    from app.db import Database

    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(path)
    legacy.execute(
        "CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, "
        "model TEXT NOT NULL, mode TEXT NOT NULL, audio_ms REAL, processing_ms REAL, "
        "rtf REAL, words INTEGER, chars INTEGER, latency_partial_ms REAL, "
        "latency_final_ms REAL, text TEXT, text_hash TEXT, expected_text TEXT)"
    )
    legacy.execute(
        "INSERT INTO runs (ts, model, mode, text, expected_text) "
        "VALUES (1.0, 'qwen3', 'batch', 'got', 'wanted')"
    )
    legacy.commit()
    legacy.close()

    db = Database(path)
    db.connect()
    try:
        rows = db.recent_runs(limit=10)
        assert len(rows) == 1
        assert rows[0]["expected_text"] == "wanted"  # preserved
        assert rows[0]["audio_path"] is None  # added, back-filled NULL

        run_id = db.insert_run(model="sensevoice", mode="asr", text="new")
        db.set_audio_path(run_id, "/data/dataset/2.ogg")
        assert db.get_run(run_id)["audio_path"] == "/data/dataset/2.ogg"

        assert db.update_expected_text(run_id, "truth") is True
        assert db.get_run(run_id)["expected_text"] == "truth"
        assert db.update_expected_text(4242, "nope") is False
    finally:
        db.close()


def test_wav_fixture_is_actually_a_wav(tiny_wav):
    """Guard the fixture the rest of the suite leans on."""
    import io

    with wave.open(io.BytesIO(tiny_wav), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16000
