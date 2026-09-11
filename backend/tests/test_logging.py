"""Logging tests.

Two halves: :mod:`app.logging_setup` on its own (level, format, stream), and
the request logging the app actually emits (one line per request, a request id
that ties an error back to its request, and never a transcript).

Global logging state is restored between tests by the autouse ``restore_logging``
fixture in ``conftest`` — ``setup_logging`` is process-wide, and so is the
lifespan call to it that every app fixture makes.
"""

from __future__ import annotations

import io
import json
import logging
import re
import traceback

import pytest

from .conftest import make_wav

RID = r"rid=([0-9a-f]{8})"


def hl_stt_messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "hl-stt"]


def request_lines(caplog) -> list[str]:
    return [m for m in hl_stt_messages(caplog) if m.startswith("request ")]


def rid_of(message: str) -> str:
    match = re.search(RID, message)
    assert match, f"no request id in {message!r}"
    return match.group(1)


# --- setup_logging -----------------------------------------------------------
def test_setup_logging_writes_to_the_given_stream_in_the_agreed_format():
    from app.logging_setup import setup_logging

    buffer = io.StringIO()
    logger = setup_logging("INFO", stream=buffer)
    logger.info("hello %s", "world")

    line = buffer.getvalue().strip()
    # timestamp, level, logger name, message
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO hl-stt: hello world$", line)


def test_setup_logging_defaults_to_stdout(capsys):
    import sys

    from app.logging_setup import setup_logging

    setup_logging("INFO")
    assert _own_handler().stream is sys.stdout

    logging.getLogger("hl-stt").info("to docker logs")
    assert "to docker logs" in capsys.readouterr().out


def test_setup_logging_applies_the_configured_level():
    from app.logging_setup import setup_logging

    buffer = io.StringIO()
    setup_logging("WARNING", stream=buffer)
    assert logging.getLogger().level == logging.WARNING

    logging.getLogger("hl-stt").info("suppressed")
    logging.getLogger("hl-stt").warning("kept")
    assert "suppressed" not in buffer.getvalue()
    assert "kept" in buffer.getvalue()

    setup_logging("DEBUG", stream=buffer)
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_takes_the_level_from_settings(monkeypatch):
    from app import logging_setup
    from app.config import Settings

    monkeypatch.setenv("STT_LOG_LEVEL", "debug")
    monkeypatch.setattr(logging_setup, "settings", Settings())

    logging_setup.setup_logging(stream=io.StringIO())
    assert logging.getLogger().level == logging.DEBUG


def test_log_level_setting_defaults_to_info_and_survives_nonsense(monkeypatch):
    from app.config import Settings
    from app.logging_setup import resolve_level

    monkeypatch.delenv("STT_LOG_LEVEL", raising=False)
    assert Settings().log_level == "INFO"

    monkeypatch.setenv("STT_LOG_LEVEL", " warning ")
    assert Settings().log_level == "WARNING"

    # A typo must not silence the app.
    assert resolve_level("LOUD") == logging.INFO
    assert resolve_level(None) == logging.INFO
    assert resolve_level(logging.DEBUG) == logging.DEBUG


def test_setup_logging_replaces_its_handler_instead_of_stacking():
    from app.logging_setup import HANDLER_NAME, setup_logging

    for _ in range(3):
        setup_logging("INFO", stream=io.StringIO())

    ours = [h for h in logging.getLogger().handlers if h.get_name() == HANDLER_NAME]
    assert len(ours) == 1


def test_setup_logging_puts_uvicorn_under_the_same_level():
    """uvicorn pins a level on its own loggers; STT_LOG_LEVEL has to win."""
    from app.logging_setup import setup_logging

    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).setLevel(logging.INFO)

    buffer = io.StringIO()
    setup_logging("WARNING", stream=buffer)

    logging.getLogger("uvicorn.error").info("Application startup complete.")
    assert buffer.getvalue() == ""

    logging.getLogger("uvicorn.error").warning("something is wrong")
    assert "something is wrong" in buffer.getvalue()


def _own_handler() -> logging.Handler:
    """The single handler setup_logging owns on the root logger."""
    from app.logging_setup import HANDLER_NAME

    root = logging.getLogger()
    return next(h for h in root.handlers if h.get_name() == HANDLER_NAME)


# --- safe() ------------------------------------------------------------------
def test_safe_strips_everything_that_could_end_a_log_line():
    from app.middleware import safe

    forged = "sensevoice\n2026-01-01 00:00:00,000 INFO hl-stt: nothing to see"
    assert "\n" not in safe(forged)
    assert safe("a\rb\tc\x00d") == "a?b?c?d"
    # U+2028 / U+2029 break a line for plenty of log viewers, too.
    assert safe("a b c") == "a?b?c"
    assert safe("plain/path") == "plain/path"


def test_safe_marks_a_truncation_instead_of_dropping_the_tail():
    from app.middleware import safe

    assert safe("x" * 200) == "x" * 200
    truncated = safe("x" * 201)
    assert truncated.startswith("x" * 200)
    assert truncated.endswith("[truncated]")


def test_a_forged_path_cannot_forge_a_log_line(caplog):
    """A %0a in the path decodes to a real newline before it reaches the app.

    Driven as raw ASGI on purpose: an HTTP client strips CR/LF out of a URL
    before sending it, so going through the test client would prove nothing
    about what the middleware does with one.
    """
    import asyncio

    from app.middleware import RequestLoggingMiddleware

    async def ok(scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    forged = "/x\nINFO hl-stt: request rid=deadbeef GET /admin -> 200 in 1.0ms"
    scope = {"type": "http", "method": "GET", "path": forged, "headers": []}

    caplog.set_level(logging.INFO)
    asyncio.run(RequestLoggingMiddleware(ok)(scope, _no_receive, _drop_send))

    line = request_lines(caplog)[0]
    assert "\n" not in line
    assert "/x?INFO" in line


async def _no_receive():  # pragma: no cover - the request has no body
    return {"type": "http.disconnect"}


async def _drop_send(message) -> None:
    return None


# --- request logging ---------------------------------------------------------
def test_middleware_logs_one_line_per_request(client, caplog):
    caplog.set_level(logging.INFO)
    assert client.get("/models").status_code == 200

    assert len(request_lines(caplog)) == 1
    line = request_lines(caplog)[0]

    assert "GET /models" in line
    assert "-> 200" in line
    assert re.search(RID, line)
    assert re.search(r"in \d+\.\d+ms", line)


def test_middleware_logs_a_client_error_at_warning(client, tiny_wav, caplog):
    caplog.set_level(logging.INFO)
    resp = client.post(
        "/transcribe",
        params={"model": "nope"},
        files={"audio": ("a.wav", tiny_wav, "audio/wav")},
    )
    assert resp.status_code == 400

    record = next(r for r in caplog.records if r.getMessage().startswith("request "))
    assert record.levelno == logging.WARNING
    assert "-> 400" in record.getMessage()


def test_healthcheck_polling_is_quiet_until_it_fails(client, caplog):
    """The compose healthcheck hits /health every 30s; INFO is not the place."""
    caplog.set_level(logging.DEBUG)
    assert client.get("/health").status_code == 200

    record = next(r for r in caplog.records if r.getMessage().startswith("request "))
    assert record.levelno == logging.DEBUG
    assert "GET /health -> 200" in record.getMessage()


def test_every_request_gets_its_own_id(client, caplog):
    caplog.set_level(logging.INFO)
    client.get("/models")
    client.get("/models")

    ids = [rid_of(line) for line in request_lines(caplog)]
    assert len(ids) == 2
    assert ids[0] != ids[1]


def test_spa_navigation_logs_the_status_the_client_gets(built_frontend, caplog):
    """The SPA fallback rewrites a 404 into a 200; the log must say 200.

    That only holds while the request logger is the outermost middleware —
    `mount_spa` installs its fallback as a middleware too, and whichever is
    outside the other decides which status gets logged.
    """
    caplog.set_level(logging.INFO)
    resp = built_frontend.get("/reports")

    assert resp.status_code == 200
    assert "<div id=root>" in resp.text

    line = next(m for m in request_lines(caplog) if "/reports" in m)
    assert "GET /reports -> 200" in line


def test_request_id_stays_bound_while_the_body_streams(client, caplog):
    """A streaming route logs under its own id, not "-".

    The id is bound for as long as the response is being written, which is the
    whole point of doing this in a plain ASGI middleware.
    """
    from starlette.responses import StreamingResponse

    from app.middleware import current_request_id

    @client.app.get("/_streamed")
    def streamed() -> StreamingResponse:
        async def chunks():
            yield b"first "
            logging.getLogger("hl-stt").info("mid-body rid=%s", current_request_id())
            yield b"second"

        return StreamingResponse(chunks(), media_type="text/plain")

    caplog.set_level(logging.INFO)
    assert client.get("/_streamed").text == "first second"

    mid_body = next(m for m in hl_stt_messages(caplog) if m.startswith("mid-body"))
    line = next(m for m in request_lines(caplog) if "/_streamed" in m)
    assert rid_of(mid_body) == rid_of(line)


def test_unhandled_exception_is_logged_with_traceback_and_request_id(client, caplog):
    from fastapi.testclient import TestClient

    @client.app.get("/_boom")
    def boom() -> dict:  # pragma: no cover - raises by design
        raise RuntimeError("kaboom")

    caplog.set_level(logging.INFO)
    # The app is already started by the `client` fixture; this second client
    # only exists to keep the 500 as a response instead of a raised exception.
    resp = TestClient(client.app, raise_server_exceptions=False).get("/_boom")

    # FastAPI's default response, unchanged.
    assert resp.status_code == 500
    assert resp.text == "Internal Server Error"

    logged = [r for r in caplog.records if r.name == "hl-stt"]
    handler_record = next(
        r for r in logged if r.getMessage().startswith("unhandled exception")
    )
    assert handler_record.levelno == logging.ERROR
    assert handler_record.exc_info is not None
    formatted = "".join(traceback.format_exception(*handler_record.exc_info))
    assert "RuntimeError: kaboom" in formatted
    assert "raise RuntimeError" in formatted

    # The middleware's line and the traceback carry the same id, which is the
    # whole point of having one.
    middleware_record = next(r for r in logged if r.getMessage().startswith("request "))
    assert "unhandled exception" in middleware_record.getMessage()
    assert rid_of(middleware_record.getMessage()) == rid_of(handler_record.getMessage())


# --- route logging -----------------------------------------------------------
def test_asr_logs_the_decode_without_the_transcript(stub_model_client, tiny_wav, caplog):
    caplog.set_level(logging.INFO)
    client = stub_model_client
    resp = client.post("/asr", files={"audio": ("note.wav", tiny_wav, "audio/wav")})
    assert resp.status_code == 200

    line = next(m for m in hl_stt_messages(caplog) if m.startswith("asr rid="))
    assert "model=sensevoice" in line
    assert f"audio_bytes={len(tiny_wav)}" in line
    assert re.search(r"inference_ms=\d+\.\d+", line)
    assert f"text_len={len(resp.json()['text'])}" in line

    _assert_transcript_never_logged(caplog, resp.json()["text"])


def test_transcribe_logs_the_decode_without_the_transcript(
    stub_model_client, tiny_wav, caplog
):
    caplog.set_level(logging.INFO)
    client = stub_model_client
    resp = client.post(
        "/transcribe",
        params={"model": client.stub_model},
        files={"audio": ("a.wav", tiny_wav, "audio/wav")},
    )
    assert resp.status_code == 200

    line = next(m for m in hl_stt_messages(caplog) if m.startswith("transcribe rid="))
    assert f"model={client.stub_model}" in line
    assert f"audio_bytes={len(tiny_wav)}" in line
    assert "text_len=11" in line

    _assert_transcript_never_logged(caplog, resp.json()["text"])


def test_stream_logs_open_and_final_without_the_transcript(stub_model_client, caplog):
    caplog.set_level(logging.INFO)
    client = stub_model_client
    pcm = make_wav(duration_s=2.0)[44:]

    with client.websocket_connect(
        f"/stream?model={client.stub_model}&sample_rate=16000"
    ) as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(pcm)
        ws.send_text(json.dumps({"type": "eof"}))
        message = ws.receive_json()
        while message["type"] == "partial":
            message = ws.receive_json()
        assert message["type"] == "final"

    messages = hl_stt_messages(caplog)
    opened = next(m for m in messages if m.startswith("stream rid=") and " open " in m)
    assert f"model={client.stub_model}" in opened

    final = next(m for m in messages if m.startswith("stream rid=") and " final " in m)
    assert f"audio_bytes={len(pcm)}" in final
    assert re.search(r"inference_ms=\d+\.\d+", final)
    assert "text_len=11" in final
    assert f"run_id={message['run_id']}" in final
    # The websocket never passes through the HTTP middleware, so the id has to
    # come from the route itself, and it is the same one all connection long.
    assert rid_of(final) == rid_of(opened)

    _assert_transcript_never_logged(caplog, message["text"])


def test_stream_failure_is_logged(stub_model_client, caplog):
    caplog.set_level(logging.INFO)
    with stub_model_client.websocket_connect("/stream?model=nope") as ws:
        assert ws.receive_json()["type"] == "error"

    line = next(m for m in hl_stt_messages(caplog) if "failed:" in m)
    assert "unknown model" in line
    assert re.search(RID, line)


def _assert_transcript_never_logged(caplog, text: str) -> None:
    assert text, "the stub should have produced some text to look for"
    for record in caplog.records:
        assert text not in record.getMessage()
