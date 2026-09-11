"""Per-request logging.

Every HTTP request gets a short id. That id goes on the request's own log line
and on anything logged while the request is in flight, so an error in the log
can be tied back to the request that produced it instead of matched by
timestamp and hope.

Only ``http`` scopes pass through here, so the websocket endpoint is not
covered — ``routes/stream.py`` binds its own id and logs its own lifecycle.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from contextvars import ContextVar, Token

from fastapi import Request, Response
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("hl-stt")

#: 8 hex chars stays readable in a log line and is plenty to tell apart the
#: handful of requests a single-worker box has in flight at once.
REQUEST_ID_LENGTH = 8

#: Stands in for a value we do not have — the request id outside a request
#: (startup, scripts).
UNKNOWN = "-"

#: Logged at DEBUG when they succeed. The compose healthcheck polls /health
#: every 30s, which is ~2900 lines a day saying nothing; a failing one is still
#: logged at WARNING like any other, because that is the case worth reading.
QUIET_PATHS: frozenset[str] = frozenset({"/health"})

_request_id: ContextVar[str] = ContextVar("hl_stt_request_id", default=UNKNOWN)

#: Anything a log reader would take as a line ending: the C0 controls, DEL, and
#: the two Unicode line separators. A request for "/%0aINFO ..." is not a line.
_LINE_BREAKERS = re.compile("[\x00-\x1f\x7f\u2028\u2029]")


def new_request_id() -> str:
    return uuid.uuid4().hex[:REQUEST_ID_LENGTH]


def current_request_id() -> str:
    """Id of the request being served on this task, or :data:`UNKNOWN`."""
    return _request_id.get()


def bind_request_id(request_id: str) -> Token[str]:
    """Make ``request_id`` current for this task. Reset the token when done."""
    return _request_id.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    _request_id.reset(token)


def safe(value: str, limit: int = 200) -> str:
    """A caller-controlled string, made safe to put in a log line.

    Strips what would otherwise end the line early, and marks a truncation
    rather than quietly dropping the tail — a value cut off without a marker
    reads as the whole value.
    """
    cleaned = _LINE_BREAKERS.sub("?", value)
    if len(cleaned) > limit:
        return f"{cleaned[:limit]}…[truncated]"
    return cleaned


class RequestLoggingMiddleware:
    """One line per request: method, path, status, duration, request id.

    Plain ASGI rather than ``BaseHTTPMiddleware`` on purpose. The id stays
    bound for as long as the response is being written, so a streaming route
    logs under its own id, and the duration covers the body instead of stopping
    at the headers.

    No client address: the only peer this process ever has is Caddy, so the
    field would be the same proxy IP on every line. Caddy's own access log
    holds the real client, and reading it out of ``X-Forwarded-For`` here would
    mean trusting a header — uvicorn can be told which proxies to trust, but
    that also rewrites the request scheme and so the URLs the app generates,
    which is more than a logging change should do.

    Added last in :func:`app.main.create_app` so it is the outermost middleware
    and sees the status the client actually gets.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = new_request_id()
        # Both, on purpose: the scope carries the id out to the exception
        # handler, which builds a Request of its own, and the context var lets
        # a route log the id without taking a Request it has no other use for.
        scope.setdefault("state", {})["request_id"] = request_id
        token = bind_request_id(request_id)

        method = safe(scope.get("method", UNKNOWN))
        path = safe(scope.get("path", ""))
        started = time.perf_counter()
        status: int | None = None
        logged = False

        async def send_logging(message: Message) -> None:
            nonlocal status, logged
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)
            done = message["type"] == "http.response.body" and not message.get(
                "more_body", False
            )
            if done and status is not None:
                logger.log(
                    _level_for(status, path),
                    "request rid=%s %s %s -> %d in %.1fms",
                    request_id,
                    method,
                    path,
                    status,
                    _elapsed_ms(started),
                )
                logged = True

        try:
            await self.app(scope, receive, send_logging)
        except Exception:
            # The traceback belongs to the exception handler, which sits
            # outside this middleware and has no timing to report.
            logger.error(
                "request rid=%s %s %s -> unhandled exception in %.1fms",
                request_id,
                method,
                path,
                _elapsed_ms(started),
            )
            logged = True
            raise
        finally:
            if not logged:
                # The response never finished: a client that hung up mid-body,
                # or a shutdown. Worth a line of its own rather than silence.
                logger.warning(
                    "request rid=%s %s %s -> %s incomplete after %.1fms",
                    request_id,
                    method,
                    path,
                    status if status is not None else "no response",
                    _elapsed_ms(started),
                )
            reset_request_id(token)


async def log_unhandled_exception(request: Request, exc: Exception) -> Response:
    """Log the traceback of an error no route handled, then 500 as usual.

    Registered for bare ``Exception``, which Starlette only consults once every
    other handler has declined, so ``HTTPException`` and validation errors keep
    their own responses. The response here is byte-for-byte the one Starlette
    would have produced on its own: clients see no change, the log gains the
    traceback and the request id.
    """
    request_id = getattr(request.state, "request_id", None) or current_request_id()
    logger.error(
        "unhandled exception rid=%s %s %s",
        request_id,
        safe(request.method),
        safe(request.url.path),
        exc_info=exc,
    )
    return PlainTextResponse("Internal Server Error", status_code=500)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _level_for(status_code: int, path: str) -> int:
    """5xx is ours to fix, 4xx is worth seeing, the rest is routine."""
    if status_code >= 500:
        return logging.ERROR
    if status_code >= 400:
        return logging.WARNING
    if path in QUIET_PATHS:
        return logging.DEBUG
    return logging.INFO
