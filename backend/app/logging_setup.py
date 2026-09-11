"""Process-wide logging setup.

One place decides where records go and what they look like, so a line from the
request middleware, a route, and a stdlib warning all arrive in the same shape.

STDOUT on purpose. The container runs a single foreground process, so writing
to stdout is what makes ``docker logs hl-stt`` useful with no logging driver
configured. uvicorn's own loggers are re-pointed at the same handler rather
than left on stderr in their own format.
"""

from __future__ import annotations

import logging
import sys
from typing import IO

from .config import settings

#: timestamp, level, logger name, message — the shape the app already used.
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

#: Names the handler this module owns so a second call replaces it instead of
#: stacking a duplicate. ``uvicorn --reload`` and the tests both re-enter.
HANDLER_NAME = "hl-stt-stdout"

#: uvicorn installs these with handlers of its own the moment it starts.
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def resolve_level(value: str | int | None) -> int:
    """``"debug"`` / ``10`` / nonsense -> a level int (INFO for nonsense)."""
    if isinstance(value, int):
        return value
    level = logging.getLevelName(str(value or "").strip().upper())
    # getLevelName returns "Level %s" for anything it does not know.
    return level if isinstance(level, int) else logging.INFO


def setup_logging(
    level: str | int | None = None,
    *,
    stream: IO[str] | None = None,
) -> logging.Logger:
    """Point every logger at one STDOUT handler at ``STT_LOG_LEVEL``.

    ``level`` and ``stream`` exist for tests and for a caller that knows better
    than the environment; both default to the configured behaviour. Returns the
    ``hl-stt`` logger so callers need not re-import :mod:`logging` for it.
    """
    resolved = resolve_level(settings.log_level if level is None else level)

    root = logging.getLogger()
    for existing in [h for h in root.handlers if h.get_name() == HANDLER_NAME]:
        root.removeHandler(existing)
        existing.close()

    handler = logging.StreamHandler(sys.stdout if stream is None else stream)
    handler.set_name(HANDLER_NAME)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(resolved)

    # uvicorn logs its startup through its own handlers on stderr. Hand its
    # records to ours instead so one stream carries everything in one format.
    # Its access log stays quiet: the request middleware emits that line
    # already, with a duration and a request id attached.
    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
        # uvicorn's own dictConfig pins a level on each of these, which would
        # otherwise outrank the root level and make STT_LOG_LEVEL a half-truth.
        uvicorn_logger.setLevel(logging.NOTSET)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    return logging.getLogger("hl-stt")
