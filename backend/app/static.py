"""Serving the built SPA from the same process as the API.

One container, one origin: uvicorn serves ``frontend/dist`` and the API routes
keep their paths, so there is no CORS hop in production.

Implemented as a **404 fallback**, not a ``Mount`` at ``/``. That distinction
matters: a mount at ``/`` matches every path, and Starlette prefers a full match
over the *partial* match it records for a right-path/wrong-method route — so a
mount would silently turn ``GET /transcribe`` (a POST-only route) from a 405
into a 404. Running after the router instead leaves routing precedence exactly
as the API defines it, and only steps in once the router has given up.

Guarded at the call site: a checkout without a built frontend (a fresh clone,
CI, the test suite) skips the fallback instead of failing to boot.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.staticfiles import StaticFiles

logger = logging.getLogger("hl-stt")

# Path prefixes owned by the API. A miss under one of these stays a 404 —
# returning the HTML shell with status 200 for a typo'd endpoint turns an
# obvious client bug into a confusing "why is my JSON an HTML page".
API_PREFIXES: tuple[str, ...] = (
    "/health",
    "/models",
    "/transcribe",
    "/runs",
    "/stream",
    "/docs",
    "/redoc",
    "/openapi.json",
)


def is_api_path(path: str) -> bool:
    """True if ``path`` belongs to the API rather than the SPA."""
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in API_PREFIXES)


def mount_spa(app: FastAPI, directory: Path) -> bool:
    """Serve the built frontend as a 404 fallback. False if it isn't there."""
    index = directory / "index.html"
    if not index.is_file():
        logger.info(
            "no built frontend at %s — API only (run `npm run build` in frontend/)",
            directory,
        )

        @app.get("/", include_in_schema=False)
        def _no_ui() -> dict:
            # Without a UI, "/" should say so rather than 404 with no explanation.
            return {
                "status": "api-only",
                "detail": f"no built frontend at {directory}",
                "hint": "build the frontend, or set STT_STATIC_DIR",
            }

        return False

    # `html=True` makes StaticFiles resolve a directory to its index.html.
    files = StaticFiles(directory=str(directory), html=True)

    @app.middleware("http")
    async def spa_fallback(request: Request, call_next) -> Response:
        response = await call_next(request)

        if response.status_code != 404:
            return response
        # 405s, POSTs to nowhere, and unknown API paths are all real errors.
        if request.method.upper() not in ("GET", "HEAD"):
            return response
        if is_api_path(request.url.path):
            return response

        relative = request.url.path.lstrip("/")
        try:
            # A real file (hashed bundle, favicon) is served verbatim...
            return await files.get_response(relative, request.scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
        try:
            # ...and anything else is a client-side route: return the shell so a
            # refresh on /reports works.
            return await files.get_response("index.html", request.scope)
        except StarletteHTTPException:  # pragma: no cover - dist deleted at runtime
            return JSONResponse(
                {"detail": "frontend index.html is missing"}, status_code=500
            )

    logger.info("serving frontend from %s", directory)
    return True


__all__ = ["API_PREFIXES", "is_api_path", "mount_spa"]
