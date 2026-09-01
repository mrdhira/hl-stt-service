"""FastAPI application for the hl-stt-services POC + benchmark harness."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import settings
from .db import db
from .models import SHERPA_AVAILABLE, SHERPA_IMPORT_ERROR, SHERPA_VERSION, registry
from .routes import (
    asr_router,
    models_router,
    runs_router,
    stream_router,
    transcribe_router,
)
from .static import mount_spa

logger = logging.getLogger("hl-stt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    db.connect()
    logger.info("runs database: %s (%d rows)", db.path, db.count_runs())

    if not SHERPA_AVAILABLE:
        logger.warning(
            "sherpa-onnx unavailable (%s) — /models will report every model as "
            "unavailable and /transcribe will return 503",
            SHERPA_IMPORT_ERROR,
        )
    else:
        logger.info("sherpa-onnx %s, %d threads", SHERPA_VERSION, settings.num_threads)

    registry.scan()
    # Models stay resident once loaded, so pay the construction cost at startup
    # rather than on the first request. Missing models are simply skipped.
    for key, status in registry.preload_available().items():
        logger.info("model %-11s %s (%s)", key, status, settings.model_dir(key))

    yield

    db.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="hl-stt-services",
        version=__version__,
        summary="Offline speech-to-text POC + benchmark harness (English + Japanese, CPU-only)",
        lifespan=lifespan,
    )

    # The Vite dev server talks to this API from another origin; in production
    # Caddy serves both from stt.home.arpa so this is a no-op.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(models_router)
    app.include_router(transcribe_router)
    app.include_router(asr_router)
    app.include_router(runs_router)
    app.include_router(stream_router)

    @app.get("/health", tags=["health"])
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "sherpa_onnx": SHERPA_VERSION,
            "models_available": registry.available_keys(),
        }

    # LAST: the SPA mount matches "/" and everything under it, so every API
    # route above must already be registered or it would be shadowed.
    mount_spa(app, settings.static_dir)

    return app


app = create_app()


def main() -> None:
    """Entry point for ``python -m app.main``."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,  # models are held in-process; a second worker doubles the RAM
        log_level="info",
    )


if __name__ == "__main__":
    main()
