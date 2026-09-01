# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# stage: frontend — build the Vite SPA
# ---------------------------------------------------------------------------
FROM node:22-alpine AS frontend

WORKDIR /build

# package files first: this layer is cached until the dependencies change.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ---------------------------------------------------------------------------
# stage: runtime — uvicorn serving both the API and the built SPA
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# curl: the compose healthcheck. ffmpeg: POST /asr decodes webm/opus and mp4,
# which libsndfile cannot read (no Matroska demuxer). sherpa-onnx and soundfile
# both ship manylinux wheels, so no build toolchain is needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STT_HOST=0.0.0.0 \
    STT_PORT=8000 \
    STT_MODELS_DIR=/data/models \
    STT_DB_PATH=/data/db/stt-runs.db \
    STT_DATASET_DIR=/data/dataset \
    STT_STATIC_DIR=/app/frontend/dist \
    STT_NUM_THREADS=4

WORKDIR /app

# Requirements before source: dependency layer survives code edits.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Application code. .dockerignore keeps .venv, __pycache__ and tests' caches out.
COPY backend/ ./backend/
COPY --from=frontend /build/dist ./frontend/dist

# Non-root. The uid must match the owner of the bind-mounted ./storage on the
# host, or SQLite cannot create the database — 1000 is the first human user on a
# typical single-user homelab box. Override for a different host uid:
#   docker compose build --build-arg APP_UID=$(id -u)
ARG APP_UID=1000
ARG APP_GID=1000

# /data is a mount point at runtime; create it here so the container still
# starts if the volume is missing.
RUN groupadd --gid ${APP_GID} appuser 2>/dev/null || true \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home \
        --shell /usr/sbin/nologin appuser 2>/dev/null || true \
    && mkdir -p /data/models /data/db /data/dataset \
    && chown -R ${APP_UID}:${APP_GID} /app /data
USER ${APP_UID}:${APP_GID}

WORKDIR /app/backend

EXPOSE 8000

# Single worker on purpose: the recognizers are held in-process, so a second
# worker would double the resident model memory.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
