"""Archiving every incoming clip as training / benchmark data.

Each decode that lands a `runs` row also writes its raw upload — the exact bytes
the client sent, not a re-encode — to ``STT_DATASET_DIR/<run_id>.<ext>``. Paired
with the run's ``expected_text`` (editable from the Reports tab) that gives a
labelled corpus for fine-tuning later, and it accumulates for free from ordinary
use.

The filename is the run id, so a clip is always traceable back to the row that
scored it, and the directory needs no index of its own. Only the **bare
filename** is stored on the run — never the absolute server path, because
``GET /runs`` is unauthenticated and would otherwise disclose the host layout.
Resolve it with :func:`resolve` when the file itself is needed.

Storage is best-effort by design: a full disk or a read-only mount must not turn
a successful transcription into a failed request, so failures are logged and the
run is simply left with ``audio_path = NULL``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .audio import dataset_extension
from .config import settings

logger = logging.getLogger("hl-stt")


def save_clip(
    run_id: int,
    data: bytes,
    directory: Path | None = None,
) -> str | None:
    """Archive `data` for `run_id`. Returns the bare filename, or None.

    The extension is sniffed from the bytes and never taken from the upload's
    filename, so a Telegram voice note lands as `.ogg` and a browser recording
    as `.webm` regardless of what the client claimed.
    """
    if not data:
        return None

    target_dir = settings.dataset_dir if directory is None else directory
    name = f"{run_id}.{dataset_extension(data)}"
    path = target_dir / name

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive create: never overwrite, and never adopt.
        with open(path, "xb") as handle:
            handle.write(data)
    except FileExistsError:
        # A file is already parked on this run id, which means the id counter
        # restarted — reset the database while storage/dataset/ survives and
        # ids begin again at 1. The existing file belongs to a *different*,
        # older run, so returning it would staple this transcript to someone
        # else's audio and quietly poison the very corpus this exists to build.
        # Leave audio_path NULL instead; a missing label beats a wrong one.
        logger.warning(
            "dataset: %s already exists — refusing to link run %d to a "
            "pre-existing file (stale dataset dir? database reset?)",
            path,
            run_id,
        )
        return None
    except OSError as exc:
        # Never fail the request over the archive.
        logger.warning("dataset: could not write %s (%s)", path, exc)
        return None

    logger.info("dataset: archived run %d -> %s (%d bytes)", run_id, name, len(data))
    return name


def resolve(audio_path: str, directory: Path | None = None) -> Path:
    """Absolute path for a stored `audio_path`, for server-side reads.

    Only the bare filename is persisted, so this is where it becomes a real
    path. `Path.name` strips any directory part, so a legacy absolute value or
    a traversal attempt both collapse to a plain filename inside the dataset
    dir.
    """
    target_dir = settings.dataset_dir if directory is None else directory
    return target_dir / Path(audio_path).name
