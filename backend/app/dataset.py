"""Archiving every incoming clip as training / benchmark data.

Each decode that lands a `runs` row also writes its raw upload — the exact bytes
the client sent, not a re-encode — to ``STT_DATASET_DIR/<run_id>.<ext>``. Paired
with the run's ``expected_text`` (editable from the Reports tab) that gives a
labelled corpus for fine-tuning later, and it accumulates for free from ordinary
use.

The filename is the run id, so a clip is always traceable back to the row that
scored it, and the directory needs no index of its own.

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
    filename: str | None = None,
    directory: Path | None = None,
) -> str | None:
    """Archive `data` for `run_id`. Returns the path written, or None on failure.

    The extension is sniffed from the bytes rather than trusted from the upload,
    so a Telegram voice note lands as `.ogg` and a browser recording as `.webm`
    regardless of what the client claimed.
    """
    if not data:
        return None

    target_dir = settings.dataset_dir if directory is None else directory
    extension = dataset_extension(data, filename)
    path = target_dir / f"{run_id}.{extension}"

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive create: a colliding run id means something is very wrong,
        # and silently overwriting a previous clip would lose data.
        with open(path, "xb") as handle:
            handle.write(data)
    except FileExistsError:
        logger.warning("dataset: %s already exists, not overwriting", path)
        return str(path)
    except OSError as exc:
        # Never fail the request over the archive.
        logger.warning("dataset: could not write %s (%s)", path, exc)
        return None

    logger.info("dataset: archived run %d -> %s (%d bytes)", run_id, path, len(data))
    return str(path)
