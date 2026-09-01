"""Shared request guards for the upload endpoints.

Both `/transcribe` and `/asr` accept unauthenticated audio, so both need the
same ceilings; keeping the check here means the two cannot drift apart.
"""

from __future__ import annotations

from fastapi import HTTPException

from ..config import settings


def reject_oversized(raw: bytes) -> None:
    """Refuse an upload larger than ``STT_MAX_UPLOAD_BYTES``.

    Checked against the bytes actually received rather than Content-Length, so
    a lying header buys nothing. This is one half of the DoS guard — it bounds
    the input. `audio.check_duration` is the other half: it bounds what that
    input is allowed to expand into, since decoding plus float conversion can
    amplify a small upload by ~1400x.
    """
    limit = settings.max_upload_bytes
    if limit > 0 and len(raw) > limit:
        raise HTTPException(
            status_code=413,
            detail=(
                f"upload is {len(raw)} bytes, over the {limit} byte limit "
                "(STT_MAX_UPLOAD_BYTES)"
            ),
        )
