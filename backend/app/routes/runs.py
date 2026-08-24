"""GET /runs — recent benchmark rows."""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..db import db
from ..schemas import RunRow, RunsResponse

router = APIRouter(tags=["runs"])


@router.get("/runs", response_model=RunsResponse)
def list_runs(
    limit: int = Query(50, ge=1, le=1000, description="Max rows, newest first"),
    model: str | None = Query(None, description="Filter by model key"),
    mode: str | None = Query(None, description="Filter by mode (batch | stream)"),
) -> RunsResponse:
    rows = db.recent_runs(limit=limit, model=model, mode=mode)
    return RunsResponse(count=len(rows), runs=[RunRow(**r) for r in rows])
