"""GET /runs — recent benchmark rows; PATCH /runs/{id} — correct ground truth."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query

from ..db import db
from ..schemas import ExpectedTextUpdate, RunRow, RunsResponse

router = APIRouter(tags=["runs"])


@router.get("/runs", response_model=RunsResponse)
def list_runs(
    limit: int = Query(50, ge=1, le=1000, description="Max rows, newest first"),
    model: str | None = Query(None, description="Filter by model key"),
    mode: str | None = Query(None, description="Filter by mode (batch | stream | asr)"),
) -> RunsResponse:
    rows = db.recent_runs(limit=limit, model=model, mode=mode)
    return RunsResponse(count=len(rows), runs=[RunRow(**r) for r in rows])


@router.patch("/runs/{run_id}", response_model=RunRow)
def update_run(
    payload: ExpectedTextUpdate,
    run_id: int = Path(..., ge=1, description="Run id to correct"),
) -> RunRow:
    """Set or clear a run's ground truth.

    This is how a recording becomes a labelled test case after the fact: the
    Reports tab shows what the model heard next to what was actually said, and
    the correction lands here. Only `expected_text` is mutable — the measured
    numbers are a record of what happened and must not be editable.
    """
    # Blank input clears the label rather than storing "", so "no ground truth"
    # has exactly one representation and WER stays undefined for that row.
    expected = (payload.expected_text or "").strip() or None

    if not db.update_expected_text(run_id, expected):
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")

    row = db.get_run(run_id)
    if row is None:  # pragma: no cover - deleted between UPDATE and SELECT
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return RunRow(**row)
