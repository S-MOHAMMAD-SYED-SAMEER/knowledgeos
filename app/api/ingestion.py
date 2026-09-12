"""Reading the state of an ingestion job.

Read-only, and in this milestone what it reads is always `queued`: the runner
that advances a job is milestone 3. That is the honest answer to "what is
happening to my document" right now — it is waiting.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.schemas import JobOut
from app.db.session import get_session
from app.models import IngestionJob

router = APIRouter(tags=["ingestion"])


@router.get("/ingestion/{job_id}", response_model=JobOut)
def get_job(
    job_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
) -> JobOut:
    job = session.get(IngestionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job")
    return JobOut.model_validate(job)


__all__ = ["router"]
