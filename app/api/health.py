"""Liveness.

This endpoint touches nothing. No database, no provider, no filesystem — and
that is the entire contract, not an optimisation. A liveness probe that failed
because PostgreSQL blinked would have an orchestrator restart a process that
was working perfectly.

Whether this process should be *sent work* is a different question, and it is
answered by `/ready`.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app import __version__
from app.config import Settings, get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse)
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=__version__,
        environment=settings.environment,
    )


__all__ = ["HealthResponse", "health", "router"]
