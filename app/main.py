"""Application entrypoint: `uvicorn app.main:app --reload`.

Milestone 2 serves the two probes, `POST /documents`,
`POST /documents/{id}/versions`, `GET /documents`, `GET /documents/{id}`
and `GET /ingestion/{job_id}`. That is the whole HTTP surface, and
`tests/test_health.py` asserts it, so a route belonging to a later
milestone cannot arrive here quietly.

`create_app()` takes its settings as an argument so a test can build an
application on a configuration of its own without reaching into a cache. The
module-level `app` is what Uvicorn imports.

Migrations are not run from here. `alembic upgrade head` is a deployment step
that runs once, before the new version starts — not something every process
races the others to do. `/ready` is what notices when it has not happened.
"""

from fastapi import FastAPI

from app import __version__
from app.api import documents, health, ingestion, ready
from app.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    app = FastAPI(
        title=resolved.app_name,
        version=__version__,
        debug=resolved.debug,
    )
    app.include_router(health.router)
    app.include_router(ready.router)
    app.include_router(documents.router)
    app.include_router(ingestion.router)
    return app


app = create_app()


__all__ = ["app", "create_app"]
