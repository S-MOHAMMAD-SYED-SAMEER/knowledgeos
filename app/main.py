"""Application entrypoint: `uvicorn app.main:app --reload`.

Milestone 5 added `POST /query`; milestone 8 adds `GET /queries/{id}`
alongside it, on the same router — both retrieval-and-generation routes live
in `app/api/query.py`. Milestone 10 adds `POST /answers/{id}/feedback`
(`app/api/feedback.py`) and the server-rendered UI (`app/ui/routes.py`,
mounted under `/ui`, `include_in_schema=False` on every route — it never
appears in `/openapi.json` and `tests/test_health.py`'s own exact-path
guard proves it). The JSON API's HTTP surface is those routes, the two
probes, `POST /documents`, `POST /documents/{id}/versions`, `GET
/documents`, `GET /documents/{id}` and `GET /ingestion/{job_id}`, and
`tests/test_health.py` asserts it, so a route belonging to a later
milestone cannot arrive here quietly.

What is new is behind it: the lifespan starts the ingestion runner, which
polls for queued jobs and parses and chunks them. The runner is held on the
application rather than in a module-level variable, so nothing is shared
between two applications built in one process.

`create_app()` takes its settings as an argument so a test can build an
application on a configuration of its own without reaching into a cache. The
module-level `app` is what Uvicorn imports.

Migrations are not run from here. `alembic upgrade head` is a deployment step
that runs once, before the new version starts — not something every process
races the others to do. `/ready` is what notices when it has not happened.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api import documents, feedback, health, ingestion, query, ready
from app.config import Settings, get_settings
from app.ingestion.runner import IngestionRunner
from app.ui import routes as ui_routes


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    app = FastAPI(
        title=resolved.app_name,
        version=__version__,
        debug=resolved.debug,
        lifespan=_lifespan,
    )
    # Kept on the application so the lifespan can reach it without a global,
    # and so two applications in one process each get their own.
    app.state.settings = resolved
    app.state.runner = None

    app.include_router(health.router)
    app.include_router(ready.router)
    app.include_router(documents.router)
    app.include_router(ingestion.router)
    app.include_router(query.router)
    app.include_router(feedback.router)
    app.include_router(ui_routes.router)
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the ingestion runner on the way up; stop it on the way down.

    Shutdown waits for a job in flight. A job holds a row lock and an open
    transaction, and stopping it mid-write is how a version ends up with half
    its chunks.
    """
    runner = IngestionRunner(settings=getattr(app.state, "settings", None))
    app.state.runner = runner
    runner.start()
    try:
        yield
    finally:
        runner.stop()
        app.state.runner = None


app = create_app()


__all__ = ["app", "create_app"]
