"""Readiness: is this process fit to be given work?

Three checks, and each one is a way a deployment is quietly broken rather than
a way it is down:

1. **The database answers.** One `SELECT 1`. Everything this system stores
   lives in PostgreSQL, so a process that cannot reach it can do nothing.
2. **The schema is the one this build expects.** The stamped Alembic revision
   against the head in this checkout. A process running ahead of its
   migrations writes to columns that do not exist; one running behind is a
   deployment half-finished.
3. **The `vector` extension is installed.** It is the foundation the whole
   retrieval half of this project is built on, it is installed by a step
   outside the application, and a database missing it fails much later and far
   less clearly. Nothing declares a vector column yet — the check is here
   because milestone 1 exists to prove this foundation is real.

No provider is called. There are none in milestone 1, and when there are, a
readiness probe that depended on somebody else's rate limit would take every
replica out of rotation for a reason unrelated to whether this service works.

The connection string never appears in a response or a log line: it carries a
password. Failures are reported by exception class name only.
"""

import logging
import pathlib
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from app.config import Settings, get_settings
from app.db.session import get_engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

EXTENSION = "vector"


class Check(BaseModel):
    ok: bool
    detail: str = ""


class ReadyResponse(BaseModel):
    status: Literal["ready", "not ready"]
    database: Check
    migrations: Check
    extension: Check


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"model": ReadyResponse}},
)
def ready(
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ReadyResponse:
    """200 when this process can do its job, 503 when it cannot."""
    del settings  # Each check reads what it needs.

    database = _database()
    # Both remaining checks need a connection, so with the database down there
    # is nothing to compare against and nothing to look in.
    unreachable = Check(ok=False, detail="not checked: the database is unreachable")
    migrations = _migrations() if database.ok else unreachable
    extension = _extension() if database.ok else unreachable

    ok = database.ok and migrations.ok and extension.ok
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadyResponse(
        status="ready" if ok else "not ready",
        database=database,
        migrations=migrations,
        extension=extension,
    )


# --- the three checks ------------------------------------------------------


def _database() -> Check:
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - every failure is the same answer
        # The class name, never the message and never the URL: one carries the
        # host, the other carries the password.
        logger.warning("Readiness: the database is unreachable (%s).", _named(exc))
        return Check(ok=False, detail=f"unreachable ({_named(exc)})")
    return Check(ok=True, detail="reachable")


def _migrations() -> Check:
    try:
        from alembic.config import Config as AlembicConfig
        from alembic.runtime.migration import MigrationContext
        from alembic.script import ScriptDirectory

        config = AlembicConfig(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "alembic"))
        head = ScriptDirectory.from_config(config).get_current_head()

        with get_engine().connect() as connection:
            current = MigrationContext.configure(connection).get_current_revision()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Readiness: cannot read the migration state (%s).", _named(exc))
        return Check(ok=False, detail=f"unknown ({_named(exc)})")

    if current == head:
        return Check(ok=True, detail=f"at head ({head})")
    return Check(
        ok=False,
        detail=f"database is at {current or 'nothing'}, this build expects {head}",
    )


def _extension() -> Check:
    try:
        with get_engine().connect() as connection:
            installed = connection.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = :name"),
                {"name": EXTENSION},
            ).scalar_one_or_none()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Readiness: cannot read the extension list (%s).", _named(exc))
        return Check(ok=False, detail=f"unknown ({_named(exc)})")

    if installed is None:
        return Check(ok=False, detail=f"{EXTENSION} is not installed")
    return Check(ok=True, detail=f"{EXTENSION} {installed}")


def _named(exc: BaseException) -> str:
    return type(exc).__name__


__all__ = ["EXTENSION", "Check", "ReadyResponse", "ready", "router"]
