"""The engine and the sessions that come from it.

Both are lazy and cached, which matters for two reasons. Importing this module
must not open a connection — otherwise collecting the test suite on a machine
with no PostgreSQL fails before a single test runs. And a cache with an
explicit `reset_engine()` is a global that can be changed on purpose, which a
module-level `engine = create_engine(...)` is not.

This is the only module in the application that builds an engine.
"""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url,
        # Reconnect rather than surfacing a stale-connection error on the
        # first query after the server has closed a pooled connection.
        pool_pre_ping=True,
        # Echoed SQL carries document titles and filenames, so it follows
        # `debug` rather than being on in development by default.
        echo=settings.debug,
    )


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a session closed when the request ends."""
    with get_sessionmaker()() as session:
        yield session


def reset_engine() -> None:
    """Drop the cached engine and sessionmaker, disposing the pool first.

    For tests, and for anything else that changes `database_url` underneath a
    running process.
    """
    if get_engine.cache_info().currsize:
        get_engine().dispose()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


__all__ = ["get_engine", "get_session", "get_sessionmaker", "reset_engine"]
