"""The one place a mutation route checks `Settings.demo_mode` before
running.

**M4 finding, fixed here.** M2's own design required demo mode to expose
"no uploaded document accepted" and "no ingestion mutation exposed" — but
that was only ever true of the visitor UI (`app/ui/demo_routes.py`, M3),
which simply never renders an upload/reindex/feedback form. The underlying
JSON API routes (`POST /documents`, `POST /documents/{id}/versions`,
`POST /documents/{id}/reindex`, `POST /answers/{id}/feedback`, and the
UI's own `POST /ui/answers/{id}/feedback`) were never gated by
`demo_mode` at all, and stayed fully live and mutable regardless of it — a
real visitor could `POST /documents` against a "credential-free,
fixed-corpus" public demo and permanently change what it serves. M4's
public-demo safety review is what found this; this module is the minimal
fix, not new infrastructure: one dependency, added to the five routes that
mutate.

Query, retrieval, reranking, citations and read-only document browsing are
all still allowed in demo mode, per M1's own boundary — this module is
never used on a `GET` route.
"""

from typing import Annotated

from fastapi import Depends, HTTPException

from app.config import Settings, get_settings


def require_mutation_allowed(
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Refuse a mutating request outright when `Settings.demo_mode` is on.

    403, not 404: unlike `app/ui/demo_routes.py`'s own guard (a *demo*
    route that does not exist on a non-demo deployment, so 404 is
    accurate), these are real, permanent, non-demo routes that do exist —
    demo mode is refusing the action they perform, not claiming the route
    itself is unknown.
    """
    if settings.demo_mode:
        raise HTTPException(
            status_code=403,
            detail=(
                "This deployment is running in demo mode, which serves a "
                "fixed corpus and accepts no document uploads, "
                "re-indexing, or feedback."
            ),
        )


__all__ = ["require_mutation_allowed"]
