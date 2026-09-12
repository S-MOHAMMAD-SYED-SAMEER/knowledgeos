"""Every ORM model, re-exported.

Importing this package registers all of them on `Base.metadata`, which is what
`alembic/env.py` imports and what autogenerate compares the live database
against. A model that is not reachable from here is invisible to migrations.
"""

from app.models.document import (
    SOURCE_TYPE_UPLOAD,
    SOURCE_TYPES,
    Document,
    DocumentVersion,
    VersionStatus,
)

__all__ = [
    "SOURCE_TYPES",
    "SOURCE_TYPE_UPLOAD",
    "Document",
    "DocumentVersion",
    "VersionStatus",
]
