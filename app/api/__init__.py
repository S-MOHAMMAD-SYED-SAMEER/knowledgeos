"""HTTP routes.

The two probes, upload and read for documents, read for an ingestion job, and
— from milestone 5 — `POST /query` for hybrid retrieval. Answer, feedback and
delete endpoints arrive with the milestones that can actually honour them.
"""

from app.api import documents, health, ingestion, query, ready

__all__ = ["documents", "health", "ingestion", "query", "ready"]
