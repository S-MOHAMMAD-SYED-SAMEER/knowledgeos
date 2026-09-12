"""HTTP routes.

Milestone 2 serves the two probes, the upload and read endpoints for
documents, and the read endpoint for an ingestion job. Query, retrieval,
answer, reindex and delete endpoints arrive with the milestones that can
actually honour them.
"""

from app.api import documents, health, ingestion, ready

__all__ = ["documents", "health", "ingestion", "ready"]
