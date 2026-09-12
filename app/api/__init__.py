"""HTTP routes.

Milestone 1 serves the two probes and nothing else. Documents, ingestion,
query and feedback endpoints arrive with the milestones that can actually
answer them.
"""

from app.api import health, ready

__all__ = ["health", "ready"]
