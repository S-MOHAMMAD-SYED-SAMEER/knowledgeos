"""Compatibility re-export.

The real implementation moved to `app.ingestion.corpus_seed` (M2): the
demo mode's corpus seeding (`app/demo/seed.py`) needed this exact,
unduplicated logic, and `app/` may never import `evals/`
(`tests/test_retrieval_scope.py::test_no_api_module_imports_the_evals_package`).
Moving it the other way — `evals/` importing from `app/` — is the
direction this project's layering has always allowed, so every existing
caller of `evals.retrieval.corpus` keeps working unchanged through this
re-export.
"""

from app.ingestion.corpus_seed import (
    DEFAULT_MANIFEST,
    MINIMUM_ACTIVE_CHUNKS,
    PINNED_CHUNK_OVERLAP_TOKENS,
    PINNED_CHUNK_SIZE_TOKENS,
    CorpusError,
    CorpusSeedResult,
    SeededDocument,
    ensure_corpus_seeded,
    load_manifest,
    verify_corpus_size,
)

__all__ = [
    "DEFAULT_MANIFEST",
    "MINIMUM_ACTIVE_CHUNKS",
    "PINNED_CHUNK_OVERLAP_TOKENS",
    "PINNED_CHUNK_SIZE_TOKENS",
    "CorpusError",
    "CorpusSeedResult",
    "SeededDocument",
    "ensure_corpus_seeded",
    "load_manifest",
    "verify_corpus_size",
]
