"""Seeding the demo database.

The demo corpus is exactly the evaluation fixture corpus
(`evals/fixtures/knowledge_base/manifest.yaml`) -- ten documents, eleven
versions -- indexed through the real ingestion pipeline
(`evals.retrieval.corpus.ensure_corpus_seeded`), the same function
`tests/test_evals_corpus.py` already exercises. The only thing the demo
changes is which `EmbeddingProvider` does the embedding:
`DemoEmbeddingProvider` replays real, precomputed BAAI/bge-small-en-v1.5
vectors instead of loading the model, so seeding a demo works with no
HuggingFace access and no model inference at seed time -- and without
touching `app/retrieval`, `app/reranking`, `app/generation`, or the
existing evaluation harness.
"""

from sqlalchemy.orm import Session

from app.storage import Storage
from demo.providers import DemoEmbeddingProvider
from evals.retrieval.corpus import CorpusSeedResult, ensure_corpus_seeded


def seed_demo_corpus(session: Session, storage: Storage) -> CorpusSeedResult:
    """Seed the demo's ten-document, 33-chunk corpus, idempotently."""
    embeddings = DemoEmbeddingProvider()
    return ensure_corpus_seeded(session, storage, embeddings)


def _main() -> None:
    """`python -m demo.seed` -- the same call docs/DEMO.md's manual snippet
    makes by hand, as a one-line command for Docker/Compose's own seed
    step. Builds the engine and storage the same way `app/` itself does
    (`app.db.session.get_engine`, `app.storage.get_storage`), both already
    driven by `KNOWLEDGEOS_DATABASE_URL`/`KNOWLEDGEOS_STORAGE_ROOT` -- no
    new configuration surface, no second seeding path.
    """
    from app.db.session import get_engine
    from app.storage import get_storage

    with Session(get_engine()) as session:
        result = seed_demo_corpus(session, get_storage())
        print(
            f"seeded {result.total_active_chunks} active chunks "
            f"across {len(result.documents)} documents "
            f"({result.newly_seeded} newly seeded, "
            f"{result.skipped_existing} already present)"
        )


if __name__ == "__main__":
    _main()


__all__ = ["seed_demo_corpus"]
