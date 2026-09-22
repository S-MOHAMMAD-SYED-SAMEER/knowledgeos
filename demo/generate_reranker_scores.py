"""Generates `demo/fixtures/reranker_scores.json` from the real
`cross-encoder/ms-marco-MiniLM-L-6-v2` model.

Run once, in an environment with the model available in the local
sentence-transformers cache::

    python -m demo.generate_reranker_scores

Unlike `demo/generate_embeddings.py` (which needs no database -- chunking
is a pure function of the manifest), a reranker score is only meaningful
for a (query, chunk) pair that milestone 5's real retrieval pipeline
would actually hand to reranking. So this script seeds the real demo
corpus into the database named by `KNOWLEDGEOS_TEST_DATABASE_URL`, runs
`app.retrieval.pipeline.retrieve()` unmodified for each of the five
flagship queries (`demo/generate_embeddings.py::FLAGSHIP_QUERIES`), and
scores exactly the top-20 RRF-fused candidates each one produces -- the
same candidate set `app.reranking.pipeline.rerank()` would receive in the
real pipeline. Never a blind full corpus x query cross-product: 5 queries
x 20 candidates = 100 (query, chunk_uid) pairs, not 5 x 33 = 165.

The database is migrated to head, seeded, scored, and migrated back down
to base when this finishes -- the same discipline `tests/conftest.py`
applies to every test that touches the database, so running this script
never leaves demo rows behind in whatever database it was pointed at.

Real scores only. `app.providers.cross_encoder.CrossEncoderRerankProvider`
is used exactly as production would use it -- unmodified, and its own
`local_files_only=True` discipline means this script raises rather than
downloading the model if it is not already cached, the same offline
guarantee every other real-model provider in this project keeps.
"""

import hashlib
import json
import os
from pathlib import Path

from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import reset_engine
from app.providers.cross_encoder import MODEL_NAME, CrossEncoderRerankProvider
from app.providers.reranker import Candidate
from app.retrieval.filters import RetrievalFilters
from app.retrieval.pipeline import retrieve
from app.storage import LocalStorage, reset_storage
from demo.generate_embeddings import FLAGSHIP_QUERIES
from demo.providers import DemoEmbeddingProvider
from demo.seed import seed_demo_corpus

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
OUTPUT_PATH = FIXTURES_DIR / "reranker_scores.json"

# The specification's number: "top 20 into reranking"
# (app.retrieval.pipeline.FINAL_CANDIDATE_LIMIT). Scoring anything beyond
# what the real pipeline would ever hand to a RerankProvider would not be
# replaying real behaviour -- it would be inventing coverage nothing asks
# for.
EXPECTED_CANDIDATES_PER_QUERY = 20


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate(output_path: Path = OUTPUT_PATH) -> None:
    database_url = os.environ.get("KNOWLEDGEOS_TEST_DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "KNOWLEDGEOS_TEST_DATABASE_URL must be set to a database whose "
            "name ends in '_test' -- this script downgrades it to base when "
            "finished, the same safety rule tests/conftest.py enforces"
        )
    if not database_url.rstrip("/").endswith("_test"):
        raise RuntimeError(
            f"refusing to migrate {database_url!r}: it must end in '_test'"
        )

    import tempfile

    os.environ["KNOWLEDGEOS_ENVIRONMENT"] = "test"
    storage_root = tempfile.mkdtemp()
    os.environ["KNOWLEDGEOS_STORAGE_ROOT"] = storage_root
    os.environ["KNOWLEDGEOS_DATABASE_URL"] = database_url
    get_settings.cache_clear()
    reset_engine()
    reset_storage()

    config = AlembicConfig(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    command.downgrade(config, "base")
    command.upgrade(config, "head")

    engine = create_engine(database_url)
    reranker = CrossEncoderRerankProvider()  # raises if the model is not cached
    if reranker.model_name != MODEL_NAME:  # pragma: no cover - defensive
        raise RuntimeError(f"unexpected reranker model: {reranker.model_name}")

    query_payloads: list[dict] = []
    try:
        with Session(engine) as session:
            storage = LocalStorage(storage_root)
            seed_demo_corpus(session, storage)

            embeddings = DemoEmbeddingProvider()
            for query_id, query_text in FLAGSHIP_QUERIES:
                [query_vector] = embeddings.embed([query_text])
                result = retrieve(
                    session,
                    normalized_text=query_text,
                    query_vector=query_vector,
                    filters=RetrievalFilters(),
                )
                candidates = result.candidates
                if len(candidates) != EXPECTED_CANDIDATES_PER_QUERY:
                    raise RuntimeError(
                        f"{query_id}: retrieve() returned "
                        f"{len(candidates)} candidate(s), expected exactly "
                        f"{EXPECTED_CANDIDATES_PER_QUERY} -- the demo "
                        f"corpus or the pinned retrieval configuration has "
                        f"changed"
                    )

                provider_candidates = [
                    Candidate(id=c.evidence.chunk_uid, text=c.evidence.text)
                    for c in candidates
                ]
                scored = reranker.rerank(query_text, provider_candidates)
                score_by_id = {item.id: item.score for item in scored}

                query_payloads.append(
                    {
                        "id": query_id,
                        "text": query_text,
                        "text_sha256": _sha256(query_text),
                        "candidates": [
                            {
                                "chunk_uid": c.evidence.chunk_uid,
                                "chunk_text_sha256": _sha256(c.evidence.text),
                                "document_title": c.evidence.document_title,
                                "sequence": c.evidence.sequence,
                                "fusion_rank": c.final_rank,
                                "rerank_score": score_by_id[c.evidence.chunk_uid],
                            }
                            for c in candidates
                        ],
                    }
                )
    finally:
        engine.dispose()
        command.downgrade(config, "base")
        get_settings.cache_clear()
        reset_engine()
        reset_storage()

    payload = {
        "model_name": reranker.model_name,
        "candidates_per_query": EXPECTED_CANDIDATES_PER_QUERY,
        "queries": query_payloads,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    generate()
