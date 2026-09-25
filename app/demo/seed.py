"""Making the fixed demo corpus available, without a second copy of it and
without exposing arbitrary ingestion.

This is a narrowly-scoped wrapper around `app.ingestion.corpus_seed.
ensure_corpus_seeded` — the same function the evaluation harness uses
(via `evals.retrieval.corpus`, now a thin re-export of this module) to
seed its own fixture corpus through the real ingestion pipeline
(`app.ingestion.service`, `app.ingestion.pipeline`). Reusing it directly
means the demo corpus is produced by the identical parse/chunk/index code
path a real upload takes, with the identical, real chunk_uids the curated
demo answers in `app/generation/demo_scenarios.py` cite — not a second,
hand-maintained copy of the corpus that could drift from the real one.
`app/ingestion/corpus_seed.py`'s own module docstring explains why the
implementation lives there rather than in `evals/`: `app/` may never
import `evals/` (`tests/test_retrieval_scope.py::
test_no_api_module_imports_the_evals_package`).

Nothing here adds a route. There is no `POST /demo/documents` and nothing
in this module accepts caller-supplied content — the only document source
is the committed `evals/fixtures/knowledge_base/` manifest, the same one
the evaluation harness already trusts.
"""

from sqlalchemy.orm import Session

from app.ingestion.corpus_seed import CorpusSeedResult, ensure_corpus_seeded
from app.ingestion.runner import get_embedding_provider
from app.storage import get_storage


def ensure_demo_corpus_seeded(session: Session) -> CorpusSeedResult:
    """Seed the fixed fixture corpus into this demo's database, once.

    Idempotent — see `ensure_corpus_seeded`'s own docstring — so this is
    safe to call on every demo-mode startup rather than needing a
    separate "has this already run" flag.

    Uses the real, cached embedding provider (`BgeEmbeddingProvider`) and
    the real, cached storage backend — the same accessors `/query` and the
    ingestion runner already use — never a fake, even in demo mode.
    """
    return ensure_corpus_seeded(session, get_storage(), get_embedding_provider())


__all__ = ["ensure_demo_corpus_seeded"]
