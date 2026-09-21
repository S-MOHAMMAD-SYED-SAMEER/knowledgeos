"""CLI entry point for the evaluation harness.

`python -m evals.run --suite retrieval` -- the specification's own command.
There is no `/evals/run` HTTP endpoint: evaluation is an operator-run,
offline process, never something the running application exposes to a
caller.

**Hard-failure path (D1, D2).** If the real local models
(`BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) are not
available from the local sentence-transformers cache, this exits non-zero
with a structural message and produces nothing: no `eval_runs` row, no JSON
artifact, no printed metrics. A canary call against each real model is made
before any database write happens, specifically so an unavailable model is
caught before the fixture corpus is touched at all -- a fake provider is
never substituted here, only a pytest-only allowance
(`tests/test_evals_*.py`) ever uses one.

**Persistence (D7, D11).** A successful run writes one `eval_runs` row to
the same configured KnowledgeOS database every other part of the
application uses -- never a second, evaluation-only database -- then writes
the JSON artifact and prints the console summary. All three are built from
the same `SuiteResult`, so they can never disagree with each other.
"""

import argparse
import sys

from sqlalchemy.orm import Session

from app.db.session import get_sessionmaker
from app.models import RETRIEVAL_SUITE, EvalRun
from app.providers.bge import BgeEmbeddingProvider
from app.providers.cross_encoder import CrossEncoderRerankProvider
from app.providers.embeddings import EmbeddingError
from app.providers.passthrough_reranker import PassthroughRerankProvider
from app.providers.reranker import Candidate, RerankError
from app.storage import get_storage
from evals.retrieval import report
from evals.retrieval.corpus import CorpusError
from evals.retrieval.questions import QuestionSetError
from evals.retrieval.suite import run_retrieval_suite

SUPPORTED_SUITES = ("retrieval",)

# A minimal, fixed probe used only to prove a model actually loads and runs
# -- never evaluation data, and never compared against anything.
_CANARY_TEXT = "retrieval evaluation model availability check"


def _unavailable_reason(
    embeddings: BgeEmbeddingProvider, cross_encoder: CrossEncoderRerankProvider
) -> str | None:
    """Try a minimal real call against each required model.

    Returns a message naming what is unavailable, or `None` once both
    models have actually produced output. Run before any database write, so
    a failing model never leaves behind a partially seeded corpus followed
    by a missing `eval_runs` row -- it leaves behind nothing at all.
    """
    try:
        embeddings.embed([_CANARY_TEXT])
    except EmbeddingError as exc:
        return f"the embedding model ({embeddings.model_name}) is unavailable: {exc}"

    try:
        cross_encoder.rerank(_CANARY_TEXT, [Candidate(id="canary", text=_CANARY_TEXT)])
    except RerankError as exc:
        return f"the reranking model ({cross_encoder.model_name}) is unavailable: {exc}"

    return None


def _fail(message: str) -> int:
    print(f"retrieval evaluation cannot run: {message}", file=sys.stderr)
    print(
        "no metrics were computed, no eval_runs row was written, and no "
        "report was produced.",
        file=sys.stderr,
    )
    return 1


def run_retrieval(session: Session) -> int:
    storage = get_storage()
    embeddings = BgeEmbeddingProvider()
    cross_encoder = CrossEncoderRerankProvider()
    passthrough = PassthroughRerankProvider()

    reason = _unavailable_reason(embeddings, cross_encoder)
    if reason is not None:
        return _fail(reason)

    try:
        result = run_retrieval_suite(session, storage, embeddings, passthrough, cross_encoder)
    except (CorpusError, QuestionSetError, EmbeddingError, RerankError) as exc:
        return _fail(str(exc))

    eval_run = EvalRun(
        suite=RETRIEVAL_SUITE,
        prompt_version=None,
        config=result.config,
        metrics=report.metrics_payload(result),
    )
    session.add(eval_run)
    session.commit()

    report_path = report.write_json_report(
        result, eval_run_id=eval_run.id, directory=report.DEFAULT_REPORT_DIR
    )

    print(report.render_console_summary(result, eval_run_id=eval_run.id))
    print(f"\nJSON report written to {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.run")
    parser.add_argument("--suite", required=True, choices=SUPPORTED_SUITES)
    args = parser.parse_args(argv)

    session_factory = get_sessionmaker()
    with session_factory() as session:
        if args.suite == "retrieval":
            return run_retrieval(session)
        raise AssertionError(f"unhandled suite {args.suite!r}")  # argparse guards this


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["SUPPORTED_SUITES", "main", "run_retrieval"]
