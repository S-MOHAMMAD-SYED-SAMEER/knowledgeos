"""CLI entry point for the evaluation harness.

`python -m evals.run --suite retrieval` and `python -m evals.run --suite
answers` -- the specification's own command, extended in milestone 9 with
its own suite name. There is no `/evals/run` HTTP endpoint: evaluation is
an operator-run, offline process, never something the running application
exposes to a caller.

**Hard-failure path (D1, D2), for both suites.** If the real local models
required for a suite are not available from the local sentence-transformers
cache (or, for `answers`, from a configured Gemini credential and model),
this exits non-zero with a structural message and produces nothing: no
`eval_runs` row, no JSON artifact, no printed metrics. A canary call
against every required model is made before any database write happens --
the `answers` suite's canary covers three models (embedding, reranking,
generation), never just the two `retrieval` needs, because it calls all
three unmodified milestone 5/6/8 functions. A fake provider is never
substituted here; that is a pytest-only allowance (`tests/test_evals_*.py`,
`tests/test_answers_*.py`).

**Persistence (D7, D11).** A successful run writes one `eval_runs` row to
the same configured KnowledgeOS database every other part of the
application uses -- never a second, evaluation-only database -- then writes
the JSON artifact and prints the console summary. All three are built from
the same result object, so they can never disagree with each other.
"""

import argparse
import sys
from dataclasses import asdict

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import get_sessionmaker
from app.generation.prompt import load_prompt
from app.models import RETRIEVAL_SUITE, EvalRun
from app.providers.bge import BgeEmbeddingProvider
from app.providers.cross_encoder import CrossEncoderRerankProvider
from app.providers.embeddings import EmbeddingError
from app.providers.gemini_llm import GeminiLLMProvider
from app.providers.llm import LLMError
from app.providers.passthrough_reranker import PassthroughRerankProvider
from app.providers.reranker import Candidate, RerankError
from app.storage import get_storage
from evals.answers import report as answers_report
from evals.answers.calibration import calibrate_threshold, collect_dev_observations
from evals.answers.semantic import check_semantic_scorer_available
from evals.answers.splits import SplitError, apply_split
from evals.answers.suite import run_answers_suite
from evals.retrieval import report
from evals.retrieval.corpus import CorpusError, ensure_corpus_seeded
from evals.retrieval.questions import QuestionSetError, load_question_set
from evals.retrieval.suite import run_retrieval_suite

SUPPORTED_SUITES = ("retrieval", "answers")

ANSWERS_SUITE = "answers"

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


def _llm_unavailable_reason(llm: GeminiLLMProvider) -> str | None:
    """The same canary discipline, for the generation provider the
    `answers` suite also requires."""
    try:
        llm.complete(system="availability check", user=_CANARY_TEXT, max_tokens=16)
    except LLMError as exc:
        return f"the language model ({llm.model_name}) is unavailable: {exc}"
    return None


def _fail(message: str, *, suite_label: str = "retrieval") -> int:
    print(f"{suite_label} evaluation cannot run: {message}", file=sys.stderr)
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


def run_answers(session: Session) -> int:
    """The `answers` suite (D1, D8): the frozen milestone 7 questions
    through the real retrieve -> rerank -> generate pipeline, scored by
    `evals/answers/metrics.py`.

    Three real models are required, not two -- embedding and reranking
    (shared with the retrieval suite) plus generation. All three are
    canary-checked before anything is written, mirroring
    `run_retrieval`'s own discipline exactly.
    """
    settings = get_settings()
    storage = get_storage()
    embeddings = BgeEmbeddingProvider()
    cross_encoder = CrossEncoderRerankProvider()

    reason = _unavailable_reason(embeddings, cross_encoder)
    if reason is not None:
        return _fail(reason, suite_label="answer")

    if settings.llm_model is None:
        return _fail(
            "KNOWLEDGEOS_LLM_MODEL is not configured -- no Gemini model has "
            "been verified for this build (see app/config.py)",
            suite_label="answer",
        )
    llm = GeminiLLMProvider(settings.llm_model)

    llm_reason = _llm_unavailable_reason(llm)
    if llm_reason is not None:
        return _fail(llm_reason, suite_label="answer")

    # Semantic grounding (D2) is a softer requirement than the three models
    # above: its own unavailability is an explicit reported state, not a
    # reason to hard-fail the whole answer suite (the deterministic layer,
    # citations, and abstention are still real and worth evaluating).
    semantic_reason = check_semantic_scorer_available(cross_encoder)
    semantic_scorer = cross_encoder if semantic_reason is None else None

    try:
        questions = load_question_set()
    except QuestionSetError as exc:
        return _fail(str(exc), suite_label="answer")

    try:
        split = apply_split(questions)
    except SplitError as exc:
        return _fail(str(exc), suite_label="answer")

    # Calibration (D3) must run, against the dev split only, before the
    # suite itself: `abstention_threshold` is one of the suite's own
    # parameters, not something it derives internally. It needs the corpus
    # seeded first -- `collect_dev_observations` calls `retrieve()`
    # directly, not through `run_answers_suite`, which would otherwise seed
    # it later. `ensure_corpus_seeded` is idempotent, so the suite's own
    # (redundant) seeding call below is a cheap no-op once this one has run.
    try:
        ensure_corpus_seeded(session, storage, embeddings)
        observations = collect_dev_observations(session, embeddings, cross_encoder, split.dev)
        calibration = calibrate_threshold(observations)
    except (CorpusError, EmbeddingError, RerankError) as exc:
        return _fail(str(exc), suite_label="answer")

    try:
        result = run_answers_suite(
            session,
            storage,
            embeddings,
            cross_encoder,
            llm,
            semantic_scorer=semantic_scorer,
            abstention_threshold=calibration.threshold,
            max_tokens=settings.llm_max_output_tokens,
        )
    except (
        CorpusError,
        QuestionSetError,
        SplitError,
        EmbeddingError,
        RerankError,
        LLMError,
    ) as exc:
        return _fail(str(exc), suite_label="answer")

    pricing_table = settings.llm_pricing_usd_per_million_tokens

    eval_run = EvalRun(
        suite=ANSWERS_SUITE,
        prompt_version=load_prompt().version,
        config={
            **result.config,
            "calibration": asdict(calibration),
            "semantic_scorer_unavailable_reason": semantic_reason,
        },
        metrics=answers_report.metrics_payload(result, pricing_table=pricing_table),
    )
    session.add(eval_run)
    session.commit()

    report_path = answers_report.write_json_report(
        result,
        eval_run_id=eval_run.id,
        pricing_table=pricing_table,
        directory=answers_report.DEFAULT_REPORT_DIR,
    )

    print(
        answers_report.render_console_summary(
            result, eval_run_id=eval_run.id, pricing_table=pricing_table
        )
    )
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
        if args.suite == "answers":
            return run_answers(session)
        raise AssertionError(f"unhandled suite {args.suite!r}")  # argparse guards this


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["SUPPORTED_SUITES", "main", "run_answers", "run_retrieval"]
