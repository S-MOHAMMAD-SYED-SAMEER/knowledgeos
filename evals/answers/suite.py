"""Running the answer evaluation suite: the frozen corpus and question
set (milestone 7, reused unmodified), through the real retrieve → rerank →
generate pipeline (milestones 5, 6, 8, all reused unmodified), scored by
`evals/answers/metrics.py`.

Every question in the frozen set is evaluated — not just the dev or test
half. The dev/test split (`evals/answers/splits.py`) exists specifically
for `evals/answers/calibration.py`'s threshold sweep, per §9's own words
("calibrated on a dev split"); nothing in §13 scopes the answer *metrics*
to a split, and milestone 7's own retrieval suite evaluates the whole
frozen set the same way. The split assignment is still recorded in the
result, for transparency about which half each question fell in.

Each question's real, successful generation is persisted through
`app.generation.persistence.persist_query` — the same function
`POST /query` calls, unmodified — so an evaluation run leaves the exact
same `queries`/`retrieved_chunks`/`answers` trail a live request would,
and `GET /queries/{id}` can inspect any evaluated question afterward. A
rejected attempt (`CitationError`/`GenerationError`) is never persisted,
for the same reason `POST /query` never persists one: there is no valid
`GeneratedAnswer` to write.

An `LLMError` (the provider itself failing, not the model's output) is not
a data point to score, and it must not throw away the answers already paid
for. The question is left pending (it is not scored as anything), the run
carries on, and after `max_consecutive_failures` failures in a row it stops
instead of hammering a service that is down or out of quota. What was done is
reported through `on_progress` after every question, so the caller can save it;
a later run resumes from `resume_from` and asks only the pending questions. A
run with any pending question is incomplete, and the caller must not report
it as a result.

`total_ms` is the sum of the three measured stages (retrieval + rerank +
generation-attempt, whether or not generation ultimately succeeded) —
each stage timed independently with `app.observability.timing.stage_timer`,
which sets its `.ms` in a `finally` block, so a stage that raised still
contributes its real measured time rather than leaving a gap.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.generation.generator import (
    FAILURE_CITATION_INVALID,
    FAILURE_MALFORMED_OUTPUT,
    FAILURE_MODEL_BLOCKED,
    generate_answer,
)
from app.generation.persistence import persist_query
from app.observability.timing import stage_timer
from app.parsing import normalize
from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMError, LLMProvider, LLMRateLimited
from app.providers.reranker import RerankProvider
from app.reranking.pipeline import rerank
from app.retrieval.filters import RetrievalFilters
from app.retrieval.fusion import DEFAULT_RRF_K
from app.retrieval.pipeline import retrieve
from app.storage import Storage
from evals.answers.metrics import AnswerAttempt, AttemptOutcome
from evals.answers.semantic import score_answer_semantic
from evals.answers.splits import DEFAULT_SPLITS_PATH, apply_split
from evals.retrieval.corpus import (
    DEFAULT_MANIFEST,
    CorpusSeedResult,
    ensure_corpus_seeded,
    verify_corpus_size,
)
from evals.retrieval.questions import DEFAULT_QUESTIONS_DIR, load_question_set

logger = logging.getLogger(__name__)

# Stop after this many provider failures in a row, unless told otherwise.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 2

ProgressCallback = Callable[[list[AnswerAttempt], list[UUID]], None]


@dataclass(frozen=True)
class AnswerSuiteResult:
    attempts: list[AnswerAttempt]
    persisted_query_ids: list[UUID]
    corpus: CorpusSeedResult
    dev_question_count: int
    test_question_count: int
    total_question_count: int
    config: dict
    # Questions with no result yet: the provider failed on them, or the run stopped before them.
    # Empty for a complete run, which is the only kind that may be reported.
    pending_question_ids: list[str] = field(default_factory=list)
    # Why the run stopped before the end, or None.
    stopped_early: str | None = None
    # One entry per provider failure: {"question_id", "error"} (never a message from the model).
    provider_errors: list[dict] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.pending_question_ids


def _filters_to_dict(filters: RetrievalFilters) -> dict:
    """JSON-safe form of `RetrievalFilters`, the same shape
    `QueryFiltersIn.model_dump(mode="json")` produces in the serving
    path -- built by hand here since a question's filters come from
    `evals.retrieval.questions`, not from a pydantic request model."""
    return {
        "document_id": str(filters.document_id) if filters.document_id else None,
        "department": filters.department,
        "category": filters.category,
        "tags": list(filters.tags),
        "include_superseded": filters.include_superseded,
    }


def run_answers_suite(
    session: Session,
    storage: Storage,
    embeddings: EmbeddingProvider,
    reranker_provider: RerankProvider,
    llm: LLMProvider,
    *,
    semantic_scorer: RerankProvider | None,
    abstention_threshold: float | None,
    max_tokens: int,
    manifest_path: Path = DEFAULT_MANIFEST,
    questions_dir: Path = DEFAULT_QUESTIONS_DIR,
    splits_path: Path = DEFAULT_SPLITS_PATH,
    rrf_k: int = DEFAULT_RRF_K,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
    resume_from: tuple[list[AnswerAttempt], list[UUID]] | None = None,
    on_progress: ProgressCallback | None = None,
) -> AnswerSuiteResult:
    """Evaluate every frozen question through the real pipeline.

    `semantic_scorer=None` skips semantic grounding entirely (every
    attempt's `semantic` stays `None`) rather than silently substituting
    a fake -- the CLI decides whether to pass one, after its own
    availability check
    (`evals/answers/semantic.py::check_semantic_scorer_available`).
    """
    corpus_result = ensure_corpus_seeded(session, storage, embeddings, manifest_path=manifest_path)
    verify_corpus_size(corpus_result)

    questions = load_question_set(questions_dir)
    split = apply_split(questions, splits_path)

    # A question the provider failed on is asked again on resume; every other result is kept.
    attempts: list[AnswerAttempt] = (
        [a for a in resume_from[0] if a.outcome != AttemptOutcome.PROVIDER_ERROR] if resume_from else []
    )
    persisted_query_ids: list[UUID] = list(resume_from[1]) if resume_from else []
    already_done = {a.question_id for a in attempts}
    provider_errors: list[dict] = []
    consecutive_failures = 0
    stopped_early: str | None = None

    for question in questions:
        if question.id in already_done:
            continue
        normalized_text = normalize(question.text)

        with stage_timer() as retrieval_timer:
            query_vector = embeddings.embed([normalized_text])[0]
            result = retrieve(
                session,
                normalized_text=normalized_text,
                query_vector=query_vector,
                filters=question.filters,
                rrf_k=rrf_k,
            )

        with stage_timer() as rerank_timer:
            reranked = rerank(normalized_text, result.candidates, reranker_provider)

        take_calls = getattr(llm, "take_calls", None)
        try:
            with stage_timer() as llm_timer:
                answer = generate_answer(
                    query_text=normalized_text,
                    candidates=reranked,
                    llm=llm,
                    abstention_threshold=abstention_threshold,
                    max_tokens=max_tokens,
                )
        except LLMError as exc:
            # The provider failed, not the model's output. Record the question as failed (with the
            # adapter's message) and carry on. Stop only on a 429 / quota error at once, or when two
            # different questions in a row have failed; nothing is retried.
            consecutive_failures += 1
            provider_errors.append(
                {
                    "question_id": question.id,
                    "error": type(exc).__name__,
                    # The adapter's own message: a status such as "generation failed (HTTP 503)",
                    # never the prompt.
                    "message": str(exc),
                    "raw_replies": take_calls() if take_calls else None,
                }
            )
            logger.warning(
                "The provider failed on question %s (%s); %d failure(s) in a row.",
                question.id,
                type(exc).__name__,
                consecutive_failures,
            )
            attempts.append(
                AnswerAttempt(
                    question_id=question.id,
                    expected_abstain=question.expected_abstain,
                    outcome=AttemptOutcome.PROVIDER_ERROR,
                    abstained=None,
                    citation_valid=None,
                    citation_coverage=None,
                    grounded=None,
                    semantic=None,
                    model_name=llm.model_name,
                    input_tokens=None,
                    output_tokens=None,
                    retrieval_ms=retrieval_timer.ms,
                    rerank_ms=rerank_timer.ms,
                    llm_ms=None,
                    total_ms=None,
                    error=str(exc),
                    raw_replies=provider_errors[-1]["raw_replies"],
                )
            )
            if on_progress is not None:
                on_progress(attempts, persisted_query_ids)
            if _is_rate_limit(exc):
                stopped_early = f"rate limit / quota error on {question.id} ({exc}); stopped, nothing retried"
                break
            if consecutive_failures >= max_consecutive_failures:
                stopped_early = (
                    f"{consecutive_failures} different questions failed in a row (last: {question.id}: {exc}); "
                    "stopped instead of retrying in a loop"
                )
                break
            continue
        raw_replies = take_calls() if take_calls else None
        if answer.model_name is not None:
            # The provider answered. (A question that never reached it, a pre-LLM abstention,
            # says nothing about whether it is back, so it does not end a run of failures.)
            consecutive_failures = 0

        if answer.verification_failure is not None:
            # The model was asked (twice, unless it declined) and nothing it said could be shown:
            # the person gets "I couldn't verify this answer". Counted on its own, and not
            # persisted, as a rejected answer never was.
            attempts.append(
                _failed_attempt(
                    question,
                    _UNVERIFIED_OUTCOME[answer.verification_failure],
                    retrieval_timer.ms,
                    rerank_timer.ms,
                    answer=answer,
                    llm_ms=llm_timer.ms,
                    raw_replies=raw_replies,
                )
            )
            if on_progress is not None:
                on_progress(attempts, persisted_query_ids)
            continue

        # `answer.model_name is None` marks a pre-LLM abstention (zero
        # candidates, or the score threshold) -- generate_answer never
        # called the provider, so there is no real generation latency to
        # report for this question.
        llm_ms = llm_timer.ms if answer.model_name is not None else None

        semantic_result = None
        if semantic_scorer is not None and answer.abstained is False:
            chunk_text_by_uid = {c.evidence.chunk_uid: c.evidence.text for c in reranked}
            semantic_result = score_answer_semantic(
                answer_text=answer.answer_text,
                chunk_text_by_uid=chunk_text_by_uid,
                scorer=semantic_scorer,
            )

        outcome = (
            AttemptOutcome.PRE_LLM_ABSTAIN
            if answer.model_name is None
            else AttemptOutcome.SUCCESS
        )
        coverage = answer.grounding_detail["deterministic"]["citation_coverage"]
        total_ms = retrieval_timer.ms + rerank_timer.ms + (llm_ms or 0.0)

        attempts.append(
            AnswerAttempt(
                question_id=question.id,
                expected_abstain=question.expected_abstain,
                outcome=outcome,
                abstained=answer.abstained,
                citation_valid=answer.citation_valid,
                citation_coverage=coverage,
                grounded=answer.grounded,
                semantic=semantic_result,
                model_name=answer.model_name,
                input_tokens=answer.input_tokens,
                output_tokens=answer.output_tokens,
                retrieval_ms=retrieval_timer.ms,
                rerank_ms=rerank_timer.ms,
                llm_ms=llm_ms,
                total_ms=total_ms,
                attempts_used=answer.attempts if answer.model_name is not None else None,
                answer_text=answer.answer_text,
                citations=list(answer.citations),
                selected_chunk_uids=[
                    item.candidate.evidence.chunk_uid for item in answer.selected if item.selected
                ],
                raw_replies=raw_replies,
            )
        )

        query_row, _ = persist_query(
            session,
            query_text=question.text,
            normalized_text=normalized_text,
            filters=_filters_to_dict(question.filters),
            candidates=reranked,
            answer=answer,
            retrieval_ms=retrieval_timer.ms,
            rerank_ms=rerank_timer.ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
        )
        persisted_query_ids.append(query_row.id)
        if on_progress is not None:
            on_progress(attempts, persisted_query_ids)

    answered = {a.question_id for a in attempts}
    pending = [q.id for q in questions if q.id not in answered]

    config = {
        "embedding_model": embeddings.model_name,
        "rerank_model": reranker_provider.model_name,
        "llm_model": llm.model_name,
        "semantic_scorer_model": semantic_scorer.model_name if semantic_scorer else None,
        "abstention_threshold": abstention_threshold,
        "rrf_k": rrf_k,
        "question_count": len(questions),
        "dev_question_count": len(split.dev),
        "test_question_count": len(split.test),
        "corpus_active_chunks": corpus_result.total_active_chunks,
    }

    return AnswerSuiteResult(
        attempts=attempts,
        persisted_query_ids=persisted_query_ids,
        corpus=corpus_result,
        dev_question_count=len(split.dev),
        test_question_count=len(split.test),
        total_question_count=len(questions),
        config=config,
        pending_question_ids=pending,
        stopped_early=stopped_early,
        provider_errors=provider_errors,
    )


def _is_rate_limit(exc: LLMError) -> bool:
    """A 429 or a quota error: the one kind of failure that ends a run at once."""
    text = str(exc).lower()
    return isinstance(exc, LLMRateLimited) or "429" in text or "quota" in text or "resource_exhausted" in text


_UNVERIFIED_OUTCOME = {
    FAILURE_CITATION_INVALID: AttemptOutcome.CITATION_INVALID,
    FAILURE_MALFORMED_OUTPUT: AttemptOutcome.MALFORMED_OUTPUT,
    FAILURE_MODEL_BLOCKED: AttemptOutcome.MODEL_BLOCKED,
}


def _failed_attempt(
    question, outcome, retrieval_ms, rerank_ms, *, answer, llm_ms, raw_replies=None
) -> AnswerAttempt:
    """A question whose answer could not be verified. The tokens are real (the model was
    asked), so they count towards cost; there is no answer to describe, so every field about
    the answer is None, and the abstention score treats it as a predicted non-abstention."""
    return AnswerAttempt(
        question_id=question.id,
        expected_abstain=question.expected_abstain,
        outcome=outcome,
        abstained=None,
        citation_valid=None,
        citation_coverage=None,
        grounded=None,
        semantic=None,
        model_name=answer.model_name,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        llm_ms=llm_ms,
        total_ms=retrieval_ms + rerank_ms + llm_ms,
        attempts_used=answer.attempts,
        raw_replies=raw_replies,
    )


__all__ = ["AnswerSuiteResult", "run_answers_suite"]
