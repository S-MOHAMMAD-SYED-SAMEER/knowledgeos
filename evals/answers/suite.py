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

An `LLMError` (the provider itself failing, not the model's output)
propagates uncaught — a real-provider evaluation run needs the provider to
keep working for its entire duration; an outage partway through is an
infrastructure failure for the run to fail on, not a data point to score.

`total_ms` is the sum of the three measured stages (retrieval + rerank +
generation-attempt, whether or not generation ultimately succeeded) —
each stage timed independently with `app.observability.timing.stage_timer`,
which sets its `.ms` in a `finally` block, so a stage that raised still
contributes its real measured time rather than leaving a gap.
"""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.generation.citations import CitationError
from app.generation.generator import GenerationError, generate_answer
from app.generation.persistence import persist_query
from app.observability.timing import stage_timer
from app.parsing import normalize
from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider
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


@dataclass(frozen=True)
class AnswerSuiteResult:
    attempts: list[AnswerAttempt]
    persisted_query_ids: list[UUID]
    corpus: CorpusSeedResult
    dev_question_count: int
    test_question_count: int
    total_question_count: int
    config: dict


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

    attempts: list[AnswerAttempt] = []
    persisted_query_ids: list[UUID] = []

    for question in questions:
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

        with stage_timer() as llm_timer:
            try:
                answer = generate_answer(
                    query_text=normalized_text,
                    candidates=reranked,
                    llm=llm,
                    abstention_threshold=abstention_threshold,
                    max_tokens=max_tokens,
                )
            except CitationError:
                attempts.append(
                    _failed_attempt(
                        question,
                        AttemptOutcome.CITATION_INVALID,
                        retrieval_timer.ms,
                        rerank_timer.ms,
                    )
                )
                continue
            except GenerationError:
                attempts.append(
                    _failed_attempt(
                        question,
                        AttemptOutcome.MALFORMED_OUTPUT,
                        retrieval_timer.ms,
                        rerank_timer.ms,
                    )
                )
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
    )


def _failed_attempt(question, outcome, retrieval_ms, rerank_ms) -> AnswerAttempt:
    return AnswerAttempt(
        question_id=question.id,
        expected_abstain=question.expected_abstain,
        outcome=outcome,
        abstained=None,
        citation_valid=None,
        citation_coverage=None,
        grounded=None,
        semantic=None,
        model_name=None,
        input_tokens=None,
        output_tokens=None,
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        llm_ms=None,
        total_ms=retrieval_ms + rerank_ms,
    )


__all__ = ["AnswerSuiteResult", "run_answers_suite"]
