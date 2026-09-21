"""Running the retrieval evaluation suite end to end, against a real
database and the real fixture corpus, but through fake providers.

The specification is explicit that only a pytest suite may use fakes
officially (D2/D1): what these tests prove is that the suite's wiring --
seeding, embedding each question, calling `retrieve()` once, reranking
twice over the identical candidate set, aggregating six metrics -- is
correct. None of the numbers these tests produce are an "official"
retrieval evaluation result; only `evals/run.py`, run against the real
local models, could produce that.
"""

import pytest

from app.providers import FakeEmbeddingProvider, FakeRerankProvider, PassthroughRerankProvider
from app.storage import LocalStorage
from evals.retrieval.questions import load_question_set
from evals.retrieval.suite import ConditionMetrics, SuiteResult, run_retrieval_suite


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


def _metric_fields_in_unit_interval(metrics: ConditionMetrics) -> bool:
    return all(
        0.0 <= value <= 1.0
        for value in (
            metrics.recall_at_5,
            metrics.recall_at_10,
            metrics.precision_at_5,
            metrics.mrr,
            metrics.ndcg_at_10,
        )
    )


def test_the_suite_runs_end_to_end_against_the_real_fixtures(
    session, storage, embeddings
) -> None:
    result = run_retrieval_suite(
        session,
        storage,
        embeddings,
        PassthroughRerankProvider(),
        FakeRerankProvider(),
    )

    assert isinstance(result, SuiteResult)
    assert _metric_fields_in_unit_interval(result.passthrough)
    assert _metric_fields_in_unit_interval(result.cross_encoder)
    assert 0.0 <= result.metadata_filter_correctness <= 1.0


def test_evaluated_plus_excluded_equals_total_questions(
    session, storage, embeddings
) -> None:
    result = run_retrieval_suite(
        session, storage, embeddings, PassthroughRerankProvider(), FakeRerankProvider()
    )

    questions = load_question_set()
    assert result.total_question_count == len(questions)
    assert result.evaluated_question_count + result.excluded_question_count == len(
        questions
    )


def test_excluded_count_matches_insufficient_evidence_questions(
    session, storage, embeddings
) -> None:
    result = run_retrieval_suite(
        session, storage, embeddings, PassthroughRerankProvider(), FakeRerankProvider()
    )

    questions = load_question_set()
    expected_excluded = sum(1 for q in questions if not q.expected_chunk_uids)
    assert result.excluded_question_count == expected_excluded


def test_filtered_question_count_matches_questions_with_non_empty_filters(
    session, storage, embeddings
) -> None:
    from app.retrieval.filters import RetrievalFilters

    result = run_retrieval_suite(
        session, storage, embeddings, PassthroughRerankProvider(), FakeRerankProvider()
    )

    questions = load_question_set()
    expected_filtered = sum(1 for q in questions if q.filters != RetrievalFilters())
    assert result.filtered_question_count == expected_filtered


def test_identical_rerank_providers_produce_identical_metrics(
    session, storage, embeddings
) -> None:
    """Both conditions rerank the same candidate set; if both providers
    preserve order the same way, the two conditions' rank-sensitive metrics
    must be numerically identical -- a sanity check on the "same
    pre-reranking candidate set" invariant, not a claim about real
    reranking's effect."""
    result = run_retrieval_suite(
        session,
        storage,
        embeddings,
        PassthroughRerankProvider(),
        PassthroughRerankProvider(),
    )

    assert result.passthrough == result.cross_encoder


def test_config_records_the_pinned_chunk_settings_and_model_names(
    session, storage, embeddings
) -> None:
    result = run_retrieval_suite(
        session, storage, embeddings, PassthroughRerankProvider(), FakeRerankProvider()
    )

    assert result.config["chunk_size_tokens"] == 512
    assert result.config["chunk_overlap_tokens"] == 64
    assert result.config["embedding_model"] == embeddings.model_name
    assert result.config["passthrough_rerank_model"] == "passthrough"
    assert result.config["cross_encoder_rerank_model"] == FakeRerankProvider().model_name
    assert result.config["question_count"] == result.total_question_count


def test_corpus_is_seeded_as_part_of_running_the_suite(
    session, storage, embeddings
) -> None:
    from evals.retrieval.corpus import MINIMUM_ACTIVE_CHUNKS

    result = run_retrieval_suite(
        session, storage, embeddings, PassthroughRerankProvider(), FakeRerankProvider()
    )

    assert result.corpus.total_active_chunks > MINIMUM_ACTIVE_CHUNKS


def test_metadata_filter_correctness_matches_direct_computation(
    session, storage, embeddings
) -> None:
    """Recomputing the metric independently from the same retrieved
    evidence, rather than trusting the suite's own aggregation, to prove
    the suite is not just returning a plausible-looking number."""
    from app.parsing import normalize
    from app.retrieval.filters import RetrievalFilters
    from app.retrieval.pipeline import retrieve
    from evals.retrieval.corpus import ensure_corpus_seeded
    from evals.retrieval.metrics import metadata_filter_correctness

    ensure_corpus_seeded(session, storage, embeddings)
    questions = load_question_set()

    pairs = []
    for question in questions:
        if question.filters == RetrievalFilters():
            continue
        normalized_text = normalize(question.text)
        query_vector = embeddings.embed([normalized_text])[0]
        result = retrieve(
            session,
            normalized_text=normalized_text,
            query_vector=query_vector,
            filters=question.filters,
        )
        pairs.append((question.filters, [c.evidence for c in result.candidates]))

    expected = metadata_filter_correctness(pairs)

    suite_result = run_retrieval_suite(
        session, storage, embeddings, PassthroughRerankProvider(), FakeRerankProvider()
    )
    assert suite_result.metadata_filter_correctness == expected
