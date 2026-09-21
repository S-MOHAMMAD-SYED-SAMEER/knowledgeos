"""Running the retrieval evaluation suite: the frozen question set against
the fixture corpus, under two reranking conditions, six metrics each.

Consumes retrieval the same way `POST /query` does -- normalize, embed,
`retrieve()`, `rerank()` -- but through direct function calls, never HTTP
(the specification's own words: no `/evals/run` endpoint, and evaluation
does not go through the API layer). The two conditions run over the
*identical* pre-reranking candidate set: `retrieve()` is called once per
question, and its `RetrievedChunk` list is reranked twice, once by each
provider. Reranking only ever reorders that list; it cannot add or remove a
candidate, so any difference between the two conditions' metrics is
attributable to reordering alone.

This module calls providers and the database, the same relationship
`evals/retrieval/corpus.py` has to both -- not bound by the "no provider
calls" layering rule that applies to `app/retrieval/` itself.

Which provider is "real" is not this module's decision: `run_retrieval_suite`
takes an `EmbeddingProvider` and two `RerankProvider`s as arguments, exactly
like `app/api/query.py` takes them as FastAPI dependencies. `evals/run.py`
is what decides they must be `BgeEmbeddingProvider` and
`CrossEncoderRerankProvider`/`PassthroughRerankProvider` for an official run;
this module's own tests pass fakes, per the specification's "fakes only
inside pytest tests" allowance (D2) -- computing metrics correctly is what
those tests are for, not producing an official number.
"""

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.parsing import normalize
from app.providers.embeddings import EmbeddingProvider
from app.providers.reranker import RerankProvider
from app.reranking.pipeline import rerank
from app.retrieval.filters import RetrievalFilters
from app.retrieval.fusion import DEFAULT_RRF_K
from app.retrieval.pipeline import retrieve
from app.storage import Storage
from evals.retrieval.corpus import (
    DEFAULT_MANIFEST,
    PINNED_CHUNK_OVERLAP_TOKENS,
    PINNED_CHUNK_SIZE_TOKENS,
    CorpusSeedResult,
    ensure_corpus_seeded,
    verify_corpus_size,
)
from evals.retrieval.metrics import (
    NDCG_DEPTH,
    PRECISION_DEPTH,
    mean_reciprocal_rank,
    metadata_filter_correctness,
    ndcg_at_10,
    precision_at_5,
    recall_at_k,
)
from evals.retrieval.questions import (
    DEFAULT_QUESTIONS_DIR,
    Question,
    load_question_set,
)

# D12, locked: the "reranking disabled" condition is the passthrough
# provider, not a skipped reranking stage -- every question still goes
# through `app/reranking/pipeline.rerank`, just with a provider that
# preserves fusion order.
CONDITION_PASSTHROUGH = "passthrough"
CONDITION_CROSS_ENCODER = "cross_encoder"


@dataclass(frozen=True)
class ConditionMetrics:
    """The five rank-sensitive metrics, for one reranking condition."""

    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    mrr: float
    ndcg_at_10: float


@dataclass(frozen=True)
class SuiteResult:
    """Everything one run of the retrieval suite produced.

    `metadata_filter_correctness` is a single number, not one per condition:
    reranking reorders candidates but never adds or removes one, so which
    candidates violate a question's filter -- the only thing this metric
    checks -- is identical under both conditions by construction.
    """

    passthrough: ConditionMetrics
    cross_encoder: ConditionMetrics
    metadata_filter_correctness: float
    evaluated_question_count: int
    excluded_question_count: int
    filtered_question_count: int
    total_question_count: int
    corpus: CorpusSeedResult
    config: dict


def _aggregate(
    scored: list[tuple[frozenset[str], list[str]]]
) -> ConditionMetrics:
    """Mean of each metric over one condition's per-question scores.

    `scored` already excludes empty-`expected_chunk_uids` questions (D4) --
    every metric function here raises on an empty expected set, so excluding
    them earlier is what makes this loop safe to call unconditionally.
    """
    n = len(scored)
    return ConditionMetrics(
        recall_at_5=sum(recall_at_k(e, r, 5) for e, r in scored) / n,
        recall_at_10=sum(recall_at_k(e, r, 10) for e, r in scored) / n,
        precision_at_5=sum(precision_at_5(e, r) for e, r in scored) / n,
        mrr=sum(mean_reciprocal_rank(e, r) for e, r in scored) / n,
        ndcg_at_10=sum(ndcg_at_10(e, r) for e, r in scored) / n,
    )


def run_retrieval_suite(
    session: Session,
    storage: Storage,
    embeddings: EmbeddingProvider,
    passthrough_provider: RerankProvider,
    cross_encoder_provider: RerankProvider,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    questions_dir: Path = DEFAULT_QUESTIONS_DIR,
    rrf_k: int = DEFAULT_RRF_K,
) -> SuiteResult:
    """Seed the fixture corpus, load the frozen question set, evaluate.

    One embedding call and one `retrieve()` call per question -- both
    reranking conditions are scored from that single candidate set, never
    from two independent retrievals, which is what keeps the A/B comparison
    honest (locked invariant: the pre-reranking candidate set is identical
    between conditions).
    """
    corpus_result = ensure_corpus_seeded(session, storage, embeddings, manifest_path=manifest_path)
    verify_corpus_size(corpus_result)

    questions: list[Question] = load_question_set(questions_dir)

    passthrough_scored: list[tuple[frozenset[str], list[str]]] = []
    cross_encoder_scored: list[tuple[frozenset[str], list[str]]] = []
    filtered_pairs: list[tuple[RetrievalFilters, list]] = []
    excluded = 0

    for question in questions:
        normalized_text = normalize(question.text)
        query_vector = embeddings.embed([normalized_text])[0]
        result = retrieve(
            session,
            normalized_text=normalized_text,
            query_vector=query_vector,
            filters=question.filters,
            rrf_k=rrf_k,
        )
        candidates = result.candidates

        if question.filters != RetrievalFilters():
            filtered_pairs.append(
                (question.filters, [c.evidence for c in candidates])
            )

        passthrough_ranked = rerank(normalized_text, candidates, passthrough_provider)
        cross_encoder_ranked = rerank(normalized_text, candidates, cross_encoder_provider)

        if not question.expected_chunk_uids:
            excluded += 1
            continue

        passthrough_scored.append(
            (
                question.expected_chunk_uids,
                [c.evidence.chunk_uid for c in passthrough_ranked],
            )
        )
        cross_encoder_scored.append(
            (
                question.expected_chunk_uids,
                [c.evidence.chunk_uid for c in cross_encoder_ranked],
            )
        )

    config = {
        "embedding_model": embeddings.model_name,
        "passthrough_rerank_model": passthrough_provider.model_name,
        "cross_encoder_rerank_model": cross_encoder_provider.model_name,
        "chunk_size_tokens": PINNED_CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": PINNED_CHUNK_OVERLAP_TOKENS,
        "rrf_k": rrf_k,
        "precision_depth": PRECISION_DEPTH,
        "ndcg_depth": NDCG_DEPTH,
        "question_count": len(questions),
        "corpus_active_chunks": corpus_result.total_active_chunks,
    }

    return SuiteResult(
        passthrough=_aggregate(passthrough_scored),
        cross_encoder=_aggregate(cross_encoder_scored),
        metadata_filter_correctness=metadata_filter_correctness(filtered_pairs),
        evaluated_question_count=len(passthrough_scored),
        excluded_question_count=excluded,
        filtered_question_count=len(filtered_pairs),
        total_question_count=len(questions),
        corpus=corpus_result,
        config=config,
    )


__all__ = [
    "CONDITION_CROSS_ENCODER",
    "CONDITION_PASSTHROUGH",
    "ConditionMetrics",
    "SuiteResult",
    "run_retrieval_suite",
]
