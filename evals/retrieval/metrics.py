"""Retrieval evaluation metrics.

Pure functions: no database, no provider, no filesystem, no hidden global
state. The specification names six metrics — Recall@5, Recall@10,
Precision@5, MRR, nDCG@10, metadata-filter correctness — but defines none of
them mathematically. Every formula here is therefore a locked project
decision, documented at its own function, rather than a specification quote.

Binary relevance throughout. The specification's question schema
(`expected_chunk_uids`, a flat list) carries no relevance grade, so there is
no graded relevance to implement: a returned `chunk_uid` either is or is not
in a question's expected set.

A question with an empty `expected_chunk_uids` (the insufficient-evidence
category, where nothing should be cited) makes Recall, Precision, MRR and
nDCG undefined — 0/0, not 0. Every function below raises `ValueError` for an
empty `expected` rather than returning a number that would silently pull an
aggregate mean down. The caller (`suite.py`) excludes such questions from
those four aggregates before ever calling these functions on them, and
reports the excluded count separately — the locked aggregation rule.
"""

import math

from app.retrieval.filters import RetrievalFilters
from app.retrieval.vector import ChunkEvidence

# MRR is computed over the entire returned top-20 list, not just the top 10
# nDCG looks at — a locked decision, distinct from nDCG's own cutoff.
DEFAULT_MRR_DEPTH = 20

# nDCG's own cutoff. Named so it is one constant, not a literal repeated at
# every call site.
NDCG_DEPTH = 10

PRECISION_DEPTH = 5


def recall_at_k(expected: set[str], retrieved: list[str], k: int) -> float:
    """`|E ∩ R@k| / |E|`.

    Raises `ValueError` for an empty `expected` — see the module docstring.
    """
    if not expected:
        raise ValueError("recall is undefined for an empty expected set")
    if k <= 0:
        raise ValueError("k must be positive")
    hits = expected & set(retrieved[:k])
    return len(hits) / len(expected)


def precision_at_5(expected: set[str], retrieved: list[str]) -> float:
    """`|E ∩ R@5| / 5` — a fixed denominator of 5, not `min(5, len(retrieved))`.

    A retriever that returns fewer than 5 candidates is not rewarded with a
    smaller denominator for having returned less.
    """
    if not expected:
        raise ValueError("precision is undefined for an empty expected set")
    hits = expected & set(retrieved[:PRECISION_DEPTH])
    return len(hits) / PRECISION_DEPTH


def mean_reciprocal_rank(
    expected: set[str], retrieved: list[str], *, depth: int = DEFAULT_MRR_DEPTH
) -> float:
    """`1 / rank` of the first relevant item within the top `depth` results
    — the whole returned top-20 list by default, a locked decision distinct
    from nDCG's top-10 cutoff. `0.0` if none of the top `depth` results is
    relevant; there is no "undefined" case here because a zero score for
    "found nothing in the ranking" is exactly what MRR means.
    """
    if not expected:
        raise ValueError("MRR is undefined for an empty expected set")
    for position, candidate_uid in enumerate(retrieved[:depth], start=1):
        if candidate_uid in expected:
            return 1.0 / position
    return 0.0


def dcg_at_k(relevances: list[int], k: int) -> float:
    """`Σ rel_i / log2(i + 1)`, `i` counted from 1, over the first `k`."""
    return sum(
        rel / math.log2(index + 1)
        for index, rel in enumerate(relevances[:k], start=1)
    )


def ndcg_at_10(expected: set[str], retrieved: list[str]) -> float:
    """`DCG@10 / IDCG@10`, binary relevance.

    IDCG places every relevant item (up to 10) first — the best ordering
    achievable for this question's expected set. `0.0` when IDCG is 0,
    guarded explicitly rather than dividing by zero (this can only happen
    for an empty `expected`, which is refused outright below, but the guard
    stays because "refuse to divide by zero" should never depend on a
    caller having gotten the precondition right).
    """
    if not expected:
        raise ValueError("nDCG is undefined for an empty expected set")
    relevances = [1 if uid in expected else 0 for uid in retrieved[:NDCG_DEPTH]]
    dcg = dcg_at_k(relevances, NDCG_DEPTH)

    ideal = [1] * min(len(expected), NDCG_DEPTH)
    ideal += [0] * (NDCG_DEPTH - len(ideal))
    idcg = dcg_at_k(ideal, NDCG_DEPTH)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def metadata_filter_violated(evidence: ChunkEvidence, filters: RetrievalFilters) -> bool:
    """Whether one retrieved chunk's metadata violates a requested filter.

    Mirrors `app.retrieval.filters.predicates()` field for field — the same
    five conditions, checked in memory instead of compiled to SQL, since a
    metric has already-retrieved evidence in hand rather than a query to
    build. Reusing `RetrievalFilters` and `ChunkEvidence` (rather than a new
    ad-hoc shape) is what keeps this check honestly identical to what M5
    actually filtered on.
    """
    if filters.document_id is not None and evidence.document_id != filters.document_id:
        return True
    if filters.department is not None and evidence.department != filters.department:
        return True
    if filters.category is not None and evidence.category != filters.category:
        return True
    if filters.tags:
        if not set(filters.tags) <= set(evidence.tags):
            return True
    if not filters.include_superseded and evidence.version_status == "superseded":
        return True
    return False


def metadata_filter_correctness(
    questions: list[tuple[RetrievalFilters, list[ChunkEvidence]]]
) -> float:
    """Fraction of filtered questions for which **zero** returned candidates
    violate the requested filter.

    Each element is `(filters, retrieved_evidence)` for one question whose
    filters are non-empty (an unfiltered question does not belong in this
    metric at all). Raises `ValueError` for an empty input — correctness
    over zero filtered questions is undefined, and the 100% gate must never
    be computed from a zero denominator.
    """
    if not questions:
        raise ValueError(
            "metadata-filter correctness needs at least one filtered question"
        )
    correct = sum(
        1
        for filters, candidates in questions
        if not any(metadata_filter_violated(c, filters) for c in candidates)
    )
    return correct / len(questions)


__all__ = [
    "DEFAULT_MRR_DEPTH",
    "NDCG_DEPTH",
    "PRECISION_DEPTH",
    "dcg_at_k",
    "mean_reciprocal_rank",
    "metadata_filter_correctness",
    "metadata_filter_violated",
    "ndcg_at_10",
    "precision_at_5",
    "recall_at_k",
]
