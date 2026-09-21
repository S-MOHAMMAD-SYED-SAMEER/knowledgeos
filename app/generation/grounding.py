"""Grounding: the specification's two-layer check (§9), kept distinct and
reported separately.

**Deterministic layer (this milestone):** citation validity, citation
coverage, selected-chunk validity, abstention-flag consistency. The
specification requires these to be "exact and must pass at 100%" — and
they do, structurally: `app/generation/generator.py` calls
`app/generation/citations.py::validate_citations` as a hard gate *before*
this module ever runs, so by the time an answer reaches here its citations
have already been proven valid. This module recomputes the same checks
independently rather than trusting that gate blindly — the same
defense-in-depth instinct `evals/retrieval/metrics.py` applies to its own
structurally-unreachable nDCG guard, documented there for the same reason
it is documented here.

**Semantic layer (NOT this milestone):** per-sentence entailment against
cited chunks, using the local cross-encoder as an NLI-style scorer. That
model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is absent from the local
model cache in every environment this project has been built in so far —
the same, unchanged condition milestones 4, 6 and 7 already documented.
This milestone reports the semantic layer as **unexercised** rather than
fabricating a score from the reranker's own relevance score, which is a
retrieval-relevance signal, not an entailment signal, and the specification
never authorizes reusing one as the other. `grounding_detail.semantic` is
always `{"status": "unavailable", "reason": ...}` here — never a number —
so a later milestone can add a real scorer without redesigning this shape.
"""

from dataclasses import dataclass

from app.generation.citations import (
    abstention_flag_consistent,
    citation_coverage,
    citation_validity,
    extract_all_inline_citations,
)

SEMANTIC_UNAVAILABLE_REASON = (
    "the local cross-encoder NLI scorer is not available in this "
    "environment; semantic grounding was not exercised for this answer"
)


@dataclass(frozen=True)
class GroundingResult:
    """`grounded` reflects the deterministic layer's verdict only — it
    says nothing about semantic entailment, which `detail.semantic`
    reports separately and honestly as unavailable rather than folded into
    this boolean."""

    grounded: bool
    citation_valid: bool
    detail: dict


def deterministic_grounding(
    *,
    answer_text: str,
    declared_citations: list[str],
    retrieved_chunk_uids: frozenset[str],
    selected_chunk_uids: frozenset[str],
    abstained: bool,
) -> GroundingResult:
    """Independently recompute the three deterministic checks and compose
    `grounding_detail`, the JSONB shape `answers.grounding_detail`
    persists."""
    if abstained:
        valid = not declared_citations and not extract_all_inline_citations(answer_text)
        coverage = 1.0
    else:
        valid = citation_validity(
            answer_text=answer_text,
            declared_citations=declared_citations,
            retrieved_chunk_uids=retrieved_chunk_uids,
            selected_chunk_uids=selected_chunk_uids,
        )
        coverage = citation_coverage(answer_text)

    consistent = abstention_flag_consistent(
        abstained=abstained, declared_citations=declared_citations, answer_text=answer_text
    )

    grounded = valid and consistent and (abstained or coverage == 1.0)

    detail = {
        "deterministic": {
            "citation_valid": valid,
            "citation_coverage": coverage,
            "abstention_flag_consistent": consistent,
        },
        "semantic": {
            "status": "unavailable",
            "reason": SEMANTIC_UNAVAILABLE_REASON,
        },
    }

    return GroundingResult(grounded=grounded, citation_valid=valid, detail=detail)


__all__ = [
    "SEMANTIC_UNAVAILABLE_REASON",
    "GroundingResult",
    "deterministic_grounding",
]
