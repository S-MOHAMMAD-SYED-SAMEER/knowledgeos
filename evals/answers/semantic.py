"""Semantic grounding: per-sentence entailment against cited chunks,
using the local cross-encoder as an NLI-style scorer (§9).

**Scope, exactly as locked.** §9: "Semantic: per-sentence entailment
against its cited chunks, using the local cross-encoder as an NLI-style
scorer. This is approximate." The scorer is the SPEC-named model,
`cross-encoder/ms-marco-MiniLM-L-6-v2` — the same model
`app/providers/cross_encoder.py` already wraps for reranking, reused here
for a different purpose (entailment-style scoring of one sentence against
its own cited chunk text), never as a substitute for the M6 reranker
*score*, which is a retrieval-relevance signal computed for entirely
different candidates (whole chunks against a query) and is not reused
here at all.

**No threshold is invented.** `cross-encoder/ms-marco-MiniLM-L-6-v2`
produces an unbounded relevance logit, not a probability — there is no
principled absolute cutoff ("0.5 means supported") without empirically
calibrating one against real model output, which this build cannot do
(the model is unavailable; see `check_semantic_scorer_available` below).
The specification gives no formula, threshold, or worked example for
"unsupported claim" anywhere (searched in full — §9, §13, §18; none
define it). Rather than pick a number "by taste" (the same discipline §9
states explicitly for the abstention threshold, applied here to an
equally undefined one), `unsupported_claim_threshold` defaults to `None`:
every sentence's **raw** entailment score is still computed and reported
— real, honest data — but the binary supported/unsupported classification,
and therefore `unsupported_claim_rate`, stays `None` until a threshold has
actually been calibrated and documented. This is an explicit,
**project-level decision**, recorded here because the specification does
not make it.

Deterministic grounding (`app/generation/grounding.py`) stays completely
untouched and is reported as a separate layer, per §9's "kept distinct and
reported separately" — this module never writes to
`answers.grounding_detail`; it is an evaluation-time computation only,
consumed by `evals/answers/metrics.py` and `evals/answers/report.py`.
"""

from dataclasses import dataclass

from app.generation.citations import extract_inline_citations, split_sentences
from app.providers.reranker import Candidate, RerankError, RerankProvider

# Never invented -- see the module docstring. `None` means "raw scores
# only, no supported/unsupported verdict, no unsupported-claim rate."
DEFAULT_UNSUPPORTED_CLAIM_THRESHOLD: float | None = None

# A minimal, fixed probe -- the same pattern `evals/run.py::_CANARY_TEXT`
# uses for the retrieval suite's own models -- to prove the scorer is
# genuinely callable before any real sentence is scored.
_CANARY_SENTENCE = "semantic grounding scorer availability check"
_CANARY_CHUNK = "a chunk of evidence text used only to verify the scorer loads"


class SemanticGroundingUnavailable(RuntimeError):
    """The local cross-encoder could not score anything.

    Never carries the sentence or chunk text that was being scored —
    only what failed structurally.
    """


@dataclass(frozen=True)
class SentenceEntailment:
    """One sentence's raw entailment-style score against the best of its
    own cited chunks. `supported` is `None` whenever no threshold was
    configured — see the module docstring."""

    sentence: str
    cited_chunk_uids: frozenset[str]
    max_score: float
    supported: bool | None


@dataclass(frozen=True)
class SemanticGroundingResult:
    """One answer's semantic layer. `status` is `"scored"` or
    `"unavailable"` — the same explicit two-state shape
    `app/generation/grounding.py` already uses for
    `grounding_detail.semantic`, kept honest here rather than a numeric
    result standing in for "we don't actually know."""

    status: str
    reason: str | None
    sentence_scores: list[SentenceEntailment]
    unsupported_claim_rate: float | None


def check_semantic_scorer_available(scorer: RerankProvider) -> str | None:
    """A canary call proving the scorer actually produces output, run
    once before any official scoring begins — the same discipline
    `evals/run.py::_unavailable_reason` applies to the retrieval suite's
    embedding and reranking models. Returns a message naming what is
    unavailable, or `None` once the scorer has genuinely responded."""
    try:
        scorer.rerank(_CANARY_SENTENCE, [Candidate(id="canary", text=_CANARY_CHUNK)])
    except RerankError as exc:
        return f"the semantic grounding scorer ({scorer.model_name}) is unavailable: {exc}"
    return None


def score_answer_semantic(
    *,
    answer_text: str,
    chunk_text_by_uid: dict[str, str],
    scorer: RerankProvider,
    unsupported_claim_threshold: float | None = DEFAULT_UNSUPPORTED_CLAIM_THRESHOLD,
) -> SemanticGroundingResult:
    """Score every citation-bearing sentence of `answer_text` against the
    text of its own cited chunk(s).

    A sentence with no inline citation marker is skipped — citation
    coverage (rule 2) is `app/generation/citations.py`'s job, not this
    module's; there is nothing to entail a claim against without a cited
    chunk. `chunk_text_by_uid` supplies the text for whichever chunk_uids
    the answer actually cites (the selected evidence, in practice).

    Returns `status="unavailable"` with an empty result — never a
    fabricated score — the moment the scorer itself fails.
    """
    scored: list[SentenceEntailment] = []

    for sentence in split_sentences(answer_text):
        cited = extract_inline_citations(sentence)
        if not cited:
            continue

        candidates = [
            Candidate(id=uid, text=chunk_text_by_uid[uid])
            for uid in sorted(cited)
            if uid in chunk_text_by_uid
        ]
        if not candidates:
            continue

        try:
            results = scorer.rerank(sentence, candidates)
        except RerankError as exc:
            return SemanticGroundingResult(
                status="unavailable",
                reason=f"the semantic scorer failed while scoring ({type(exc).__name__})",
                sentence_scores=[],
                unsupported_claim_rate=None,
            )

        best_score = max(item.score for item in results)
        supported = (
            None
            if unsupported_claim_threshold is None
            else best_score >= unsupported_claim_threshold
        )
        scored.append(
            SentenceEntailment(
                sentence=sentence,
                cited_chunk_uids=cited,
                max_score=best_score,
                supported=supported,
            )
        )

    if unsupported_claim_threshold is None or not scored:
        return SemanticGroundingResult(
            status="scored", reason=None, sentence_scores=scored, unsupported_claim_rate=None
        )

    unsupported = sum(1 for entry in scored if entry.supported is False)
    return SemanticGroundingResult(
        status="scored",
        reason=None,
        sentence_scores=scored,
        unsupported_claim_rate=unsupported / len(scored),
    )


__all__ = [
    "DEFAULT_UNSUPPORTED_CLAIM_THRESHOLD",
    "SemanticGroundingResult",
    "SemanticGroundingUnavailable",
    "SentenceEntailment",
    "check_semantic_scorer_available",
    "score_answer_semantic",
]
