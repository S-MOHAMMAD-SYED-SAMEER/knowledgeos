"""Orchestrating one query's generation: select evidence, decide
pre-LLM abstention, call the model, parse, validate, ground.

Pure orchestration over an already-computed reranked candidate list and an
injected `LLMProvider` — this module makes at most one provider call and
never touches the database. Persistence is `app/generation/persistence.py`'s
own, separate job, kept apart so this module is fully testable against the
scripted fake with no database at all.

**Invalid output is a rejection, not a degraded result.** The
specification's own words for rule 1: "invalid citation, answer rejected."
`validate_citations` raises `CitationError` (a `ValueError` subclass) on
any violation, and this module does not catch it — it propagates to
whichever caller invoked `generate_answer`, which for `POST /query` is the
API layer's own 502 mapping. Nothing is persisted for a rejected answer
(`app/generation/persistence.py` is never reached), so every `Answer` row
this milestone ever writes has `citation_valid = True` by construction —
documented here because it is the direct, intended consequence of treating
"rejected" as "not written," the same discipline the specification applies
to fabricated metrics (§18): a number that did not pass validation is never
recorded as if it had.
"""

import json
from dataclasses import dataclass

from app.generation.abstention import (
    DEFAULT_ABSTENTION_TEXT,
    post_llm_abstention,
    pre_llm_abstention,
)
from app.generation.citations import validate_citations
from app.generation.evidence import SelectedEvidence, select_evidence
from app.generation.grounding import deterministic_grounding
from app.generation.prompt import LoadedPrompt, load_prompt, serialize_evidence
from app.providers.llm import LLMProvider
from app.reranking.pipeline import RerankedChunk

# The specification's own three fields, verbatim, and exactly these three.
REQUIRED_FIELDS = frozenset({"answer", "citations", "sufficient_evidence"})


class GenerationError(RuntimeError):
    """The model's raw output could not be parsed or did not match the
    required structure.

    Never carries the prompt, the evidence text, or the model's raw
    output — only what kind of structural failure occurred.
    """


@dataclass(frozen=True)
class GeneratedAnswer:
    """Everything one query's generation produced, ready to persist or
    to serve."""

    answer_text: str
    citations: list[str]
    sufficient_evidence: bool
    abstained: bool
    abstain_reason: str | None
    citation_valid: bool
    grounded: bool
    grounding_detail: dict
    selected: list[SelectedEvidence]
    model_name: str | None  # None only when the LLM was never called
    prompt_version: str
    prompt_content_hash: str
    input_tokens: int
    output_tokens: int


def generate_answer(
    *,
    query_text: str,
    candidates: list[RerankedChunk],
    llm: LLMProvider,
    abstention_threshold: float | None,
    max_tokens: int,
    prompt: LoadedPrompt | None = None,
) -> GeneratedAnswer:
    """Run the full generation decision for one query.

    `candidates` is milestone 6's already-reranked top-20 (or fewer) —
    this function never calls `app.retrieval` or `app.reranking` itself,
    consistent with the specification's layering rule. An empty list is
    the zero-candidate pre-LLM abstention case, handled with no provider
    call at all.
    """
    resolved_prompt = prompt or load_prompt()
    selected = select_evidence(candidates)
    retrieved_uids = frozenset(c.evidence.chunk_uid for c in candidates)
    selected_uids = frozenset(
        s.candidate.evidence.chunk_uid for s in selected if s.selected
    )

    top_rerank_score = candidates[0].rerank_score if candidates else None
    pre = pre_llm_abstention(
        candidate_count=len(candidates),
        top_rerank_score=top_rerank_score,
        threshold=abstention_threshold,
    )

    if pre.abstain:
        return _finish(
            answer_text=DEFAULT_ABSTENTION_TEXT,
            citations=[],
            sufficient_evidence=False,
            abstained=True,
            abstain_reason=pre.reason,
            selected=selected,
            retrieved_uids=retrieved_uids,
            selected_uids=selected_uids,
            model_name=None,
            prompt=resolved_prompt,
            input_tokens=0,
            output_tokens=0,
        )

    evidence_block = serialize_evidence(selected)
    user_prompt = f"Question:\n{query_text}\n\nEvidence:\n{evidence_block}"

    # LLMError propagates uncaught — the API layer's 503 mapping.
    result = llm.complete(
        system=resolved_prompt.text, user=user_prompt, max_tokens=max_tokens
    )

    parsed = _parse(result.text)
    post = post_llm_abstention(sufficient_evidence=parsed["sufficient_evidence"])

    return _finish(
        answer_text=parsed["answer"],
        citations=parsed["citations"],
        sufficient_evidence=parsed["sufficient_evidence"],
        abstained=post.abstain,
        abstain_reason=post.reason,
        selected=selected,
        retrieved_uids=retrieved_uids,
        selected_uids=selected_uids,
        model_name=llm.model_name,
        prompt=resolved_prompt,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def _parse(text: str) -> dict:
    """Parse and validate the model's raw text against the specification's
    exact three-field structure. No silent repair: a missing field, an
    extra field, or a wrong type all raise `GenerationError` rather than
    being coerced or defaulted.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenerationError(
            f"model output is not valid JSON ({type(exc).__name__})"
        ) from exc

    if not isinstance(data, dict):
        raise GenerationError("model output is not a JSON object")

    if set(data) != REQUIRED_FIELDS:
        raise GenerationError(
            f"model output has field(s) {sorted(data)}, expected exactly "
            f"{sorted(REQUIRED_FIELDS)}"
        )

    answer = data["answer"]
    citations = data["citations"]
    sufficient_evidence = data["sufficient_evidence"]

    if not isinstance(answer, str):
        raise GenerationError("'answer' must be a string")
    if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
        raise GenerationError("'citations' must be a list of strings")
    if not isinstance(sufficient_evidence, bool):
        raise GenerationError("'sufficient_evidence' must be a boolean")

    return {
        "answer": answer,
        "citations": citations,
        "sufficient_evidence": sufficient_evidence,
    }


def _finish(
    *,
    answer_text: str,
    citations: list[str],
    sufficient_evidence: bool,
    abstained: bool,
    abstain_reason: str | None,
    selected: list[SelectedEvidence],
    retrieved_uids: frozenset[str],
    selected_uids: frozenset[str],
    model_name: str | None,
    prompt: LoadedPrompt,
    input_tokens: int,
    output_tokens: int,
) -> GeneratedAnswer:
    # Raises CitationError (uncaught) on any violation — the answer is
    # rejected, per the specification's own words, and nothing below this
    # call ever executes for a rejected answer.
    validate_citations(
        answer_text=answer_text,
        declared_citations=citations,
        retrieved_chunk_uids=retrieved_uids,
        selected_chunk_uids=selected_uids,
        abstained=abstained,
    )

    grounding = deterministic_grounding(
        answer_text=answer_text,
        declared_citations=citations,
        retrieved_chunk_uids=retrieved_uids,
        selected_chunk_uids=selected_uids,
        abstained=abstained,
    )

    return GeneratedAnswer(
        answer_text=answer_text,
        citations=citations,
        sufficient_evidence=sufficient_evidence,
        abstained=abstained,
        abstain_reason=abstain_reason,
        citation_valid=True,  # validate_citations did not raise
        grounded=grounding.grounded,
        grounding_detail=grounding.detail,
        selected=selected,
        model_name=model_name,
        prompt_version=prompt.version,
        prompt_content_hash=prompt.content_hash,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


__all__ = [
    "REQUIRED_FIELDS",
    "GeneratedAnswer",
    "GenerationError",
    "generate_answer",
]
