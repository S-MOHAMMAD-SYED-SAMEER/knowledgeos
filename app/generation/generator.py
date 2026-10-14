"""Orchestrating one query's generation: select evidence, decide
pre-LLM abstention, call the model, parse, validate, ground.

Pure orchestration over an already-computed reranked candidate list and an
injected `LLMProvider` — this module makes at most one provider call and
never touches the database. Persistence is `app/generation/persistence.py`'s
own, separate job, kept apart so this module is fully testable against the
scripted fake with no database at all.

**Invalid output is rejected, never shown, and never repaired.** The
specification's own words for rule 1: "invalid citation, answer rejected."
`validate_citations` raises `CitationError` (a `ValueError` subclass) on
any violation, and the output parser raises `GenerationError` on anything
that is not exactly the three-field JSON object. Neither is ever softened:
a rejected answer is not displayed, and there is no silent repair (no
stripping of code fences, no guessing at a missing field).

What changed is what the person sees. A model that forgets one citation marker
on one sentence is not a reason to show an error page. So a rejection is
**retried once** (the same request; a second sample from the model), and if the
second reply is rejected too, the caller gets a fixed, honest reply —
`UNVERIFIED_ANSWER_TEXT`, "I couldn't verify this answer" — marked
`verification_failure`, with nothing the model wrote in it. Both rejections are
logged (what failed, never the text) and the evaluation counts them separately
from abstentions and from successes. Tokens from every attempt are summed, so
cost is what was actually spent.

A model that declines to answer at all (`LLMBlocked`) is not retried: asking
again would only be asking the same question twice.

Provider failures (`LLMError` and its other subclasses: rate limit, timeout,
outage) are not answers and are not softened here; they propagate to the API
layer, which turns them into a clear, safe 503.
"""

import json
import logging
from dataclasses import dataclass

from app.generation.abstention import (
    DEFAULT_ABSTENTION_TEXT,
    post_llm_abstention,
    pre_llm_abstention,
)
from app.generation.citations import CitationError, validate_citations
from app.generation.evidence import SelectedEvidence, select_evidence
from app.generation.grounding import SEMANTIC_UNAVAILABLE_REASON, deterministic_grounding
from app.generation.prompt import LoadedPrompt, load_prompt, serialize_evidence
from app.providers.llm import LLMBlocked, LLMProvider
from app.reranking.pipeline import RerankedChunk

logger = logging.getLogger(__name__)

# The specification's own three fields, verbatim, and exactly these three.
REQUIRED_FIELDS = frozenset({"answer", "citations", "sufficient_evidence"})

# The same three fields as a JSON Schema, sent to a provider that can be asked for
# JSON mode. It makes malformed output rarer; `_parse` below is still the check.
ANSWER_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "sufficient_evidence": {"type": "boolean"},
    },
    "required": ["answer", "citations", "sufficient_evidence"],
}

# One try, and one retry. Two rejected replies in a row is a verdict, not bad luck.
MAX_GENERATION_ATTEMPTS = 2

# What the person sees when no reply could be verified. Fixed text: nothing the model
# wrote is ever shown from a rejected attempt.
UNVERIFIED_ANSWER_TEXT = (
    "I couldn't verify this answer against the documents, so I'm not going to show it. "
    "Please try again, or rephrase your question."
)

# Why a reply could not be verified (`GeneratedAnswer.verification_failure`).
FAILURE_CITATION_INVALID = "citation_invalid"
FAILURE_MALFORMED_OUTPUT = "malformed_output"
FAILURE_MODEL_BLOCKED = "model_blocked"


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
    # How many times the model was asked (1, or 2 after a rejected first reply).
    attempts: int = 1
    # None for a verified answer. Otherwise why no reply could be verified; the text
    # above is then `UNVERIFIED_ANSWER_TEXT`, not anything the model wrote.
    verification_failure: str | None = None


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

    input_tokens = 0
    output_tokens = 0
    failure: Exception | None = None

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        try:
            result = llm.complete(
                system=resolved_prompt.text,
                user=user_prompt,
                max_tokens=max_tokens,
                json_schema=ANSWER_JSON_SCHEMA,
            )
        except LLMBlocked as exc:
            # The model declined. Asking again is asking the same thing twice.
            logger.warning(
                "The model declined to answer (%s); showing the unverified-answer reply.",
                exc.reason or "no reason given",
            )
            return _unverified(
                kind=FAILURE_MODEL_BLOCKED,
                attempts=attempt,
                selected=selected,
                model_name=llm.model_name,
                prompt=resolved_prompt,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        # Any other LLMError propagates uncaught: the API layer's 503 mapping.

        input_tokens += result.input_tokens
        output_tokens += result.output_tokens

        try:
            parsed = _parse(result.text, finish_reason=result.finish_reason)
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
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                attempts=attempt,
            )
        except (GenerationError, CitationError) as exc:
            failure = exc
            logger.warning(
                "The model's reply was rejected on attempt %d of %d: %s",
                attempt,
                MAX_GENERATION_ATTEMPTS,
                exc,
            )

    logger.warning(
        "No reply could be verified after %d attempts (%s); showing the unverified-answer reply.",
        MAX_GENERATION_ATTEMPTS,
        type(failure).__name__,
    )
    return _unverified(
        kind=FAILURE_CITATION_INVALID if isinstance(failure, CitationError) else FAILURE_MALFORMED_OUTPUT,
        attempts=MAX_GENERATION_ATTEMPTS,
        selected=selected,
        model_name=llm.model_name,
        prompt=resolved_prompt,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _unverified(
    *,
    kind: str,
    attempts: int,
    selected: list[SelectedEvidence],
    model_name: str | None,
    prompt: LoadedPrompt,
    input_tokens: int,
    output_tokens: int,
) -> GeneratedAnswer:
    """The reply for a question whose answer could not be verified.

    No claims, so no citations and nothing to validate (`citation_valid` is
    vacuously true, as for an abstention); `grounded` is false because nothing
    was checked against evidence; it is not an abstention, because the evidence
    was never judged insufficient. `grounding_detail["generation"]` is how every
    later reader (the API, the UI, the evaluation) tells it apart.
    """
    return GeneratedAnswer(
        answer_text=UNVERIFIED_ANSWER_TEXT,
        citations=[],
        sufficient_evidence=False,
        abstained=False,
        abstain_reason=None,
        citation_valid=True,
        grounded=False,
        grounding_detail={
            "deterministic": {
                "status": "not_applicable",
                "reason": "no reply from the model could be verified",
            },
            "semantic": {"status": "unavailable", "reason": SEMANTIC_UNAVAILABLE_REASON},
            "generation": {"verified": False, "attempts": attempts, "failure": kind},
        },
        selected=selected,
        model_name=model_name,
        prompt_version=prompt.version,
        prompt_content_hash=prompt.content_hash,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        attempts=attempts,
        verification_failure=kind,
    )


def _parse(text: str, finish_reason: str | None = None) -> dict:
    """Parse and validate the model's raw text against the specification's
    exact three-field structure. No silent repair: a missing field, an
    extra field, or a wrong type all raise `GenerationError` rather than
    being coerced or defaulted.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        if finish_reason == "MAX_TOKENS":
            raise GenerationError(
                "model output was cut off at the token limit (finish reason MAX_TOKENS)"
            ) from exc
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
    attempts: int = 1,
) -> GeneratedAnswer:
    # Raises CitationError on any violation — the answer is rejected, per the
    # specification's own words, and nothing below this call ever executes for a
    # rejected answer. `generate_answer` catches it to retry once, then to give the
    # unverified-answer reply; it never shows the rejected text.
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

    detail = grounding.detail
    if attempts > 1:
        # Only when it happened, so an ordinary answer's detail is unchanged.
        detail = {**detail, "generation": {"verified": True, "attempts": attempts}}

    return GeneratedAnswer(
        answer_text=answer_text,
        citations=citations,
        sufficient_evidence=sufficient_evidence,
        abstained=abstained,
        abstain_reason=abstain_reason,
        citation_valid=True,  # validate_citations did not raise
        grounded=grounding.grounded,
        grounding_detail=detail,
        selected=selected,
        model_name=model_name,
        prompt_version=prompt.version,
        prompt_content_hash=prompt.content_hash,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        attempts=attempts,
    )


__all__ = [
    "ANSWER_JSON_SCHEMA",
    "FAILURE_CITATION_INVALID",
    "FAILURE_MALFORMED_OUTPUT",
    "FAILURE_MODEL_BLOCKED",
    "MAX_GENERATION_ATTEMPTS",
    "REQUIRED_FIELDS",
    "UNVERIFIED_ANSWER_TEXT",
    "GeneratedAnswer",
    "GenerationError",
    "generate_answer",
]
