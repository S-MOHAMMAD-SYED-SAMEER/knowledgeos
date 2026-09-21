"""Building reranked candidates and canned model JSON, for generation
tests that don't need a real database.
"""

import json
import re
import uuid

from app.providers.llm import LLMResult
from app.reranking.pipeline import RerankedChunk
from app.retrieval.vector import ChunkEvidence

_CHUNK_UID_ATTR = re.compile(r'chunk_uid="([0-9a-f]{32})"')


def make_reranked(
    final_rank: int,
    *,
    uid: str | None = None,
    text: str = "some chunk text",
    document_title: str = "Some Document",
    rerank_score: float | None = None,
) -> RerankedChunk:
    """One reranked candidate, with a real 32-char lowercase-hex chunk_uid
    derived deterministically from `final_rank` unless one is given."""
    resolved_uid = uid or f"{final_rank:032x}"
    evidence = ChunkEvidence(
        chunk_uid=resolved_uid,
        text=text,
        sequence=final_rank,
        page=None,
        section=None,
        char_start=0,
        char_end=len(text),
        token_count=len(text.split()),
        document_id=uuid.uuid4(),
        document_title=document_title,
        department=None,
        category=None,
        tags=(),
        version_id=uuid.uuid4(),
        version_number=1,
        version_status="active",
    )
    return RerankedChunk(
        evidence=evidence,
        lexical_rank=final_rank,
        vector_rank=final_rank,
        rrf_score=1.0 / final_rank,
        fusion_rank=final_rank,
        rerank_score=rerank_score if rerank_score is not None else (100.0 - final_rank),
        final_rank=final_rank,
    )


def make_candidates(count: int) -> list[RerankedChunk]:
    """`count` candidates, `final_rank` 1..count, contiguous."""
    return [make_reranked(i) for i in range(1, count + 1)]


def valid_json(answer: str, citations: list[str], sufficient_evidence: bool = True) -> str:
    return json.dumps(
        {
            "answer": answer,
            "citations": citations,
            "sufficient_evidence": sufficient_evidence,
        }
    )


def cited_answer_text(*uids: str) -> str:
    """One sentence per uid, each carrying exactly that uid's marker."""
    return " ".join(f"Fact about chunk {i + 1} [{uid}]." for i, uid in enumerate(uids))


class AutoCitingLLMProvider:
    """A reactive test double, unlike the scripted `FakeLLMProvider`: reads
    every `chunk_uid="..."` attribute out of whatever evidence it was
    actually given (`app.generation.prompt.serialize_evidence`'s own
    format) and cites all of them in a well-formed, fully-covered answer.

    For the many pre-existing retrieval/reranking API tests
    (`tests/test_query_api.py`, `tests/test_reranking_api.py`) that seed
    varying, per-test content and expect the *existing* retrieval/
    reranking assertions to keep working unmodified now that `/query`
    also generates -- a scripted fake would need to know each test's exact
    chunk_uids in advance, which is exactly what this avoids. Deterministic
    and content-derived, the same spirit as `FakeRerankProvider`, but for
    generation.

    Nothing in the application ever selects this provider. Only a test
    does.
    """

    model_name = "auto-citing-test-double"

    def complete(self, *, system: str, user: str, max_tokens: int) -> LLMResult:
        uids = _CHUNK_UID_ATTR.findall(user)
        answer = cited_answer_text(*uids) if uids else "No evidence was provided."
        text = valid_json(answer, uids, sufficient_evidence=bool(uids))
        return LLMResult(
            text=text,
            input_tokens=len(system.split()) + len(user.split()),
            output_tokens=len(answer.split()),
        )


__all__ = [
    "AutoCitingLLMProvider",
    "cited_answer_text",
    "make_candidates",
    "make_reranked",
    "valid_json",
]
