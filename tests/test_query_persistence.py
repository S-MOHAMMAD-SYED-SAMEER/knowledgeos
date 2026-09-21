"""Persisting `queries` / `retrieved_chunks` / `answers`, against a real
database and real retrieval/reranking output -- the only thing faked is
the LLM.
"""

import pytest
from sqlalchemy import select

from app.generation.generator import generate_answer
from app.generation.persistence import persist_query
from app.generation.prompt import load_prompt
from app.models import Answer, Chunk, Query, RetrievedChunk
from app.parsing import normalize
from app.providers import FakeEmbeddingProvider, PassthroughRerankProvider
from app.providers.fake_llm import FakeLLMProvider
from app.reranking.pipeline import rerank
from app.retrieval.filters import RetrievalFilters
from app.retrieval.pipeline import retrieve
from app.storage import LocalStorage

from .generation_fixtures import cited_answer_text, valid_json
from .retrieval_fixtures import seed_active_version

PROMPT = load_prompt()


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


def _reranked_candidates(session, storage, embeddings, *, query_text: str):
    """Seed one document, then run the real retrieve() + rerank() pipeline
    against it -- genuine `RerankedChunk`s with real, database-backed
    chunk_uids."""
    seed_active_version(
        session, storage, embeddings=embeddings,
        text="production database access requires manager approval and expires after seven days",
    )
    normalized = normalize(query_text)
    vector = embeddings.embed([normalized])[0]
    result = retrieve(
        session,
        normalized_text=normalized,
        query_vector=vector,
        filters=RetrievalFilters(),
    )
    return normalized, rerank(normalized, result.candidates, PassthroughRerankProvider())


def test_a_successful_query_persists_one_row_in_each_table(
    session, storage, embeddings
) -> None:
    normalized, candidates = _reranked_candidates(
        session, storage, embeddings, query_text="production database access"
    )
    assert candidates  # the fixture must have actually retrieved something

    uids = [c.evidence.chunk_uid for c in candidates]
    fake_llm = FakeLLMProvider([valid_json(cited_answer_text(*uids), uids)])

    answer = generate_answer(
        query_text=normalized,
        candidates=candidates,
        llm=fake_llm,
        abstention_threshold=None,
        max_tokens=200,
        prompt=PROMPT,
    )

    query_row, answer_row = persist_query(
        session,
        query_text="production database access",
        normalized_text=normalized,
        filters={},
        candidates=candidates,
        answer=answer,
    )

    assert session.execute(select(Query)).scalars().all() == [query_row]
    assert session.execute(select(Answer)).scalars().all() == [answer_row]
    retrieved_rows = session.execute(select(RetrievedChunk)).scalars().all()
    assert len(retrieved_rows) == len(candidates)


def test_the_query_row_records_the_model_and_prompt_version(
    session, storage, embeddings
) -> None:
    normalized, candidates = _reranked_candidates(
        session, storage, embeddings, query_text="production database access"
    )
    uids = [c.evidence.chunk_uid for c in candidates]
    fake_llm = FakeLLMProvider([valid_json(cited_answer_text(*uids), uids)])

    answer = generate_answer(
        query_text=normalized, candidates=candidates, llm=fake_llm,
        abstention_threshold=None, max_tokens=200, prompt=PROMPT,
    )
    query_row, _ = persist_query(
        session, query_text="q", normalized_text=normalized, filters={},
        candidates=candidates, answer=answer,
    )

    assert query_row.model == fake_llm.model_name
    assert query_row.prompt_version == "knowledge_answer_v1"
    assert query_row.input_tokens is not None
    assert query_row.output_tokens is not None


def test_m9_owned_timing_and_cost_columns_are_null(
    session, storage, embeddings
) -> None:
    """Locked decision: nullable, and never a fabricated zero."""
    normalized, candidates = _reranked_candidates(
        session, storage, embeddings, query_text="production database access"
    )
    uids = [c.evidence.chunk_uid for c in candidates]
    fake_llm = FakeLLMProvider([valid_json(cited_answer_text(*uids), uids)])
    answer = generate_answer(
        query_text=normalized, candidates=candidates, llm=fake_llm,
        abstention_threshold=None, max_tokens=200, prompt=PROMPT,
    )
    query_row, _ = persist_query(
        session, query_text="q", normalized_text=normalized, filters={},
        candidates=candidates, answer=answer,
    )

    assert query_row.retrieval_ms is None
    assert query_row.rerank_ms is None
    assert query_row.llm_ms is None
    assert query_row.total_ms is None
    assert query_row.cost_usd is None


def test_retrieved_chunks_selected_flag_matches_the_top_eight(
    session, storage, embeddings
) -> None:
    for i in range(12):
        seed_active_version(
            session, storage,
            embeddings=embeddings,
            text=f"widget policy number {i} about access control",
            title=f"doc{i}.txt",
        )
    normalized = normalize("widget policy access control")
    vector = embeddings.embed([normalized])[0]
    result = retrieve(
        session, normalized_text=normalized, query_vector=vector,
        filters=RetrievalFilters(),
    )
    candidates = rerank(normalized, result.candidates, PassthroughRerankProvider())
    assert len(candidates) > 8

    top_uids = [c.evidence.chunk_uid for c in candidates[:8]]
    fake_llm = FakeLLMProvider([valid_json(cited_answer_text(*top_uids), top_uids)])
    answer = generate_answer(
        query_text=normalized, candidates=candidates, llm=fake_llm,
        abstention_threshold=None, max_tokens=200, prompt=PROMPT,
    )
    persist_query(
        session, query_text="q", normalized_text=normalized, filters={},
        candidates=candidates, answer=answer,
    )

    rows = session.execute(select(RetrievedChunk)).scalars().all()
    selected_chunk_ids = {row.chunk_id for row in rows if row.selected}
    assert len(selected_chunk_ids) == 8

    top_chunk_ids = set(
        session.execute(
            select(Chunk.id).where(Chunk.chunk_uid.in_(top_uids))
        ).scalars().all()
    )
    assert selected_chunk_ids == top_chunk_ids


def test_an_abstained_query_still_persists_with_no_retrieved_chunks(
    session,
) -> None:
    """Zero candidates: the pre-LLM abstention case. Still persisted --
    an abstention is a real, valid answer at this layer."""
    fake_llm = FakeLLMProvider([])  # must never be called

    answer = generate_answer(
        query_text="anything", candidates=[], llm=fake_llm,
        abstention_threshold=None, max_tokens=200, prompt=PROMPT,
    )
    query_row, answer_row = persist_query(
        session, query_text="q", normalized_text="q", filters={},
        candidates=[], answer=answer,
    )

    assert answer_row.abstained
    assert session.execute(
        select(RetrievedChunk).where(RetrievedChunk.query_id == query_row.id)
    ).scalars().all() == []


def test_answer_row_citation_valid_is_always_true(
    session, storage, embeddings
) -> None:
    """Every row this milestone writes has citation_valid = True -- an
    invalid-citation answer is rejected before persistence is ever
    reached (see app/generation/generator.py)."""
    normalized, candidates = _reranked_candidates(
        session, storage, embeddings, query_text="production database access"
    )
    uids = [c.evidence.chunk_uid for c in candidates]
    fake_llm = FakeLLMProvider([valid_json(cited_answer_text(*uids), uids)])
    answer = generate_answer(
        query_text=normalized, candidates=candidates, llm=fake_llm,
        abstention_threshold=None, max_tokens=200, prompt=PROMPT,
    )
    _, answer_row = persist_query(
        session, query_text="q", normalized_text=normalized, filters={},
        candidates=candidates, answer=answer,
    )

    assert answer_row.citation_valid is True


def test_answers_table_has_at_most_one_row_per_query(
    session, storage, embeddings
) -> None:
    """The unique constraint on `answers.query_id` -- a locked project
    decision, since the specification names no explicit rule."""
    import sqlalchemy.exc

    normalized, candidates = _reranked_candidates(
        session, storage, embeddings, query_text="production database access"
    )
    uids = [c.evidence.chunk_uid for c in candidates]
    fake_llm = FakeLLMProvider(
        [valid_json(cited_answer_text(*uids), uids), valid_json(cited_answer_text(*uids), uids)]
    )
    answer = generate_answer(
        query_text=normalized, candidates=candidates, llm=fake_llm,
        abstention_threshold=None, max_tokens=200, prompt=PROMPT,
    )
    query_row, _ = persist_query(
        session, query_text="q", normalized_text=normalized, filters={},
        candidates=candidates, answer=answer,
    )

    from app.models import Answer as AnswerModel

    duplicate = AnswerModel(
        query_id=query_row.id,
        answer_text="second answer for the same query",
        abstained=False,
        citations=[],
        citation_valid=True,
        grounded=True,
        grounding_detail={},
    )
    session.add(duplicate)
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        session.commit()
