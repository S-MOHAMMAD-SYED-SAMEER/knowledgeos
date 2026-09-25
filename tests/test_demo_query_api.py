"""`POST /query` in demo mode, end to end against a real database — the
one place M2 actually decides demo mode (`app/api/query.py::llm_provider`)
exercised the way a real visitor request would reach it, not through a
dependency override of `llm_provider` itself.

`embedding_provider` and `rerank_provider` are overridden to test-only/
real-shipped substitutes, the same as `tests/test_query_api.py` — the real
local BGE model and cross-encoder are unavailable in this environment (no
network, no model cache; see the M2 report). `llm_provider` is
deliberately left un-overridden: exercising the *real* dependency function,
with `Settings.demo_mode=True`, is the entire point of this file.

**Why these tests seed only the 1-2 real documents a scenario actually
cites, not the full 10-document corpus.** `FakeEmbeddingProvider`'s own
docstring calls its vectors "meaningless" for real retrieval, and an
empirical probe run while building this milestone (seeding the full
corpus with it) confirmed that first-hand: several curated citations
landed outside the real reranked top-8, and two fell outside the top-20
fused candidates entirely — noise from the fake vector channel, fused via
RRF, outranking a real, strong lexical match. Scoping each test to just
the document(s) it needs removes that noise (little else competes
lexically), so the real lexical channel — real `ts_rank_cd`, not a stand-
in — reliably surfaces the correct chunk. This is a real, reproducible
property of *this test's* narrowed corpus, not a claim about
production-scale relevance; the full-corpus caveat belongs in, and is
documented in, the M2 report's limitations section. Every document seeded
here is still real fixture content (`evals/fixtures/knowledge_base/`),
under its real manifest `document_id`/`version_id`, run through the real,
unmodified ingestion pipeline (`app.ingestion.service`/`pipeline`) — the
same functions `tests/retrieval_fixtures.py::seed_active_version` and
`evals/retrieval/corpus.py::ensure_corpus_seeded` already use, not a
second implementation of either.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.api.query import DEMO_UNSUPPORTED_MODEL_NAME, DEMO_UNSUPPORTED_TEXT
from app.generation.abstention import DEFAULT_ABSTENTION_TEXT
from app.generation.demo_scenarios import DEMO_SCENARIOS
from app.providers.demo_llm import DEMO_MODEL_NAME

from .demo_fixtures import SCENARIO_DOCUMENTS, seed_demo_document

# `storage`, `embeddings`, `demo_client`, `fake_demo_client` are fixtures
# from `tests/demo_fixtures.py`, registered as a plugin in
# `tests/conftest.py` -- pytest resolves them by parameter name below with
# no import needed (see that plugin registration's own comment for why).


# --- matched curated scenarios, through real retrieval -----------------


@pytest.mark.parametrize(
    "scenario",
    [s for s in DEMO_SCENARIOS if s.question_id != "ie001"],
    ids=lambda s: s.question_id,
)
def test_matched_scenario_returns_the_hand_verified_grounded_answer(
    scenario, fake_demo_client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        for key in SCENARIO_DOCUMENTS[scenario.question_id]:
            seed_demo_document(session, storage, embeddings, key)

    body = fake_demo_client.post("/query", json={"query": scenario.question_text}).json()

    assert body["answer"] == scenario.answer_text
    assert body["citations"] == list(scenario.citations)
    assert body["citation_valid"] is True
    assert body["grounded"] is True
    assert body["abstained"] is False
    assert body["model"] == DEMO_MODEL_NAME


def test_a_matched_scenario_tolerates_case_and_whitespace_over_http(
    fake_demo_client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    scenario = next(s for s in DEMO_SCENARIOS if s.question_id == "da001")
    with Session(migrated_engine) as session:
        seed_demo_document(session, storage, embeddings, "remote-work")

    noisy = "  " + scenario.question_text.upper() + "  "
    body = fake_demo_client.post("/query", json={"query": noisy}).json()

    assert body["answer"] == scenario.answer_text
    assert body["grounded"] is True


# --- ie001: real post-LLM abstention, real abstention text -------------


def test_ie001_abstains_through_http_with_the_real_abstention_text(
    fake_demo_client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    scenario = next(s for s in DEMO_SCENARIOS if s.question_id == "ie001")
    with Session(migrated_engine) as session:
        # Any nonzero real corpus — irrelevant to the question, on purpose:
        # this must abstain because the demo provider itself reports
        # insufficient evidence, not because retrieval found nothing.
        seed_demo_document(session, storage, embeddings, "incident-response")

    body = fake_demo_client.post("/query", json={"query": scenario.question_text}).json()

    assert body["answer"] == DEFAULT_ABSTENTION_TEXT
    assert body["abstained"] is True
    assert body["citations"] == []
    assert body["citation_valid"] is True
    assert body["model"] == DEMO_MODEL_NAME


# --- unmatched questions: honest, distinct, never grounded --------------


def test_unmatched_question_returns_the_honest_unsupported_response(
    fake_demo_client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        seed_demo_document(session, storage, embeddings, "remote-work")

    body = fake_demo_client.post(
        "/query", json={"query": "What is the weather like today?"}
    ).json()

    assert body["answer"] == DEMO_UNSUPPORTED_TEXT
    assert body["model"] == DEMO_UNSUPPORTED_MODEL_NAME
    assert body["citations"] == []
    # The explicit M2 requirement: retrieval finding chunks must never, by
    # itself, be reported as grounding for an answer nothing generated.
    assert body["grounded"] is False
    assert body["abstained"] is False


def test_a_near_miss_of_a_curated_question_is_also_unsupported_not_a_match(
    fake_demo_client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    scenario = next(s for s in DEMO_SCENARIOS if s.question_id == "da001")
    with Session(migrated_engine) as session:
        seed_demo_document(session, storage, embeddings, "remote-work")

    close_but_not_it = scenario.question_text.rstrip("?") + ", roughly speaking?"
    body = fake_demo_client.post("/query", json={"query": close_but_not_it}).json()

    assert body["answer"] == DEMO_UNSUPPORTED_TEXT
    assert body["model"] == DEMO_UNSUPPORTED_MODEL_NAME


# --- Gemini is structurally unreachable, over HTTP, in demo mode --------


def test_gemini_is_never_reached_over_http_for_a_matched_or_unmatched_question(
    fake_demo_client: TestClient,
    migrated_engine: Engine,
    storage,
    embeddings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.query as query_module

    def _fail_if_called() -> None:
        raise AssertionError("the real Gemini accessor must not be reached in demo mode")

    monkeypatch.setattr(query_module, "get_llm_provider", _fail_if_called)

    scenario = next(s for s in DEMO_SCENARIOS if s.question_id == "da001")
    with Session(migrated_engine) as session:
        seed_demo_document(session, storage, embeddings, "remote-work")

    matched = fake_demo_client.post("/query", json={"query": scenario.question_text})
    unmatched = fake_demo_client.post("/query", json={"query": "unrelated question"})

    assert matched.status_code == 200
    assert unmatched.status_code == 200


# --- schema: unchanged by demo mode -------------------------------------


def test_response_schema_is_unchanged_in_demo_mode(
    fake_demo_client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    scenario = next(s for s in DEMO_SCENARIOS if s.question_id == "da001")
    with Session(migrated_engine) as session:
        seed_demo_document(session, storage, embeddings, "remote-work")

    body = fake_demo_client.post("/query", json={"query": scenario.question_text}).json()

    assert set(body) == {
        "query",
        "normalized_query",
        "filters",
        "candidates",
        "counts",
        "query_id",
        "answer_id",
        "answer",
        "abstained",
        "citations",
        "citation_valid",
        "grounded",
        "grounding_detail",
        "model",
        "prompt_version",
    }
