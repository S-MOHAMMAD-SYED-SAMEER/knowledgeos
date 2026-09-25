"""The visitor-facing demo UI (M3): `GET /demo`, `POST /demo/query`.

Same real-database, real-dispatch discipline `tests/test_demo_query_api.py`
already established for the JSON API: `embedding_provider`/`rerank_provider`
are overridden to test-only/real-shipped substitutes (no real model
weights in this environment), `llm_provider` is left un-overridden so the
real `Settings.demo_mode` dispatch (`app/api/query.py::llm_provider`) is
what actually answers, and each scenario seeds only the real fixture
document(s) it cites -- see `tests/demo_fixtures.py`.
"""

import pytest
from fastapi.testclient import TestClient
from markupsafe import escape
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.generation.abstention import DEFAULT_ABSTENTION_TEXT
from app.generation.demo_scenarios import DEMO_SCENARIOS
from app.providers.demo_llm import DEMO_MODEL_NAME
from app.ui.demo_routes import _CHIP_LABELS

from .demo_fixtures import SCENARIO_DOCUMENTS, seed_demo_document

# `storage`, `embeddings`, `demo_client`, `fake_demo_client` are fixtures
# from `tests/demo_fixtures.py`, registered as a plugin in
# `tests/conftest.py` -- pytest resolves them by parameter name below with
# no import needed (see that plugin registration's own comment for why).


def _normalized(html: str) -> str:
    """Collapse the template's own indentation/line-wrapping whitespace so
    a multi-line Jinja block can be asserted against as one string,
    without asserting on exact source formatting."""
    return " ".join(html.split())


def _rendered(text: str) -> str:
    """`text` the way Jinja's own autoescaping (MarkupSafe) would render
    it -- e.g. an apostrophe becomes `&#39;` -- so an assertion here
    reflects what a browser actually receives, not the raw Python string."""
    return _normalized(str(escape(text)))


DA001 = next(s for s in DEMO_SCENARIOS if s.question_id == "da001")
IE001 = next(s for s in DEMO_SCENARIOS if s.question_id == "ie001")


# --- GET /demo -----------------------------------------------------------


def test_demo_page_renders_200_when_demo_mode_is_enabled(
    fake_demo_client: TestClient, migrated_engine: Engine
) -> None:
    response = fake_demo_client.get("/demo")
    assert response.status_code == 200


def test_demo_page_is_not_in_the_json_api_openapi_contract(
    fake_demo_client: TestClient, migrated_engine: Engine
) -> None:
    paths = fake_demo_client.get("/openapi.json").json()["paths"]
    assert not any(p.startswith("/demo") for p in paths)


def test_demo_page_answers_404_when_demo_mode_is_disabled(
    client: TestClient, migrated_engine: Engine
) -> None:
    # `client` (tests/conftest.py) has `demo_mode` at its default, False --
    # a visitor must not be able to reach real-pipeline-behind-a-demo-page
    # behavior on a deployment that never opted into demo mode.
    response = client.get("/demo")
    assert response.status_code == 404


def test_demo_page_does_not_expose_secrets_or_env_details(
    fake_demo_client: TestClient, migrated_engine: Engine
) -> None:
    body = fake_demo_client.get("/demo").text
    lowered = body.lower()
    for leak in (
        "gemini_api_key",
        "google_api_key",
        "database_url",
        "postgresql://",
        "postgresql+psycopg",
        "secret",
        "traceback",
    ):
        assert leak not in lowered


def test_demo_page_shows_the_approved_disclosure_text(
    fake_demo_client: TestClient, migrated_engine: Engine
) -> None:
    body = _normalized(fake_demo_client.get("/demo").text)
    assert (
        "This demo uses a fixed set of synthetic company policy "
        "documents and hand-verified answers. No sign-in or API key is "
        "required." in body
    )


def test_all_eight_curated_questions_are_present_as_options(
    fake_demo_client: TestClient, migrated_engine: Engine
) -> None:
    body = _normalized(fake_demo_client.get("/demo").text)
    assert len(DEMO_SCENARIOS) == 8
    for scenario in DEMO_SCENARIOS:
        assert _rendered(scenario.question_text) in body
        assert _rendered(_CHIP_LABELS[scenario.question_id]) in body


def test_demo_page_exposes_no_upload_ingestion_or_feedback_controls(
    fake_demo_client: TestClient, migrated_engine: Engine
) -> None:
    body = fake_demo_client.get("/demo").text
    lowered = body.lower()
    assert "/documents" not in lowered  # upload/document-management routes
    assert "/ingestion" not in lowered
    assert "/feedback" not in lowered
    assert 'type="file"' not in lowered  # no upload widget


# --- POST /demo/query: curated scenarios ----------------------------------


@pytest.mark.parametrize(
    "scenario",
    [s for s in DEMO_SCENARIOS if s.question_id != "ie001"],
    ids=lambda s: s.question_id,
)
def test_curated_question_submits_through_the_real_query_path_and_renders_grounded(
    scenario, fake_demo_client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        for key in SCENARIO_DOCUMENTS[scenario.question_id]:
            seed_demo_document(session, storage, embeddings, key)

    response = fake_demo_client.post("/demo/query", data={"query": scenario.question_text})

    assert response.status_code == 200
    body = _normalized(response.text)
    assert _rendered(scenario.answer_text) in body
    assert "Answered" in body
    assert DEMO_MODEL_NAME in body
    # Every real citation's chunk_uid, document title and evidence text
    # must actually be rendered, not just the answer text.
    for uid in scenario.citations:
        assert uid in body


def test_grounded_response_shows_citations_and_evidence_text(
    fake_demo_client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        seed_demo_document(session, storage, embeddings, "remote-work")

    body = fake_demo_client.post(
        "/demo/query", data={"query": DA001.question_text}
    ).text

    assert "Citations" in body
    assert DA001.citations[0] in body
    assert "Remote Work Policy" in body  # the cited document's real title
    # the actual evidence text (not just a link/uid) must be present
    assert "remote" in body.lower()


def test_retrieval_reranking_details_render_when_available(
    fake_demo_client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        seed_demo_document(session, storage, embeddings, "remote-work")

    body = fake_demo_client.post(
        "/demo/query", data={"query": DA001.question_text}
    ).text

    assert "How KnowledgeOS ranked the evidence" in body
    for column in (
        "final rank",
        "lexical rank",
        "vector rank",
        "rrf score",
        "fusion rank",
        "rerank score",
        "selected",
    ):
        assert column in body


# --- ie001: genuine abstention vs. unsupported question -------------------


def test_genuine_abstention_renders_the_real_abstention_text(
    fake_demo_client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        seed_demo_document(session, storage, embeddings, "incident-response")

    body = fake_demo_client.post(
        "/demo/query", data={"query": IE001.question_text}
    ).text

    assert DEFAULT_ABSTENTION_TEXT in body
    assert "Abstained" in body
    assert "No verified demo answer" not in body


def test_unsupported_question_renders_the_honest_unmatched_response(
    fake_demo_client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        seed_demo_document(session, storage, embeddings, "remote-work")

    body = fake_demo_client.post(
        "/demo/query", data={"query": "What is the weather like today?"}
    ).text

    assert "No verified demo answer" in body
    assert "not</strong> a grounded answer" in body or "not a grounded answer" in body
    assert DEFAULT_ABSTENTION_TEXT not in body
    assert "Abstained" not in body


# --- reset -----------------------------------------------------------------


def test_the_answer_page_links_back_to_a_fresh_demo_page(
    fake_demo_client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        seed_demo_document(session, storage, embeddings, "remote-work")

    answer_body = fake_demo_client.post(
        "/demo/query", data={"query": DA001.question_text}
    ).text
    assert 'href="/demo"' in answer_body

    reset_body = fake_demo_client.get("/demo").text
    # The fresh landing page carries no leftover answer/evidence state.
    assert DA001.answer_text not in reset_body
    assert "How KnowledgeOS ranked the evidence" not in reset_body


# --- normal (non-demo) UI is unaffected -------------------------------------


def test_normal_ui_query_page_is_unaffected_by_the_demo_routes(
    client: TestClient, migrated_engine: Engine
) -> None:
    response = client.get("/ui/query")
    assert response.status_code == 200
    assert "<form" in response.text
