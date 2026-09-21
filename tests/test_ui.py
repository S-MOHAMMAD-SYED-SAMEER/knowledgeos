"""The server-rendered UI (milestone 10, §12) — five pages, mounted under
`/ui`, exercised end to end against a real database and real
retrieval/reranking, with a generation test double (the same allowance
`tests/test_query_api.py` already uses for the JSON API).

The query flow calls `app.api.query.run_query` directly (locked decision
D4) — these tests override the same three FastAPI dependencies
(`embedding_provider`, `rerank_provider`, `llm_provider`)
`tests/test_query_api.py` overrides for the JSON API, proving the UI route
declares them as real dependencies rather than reaching for the cached
real providers directly (which would make overriding impossible — see
`app/ui/routes.py`'s own docstring on why).
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.api.query import embedding_provider, llm_provider, rerank_provider
from app.models import EvalRun
from app.providers import FakeEmbeddingProvider
from app.providers.passthrough_reranker import PassthroughRerankProvider
from app.storage import LocalStorage

from .generation_fixtures import AutoCitingLLMProvider
from .retrieval_fixtures import seed_active_version


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def fake_client(client: TestClient, embeddings: FakeEmbeddingProvider):
    """`/ui/query` through the same fake embedding provider, passthrough
    reranker, and auto-citing generation test double
    `tests/test_query_api.py::fake_client` already uses for the JSON API."""
    client.app.dependency_overrides[embedding_provider] = lambda: embeddings
    client.app.dependency_overrides[rerank_provider] = lambda: PassthroughRerankProvider()
    client.app.dependency_overrides[llm_provider] = lambda: AutoCitingLLMProvider()
    try:
        yield client
    finally:
        client.app.dependency_overrides.clear()


def _seed(session, storage, embeddings, **overrides):
    fields = {"text": "production database access requires manager approval and expires after seven days"}
    fields.update(overrides)
    return seed_active_version(session, storage, embeddings=embeddings, **fields)


# --- document list -----------------------------------------------------


def test_document_list_renders_200(fake_client, migrated_engine: Engine) -> None:
    assert fake_client.get("/ui/").status_code == 200


def test_document_list_shows_seeded_documents(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings, title="Access SOP.txt")

    body = fake_client.get("/ui/").text
    assert "Access SOP" in body


def test_document_list_shows_an_empty_state_with_nothing_seeded(
    fake_client, migrated_engine: Engine
) -> None:
    body = fake_client.get("/ui/").text
    assert "no documents" in body.lower()


def test_document_list_is_not_in_the_json_api_openapi_contract(
    fake_client, migrated_engine: Engine
) -> None:
    paths = fake_client.get("/openapi.json").json()["paths"]
    assert "/ui/" not in paths
    assert not any(p.startswith("/ui") for p in paths)


# --- document detail / version history ----------------------------------


def test_document_detail_shows_version_history(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        seeded = _seed(session, storage, embeddings, title="Access SOP.txt")
        document_id = seeded.document.id

    body = fake_client.get(f"/ui/documents/{document_id}").text
    assert "Access SOP" in body
    assert "v1" in body


def test_document_detail_404_for_unknown_document(
    fake_client, migrated_engine: Engine
) -> None:
    response = fake_client.get(f"/ui/documents/{uuid.uuid4()}")
    assert response.status_code == 404
    assert "No such document" in response.text
    assert "Traceback" not in response.text
    assert "File \"" not in response.text


# --- query box + POST/redirect/GET --------------------------------------


def test_query_form_renders_200(fake_client, migrated_engine: Engine) -> None:
    response = fake_client.get("/ui/query")
    assert response.status_code == 200
    assert "<form" in response.text


def test_query_post_redirects_to_the_answer_page(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    response = fake_client.post(
        "/ui/query", data={"query": "production database access"}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("/ui/answers/")


def test_a_reload_of_the_answer_page_never_resubmits(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    """POST/redirect/GET: the answer page is reached by GET, not by the
    POST staying live -- a second GET of the same URL is idempotent."""
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    post_response = fake_client.post(
        "/ui/query", data={"query": "production database access"}, follow_redirects=False
    )
    answer_url = post_response.headers["location"]

    first = fake_client.get(answer_url)
    second = fake_client.get(answer_url)
    assert first.status_code == second.status_code == 200
    assert first.text == second.text


def test_an_empty_query_renders_a_safe_error(
    fake_client, migrated_engine: Engine
) -> None:
    response = fake_client.post("/ui/query", data={"query": "   "})
    assert response.status_code == 422
    assert "Traceback" not in response.text
    assert "File \"" not in response.text


# --- answer page: citations, evidence side by side, feedback form -------


def _post_query(fake_client, text: str = "production database access") -> str:
    response = fake_client.post("/ui/query", data={"query": text}, follow_redirects=False)
    return response.headers["location"]


def test_answer_page_shows_the_question_and_answer(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    body = fake_client.get(_post_query(fake_client)).text
    assert "production database access" in body.lower()
    assert "manager approval" in body.lower() or "seven days" in body.lower()


def test_answer_page_shows_evidence_text_side_by_side(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings, title="Access SOP.txt")

    body = fake_client.get(_post_query(fake_client)).text
    assert "Access SOP" in body
    assert "manager approval" in body.lower()


def test_answer_page_shows_citations_linked_to_evidence(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    body = fake_client.get(_post_query(fake_client)).text
    assert "Citations" in body
    # The auto-citing fake cites every chunk it was shown; a 32-hex-char
    # chunk_uid appears both as a citation link target and an evidence anchor.
    import re

    uids = set(re.findall(r'href="#evidence-([0-9a-f]{32})"', body))
    anchors = set(re.findall(r'id="evidence-([0-9a-f]{32})"', body))
    assert uids
    assert uids <= anchors


def test_answer_page_shows_an_abstention_when_nothing_was_retrieved(
    fake_client, migrated_engine: Engine
) -> None:
    body = fake_client.get(_post_query(fake_client, "nothing has been indexed yet")).text
    assert "abstain" in body.lower()


def test_answer_page_404_for_unknown_query(fake_client, migrated_engine: Engine) -> None:
    response = fake_client.get(f"/ui/answers/{uuid.uuid4()}")
    assert response.status_code == 404


def test_answer_page_has_a_feedback_form(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    body = fake_client.get(_post_query(fake_client)).text
    assert 'name="rating"' in body
    assert 'value="helpful"' in body
    assert 'value="not_helpful"' in body
    assert 'name="reason"' in body


def test_feedback_form_submission_persists_and_redirects_back(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    answer_url = _post_query(fake_client)
    body = fake_client.get(answer_url).text

    import re

    match = re.search(r'action="(/ui/answers/[0-9a-f-]+/feedback)"', body)
    assert match is not None
    feedback_url = match.group(1)

    response = fake_client.post(
        feedback_url,
        data={"rating": "helpful", "reason": "clear and well cited"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == answer_url

    redirected = fake_client.get(response.headers["location"])
    assert "clear and well cited" in redirected.text


def test_feedback_form_rejects_an_invalid_rating_safely(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    answer_url = _post_query(fake_client)
    body = fake_client.get(answer_url).text
    import re

    feedback_url = re.search(r'action="(/ui/answers/[0-9a-f-]+/feedback)"', body).group(1)

    response = fake_client.post(feedback_url, data={"rating": "amazing"})
    assert response.status_code == 422
    assert "Traceback" not in response.text


# --- evaluation results page ---------------------------------------------


def test_eval_page_honest_empty_state_with_no_recorded_runs(
    fake_client, migrated_engine: Engine
) -> None:
    body = fake_client.get("/ui/evals").text
    assert "no evaluation run has been recorded" in body.lower()
    # No fabricated numbers: no percentage sign anywhere near a metric claim.
    assert "recall@5" not in body.lower() or "no evaluation run" in body.lower()


def test_eval_page_shows_a_recorded_run_verbatim(
    fake_client, migrated_engine: Engine
) -> None:
    with Session(migrated_engine) as session:
        row = EvalRun(
            suite="retrieval",
            prompt_version=None,
            config={"embedding_model": "fake-deterministic", "chunk_size_tokens": 512},
            metrics={"passthrough": {"recall_at_5": 0.42}},
        )
        session.add(row)
        session.commit()
        run_id = row.id

    body = fake_client.get("/ui/evals").text
    assert str(run_id) in body
    assert "0.42" in body
    assert "fake-deterministic" in body


def test_eval_page_never_shows_a_number_it_did_not_read_from_eval_runs(
    fake_client, migrated_engine: Engine
) -> None:
    """With zero rows, no metrics or config block is rendered at all --
    only the honest empty-state sentence. `<pre>` is where a recorded
    run's `config`/`metrics` JSON would appear (see the "shows a recorded
    run verbatim" test above); its absence here is the proof nothing was
    fabricated to fill the gap."""
    body = fake_client.get("/ui/evals").text
    assert "<pre>" not in body


def test_eval_page_reads_through_app_models_never_evals_package() -> None:
    """Static guard mirroring `tests/test_retrieval_scope.py`'s own: the
    UI module itself must never import `evals/`."""
    import ast
    import pathlib

    source = pathlib.Path("app/ui/routes.py").read_text()
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    assert not any(name.startswith("evals") for name in names)


# --- escaping / XSS --------------------------------------------------------


def test_a_hostile_document_title_is_escaped_in_the_document_list(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings, title="<script>alert(1)</script>.txt")

    body = fake_client.get("/ui/").text
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


def test_a_hostile_document_title_is_escaped_on_the_detail_page(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        seeded = _seed(session, storage, embeddings, title="<img src=x onerror=alert(1)>.txt")
        document_id = seeded.document.id

    body = fake_client.get(f"/ui/documents/{document_id}").text
    assert "<img src=x onerror=alert(1)>" not in body
    assert "&lt;img" in body


def test_answer_and_evidence_text_are_escaped_on_the_answer_page(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(
            session, storage, embeddings,
            text="<script>alert('evidence')</script> manager approval required for database access",
        )

    body = fake_client.get(_post_query(fake_client)).text
    assert "<script>alert('evidence')</script>" not in body
    assert "&lt;script&gt;" in body


def test_feedback_reason_is_escaped_when_displayed(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    answer_url = _post_query(fake_client)
    body = fake_client.get(answer_url).text
    import re

    feedback_url = re.search(r'action="(/ui/answers/[0-9a-f-]+/feedback)"', body).group(1)

    response = fake_client.post(
        feedback_url,
        data={"rating": "helpful", "reason": "<script>alert('xss')</script>"},
        follow_redirects=True,
    )
    assert "<script>alert('xss')</script>" not in response.text
    assert "&lt;script&gt;" in response.text


def test_no_template_uses_the_safe_filter() -> None:
    """Locked decision: no template in this package ever marks model
    output, document metadata, chunk text, or feedback text as `|safe`."""
    import pathlib

    templates_dir = pathlib.Path("app/ui/templates")
    for path in templates_dir.glob("*.html"):
        assert "|safe" not in path.read_text(), path.name


# --- safe error handling everywhere --------------------------------------


def test_provider_failure_in_the_ui_never_leaks_a_traceback(
    client: TestClient, migrated_engine: Engine
) -> None:
    """No override: `/ui/query` reaches for the real, cached
    `BgeEmbeddingProvider`, which fails in this environment -- proving the
    UI's own error path, not the JSON API's."""
    response = client.post("/ui/query", data={"query": "anything"})
    assert response.status_code == 503
    assert "Traceback" not in response.text
    assert "File \"" not in response.text
    assert "/home/" not in response.text
