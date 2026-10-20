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

import re
import uuid

import pytest
from fastapi.testclient import TestClient
from markupsafe import escape
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


def test_eval_page_reads_the_committed_results_and_shows_counts_with_their_samples(
    fake_client, migrated_engine: Engine
) -> None:
    body = fake_client.get("/ui/evals").text
    lowered = body.lower()
    # The committed files are shown, each figure as a count over a named denominator.
    assert "Answers (real model)" in body
    assert "Grounded answers" in body
    assert re.search(r"<strong>\d+ of \d+</strong>", body)
    assert "Abstention recall" in body and "Abstention precision" in body
    # And what they are scored against, and that the threshold is in-sample.
    assert "gold" in lowered
    assert "in-sample" in lowered
    # The database list is honest when nothing was recorded there.
    assert "Runs recorded in this database" in body
    assert "None. A run you start here" in body


def test_eval_page_headline_numbers_come_from_the_results_file_not_the_template(
    fake_client, migrated_engine: Engine, tmp_path, monkeypatch
) -> None:
    import json

    import app.ui.results_view as results_view

    (tmp_path / "2099-01-01-answers-eval-x.json").write_text(
        json.dumps(
            {
                "suite": "answers",
                "gold_label": "a label",
                "config": {
                    "llm_model": "some-model",
                    "prompt_version": "knowledge_answer_v9",
                    "gold_version": "gold-v9",
                    "abstention_threshold": 1.5,
                    "question_count": 10,
                    "dev_question_count": 6,
                    "test_question_count": 4,
                    "model_requests": {"total": 12},
                },
                "metrics": {
                    "total_question_count": 10,
                    "verification": {"asked_count": 8, "unverified": 3, "verified_after_retry": 1},
                    "abstention": {"true_positives": 2, "false_positives": 1, "positive_count": 3},
                    "outcome_counts": {"pre_llm_abstain": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(results_view, "RESULTS_DIR", tmp_path)

    body = fake_client.get("/ui/evals").text

    assert "knowledge_answer_v9" in body and "gold-v9" in body
    assert "<strong>5 of 8</strong>" in body  # grounded: 8 reached the model, 3 rejected
    assert "<strong>3 of 8</strong>" in body  # rejected by the citation check
    assert "<strong>2 of 3</strong>" in body  # abstention recall
    assert "<strong>2 of 3</strong>" in body  # abstention precision: 2 of (2 + 1)
    assert "1.5000" in body


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

    source = pathlib.Path("app/ui/routes.py").read_text(encoding="utf-8")
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
        assert "|safe" not in path.read_text(encoding="utf-8"), path.name


# --- safe error handling everywhere --------------------------------------


@pytest.mark.usefixtures("models_not_in_cache")
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


# --- the "could not verify" reply -----------------------------------------------------


def test_an_unverified_answer_is_labelled_as_such_and_not_as_answered(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    from app.generation.generator import UNVERIFIED_ANSWER_TEXT
    from app.providers.fake_llm import FakeLLMProvider

    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    client.app.dependency_overrides[embedding_provider] = lambda: embeddings
    client.app.dependency_overrides[rerank_provider] = lambda: PassthroughRerankProvider()
    client.app.dependency_overrides[llm_provider] = lambda: FakeLLMProvider(["not json"] * 2)
    try:
        response = client.post("/ui/query", data={"query": "production database access"})
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 200  # the redirect was followed to the answer page
    assert str(escape(UNVERIFIED_ANSWER_TEXT)) in response.text  # the template escapes the apostrophe
    assert "Could not verify this answer" in response.text
    assert "badge answered" not in response.text
    assert "Citations verified" not in response.text


def test_a_verified_answer_after_a_retry_looks_like_any_other_answer(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    from sqlalchemy import select

    from app.models import Chunk
    from app.providers.fake_llm import FakeLLMProvider

    from .generation_fixtures import valid_json

    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)
        uids = session.execute(select(Chunk.chunk_uid)).scalars().all()
    good = valid_json(" ".join(f"Fact [{u}]." for u in uids), uids)

    client.app.dependency_overrides[embedding_provider] = lambda: embeddings
    client.app.dependency_overrides[rerank_provider] = lambda: PassthroughRerankProvider()
    client.app.dependency_overrides[llm_provider] = lambda: FakeLLMProvider(["not json", good])
    try:
        response = client.post("/ui/query", data={"query": "production database access"})
    finally:
        client.app.dependency_overrides.clear()

    assert "badge answered" in response.text
    assert "Could not verify this answer" not in response.text


# --- errors are pages, not JSON ------------------------------------------------------------


def test_a_malformed_id_in_a_ui_path_is_a_styled_404_not_raw_json(fake_client, migrated_engine: Engine) -> None:
    for path in ("/ui/documents/not-a-uuid", "/ui/answers/not-a-uuid"):
        response = fake_client.get(path)

        assert response.status_code == 404, path
        assert response.headers["content-type"].startswith("text/html"), path
        assert "No such page" in response.text
        assert "uuid" not in response.text.lower()  # the parser's own message is not shown
        assert "not-a-uuid" not in response.text


def test_a_malformed_form_post_is_a_styled_422(fake_client, migrated_engine: Engine) -> None:
    response = fake_client.post("/ui/query", data={})  # the required field is missing

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("text/html")
    assert "That request was not valid" in response.text


def test_the_json_api_keeps_its_own_validation_errors(fake_client, migrated_engine: Engine) -> None:
    response = fake_client.get("/documents/not-a-uuid")

    assert response.headers["content-type"].startswith("application/json")
    assert response.status_code == 422


def test_a_refusal_raised_by_a_dependency_is_a_styled_page_in_the_ui(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    """The demo's mutation guard raises from a route dependency, which used to reach the browser as
    FastAPI's raw JSON 403."""
    from fastapi import HTTPException

    from app.api.demo_guard import require_mutation_allowed

    def _refuse() -> None:
        raise HTTPException(status_code=403, detail="This deployment accepts no feedback.")

    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)
    fake_client.app.dependency_overrides[require_mutation_allowed] = _refuse

    response = fake_client.post(
        "/ui/answers/00000000-0000-0000-0000-000000000000/feedback", data={"rating": "helpful"}
    )

    assert response.status_code == 403
    assert response.headers["content-type"].startswith("text/html")
    assert "This deployment accepts no feedback." in response.text


# --- the Ask page and the answer page ---------------------------------------------------------


def test_the_ask_page_lists_example_questions(fake_client) -> None:
    from app.ui.routes import DEFAULT_EXAMPLE_QUESTIONS

    body = fake_client.get("/ui/query").text

    for question in DEFAULT_EXAMPLE_QUESTIONS:
        assert str(escape(question)) in body


def test_a_deployment_can_replace_the_example_questions(fake_client) -> None:
    from app.ui.routes import example_questions

    fake_client.app.dependency_overrides[example_questions] = lambda: ("Only this one?",)

    body = fake_client.get("/ui/query").text

    assert "Only this one?" in body
    assert "work remotely" not in body


def test_evidence_is_collapsed_by_default_and_a_citation_link_opens_it(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    body = fake_client.post("/ui/query", data={"query": "production database access"}).text

    # Nothing is open on arrival: the page stays short however many passages were retrieved.
    assert "<details open>" not in body
    assert re.search(r"\d+ passages retrieved, \d+ sent to the model", body)
    # The text is still in the page (collapsed, not removed), and a link to a passage opens it.
    assert "production database access requires manager approval" in body
    assert "node.open = true" in body


def test_citations_show_the_document_and_a_snippet_and_keep_the_id_in_a_tooltip(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    body = fake_client.post("/ui/query", data={"query": "production database access"}).text

    assert "<h2>Sources</h2>" in body
    # A numbered source with its document title; the 32-character id is only ever in a title attribute
    # or an anchor, never printed as text a person has to read.
    assert re.search(r'<li id="cite-1" value="1" title="chunk [0-9a-f]{32}">', body)
    assert "snippet" in body
    visible = re.sub(r"<[^>]*>", " ", body)  # the text between tags
    assert not re.search(r"[0-9a-f]{32}", visible)


def test_cited_turns_markers_into_numbered_links_and_escapes_everything_else() -> None:
    from app.ui.routes import citation_numbers, cited

    uid_a, uid_b = "a" * 32, "b" * 32
    text = f"Two days <b>[{uid_a}] and a week [{uid_b}] [{uid_a}] [not-a-marker]"
    numbers = citation_numbers(text, [uid_b, uid_a])

    assert numbers == {uid_a: 1, uid_b: 2}  # in order of first appearance in the text
    html = str(cited(text, numbers))
    assert f'href="#evidence-{uid_a}"' in html and f'title="chunk {uid_a}">[1]</a>' in html
    assert ">[2]</a>" in html
    assert "&lt;b&gt;" in html  # the answer's own markup is text, not markup
    assert "[not-a-marker]" in html


def test_a_generation_failure_shows_the_model_is_busy_with_a_retry_button_not_error_503(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    from app.api.query import llm_provider
    from app.providers.llm import LLMError

    class Failing:
        model_name = "failing"

        def complete(self, *, system, user, max_tokens, json_schema=None):
            raise LLMError("generation failed (HTTP 503)")

    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    fake_client.app.dependency_overrides[llm_provider] = lambda: Failing()
    response = fake_client.post("/ui/query", data={"query": "production database access"})

    assert response.status_code == 503  # still a 503 on the wire
    body = response.text
    assert "The model is busy" in body and "try again" in body
    assert "Error 503" not in body
    assert ">Retry</button>" in body
    assert "production database access" in body  # the question is kept for the retry


def test_the_badge_says_citations_were_verified_not_that_the_answer_is_true(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    body = fake_client.post("/ui/query", data={"query": "production database access"}).text

    assert "Citations verified" in body
    assert "Grounded in evidence" not in body
    assert "does not check that the passage supports the sentence" in body


def test_the_snippet_is_the_sentence_that_matches_the_answer_not_the_start_of_the_document() -> None:
    from app.ui.routes import _snippet

    passage = (
        "# Policy\n\nThis policy describes the company and who owns it. "
        "Standard production access lasts three days and then expires. "
        "Contact the security team with questions."
    )
    assert _snippet(passage, "Standard access lasts three days.") == "Standard production access lasts three days and then expires."
    # Nothing shared: the first sentence, with the heading left out.
    assert _snippet(passage, "unrelated words entirely").startswith("This policy describes")
    long = "word " * 80 + "end."
    assert len(_snippet(long, "")) <= 152 and _snippet(long, "").endswith("…")
