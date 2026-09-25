"""M4 finding: demo mode did not actually block the JSON API's own
mutation routes -- only the M3 visitor UI never rendered a form for them.
`app.api.demo_guard.require_mutation_allowed` is the fix; this file proves
it blocks every route that writes, and touches nothing else.

The guard runs before each route's own body (a FastAPI dependency, or an
explicit call at the top of `app/ui/routes.py::ui_submit_feedback`), so a
403 here for a nonexistent document/answer id proves demo mode refused the
request outright -- not that the id happened not to exist.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings


@pytest.fixture
def demo_client(settings_env, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from app.main import create_app

    monkeypatch.setenv("KNOWLEDGEOS_DEMO_MODE", "true")
    get_settings.cache_clear()
    return TestClient(create_app())


RANDOM_ID = uuid.uuid4()


# --- demo mode refuses every mutation route -----------------------------


def test_demo_mode_refuses_document_upload(
    demo_client: TestClient, migrated_engine
) -> None:
    response = demo_client.post(
        "/documents",
        files={"file": ("x.txt", b"hello", "text/plain")},
        data={"title": "x"},
    )
    assert response.status_code == 403
    assert "demo mode" in response.json()["detail"].lower()


def test_demo_mode_refuses_a_new_document_version(
    demo_client: TestClient, migrated_engine
) -> None:
    response = demo_client.post(
        f"/documents/{RANDOM_ID}/versions",
        files={"file": ("x.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 403


def test_demo_mode_refuses_reindex(demo_client: TestClient, migrated_engine) -> None:
    response = demo_client.post(f"/documents/{RANDOM_ID}/reindex")
    assert response.status_code == 403


def test_demo_mode_refuses_json_api_feedback(
    demo_client: TestClient, migrated_engine
) -> None:
    response = demo_client.post(
        f"/answers/{RANDOM_ID}/feedback", json={"rating": "helpful"}
    )
    assert response.status_code == 403


def test_demo_mode_refuses_ui_feedback(demo_client: TestClient, migrated_engine) -> None:
    response = demo_client.post(
        f"/ui/answers/{RANDOM_ID}/feedback", data={"rating": "helpful"}
    )
    # The UI renders its own HTML error page (`error.html`), not a JSON
    # body, but `_error()` still sets the real status code from the
    # exception -- 403, matching the JSON API's own response for the same
    # refusal.
    assert response.status_code == 403
    assert "demo mode" in response.text.lower()


def test_demo_mode_still_allows_reading_documents(
    demo_client: TestClient, migrated_engine
) -> None:
    assert demo_client.get("/documents").status_code == 200


# --- normal (non-demo) mode is unaffected: regression check -------------


def test_normal_mode_document_upload_still_succeeds(
    client: TestClient, migrated_engine
) -> None:
    response = client.post(
        "/documents",
        files={"file": ("x.txt", b"hello", "text/plain")},
        data={"title": "x"},
    )
    assert response.status_code == 201


def test_normal_mode_reindex_still_reaches_its_own_404(
    client: TestClient, migrated_engine
) -> None:
    """Proves the guard adds nothing when demo mode is off: a nonexistent
    document reaches the route's own 404, not a 403 from the guard."""
    response = client.post(f"/documents/{RANDOM_ID}/reindex")
    assert response.status_code == 404


def test_normal_mode_json_api_feedback_reaches_its_own_404(
    client: TestClient, migrated_engine
) -> None:
    response = client.post(
        f"/answers/{RANDOM_ID}/feedback", json={"rating": "helpful"}
    )
    assert response.status_code == 404
