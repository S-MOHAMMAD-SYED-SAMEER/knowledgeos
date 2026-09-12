"""The upload endpoints, end to end against a real database and real disk.

The assertions that matter most are the negative ones: an uploaded version is
never active, nothing supersedes anything on upload, and no error body ever
tells the caller where a file went.
"""

import io
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.models import Document, DocumentVersion, IngestionJob

PDF = b"%PDF-1.7\nnot parsed by this milestone\n"
TEXT = b"How do I request production database access?\n"


def _upload(client: TestClient, content: bytes = PDF, name: str = "access-sop.pdf", **fields):
    return client.post(
        "/documents",
        files={"file": (name, io.BytesIO(content), "application/pdf")},
        data=fields or None,
    )


@pytest.fixture
def uploaded(client: TestClient, migrated_engine: Engine):
    response = _upload(client)
    assert response.status_code == 201, response.text
    return response.json()


# --- a successful upload ----------------------------------------------------


def test_an_upload_is_created(client: TestClient, migrated_engine: Engine) -> None:
    assert _upload(client).status_code == 201


def test_the_response_carries_the_document_version_and_job(uploaded) -> None:
    assert set(uploaded) == {"document", "version", "job"}
    assert uploaded["document"]["title"] == "access-sop.pdf"
    assert uploaded["document"]["source_type"] == "upload"


def test_the_first_version_is_number_one_and_a_draft(uploaded) -> None:
    assert uploaded["version"]["version_number"] == 1
    assert uploaded["version"]["status"] == "draft"


def test_the_job_is_queued(uploaded) -> None:
    """There is no runner yet, so queued is where it stays."""
    assert uploaded["job"]["status"] == "queued"
    assert uploaded["job"]["attempts"] == 0
    assert uploaded["job"]["started_at"] is None


def test_checksum_and_page_count_are_null(uploaded) -> None:
    assert uploaded["version"]["checksum"] is None
    assert uploaded["version"]["page_count"] is None


def test_all_three_rows_are_in_the_database(
    uploaded, migrated_engine: Engine
) -> None:
    with Session(migrated_engine) as session:
        assert session.execute(select(Document)).scalars().all()
        assert session.execute(select(DocumentVersion)).scalars().all()
        assert session.execute(select(IngestionJob)).scalars().all()


def test_the_file_is_on_disk_with_the_bytes_that_were_sent(
    uploaded, migrated_engine: Engine
) -> None:
    from app.storage import get_storage

    with Session(migrated_engine) as session:
        version = session.execute(select(DocumentVersion)).scalar_one()

    with get_storage().open(version.storage_path) as handle:
        assert handle.read() == PDF


def test_the_storage_path_is_relative_and_built_from_identifiers(
    uploaded, migrated_engine: Engine
) -> None:
    with Session(migrated_engine) as session:
        version = session.execute(select(DocumentVersion)).scalar_one()

    assert version.storage_path.startswith("documents/")
    assert version.original_filename not in version.storage_path


def test_the_storage_path_names_the_real_document_and_version(
    uploaded, migrated_engine: Engine
) -> None:
    """A file filed under an id nothing resolves to is a file nobody can find
    from the row that owns it."""
    with Session(migrated_engine) as session:
        version = session.execute(select(DocumentVersion)).scalar_one()

    assert version.storage_path == (
        f"documents/{version.document_id}/{version.id}.pdf"
    )
    assert str(uploaded["document"]["id"]) in version.storage_path


def test_optional_metadata_is_recorded(
    client: TestClient, migrated_engine: Engine
) -> None:
    response = _upload(
        client, title="Access SOP", department="Engineering", category="policy"
    )
    body = response.json()["document"]

    assert body["title"] == "Access SOP"
    assert body["department"] == "Engineering"
    assert body["category"] == "policy"


def test_two_uploads_create_two_independent_documents(
    client: TestClient, migrated_engine: Engine
) -> None:
    first = _upload(client).json()
    second = _upload(client).json()

    assert first["document"]["id"] != second["document"]["id"]
    assert second["version"]["version_number"] == 1


def test_a_text_upload_works_too(client: TestClient, migrated_engine: Engine) -> None:
    response = client.post(
        "/documents",
        files={"file": ("notes.md", io.BytesIO(TEXT), "text/markdown")},
    )

    assert response.status_code == 201


# --- refusals ---------------------------------------------------------------


def test_an_unsupported_extension_is_415(
    client: TestClient, migrated_engine: Engine
) -> None:
    response = client.post(
        "/documents",
        files={"file": ("payload.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
    )

    assert response.status_code == 415


def test_a_file_that_is_not_what_it_claims_is_415(
    client: TestClient, migrated_engine: Engine
) -> None:
    response = _upload(client, content=b"\x7fELF not a pdf")

    assert response.status_code == 415


def test_an_empty_file_is_400(client: TestClient, migrated_engine: Engine) -> None:
    assert _upload(client, content=b"").status_code == 400


def test_an_oversized_file_is_413(
    client: TestClient, migrated_engine: Engine
) -> None:
    """Overridden through the dependency, which is how the route reads it."""
    from app.config import get_settings

    small = get_settings().model_copy(update={"max_upload_bytes": 16})
    client.app.dependency_overrides[get_settings] = lambda: small
    try:
        assert _upload(client, content=PDF + b"x" * 500).status_code == 413
    finally:
        client.app.dependency_overrides.clear()


def test_an_oversized_upload_leaves_nothing_behind(
    client: TestClient, migrated_engine: Engine
) -> None:
    from app.config import get_settings
    from app.storage import get_storage

    small = get_settings().model_copy(update={"max_upload_bytes": 16})
    client.app.dependency_overrides[get_settings] = lambda: small
    try:
        _upload(client, content=PDF + b"x" * 500)
    finally:
        client.app.dependency_overrides.clear()

    with Session(migrated_engine) as session:
        assert session.execute(select(Document)).scalars().all() == []
    assert list(get_storage().root.rglob("*.pdf")) == []


def test_a_missing_file_part_is_422(
    client: TestClient, migrated_engine: Engine
) -> None:
    assert client.post("/documents", data={"title": "no file"}).status_code == 422


def test_a_refused_upload_writes_no_rows(
    client: TestClient, migrated_engine: Engine
) -> None:
    _upload(client, content=b"not a pdf at all", name="x.pdf")

    with Session(migrated_engine) as session:
        assert session.execute(select(Document)).scalars().all() == []
        assert session.execute(select(IngestionJob)).scalars().all() == []


def test_a_refused_upload_writes_no_file(
    client: TestClient, migrated_engine: Engine
) -> None:
    from app.storage import get_storage

    _upload(client, content=b"not a pdf at all", name="x.pdf")
    root = get_storage().root

    assert list(root.rglob("*.pdf")) == []


# --- a second version -------------------------------------------------------


def test_a_second_version_is_created(
    client: TestClient, uploaded, migrated_engine: Engine
) -> None:
    document_id = uploaded["document"]["id"]

    response = client.post(
        f"/documents/{document_id}/versions",
        files={"file": ("access-sop-v2.pdf", io.BytesIO(PDF), "application/pdf")},
    )

    assert response.status_code == 201
    assert response.json()["version"]["version_number"] == 2
    assert response.json()["version"]["status"] == "draft"


def test_uploading_a_version_does_not_supersede_the_active_one(
    client: TestClient, uploaded, migrated_engine: Engine
) -> None:
    """The invariant, tested through the API: v1 has been indexed in this
    scenario and v2 has not, so v1 keeps answering questions."""
    from app.ingestion import service

    document_id = uploaded["document"]["id"]
    with Session(migrated_engine) as session:
        first = session.get(DocumentVersion, uuid.UUID(uploaded["version"]["id"]))
        service.promote_version(session, first)
        session.commit()

    client.post(
        f"/documents/{document_id}/versions",
        files={"file": ("v2.pdf", io.BytesIO(PDF), "application/pdf")},
    )

    with Session(migrated_engine) as session:
        statuses = {
            row.version_number: row.status
            for row in session.execute(select(DocumentVersion)).scalars()
        }

    assert statuses == {1: "active", 2: "draft"}


def test_a_version_for_an_unknown_document_is_404(
    client: TestClient, migrated_engine: Engine
) -> None:
    response = client.post(
        f"/documents/{uuid.uuid4()}/versions",
        files={"file": ("v2.pdf", io.BytesIO(PDF), "application/pdf")},
    )

    assert response.status_code == 404


def test_a_refused_version_upload_stores_nothing(
    client: TestClient, uploaded, migrated_engine: Engine
) -> None:
    from app.storage import get_storage

    document_id = uploaded["document"]["id"]
    before = len(list(get_storage().root.rglob("*")))

    client.post(
        f"/documents/{document_id}/versions",
        files={"file": ("v2.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
    )

    assert len(list(get_storage().root.rglob("*"))) == before


# --- reading back -----------------------------------------------------------


def test_the_document_list_is_returned(
    client: TestClient, uploaded, migrated_engine: Engine
) -> None:
    body = client.get("/documents").json()

    assert len(body["documents"]) == 1
    assert body["limit"] == 50
    assert body["offset"] == 0


def test_the_list_paginates(client: TestClient, migrated_engine: Engine) -> None:
    for _ in range(3):
        _upload(client)

    assert len(client.get("/documents?limit=2").json()["documents"]) == 2
    assert len(client.get("/documents?limit=2&offset=2").json()["documents"]) == 1


def test_the_page_size_is_bounded(client: TestClient, migrated_engine: Engine) -> None:
    """An unbounded list is a slow endpoint waiting to happen."""
    assert client.get("/documents?limit=100000").json()["limit"] == 100


def test_the_detail_includes_the_version_history(
    client: TestClient, uploaded, migrated_engine: Engine
) -> None:
    document_id = uploaded["document"]["id"]
    client.post(
        f"/documents/{document_id}/versions",
        files={"file": ("v2.pdf", io.BytesIO(PDF), "application/pdf")},
    )

    body = client.get(f"/documents/{document_id}").json()

    assert [v["version_number"] for v in body["versions"]] == [1, 2]


def test_an_unknown_document_is_404(
    client: TestClient, migrated_engine: Engine
) -> None:
    assert client.get(f"/documents/{uuid.uuid4()}").status_code == 404


def test_a_job_can_be_read_back(
    client: TestClient, uploaded, migrated_engine: Engine
) -> None:
    body = client.get(f"/ingestion/{uploaded['job']['id']}").json()

    assert body["status"] == "queued"
    assert body["document_version_id"] == uploaded["version"]["id"]


def test_an_unknown_job_is_404(client: TestClient, migrated_engine: Engine) -> None:
    assert client.get(f"/ingestion/{uuid.uuid4()}").status_code == 404


# --- what responses must never contain --------------------------------------


def test_no_response_exposes_a_storage_path(uploaded) -> None:
    """`storage_path` is where this system keeps a file. Nobody else's business."""
    assert "storage_path" not in str(uploaded)


@pytest.mark.parametrize(
    ("content", "name"),
    [(b"", "a.pdf"), (b"MZ", "a.exe"), (b"not a pdf", "a.pdf")],
)
def test_no_error_body_exposes_the_filesystem(
    client: TestClient, migrated_engine: Engine, content: bytes, name: str
) -> None:
    from app.storage import get_storage

    body = _upload(client, content=content, name=name).text
    root = str(get_storage().root)

    assert root not in body
    assert "Traceback" not in body
    assert "/home/" not in body
    assert "var/documents" not in body
