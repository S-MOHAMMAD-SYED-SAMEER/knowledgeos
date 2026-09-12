"""What survives when a step fails.

The ordering under test: the file is written first and the rows last, so a
database failure can take the file back out again. The reverse ordering would
leave a row pointing at a file that was never written — a version this system
believes it has, which breaks the moment milestone 3 opens it.

One residue is accepted and not engineered away: a crash between the storage
write and the database commit leaves an orphan file. The last test here
documents that window rather than pretending it is closed.
"""

import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.models import Document, DocumentVersion, IngestionJob
from app.storage import StorageError, get_storage

PDF = b"%PDF-1.7\nnot parsed here\n"


def _upload(client: TestClient, name: str = "a.pdf", content: bytes = PDF):
    return client.post(
        "/documents",
        files={"file": (name, io.BytesIO(content), "application/pdf")},
    )


def _rows(engine: Engine) -> tuple[int, int, int]:
    with Session(engine) as session:
        return (
            len(session.execute(select(Document)).scalars().all()),
            len(session.execute(select(DocumentVersion)).scalars().all()),
            len(session.execute(select(IngestionJob)).scalars().all()),
        )


# --- storage failure --------------------------------------------------------


def test_a_storage_failure_writes_no_rows(
    client: TestClient, migrated_engine: Engine, monkeypatch
) -> None:
    def explode(self, key, source):
        raise StorageError("the disk went away")

    monkeypatch.setattr("app.storage.local.LocalStorage.write", explode)

    assert _upload(client).status_code == 500
    assert _rows(migrated_engine) == (0, 0, 0)


def test_a_storage_failure_says_nothing_about_the_filesystem(
    client: TestClient, migrated_engine: Engine, monkeypatch
) -> None:
    def explode(self, key, source):
        raise StorageError("could not store 'documents/abc/def.pdf'")

    monkeypatch.setattr("app.storage.local.LocalStorage.write", explode)
    body = _upload(client).text

    assert "documents/abc" not in body
    assert "StorageError" not in body
    assert "Traceback" not in body


# --- database failure, and the compensating delete --------------------------


def test_a_database_failure_removes_the_file_it_had_already_written(
    client: TestClient, migrated_engine: Engine, monkeypatch
) -> None:
    """The compensation that makes 'storage first' safe."""
    deleted: list[str] = []
    original_delete = get_storage().__class__.delete

    def record(self, key):
        deleted.append(key)
        return original_delete(self, key)

    def explode(*args, **kwargs):
        raise RuntimeError("the database went away")

    monkeypatch.setattr("app.storage.local.LocalStorage.delete", record)
    monkeypatch.setattr("app.api.documents.service.create_document_with_version", explode)

    assert _upload(client).status_code == 500
    assert len(deleted) == 1
    assert get_storage().exists(deleted[0]) is False


def test_a_database_failure_writes_no_rows(
    client: TestClient, migrated_engine: Engine, monkeypatch
) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError("the database went away")

    monkeypatch.setattr("app.api.documents.service.create_document_with_version", explode)
    _upload(client)

    assert _rows(migrated_engine) == (0, 0, 0)


def test_a_job_creation_failure_rolls_back_the_document_and_version(
    client: TestClient, migrated_engine: Engine, monkeypatch
) -> None:
    """All three rows are one transaction: a job that cannot be written must
    not leave a version nothing will ever ingest."""
    def explode(*args, **kwargs):
        raise RuntimeError("the job could not be queued")

    monkeypatch.setattr("app.ingestion.service._queue_job", explode)

    assert _upload(client).status_code == 500
    assert _rows(migrated_engine) == (0, 0, 0)


def test_a_job_creation_failure_also_removes_the_file(
    client: TestClient, migrated_engine: Engine, monkeypatch
) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError("the job could not be queued")

    monkeypatch.setattr("app.ingestion.service._queue_job", explode)
    _upload(client)

    assert list(get_storage().root.rglob("*.pdf")) == []


def test_a_version_upload_failure_removes_its_file_too(
    client: TestClient, migrated_engine: Engine, monkeypatch
) -> None:
    first = _upload(client).json()
    before = {path for path in get_storage().root.rglob("*.pdf")}

    def explode(*args, **kwargs):
        raise RuntimeError("the database went away")

    monkeypatch.setattr("app.api.documents.service.add_version", explode)
    client.post(
        f"/documents/{first['document']['id']}/versions",
        files={"file": ("v2.pdf", io.BytesIO(PDF), "application/pdf")},
    )

    assert {path for path in get_storage().root.rglob("*.pdf")} == before


# --- validation failure -----------------------------------------------------


def test_a_validation_failure_touches_neither_disk_nor_database(
    client: TestClient, migrated_engine: Engine
) -> None:
    """Refused before anything is committed anywhere."""
    _upload(client, name="a.pdf", content=b"not a pdf")

    assert _rows(migrated_engine) == (0, 0, 0)
    assert list(get_storage().root.rglob("*")) == []


# --- the window that is not closed ------------------------------------------


def test_a_crash_between_storage_and_commit_leaves_an_orphan_file(
    client: TestClient, migrated_engine: Engine, monkeypatch
) -> None:
    """An accepted limitation, asserted so it stays a known one.

    The compensating delete is what normally removes this file. A process that
    dies outright never runs it, and milestone 2 ships no reaper — an
    unreferenced file costs disk and nothing else, which is the cheaper of the
    two failures.
    """
    class _Died(BaseException):
        """Not an Exception: nothing in the request path catches this."""

    def die(*args, **kwargs):
        raise _Died()

    monkeypatch.setattr("app.api.documents.service.create_document_with_version", die)

    with pytest.raises(_Died):
        _upload(client)

    assert _rows(migrated_engine) == (0, 0, 0)
    assert len(list(get_storage().root.rglob("*.pdf"))) == 1
