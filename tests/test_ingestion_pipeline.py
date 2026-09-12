"""Parsing and chunking a real stored file, into a real database.

The end state is the point of this milestone: a successful job stops at
`indexing`, because parsing and chunking are done and indexing is milestone
4's. Nothing here marks a version ready, and nothing promotes one to active.
"""

import io
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.ingestion import service
from app.ingestion.pipeline import StageFailed, parse_and_normalize, run_job
from app.models import Chunk, DocumentVersion, IngestionJob, JobStatus
from app.storage import LocalStorage

from .fixtures import docx_bytes, pdf_bytes, plain_zip_bytes, words


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, chunk_size_tokens=50, chunk_overlap_tokens=5)


def _ingest(session: Session, storage: LocalStorage, content: bytes, filename: str):
    """An uploaded version with its file really in storage, as M2 leaves it."""
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    extension = "." + filename.rsplit(".", 1)[-1]
    key = service.storage_key(document_id, version_id, extension)
    storage.write(key, io.BytesIO(content))

    result = service.create_document_with_version(
        session,
        document_id=document_id,
        title=filename,
        department=None,
        category=None,
        tags=[],
        original_filename=filename,
        storage_path=key,
        effective_date=None,
        version_id=version_id,
    )
    session.commit()
    return result


# --- a successful run -------------------------------------------------------


def test_a_text_document_becomes_chunks(session, storage, settings) -> None:
    result = _ingest(session, storage, words(200).encode(), "notes.txt")

    run_job(session, result.job, storage, settings)
    session.commit()

    chunks = session.execute(select(Chunk)).scalars().all()
    assert len(chunks) > 1
    assert [chunk.sequence for chunk in chunks] == list(range(len(chunks)))


def test_a_successful_job_stops_at_indexing(session, storage, settings) -> None:
    """The milestone 3 to milestone 4 handoff, asserted."""
    result = _ingest(session, storage, words(200).encode(), "notes.txt")

    run_job(session, result.job, storage, settings)
    session.commit()

    assert result.job.status == JobStatus.INDEXING


def test_a_successful_job_does_not_reach_ready(session, storage, settings) -> None:
    """`ready` must not be claimed before indexing has happened."""
    result = _ingest(session, storage, words(200).encode(), "notes.txt")

    run_job(session, result.job, storage, settings)
    session.commit()

    assert result.job.status != JobStatus.READY


def test_the_version_stays_a_draft(session, storage, settings) -> None:
    """active implies indexed, and nothing has been indexed."""
    result = _ingest(session, storage, words(200).encode(), "notes.txt")

    run_job(session, result.job, storage, settings)
    session.commit()
    session.expire_all()

    assert session.get(DocumentVersion, result.version.id).status == "draft"


def test_the_checksum_is_of_the_normalized_text(session, storage, settings) -> None:
    from hashlib import sha256

    from app.parsing import normalize

    body = "Line one.\r\n\r\n\r\nLine two.  \n"
    result = _ingest(session, storage, body.encode(), "notes.txt")

    run_job(session, result.job, storage, settings)
    session.commit()

    expected = sha256(normalize(body).encode("utf-8")).hexdigest()
    assert result.version.checksum == expected
    assert len(result.version.checksum) == 64


def test_the_checksum_is_not_of_the_raw_bytes(session, storage, settings) -> None:
    from hashlib import sha256

    body = b"  Text with trailing space   \r\n"
    result = _ingest(session, storage, body, "notes.txt")

    run_job(session, result.job, storage, settings)
    session.commit()

    assert result.version.checksum != sha256(body).hexdigest()


def test_chunk_offsets_index_the_normalized_document(
    session, storage, settings
) -> None:
    body = words(300)
    result = _ingest(session, storage, body.encode(), "notes.txt")

    document = parse_and_normalize(result.version, storage)
    run_job(session, result.job, storage, settings)
    session.commit()

    for chunk in session.execute(select(Chunk)).scalars():
        assert document.text[chunk.char_start : chunk.char_end] == chunk.text


# --- per-format behaviour ---------------------------------------------------


def test_a_pdf_records_its_real_page_count(session, storage, settings) -> None:
    content = pdf_bytes([words(30), words(30, prefix="b"), words(30, prefix="c")])
    result = _ingest(session, storage, content, "policy.pdf")

    run_job(session, result.job, storage, settings)
    session.commit()

    assert result.version.page_count == 3


def test_pdf_chunks_carry_a_page(session, storage, settings) -> None:
    content = pdf_bytes([words(30), words(30, prefix="b")])
    result = _ingest(session, storage, content, "policy.pdf")

    run_job(session, result.job, storage, settings)
    session.commit()

    pages = {chunk.page for chunk in session.execute(select(Chunk)).scalars()}
    assert pages <= {1, 2}
    assert None not in pages


@pytest.mark.parametrize(
    ("content", "filename"),
    [
        (words(100).encode(), "notes.txt"),
        (b"# Title\n\n" + words(100).encode(), "notes.md"),
        (docx_bytes([("Title", "Heading1"), (words(100), None)]), "handbook.docx"),
    ],
)
def test_page_count_is_null_for_formats_without_pages(
    session, storage, settings, content, filename
) -> None:
    """Never invented: not 1, not zero."""
    result = _ingest(session, storage, content, filename)

    run_job(session, result.job, storage, settings)
    session.commit()

    assert result.version.page_count is None


def test_markdown_chunks_carry_their_section(session, storage, settings) -> None:
    body = "# Access SOP\n\n" + words(60) + "\n\n## Appendix\n\n" + words(60, prefix="z")
    result = _ingest(session, storage, body.encode(), "sop.md")

    run_job(session, result.job, storage, settings)
    session.commit()

    sections = {chunk.section for chunk in session.execute(select(Chunk)).scalars()}
    assert "Access SOP" in sections


def test_a_renamed_zip_fails_rather_than_being_read_as_text(
    session, storage, settings
) -> None:
    result = _ingest(session, storage, plain_zip_bytes(), "handbook.docx")

    with pytest.raises(StageFailed) as raised:
        run_job(session, result.job, storage, settings)

    assert raised.value.stage == JobStatus.PARSING


# --- failure ----------------------------------------------------------------


def test_an_empty_document_fails(session, storage, settings) -> None:
    """A version with no text can never answer anything."""
    result = _ingest(session, storage, b"   \n\n   ", "notes.txt")

    with pytest.raises(StageFailed, match="no extractable text"):
        run_job(session, result.job, storage, settings)


def test_a_missing_stored_file_fails_without_naming_the_path(
    session, storage, settings
) -> None:
    result = _ingest(session, storage, words(50).encode(), "notes.txt")
    storage.delete(result.version.storage_path)

    with pytest.raises(StageFailed) as raised:
        run_job(session, result.job, storage, settings)

    assert result.version.storage_path not in raised.value.detail
    assert str(storage.root) not in raised.value.detail


def test_a_stage_error_quotes_no_document_content(
    session, storage, settings
) -> None:
    secret = "the approval chain is the on-call DBA"
    result = _ingest(session, storage, plain_zip_bytes(), "handbook.docx")

    with pytest.raises(StageFailed) as raised:
        run_job(session, result.job, storage, settings)

    assert secret not in raised.value.detail


# --- idempotency ------------------------------------------------------------


def test_rerunning_produces_identical_chunks(session, storage, settings) -> None:
    """The point of a content-derived identifier."""
    result = _ingest(session, storage, words(300).encode(), "notes.txt")

    run_job(session, result.job, storage, settings)
    session.commit()
    first = [
        (chunk.sequence, chunk.chunk_uid)
        for chunk in session.execute(select(Chunk).order_by(Chunk.sequence)).scalars()
    ]

    run_job(session, result.job, storage, settings)
    session.commit()
    second = [
        (chunk.sequence, chunk.chunk_uid)
        for chunk in session.execute(select(Chunk).order_by(Chunk.sequence)).scalars()
    ]

    assert first == second


def test_rerunning_creates_no_duplicates(session, storage, settings) -> None:
    result = _ingest(session, storage, words(300).encode(), "notes.txt")

    run_job(session, result.job, storage, settings)
    session.commit()
    count = len(session.execute(select(Chunk)).scalars().all())

    run_job(session, result.job, storage, settings)
    session.commit()

    assert len(session.execute(select(Chunk)).scalars().all()) == count


def test_rerunning_replaces_only_that_versions_chunks(
    session, storage, settings
) -> None:
    first = _ingest(session, storage, words(200).encode(), "one.txt")
    second = _ingest(session, storage, words(200, prefix="z").encode(), "two.txt")

    run_job(session, first.job, storage, settings)
    run_job(session, second.job, storage, settings)
    session.commit()

    untouched = {
        chunk.chunk_uid
        for chunk in session.execute(
            select(Chunk).where(Chunk.document_version_id == second.version.id)
        ).scalars()
    }

    run_job(session, first.job, storage, settings)
    session.commit()

    after = {
        chunk.chunk_uid
        for chunk in session.execute(
            select(Chunk).where(Chunk.document_version_id == second.version.id)
        ).scalars()
    }
    assert after == untouched


def test_two_versions_keep_their_own_chunks(session, storage, settings) -> None:
    first = _ingest(session, storage, words(200).encode(), "one.txt")
    second = _ingest(session, storage, words(200).encode(), "two.txt")

    run_job(session, first.job, storage, settings)
    run_job(session, second.job, storage, settings)
    session.commit()

    owners = {
        chunk.document_version_id for chunk in session.execute(select(Chunk)).scalars()
    }
    assert owners == {first.version.id, second.version.id}


def test_identical_text_in_two_versions_yields_different_uids(
    session, storage, settings
) -> None:
    """The version is part of the identifier, so there is no collision."""
    body = words(100).encode()
    first = _ingest(session, storage, body, "one.txt")
    second = _ingest(session, storage, body, "two.txt")

    run_job(session, first.job, storage, settings)
    run_job(session, second.job, storage, settings)
    session.commit()

    uids = [chunk.chunk_uid for chunk in session.execute(select(Chunk)).scalars()]
    assert len(uids) == len(set(uids))


# --- constraints ------------------------------------------------------------


def test_a_duplicate_sequence_is_refused_by_the_database(
    session, storage, settings
) -> None:
    from sqlalchemy.exc import IntegrityError

    result = _ingest(session, storage, words(100).encode(), "notes.txt")
    run_job(session, result.job, storage, settings)
    session.commit()

    session.add(
        Chunk(
            document_version_id=result.version.id,
            sequence=0,
            chunk_uid="f" * 32,
            text="duplicate position",
            page=None,
            section=None,
            char_start=0,
            char_end=1,
            token_count=1,
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_a_duplicate_uid_is_refused_by_the_database(
    session, storage, settings
) -> None:
    from sqlalchemy.exc import IntegrityError

    result = _ingest(session, storage, words(100).encode(), "notes.txt")
    run_job(session, result.job, storage, settings)
    session.commit()

    existing = session.execute(select(Chunk)).scalars().first()
    session.add(
        Chunk(
            document_version_id=result.version.id,
            sequence=999,
            chunk_uid=existing.chunk_uid,
            text="different position, same identity",
            page=None,
            section=None,
            char_start=0,
            char_end=1,
            token_count=1,
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_deleting_a_version_deletes_its_chunks(session, storage, settings) -> None:
    result = _ingest(session, storage, words(100).encode(), "notes.txt")
    run_job(session, result.job, storage, settings)
    session.commit()

    session.delete(session.get(DocumentVersion, result.version.id))
    session.commit()

    assert session.execute(select(Chunk)).scalars().all() == []
