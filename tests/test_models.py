"""Documents and versions, measured against a really-migrated PostgreSQL.

The constraints are the point of this milestone. Each one below is asserted by
provoking it, because a constraint that is only declared is a comment.
"""

from datetime import date

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Document, DocumentVersion, VersionStatus

from .conftest import version


# --- documents -------------------------------------------------------------


def test_a_document_is_saved_with_its_defaults(session: Session) -> None:
    row = Document(title="Employee Handbook")
    session.add(row)
    session.commit()

    assert row.id is not None
    assert row.source_type == "upload"
    assert row.tags == []
    assert row.created_at is not None
    assert row.updated_at is not None


def test_tags_hold_a_list_of_strings(session: Session) -> None:
    row = Document(title="Security Policy", tags=["security", "annual-review"])
    session.add(row)
    session.commit()
    session.expire_all()

    assert session.get(Document, row.id).tags == ["security", "annual-review"]


def test_tags_cannot_be_an_object(session: Session) -> None:
    """A flat set of labels, so the containment filter has one shape to read."""
    session.add(Document(title="Remote Work Policy", tags={"team": "eng"}))

    with pytest.raises(IntegrityError):
        session.commit()


def test_an_unknown_source_type_is_refused(session: Session) -> None:
    """Provenance, and upload is the only one v1 has."""
    session.add(Document(title="Incident Response", source_type="notion"))

    with pytest.raises(IntegrityError):
        session.commit()


def test_department_and_category_are_optional(session: Session) -> None:
    row = Document(title="Glossary")
    session.add(row)
    session.commit()

    assert row.department is None
    assert row.category is None


# --- versions --------------------------------------------------------------


def test_a_version_is_saved_with_its_defaults(session: Session, document) -> None:
    row = version(document.id)
    session.add(row)
    session.commit()

    assert row.status == VersionStatus.DRAFT
    assert row.checksum is None
    assert row.page_count is None
    assert row.effective_date is None
    assert row.created_at is not None


def test_a_version_belongs_to_its_document(session: Session, document) -> None:
    session.add(version(document.id))
    session.commit()
    session.expire_all()

    stored = session.get(Document, document.id)
    assert len(stored.versions) == 1
    assert stored.versions[0].document.id == document.id


def test_versions_come_back_in_order(session: Session, document) -> None:
    session.add_all(
        [
            version(document.id, 2, status=VersionStatus.ACTIVE),
            version(document.id, 1, status=VersionStatus.SUPERSEDED),
        ]
    )
    session.commit()
    session.expire_all()

    numbers = [v.version_number for v in session.get(Document, document.id).versions]
    assert numbers == [1, 2]


@pytest.mark.parametrize(
    "status", [VersionStatus.DRAFT, VersionStatus.ACTIVE, VersionStatus.SUPERSEDED]
)
def test_each_valid_status_is_accepted(session: Session, document, status) -> None:
    session.add(version(document.id, 1, status=status))
    session.commit()

    assert session.execute(select(DocumentVersion)).scalar_one().status == status


def test_a_status_outside_the_three_is_refused(session: Session, document) -> None:
    """The check constraint, not the Python enum, is what enforces this."""
    session.add(version(document.id, 1, status="published"))

    with pytest.raises(IntegrityError):
        session.commit()


def test_a_version_number_must_be_positive(session: Session, document) -> None:
    session.add(version(document.id, 0))

    with pytest.raises(IntegrityError):
        session.commit()


def test_a_version_number_is_unique_within_a_document(
    session: Session, document
) -> None:
    session.add_all([version(document.id, 1), version(document.id, 1)])

    with pytest.raises(IntegrityError):
        session.commit()


def test_two_documents_may_each_have_a_version_one(session: Session) -> None:
    first = Document(title="Handbook")
    second = Document(title="Security Policy")
    session.add_all([first, second])
    session.commit()

    session.add_all([version(first.id, 1), version(second.id, 1)])
    session.commit()

    assert len(session.execute(select(DocumentVersion)).scalars().all()) == 2


def test_a_checksum_must_look_like_a_sha256(session: Session, document) -> None:
    session.add(version(document.id, 1, checksum="not-a-hash"))

    with pytest.raises(IntegrityError):
        session.commit()


def test_a_real_sha256_is_accepted(session: Session, document) -> None:
    digest = "a" * 64
    session.add(version(document.id, 1, checksum=digest))
    session.commit()

    assert session.execute(select(DocumentVersion)).scalar_one().checksum == digest


def test_a_page_count_of_zero_is_refused(session: Session, document) -> None:
    session.add(version(document.id, 1, page_count=0))

    with pytest.raises(IntegrityError):
        session.commit()


def test_an_effective_date_is_a_calendar_day(session: Session, document) -> None:
    session.add(version(document.id, 1, effective_date=date(2026, 4, 1)))
    session.commit()

    assert session.execute(
        select(DocumentVersion)
    ).scalar_one().effective_date == date(2026, 4, 1)


# --- the rule the whole versioning model rests on --------------------------


def test_one_document_may_have_one_active_version(
    session: Session, document
) -> None:
    session.add(version(document.id, 1, status=VersionStatus.ACTIVE))
    session.commit()

    assert session.execute(select(DocumentVersion)).scalar_one().status == "active"


def test_a_second_active_version_is_refused(session: Session, document) -> None:
    """Enforced by PostgreSQL, not by whichever code path promotes a version:
    the alternative is two uploads racing and a knowledge base that answers
    from two versions at once."""
    session.add(version(document.id, 1, status=VersionStatus.ACTIVE))
    session.commit()

    session.add(version(document.id, 2, status=VersionStatus.ACTIVE))
    with pytest.raises(IntegrityError):
        session.commit()


def test_drafts_and_superseded_versions_may_sit_beside_the_active_one(
    session: Session, document
) -> None:
    """Partial, so history stays queryable for audit."""
    session.add_all(
        [
            version(document.id, 1, status=VersionStatus.SUPERSEDED),
            version(document.id, 2, status=VersionStatus.SUPERSEDED),
            version(document.id, 3, status=VersionStatus.ACTIVE),
            version(document.id, 4, status=VersionStatus.DRAFT),
        ]
    )
    session.commit()

    assert len(session.execute(select(DocumentVersion)).scalars().all()) == 4


def test_each_document_may_have_its_own_active_version(session: Session) -> None:
    first = Document(title="Handbook")
    second = Document(title="Security Policy")
    session.add_all([first, second])
    session.commit()

    session.add_all(
        [
            version(first.id, 1, status=VersionStatus.ACTIVE),
            version(second.id, 1, status=VersionStatus.ACTIVE),
        ]
    )
    session.commit()

    assert len(session.execute(select(DocumentVersion)).scalars().all()) == 2


# --- deletion ---------------------------------------------------------------


def test_deleting_a_document_deletes_its_versions(session: Session, document) -> None:
    """A version without its document is not a record of anything."""
    session.add_all([version(document.id, 1), version(document.id, 2)])
    session.commit()

    session.delete(document)
    session.commit()

    assert session.execute(select(DocumentVersion)).scalars().all() == []
    assert session.execute(select(Document)).scalars().all() == []


def test_a_version_cannot_reference_a_document_that_is_not_there(
    session: Session
) -> None:
    import uuid

    session.add(version(uuid.uuid4(), 1))

    with pytest.raises(IntegrityError):
        session.commit()
