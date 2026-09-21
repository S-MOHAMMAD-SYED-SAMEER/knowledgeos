"""Seeding the evaluation fixture corpus, against a real database.

Uses `FakeEmbeddingProvider` throughout, per the specification's allowance
for unit tests — these tests are about idempotency, version lifecycle and
chunk-count verification, none of which needs the real model to prove.
"""

import pytest
from sqlalchemy import select

from app.models import Chunk, Document, DocumentVersion, VersionStatus
from app.providers import FakeEmbeddingProvider
from app.storage import LocalStorage
from evals.retrieval.corpus import (
    MINIMUM_ACTIVE_CHUNKS,
    PINNED_CHUNK_OVERLAP_TOKENS,
    PINNED_CHUNK_SIZE_TOKENS,
    CorpusError,
    ensure_corpus_seeded,
    load_manifest,
    verify_corpus_size,
)


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


# --- the manifest itself ----------------------------------------------


def test_the_real_manifest_loads() -> None:
    manifest = load_manifest()
    assert manifest["chunk_size_tokens"] == PINNED_CHUNK_SIZE_TOKENS
    assert manifest["chunk_overlap_tokens"] == PINNED_CHUNK_OVERLAP_TOKENS


def test_the_real_manifest_has_between_eight_and_twelve_documents() -> None:
    manifest = load_manifest()
    assert 8 <= len(manifest["documents"]) <= 12


def test_exactly_one_document_has_two_versions() -> None:
    """The specification's own words: "a v2 of one of them with a
    materially changed answer"."""
    manifest = load_manifest()
    counts = [len(doc["versions"]) for doc in manifest["documents"]]
    assert counts.count(2) == 1
    assert all(c in (1, 2) for c in counts)


def test_a_manifest_with_the_wrong_chunk_size_is_refused(tmp_path) -> None:
    bad = tmp_path / "manifest.yaml"
    bad.write_text(
        "documents: []\nchunk_size_tokens: 256\nchunk_overlap_tokens: 64\n"
    )
    with pytest.raises(CorpusError, match="chunk_size_tokens"):
        load_manifest(bad)


def test_a_manifest_with_the_wrong_overlap_is_refused(tmp_path) -> None:
    bad = tmp_path / "manifest.yaml"
    bad.write_text(
        "documents: []\nchunk_size_tokens: 512\nchunk_overlap_tokens: 32\n"
    )
    with pytest.raises(CorpusError, match="chunk_overlap_tokens"):
        load_manifest(bad)


def test_a_manifest_missing_the_documents_key_is_refused(tmp_path) -> None:
    bad = tmp_path / "manifest.yaml"
    bad.write_text("chunk_size_tokens: 512\nchunk_overlap_tokens: 64\n")
    with pytest.raises(CorpusError, match="documents"):
        load_manifest(bad)


# --- seeding, against a real database -----------------------------------


def test_seeding_the_real_corpus_produces_more_than_twenty_active_chunks(
    session, storage, embeddings
) -> None:
    result = ensure_corpus_seeded(session, storage, embeddings)

    assert result.total_active_chunks > MINIMUM_ACTIVE_CHUNKS
    verify_corpus_size(result)  # must not raise


def test_seeding_produces_one_active_version_per_document(
    session, storage, embeddings
) -> None:
    result = ensure_corpus_seeded(session, storage, embeddings)

    manifest = load_manifest()
    assert len(result.documents) == len(manifest["documents"])

    for seeded in result.documents:
        active = session.execute(
            select(DocumentVersion).where(
                DocumentVersion.document_id == seeded.document_id,
                DocumentVersion.status == VersionStatus.ACTIVE,
            )
        ).scalars().all()
        assert len(active) == 1
        assert active[0].id == seeded.active_version_id


def test_the_two_version_document_has_v2_active_and_v1_superseded(
    session, storage, embeddings
) -> None:
    ensure_corpus_seeded(session, storage, embeddings)
    manifest = load_manifest()

    two_version_docs = [d for d in manifest["documents"] if len(d["versions"]) == 2]
    assert len(two_version_docs) == 1
    entry = two_version_docs[0]

    import uuid

    v1_id = uuid.UUID(str(entry["versions"][0]["version_id"]))
    v2_id = uuid.UUID(str(entry["versions"][1]["version_id"]))

    v1 = session.get(DocumentVersion, v1_id)
    v2 = session.get(DocumentVersion, v2_id)
    assert v1.status == VersionStatus.SUPERSEDED
    assert v2.status == VersionStatus.ACTIVE


def test_seeding_is_idempotent_no_duplicate_chunks(session, storage, embeddings) -> None:
    first = ensure_corpus_seeded(session, storage, embeddings)
    session.expire_all()
    second = ensure_corpus_seeded(session, storage, embeddings)

    assert second.newly_seeded == 0
    assert second.skipped_existing == first.newly_seeded
    assert second.total_active_chunks == first.total_active_chunks

    # No document_version_id has more chunks than it should: a real
    # duplicate would double every active version's chunk count.
    for seeded in second.documents:
        count = session.execute(
            select(Chunk).where(Chunk.document_version_id == seeded.active_version_id)
        ).scalars().all()
        assert len(count) == seeded.chunk_count


def test_seeded_documents_carry_the_manifests_metadata(
    session, storage, embeddings
) -> None:
    ensure_corpus_seeded(session, storage, embeddings)
    manifest = load_manifest()

    import uuid

    for entry in manifest["documents"]:
        document = session.get(Document, uuid.UUID(str(entry["document_id"])))
        assert document is not None
        assert document.title == entry["title"]
        assert document.department == entry.get("department")
        assert document.category == entry.get("category")
        assert document.tags == (entry.get("tags") or [])


def test_chunk_uids_are_produced_by_the_real_chunker_not_invented(
    session, storage, embeddings
) -> None:
    """Every chunk_uid seeded is exactly what
    `app.chunking.uid.chunk_uid(version_id, sequence, normalized_text)`
    would compute -- proving the corpus never writes a fabricated id."""
    from app.chunking.uid import chunk_uid as compute_chunk_uid

    ensure_corpus_seeded(session, storage, embeddings)

    chunks = session.execute(select(Chunk)).scalars().all()
    assert chunks
    for chunk in chunks:
        expected = compute_chunk_uid(chunk.document_version_id, chunk.sequence, chunk.text)
        assert chunk.chunk_uid == expected
