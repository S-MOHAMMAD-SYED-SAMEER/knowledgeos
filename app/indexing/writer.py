"""Making a version's chunks findable.

The stage the specification calls *index writes*, and it is the last one: when
it commits, the job is `ready` and the version is `active`.

**Where the transaction starts matters.** Embeddings are generated *before* the
transaction opens, not inside it. The specification requires index **writes**
to be transactional and says nothing about inference — and holding a row lock
and a pooled connection open across model inference for every chunk of a
document would be a long transaction for no benefit. So: generate, then open
the transaction, write, flip, commit.

**What commits together.** The embeddings, the job's move to `ready`, and the
version's promotion to `active` are one transaction. A failure leaves none of
it, which is what the specification means by a failed job never leaving a
version in a partial state. The `tsv` column needs no writing at all —
PostgreSQL maintains it from `text` — so there is no second thing to keep in
step.

**Promotion happens here and nowhere else.** A version becomes `active` when
its content can answer a question, which is precisely now. The previous active
version becomes `superseded` in the same transaction, so default retrieval
never sees two and never sees none.
"""

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ingestion.service import promote_version
from app.models import Chunk, DocumentVersion
from app.providers.embeddings import EmbeddingError, EmbeddingProvider

logger = logging.getLogger(__name__)


class IndexingFailed(RuntimeError):
    """The indexing stage could not complete.

    Carries a reason safe to store in `stage_error`: it names what failed
    structurally and never the text that was being embedded.
    """


def chunks_to_embed(session: Session, version_id) -> list[Chunk]:
    """This version's chunks, in order.

    Every chunk, not only those missing an embedding: re-indexing rewrites
    the lot, so a model change or a re-run produces a consistent set rather
    than a mixture of old and new vectors.
    """
    return list(
        session.execute(
            select(Chunk)
            .where(Chunk.document_version_id == version_id)
            .order_by(Chunk.sequence)
        ).scalars()
    )


def texts_of(chunks: list[Chunk]) -> list[str]:
    """The chunk texts, detached from the session.

    Read while a transaction is open; the strings outlive it. What is handed
    to the provider is plain text, so inference cannot touch the database by
    lazily refreshing an expired attribute.
    """
    return [chunk.text for chunk in chunks]


def generate_embeddings(
    texts: list[str], provider: EmbeddingProvider
) -> list[list[float]]:
    """Vectors for these texts. Takes strings, not rows, and takes no session.

    It cannot open a transaction because it is given nothing that could: that
    is the point of the signature, and it is what keeps inference outside the
    database's transaction whatever a caller does.

    The width is checked by the provider before anything reaches the
    database, so a model returning the wrong shape fails here with a clear
    reason rather than as an opaque column error.
    """
    if not texts:
        raise IndexingFailed("the version has no chunks to index")

    try:
        return provider.embed(texts)
    except EmbeddingError as exc:
        raise IndexingFailed(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - a provider must never escape
        raise IndexingFailed(
            f"the embedding provider failed ({type(exc).__name__})"
        ) from exc


def write_index(
    session: Session,
    version: DocumentVersion,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> None:
    """Attach the vectors to their chunks and promote the version.

    Everything here is in the caller's transaction. `tsv` is not written:
    PostgreSQL computes it from `text`, so it is already correct and writing
    to a generated column is an error.
    """
    if len(vectors) != len(chunks):
        raise IndexingFailed(
            f"{len(vectors)} vectors were produced for {len(chunks)} chunks"
        )

    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.embedding = vector
    session.flush()

    # Last, and in the same transaction: a version is active only once its
    # content is really there.
    promote_version(session, version)


__all__ = [
    "IndexingFailed",
    "chunks_to_embed",
    "generate_embeddings",
    "texts_of",
    "write_index",
]
