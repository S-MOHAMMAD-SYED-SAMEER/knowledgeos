"""Lexical retrieval: PostgreSQL full-text search over `chunks.tsv`.

This is **PostgreSQL full-text search, not BM25** — the specification is
explicit that Postgres FTS must never be called BM25 in code, comments or
docs, because `ts_rank_cd` is a different ranking model with different
behaviour. Nothing in this module or its docstrings uses that name.

Matters for exactly the case vector search blurs: an error code, a SKU, a
policy title, an acronym — an exact term a paraphrase-tolerant embedding
would happily match to something else entirely.

`websearch_to_tsquery`, not `to_tsquery`. Arbitrary user text — stray
quotes, `&`, `|`, `!`, `(` — is a syntax error to `to_tsquery` and is quietly
tolerated by `websearch_to_tsquery`, which is built to parse exactly the kind
of free text a search box receives.
"""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Chunk, Document, DocumentVersion
from app.retrieval.filters import RetrievalFilters, predicates
from app.retrieval.vector import Candidate, evidence_of

# The specification's number: "Retrieve top 50 from each". See vector.py for
# why this is fixed rather than configurable.
CANDIDATE_LIMIT = 50

# The configuration the specification names. A generated column already
# fixes this for storage (`chunks.tsv`); the query side has to name it too,
# or a mismatched configuration would silently rank stems that do not agree
# with what was indexed.
TS_CONFIG = "english"


def search(
    session: Session,
    *,
    normalized_text: str,
    filters: RetrievalFilters,
    limit: int = CANDIDATE_LIMIT,
) -> list[Candidate]:
    """The top `limit` chunks by `ts_rank_cd`, filtered and ordered.

    A query that is only stopwords (`"the of and"`) produces an empty
    `tsquery`. PostgreSQL logs a notice and the `@@` match returns nothing —
    that is correct behaviour, not a failure this function needs to handle
    specially: it simply returns an empty list.
    """
    query = func.websearch_to_tsquery(TS_CONFIG, normalized_text)
    rank = func.ts_rank_cd(Chunk.tsv, query).label("rank")

    stmt = (
        select(Chunk, DocumentVersion, Document, rank)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(Chunk.tsv.op("@@")(query))
        .where(*predicates(filters))
        # Descending: higher `ts_rank_cd` is a better match. `chunk_uid`
        # breaks ties — and real ties happen, including at a rank of exactly
        # zero, so this is load-bearing rather than defensive.
        .order_by(rank.desc(), Chunk.chunk_uid.asc())
        .limit(limit)
    )

    rows = session.execute(stmt).all()
    return [
        Candidate(
            evidence=evidence_of(chunk, version, document),
            rank=index + 1,
            score=float(row_rank),
        )
        for index, (chunk, version, document, row_rank) in enumerate(rows)
    ]


__all__ = ["CANDIDATE_LIMIT", "TS_CONFIG", "search"]
