"""The five UI pages (§12), plus the query POST and the feedback POST that
drive them.

**Reuse, not reimplementation (D4).** `POST /ui/query` calls
`app.api.query.run_query` directly, with the application's own real
dependencies (`app.api.query.embedding_provider`, `rerank_provider`,
`llm_provider`, `app.config.get_settings`) — the identical function
`POST /query` calls, called the identical way. Nothing about retrieval,
reranking, generation, citation validation, grounding, or abstention is
reimplemented here. The document list/detail pages likewise call
`app.api.documents.list_documents`/`get_document` directly rather than
querying a second time.

**Reading persisted evidence (answer page).** `GET /queries/{id}` (the
JSON API) intentionally does not carry chunk text or document/version
metadata — `app/api/query.py`'s own `RetrievedChunkOut` is a ranking
record, not an evidence viewer. The answer page needs both, so this module
reads `retrieved_chunks` joined to `chunks`, `document_versions` and
`documents` directly — a read-only join over already-persisted rows, not a
second retrieval.

**Escaping (D-safety).** Every value interpolated into a template is
plain text through Jinja's default autoescaping (`autoescape=True` below,
matching Starlette's own `select_autoescape()` default) — model answers,
document titles, chunk text, and feedback reasons all pass through
unescaped Python strings and are escaped by the template engine, never by
this module. **No template in this package uses `|safe` anywhere.** A
citation link is built only from an already-validated `chunk_uid` (32 lower
-case hex characters, `app/generation/citations.py`'s own validated shape)
used as an HTML anchor fragment — never from raw model output.

**Errors never carry a traceback, a file path, or a raw provider message.**
`HTTPException.detail` from the reused functions is already the sanitized,
public-facing string the JSON API returns to its own callers; this module
renders it as-is in `error.html` and nothing more.
"""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.demo_guard import require_mutation_allowed
from app.api.documents import DEFAULT_PAGE_SIZE, get_document, list_documents
from app.api.feedback import create_feedback
from app.api.query import embedding_provider, llm_provider, rerank_provider, run_query
from app.api.rate_limit import rate_limit_demo_query
from app.api.schemas import QueryFiltersIn, QueryRequest
from app.config import Settings, get_settings
from app.db.session import get_session
from app.models import Answer, Chunk, Document, DocumentVersion, EvalRun, Feedback, Query, RetrievedChunk
from app.models.feedback import MAX_REASON_LENGTH, RATINGS
from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider
from app.providers.reranker import RerankProvider

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(prefix="/ui", tags=["ui"], include_in_schema=False)


def _error(request: Request, exc: HTTPException) -> "object":
    return templates.TemplateResponse(
        request,
        "error.html",
        {"status_code": exc.status_code, "detail": exc.detail},
        status_code=exc.status_code,
    )


# --- document list / detail -------------------------------------------------


@router.get("/")
def ui_document_list(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    limit: int = DEFAULT_PAGE_SIZE,
    offset: int = 0,
):
    page = list_documents(session, limit=limit, offset=offset)
    return templates.TemplateResponse(
        request, "documents.html", {"page": page}
    )


@router.get("/documents/{document_id}")
def ui_document_detail(
    request: Request,
    document_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
):
    try:
        detail = get_document(document_id, session)
    except HTTPException as exc:
        return _error(request, exc)
    return templates.TemplateResponse(
        request, "document_detail.html", {"detail": detail}
    )


# --- query box (GET the form, POST runs the pipeline) -----------------------


@router.get("/query")
def ui_query_form(request: Request):
    return templates.TemplateResponse(request, "query.html", {})


@router.post("/query", dependencies=[Depends(rate_limit_demo_query)])
def ui_query_submit(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    embeddings: Annotated[EmbeddingProvider, Depends(embedding_provider)],
    reranker: Annotated[RerankProvider, Depends(rerank_provider)],
    llm: Annotated[LLMProvider, Depends(llm_provider)],
    query: Annotated[str, Form()],
):
    """Calls `run_query` directly — the same function `POST /query` calls
    — with providers sourced through the identical `Depends(...)`
    dependencies `run_query` itself declares (`embedding_provider`,
    `rerank_provider`, `llm_provider`, all from `app.api.query`, reused
    unmodified). Declaring them as dependencies here, rather than calling
    the accessor functions directly, is what lets a test override them the
    same way `tests/test_query_api.py` already does for the JSON API —
    calling them directly would silently bypass `app.dependency_overrides`
    and always reach for the real providers, even under test. Then
    redirects (303, so a page reload never resubmits the query) to the
    answer page."""
    try:
        response = run_query(
            QueryRequest(query=query, filters=QueryFiltersIn()),
            session,
            settings,
            embeddings,
            reranker,
            llm,
        )
    except HTTPException as exc:
        return _error(request, exc)

    return RedirectResponse(
        url=f"/ui/answers/{response.query_id}", status_code=303
    )


# --- answer page: question, answer, citations, evidence, feedback ----------


@dataclass(frozen=True)
class EvidenceRow:
    """One retrieved chunk, with the document/version metadata the answer
    page shows beside it -- built from a join `GET /queries/{id}`
    deliberately does not carry (see the module docstring)."""

    chunk_uid: str
    text: str
    page: int | None
    section: str | None
    final_rank: int
    rerank_score: float
    selected: bool
    document_title: str
    document_department: str | None
    document_category: str | None
    version_number: int
    version_status: str


def _citation_label(row: EvidenceRow) -> str:
    """A non-engineering-reviewer-facing label for one cited chunk, built
    only from evidence fields already fetched by `_evidence_for` -- never
    the raw `chunk_uid` itself, which stays reserved for the anchor id/href
    pair (`#evidence-{chunk_uid}`) that links a citation to its evidence."""
    return f"{row.document_title} — {row.section or 'General'}"


def _evidence_for(session: Session, query_id: uuid.UUID) -> list[EvidenceRow]:
    rows = session.execute(
        select(RetrievedChunk, Chunk, DocumentVersion, Document)
        .join(Chunk, RetrievedChunk.chunk_id == Chunk.id)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(RetrievedChunk.query_id == query_id)
        .order_by(RetrievedChunk.final_rank)
    ).all()
    return [
        EvidenceRow(
            chunk_uid=chunk.chunk_uid,
            text=chunk.text,
            page=chunk.page,
            section=chunk.section,
            final_rank=retrieved.final_rank,
            rerank_score=retrieved.rerank_score,
            selected=retrieved.selected,
            document_title=document.title,
            document_department=document.department,
            document_category=document.category,
            version_number=version.version_number,
            version_status=version.status,
        )
        for retrieved, chunk, version, document in rows
    ]


@router.get("/answers/{query_id}")
def ui_answer_page(
    request: Request,
    query_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
):
    query_row = session.get(Query, query_id)
    if query_row is None:
        return _error(
            request, HTTPException(status_code=404, detail="No such query")
        )

    answer_row = session.execute(
        select(Answer).where(Answer.query_id == query_id)
    ).scalar_one_or_none()

    evidence = _evidence_for(session, query_id)
    citation_labels = {row.chunk_uid: _citation_label(row) for row in evidence}
    feedback_rows = []
    if answer_row is not None:
        feedback_rows = list(
            session.execute(
                select(Feedback)
                .where(Feedback.answer_id == answer_row.id)
                .order_by(Feedback.created_at.desc())
            )
            .scalars()
            .all()
        )

    return templates.TemplateResponse(
        request,
        "answer.html",
        {
            "query": query_row,
            "answer": answer_row,
            "evidence": evidence,
            "citation_labels": citation_labels,
            "feedback_rows": feedback_rows,
            "ratings": RATINGS,
            "max_reason_length": MAX_REASON_LENGTH,
        },
    )


@router.post(
    "/answers/{answer_id}/feedback",
    dependencies=[Depends(require_mutation_allowed)],
)
def ui_submit_feedback(
    request: Request,
    answer_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
    rating: Annotated[str, Form()],
    reason: Annotated[str | None, Form()] = None,
):
    # M4: the same guard the JSON API's own `POST /answers/{id}/feedback`
    # uses (`app.api.demo_guard.require_mutation_allowed`), now declared
    # the same way that route declares it -- a route dependency, not an
    # explicit call -- so `demo.app.create_demo_app()`'s override
    # (`app.dependency_overrides[require_mutation_allowed]`) can refuse
    # this route too. The one accepted consequence: a demo-mode refusal
    # here is FastAPI's own default JSON 403 body, not this route's own
    # `error.html` -- unlike this function's other three errors below,
    # which remain hand-raised and still render through `_error()`.

    answer_row = session.get(Answer, answer_id)
    if answer_row is None:
        return _error(
            request, HTTPException(status_code=404, detail="No such answer")
        )
    if rating not in RATINGS:
        return _error(
            request,
            HTTPException(
                status_code=422,
                detail=f"rating must be one of {', '.join(RATINGS)}",
            ),
        )
    if reason is not None and len(reason) > MAX_REASON_LENGTH:
        return _error(
            request,
            HTTPException(
                status_code=422,
                detail=f"reason must be at most {MAX_REASON_LENGTH} characters",
            ),
        )

    create_feedback(
        session, answer_id=answer_id, rating=rating, reason=reason or None
    )
    return RedirectResponse(
        url=f"/ui/answers/{answer_row.query_id}", status_code=303
    )


# --- evaluation results -----------------------------------------------------


@router.get("/evals")
def ui_eval_results(
    request: Request, session: Annotated[Session, Depends(get_session)]
):
    """Reads `eval_runs` through `app.models.EvalRun` only -- never
    `evals/`, preserving the existing guard that no `app` module imports
    the evaluation package. An honest empty state when nothing has been
    recorded (D5); recorded values, verbatim, when it has -- never a
    placeholder number."""
    runs = list(
        session.execute(select(EvalRun).order_by(EvalRun.created_at.desc()))
        .scalars()
        .all()
    )
    return templates.TemplateResponse(request, "evals.html", {"runs": runs})


__all__ = ["router"]
