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
this module. **No template in this package uses `|safe` anywhere.** The one place a
hyperlink is built from model text, the `cited` filter, escapes everything that is not a
32-hex marker and builds the anchor itself with `Markup.format`. A
citation link is built only from an already-validated `chunk_uid` (32 lower
-case hex characters, `app/generation/citations.py`'s own validated shape)
used as an HTML anchor fragment — never from raw model output.

**Errors never carry a traceback, a file path, or a raw provider message.**
`HTTPException.detail` from the reused functions is already the sanitized,
public-facing string the JSON API returns to its own callers; this module
renders it as-is in `error.html` and nothing more.
"""

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
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
from app.models import (
    Answer,
    Chunk,
    Document,
    DocumentVersion,
    EvalRun,
    Feedback,
    Query,
    RetrievedChunk,
)
from app.models.feedback import MAX_REASON_LENGTH, RATINGS
from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider
from app.providers.reranker import RerankProvider
from app.ui.results_view import load_results

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(prefix="/ui", tags=["ui"], include_in_schema=False)

_MARKER = re.compile(r"\[([0-9a-f]{32})\]")


def citation_numbers(answer_text: str, citations: list[str]) -> dict[str, int]:
    """A short number for each cited chunk, in the order the answer first cites it.

    The 32-character id is not something a person reads; the answer shows `[1]`, `[2]` and the id
    stays in the tooltip. Citations the text never marks (the check refuses those, so this is only
    defensive) follow in the order they were declared.
    """
    numbers: dict[str, int] = {}
    for uid in _MARKER.findall(answer_text or ""):
        numbers.setdefault(uid, len(numbers) + 1)
    for uid in citations:
        numbers.setdefault(uid, len(numbers) + 1)
    return numbers


def cited(answer_text: str, numbers: dict[str, int]) -> Markup:
    """The answer text with each `[chunk_uid]` marker shown as a numbered link to its passage.

    Everything that is not a marker is escaped as plain text; a marker becomes a link only if it is
    one of the known 32-hex ids, and the id goes only into the anchor fragment and the tooltip.
    """
    out: list[Markup | str] = []
    position = 0
    for match in _MARKER.finditer(answer_text or ""):
        out.append(escape((answer_text or "")[position : match.start()]))
        uid = match.group(1)
        number = numbers.get(uid)
        if number is None:
            out.append(escape(match.group(0)))
        else:
            out.append(
                Markup('<a class="cite" href="#evidence-{uid}" title="chunk {uid}">[{number}]</a>').format(
                    uid=uid, number=number
                )
            )
        position = match.end()
    out.append(escape((answer_text or "")[position:]))
    return Markup("").join(out)


templates.env.filters["cited"] = cited


def _error(request: Request, exc: HTTPException) -> "object":
    return templates.TemplateResponse(
        request,
        "error.html",
        {"status_code": exc.status_code, "detail": exc.detail},
        status_code=exc.status_code,
    )


def _is_generation_failure(exc: HTTPException) -> bool:
    """The language model failed (busy, rate limited, timed out, unavailable). `run_query` words all
    of these as "the language model ...", and none of them says anything the person can act on
    except "try again"."""
    return exc.status_code in (429, 503) and str(exc.detail).startswith("the language model")


def _busy(request: Request, exc: HTTPException, query_text: str):
    """"The model is busy -- try again", with the question kept and a Retry button, instead of
    "Error 503". Still a 503 on the wire, with `Retry-After` when the service gave one."""
    retry_after = (exc.headers or {}).get("Retry-After")
    return templates.TemplateResponse(
        request,
        "busy.html",
        {"query_text": query_text, "retry_after": retry_after},
        status_code=exc.status_code,
        headers=exc.headers,
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


# What the Ask page suggests. The standalone demo replaces this (see `demo/app.py`) with the
# questions it can actually answer, because its embedder knows only those.
DEFAULT_EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "How many days per week may an employee work remotely?",
    "What is the approval process for production database access?",
    "Within how many minutes must an executive be notified of a severity-one incident?",
    "What is the company's policy on using generative AI tools for writing code?",
)


def example_questions() -> tuple[str, ...]:
    """The example questions the Ask page lists. A dependency, so a deployment can override it."""
    return DEFAULT_EXAMPLE_QUESTIONS


@router.get("/query")
def ui_query_form(
    request: Request,
    examples: Annotated[tuple[str, ...], Depends(example_questions)],
):
    return templates.TemplateResponse(request, "query.html", {"examples": examples})


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
        if _is_generation_failure(exc):
            return _busy(request, exc, query)
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


@dataclass(frozen=True)
class CitationRow:
    """One cited source as the answer page lists it: a number, the document, and a short snippet."""

    number: int
    chunk_uid: str
    title: str
    version_number: int | None
    section: str | None
    snippet: str


SNIPPET_LENGTH = 150


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]{4,}")
_MARKDOWN_MARKS = re.compile(r"\*\*|__|`")


def _snippet(text: str, answer_text: str = "") -> str:
    """One sentence of the passage, in one line, headings left out (a heading only repeats the title).

    The sentence that shares the most words with the answer, so the snippet shows what the citation is
    about rather than the start of the document; the first sentence when nothing is shared. Cut to
    `SNIPPET_LENGTH` characters at a word boundary.
    """
    body = " ".join(
        line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    )
    # The passage is Markdown; the snippet is a quotation shown as plain text, so the emphasis marks go.
    body = _MARKDOWN_MARKS.sub("", body)
    sentences = [s for s in _SENTENCE_BREAK.split(" ".join(body.split())) if s]
    if not sentences:
        return ""
    wanted = set(_WORD.findall(answer_text.lower()))
    best = max(sentences, key=lambda s: len(wanted & set(_WORD.findall(s.lower()))))
    chosen = best if wanted & set(_WORD.findall(best.lower())) else sentences[0]
    return chosen if len(chosen) <= SNIPPET_LENGTH else chosen[:SNIPPET_LENGTH].rsplit(" ", 1)[0] + "…"


def citation_rows(
    numbers: dict[str, int], evidence: list["EvidenceRow"], citations: list[str], answer_text: str = ""
) -> list[CitationRow]:
    by_uid = {row.chunk_uid: row for row in evidence}
    rows = []
    for uid, number in sorted(numbers.items(), key=lambda item: item[1]):
        if uid not in citations:
            continue
        found = by_uid.get(uid)
        rows.append(
            CitationRow(
                number=number,
                chunk_uid=uid,
                title=found.document_title if found else "Cited source",
                version_number=found.version_number if found else None,
                section=found.section if found else None,
                snippet=_snippet(found.text, answer_text) if found else "",
            )
        )
    return rows


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
    answer_citations = list(answer_row.citations) if answer_row is not None and answer_row.citations else []
    numbers = citation_numbers(answer_row.answer_text if answer_row is not None else "", answer_citations)
    citations_view = citation_rows(
        numbers, evidence, answer_citations, answer_row.answer_text if answer_row is not None else ""
    )
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
            "citation_numbers": numbers,
            "citations_view": citations_view,
            "selected_count": sum(1 for row in evidence if row.selected),
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
    """The committed results (`docs/results/*.json`, read by `app.ui.results_view`, which does not
    import `evals/`) with the sample size behind every figure, then any runs recorded in this
    database through `app.models.EvalRun`. Recorded values verbatim -- never a placeholder number."""
    runs = list(
        session.execute(select(EvalRun).order_by(EvalRun.created_at.desc()))
        .scalars()
        .all()
    )
    answers, retrieval = load_results()
    return templates.TemplateResponse(
        request,
        "evals.html",
        {"runs": runs, "answers": answers, "retrieval": retrieval},
    )


def error_page(request: Request, *, status_code: int, detail: str):
    """The styled error page, for a handler that has no `HTTPException` of its own to hand `_error`."""
    return _error(request, HTTPException(status_code=status_code, detail=detail))


__all__ = ["DEFAULT_EXAMPLE_QUESTIONS", "error_page", "example_questions", "router"]
