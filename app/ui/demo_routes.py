"""The visitor-facing demo UI: `GET /demo` (landing) and `POST /demo/query`
(submission) -- a clearly isolated entry point built on top of M2's demo
mode, never a second query pipeline.

**Reuse, not reimplementation.** `POST /demo/query` calls
`app.api.query.run_query` directly, through the identical FastAPI
dependency functions (`embedding_provider`, `rerank_provider`,
`llm_provider`, `get_settings`) the JSON API and
`app/ui/routes.py::ui_query_submit` already use. Demo mode itself is
decided once, application-wide, by `Settings.demo_mode`
(`app/api/query.py::llm_provider`, M2) -- this module never sets it, and
checks it only to refuse to serve when it is off; it does not duplicate
that dispatch logic.

**Why this renders the live `QueryResponse` directly, unlike `/ui/query`'s
redirect-then-read-back pattern.** `/ui/query` redirects to
`/ui/answers/{id}` and reads evidence back from the persisted
`retrieved_chunks` table, which has no `fusion_rank` column
(`app/models/retrieved_chunk.py`) -- only `app.api.schemas.
QueryCandidateOut`, the live response shape, carries it. Rendering the
`QueryResponse` `run_query` already returned, in the same request, is the
only way to show the complete, real record it documents (lexical rank,
vector rank, RRF score, fusion rank, rerank score, final rank, selected)
without inventing or recomputing any of it.

**Demo mode is a deployment-wide setting, not a per-request choice.** A
visitor cannot turn it on by visiting `/demo`; if the running application
was not started with `Settings.demo_mode = True`, both routes below answer
404 rather than silently running the real, credentialed Gemini pipeline
behind a page that claims to be a free, credential-free demo.

**No mutation surface.** Nothing here renders an upload form, an
ingestion control, or a feedback form -- `demo.html`/`demo_answer.html`
render only a question, an answer, its citations, and its retrieval/
reranking record.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.api.query import (
    DEMO_UNSUPPORTED_MODEL_NAME,
    embedding_provider,
    llm_provider,
    rerank_provider,
    run_query,
)
from app.api.schemas import QueryFiltersIn, QueryRequest, QueryResponse
from app.config import Settings, get_settings
from app.db.session import get_session
from app.generation.abstention import DEFAULT_ABSTENTION_TEXT
from app.generation.demo_scenarios import DEMO_SCENARIOS
from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider
from app.providers.reranker import RerankProvider
from app.ui.routes import TEMPLATES_DIR

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(prefix="/demo", tags=["demo"], include_in_schema=False)

# Concise, visitor-facing labels for the 8 approved scenarios. The label is
# UI-only; the text actually submitted is always `scenario.question_text`,
# read from `app.generation.demo_scenarios.DEMO_SCENARIOS` below, never
# retyped here.
_CHIP_LABELS: dict[str, str] = {
    "da001": "Remote work: days per week",
    "md001": "Break-glass access during an incident",
    "cv001": "Current access duration",
    "am001": "Approval process (an ambiguous question)",
    "mf001": "Access approval, filtered to Engineering",
    "cs001": "Executive notification for a severity-1 incident",
    "adv003": "An outdated policy premise (adversarial)",
    "ie001": "AI coding tools policy (expect: no verified answer)",
}

_MISSING_LABELS = {s.question_id for s in DEMO_SCENARIOS} - set(_CHIP_LABELS)
if _MISSING_LABELS:
    raise RuntimeError(
        f"app/ui/demo_routes.py: no chip label for scenario id(s) "
        f"{sorted(_MISSING_LABELS)} -- every DEMO_SCENARIOS entry must "
        f"have one"
    )


def _error(request: Request, exc: HTTPException):
    return templates.TemplateResponse(
        request,
        "error.html",
        {
            "status_code": exc.status_code,
            "detail": exc.detail,
            "back_url": "/demo",
            "back_label": "Back to the demo",
        },
        status_code=exc.status_code,
    )


def _require_demo_mode(settings: Settings) -> None:
    if not settings.demo_mode:
        raise HTTPException(
            status_code=404,
            detail="The demo is not enabled on this deployment.",
        )


@router.get("")
def demo_index(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
):
    try:
        _require_demo_mode(settings)
    except HTTPException as exc:
        return _error(request, exc)

    chips = [
        {
            "id": scenario.question_id,
            "label": _CHIP_LABELS[scenario.question_id],
            "question": scenario.question_text,
        }
        for scenario in DEMO_SCENARIOS
    ]
    return templates.TemplateResponse(request, "demo.html", {"chips": chips})


def _demo_answer_context(response: QueryResponse) -> dict:
    """Everything `demo_answer.html` needs, derived only from the response
    `run_query` already returned -- no recomputation, no second lookup."""
    is_unsupported = response.model == DEMO_UNSUPPORTED_MODEL_NAME
    is_genuine_abstention = response.abstained and not is_unsupported
    citation_rows = [
        candidate
        for candidate in response.candidates
        if candidate.chunk_uid in response.citations
    ]
    return {
        "response": response,
        "is_unsupported": is_unsupported,
        "is_genuine_abstention": is_genuine_abstention,
        "default_abstention_text": DEFAULT_ABSTENTION_TEXT,
        "citation_rows": citation_rows,
    }


@router.post("/query")
def demo_query_submit(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    embeddings: Annotated[EmbeddingProvider, Depends(embedding_provider)],
    reranker: Annotated[RerankProvider, Depends(rerank_provider)],
    llm: Annotated[LLMProvider, Depends(llm_provider)],
    query: Annotated[str, Form()],
):
    """Calls `run_query` directly -- the same function `POST /query` and
    `POST /ui/query` call -- with providers sourced through the identical
    `Depends(...)` dependencies `run_query` itself declares. Renders the
    returned `QueryResponse` straight into `demo_answer.html`; see the
    module docstring for why this does not redirect to a persisted-answer
    page the way `/ui/query` does.
    """
    try:
        _require_demo_mode(settings)
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

    return templates.TemplateResponse(
        request, "demo_answer.html", _demo_answer_context(response)
    )


__all__ = ["router"]
