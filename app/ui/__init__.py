"""Milestone 10: the server-rendered UI.

Jinja2, minimal, five pages (§12): document list, document detail with
version history, a query box, an answer page showing citations and
retrieved evidence side by side, and an evaluation results page. No React,
no build step — every page is HTML returned by a FastAPI route.

Mounted under `/ui` (locked decision D2), with every route
`include_in_schema=False`: the JSON API's OpenAPI contract (`/openapi.json`)
is unchanged by this package's existence, and `tests/test_health.py`'s own
exact-path guard proves it.

Nothing here reimplements retrieval, reranking, generation, citation
validation, grounding, or abstention. The query flow calls
`app.api.query.run_query` directly — the same function `POST /query`
calls — and every other page reads already-persisted rows through the same
models (`app.models`) the JSON API reads. This package never imports
`evals/`; the evaluation results page reads `eval_runs` through
`app.models.EvalRun`.
"""
