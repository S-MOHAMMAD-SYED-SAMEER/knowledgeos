# KnowledgeOS

**Retrieval-grounded answers over internal documents, with citation validation and honest abstention.**

Ask "how do I request production database access?" and get an answer built
only from your own documents — the exact chunk behind every claim, citations
checked programmatically before the answer is shown, and an explicit refusal
when the knowledge base does not contain the answer.

**Status: milestone 10 of 10 — v1 complete.**
**GitHub** [S-MOHAMMAD-SYED-SAMEER/knowledgeos](https://github.com/S-MOHAMMAD-SYED-SAMEER/knowledgeos) ·
**Live demo** No public instance is deployed — two independent, local, keyless demo modes exist instead, see **Running modes** below ·
**[Engineering deep dive](docs/ENGINEERING.md)**

> **Two demo modes, not one.** This repository has two independent,
> keyless ways to try the real query pipeline with no Gemini credential
> and no runtime model download: an **integrated demo mode**
> (`Settings.demo_mode`, `GET /demo`) built directly into `app/`, and a
> **standalone fixture-replay demo** (`demo/`, walkthrough in
> [docs/DEMO.md](docs/DEMO.md)). Neither replaces the other — see
> **Running modes** below for what distinguishes them.

---

## The problem

An organization has 100+ SOPs, policies and manuals. An employee asks a
question and either digs through Confluence for twenty minutes, or asks an
LLM that confidently invents a process.

The second failure is worse. **A wrong answer that cites nothing is
indistinguishable from a right one.**

## The solution

KnowledgeOS answers only from retrieved documents, cites the exact chunk
behind each claim, validates those citations in Python before returning the
answer, and abstains when the evidence is insufficient.

Four things are measured separately, because they fail separately:

| Question | What it measures |
| --- | --- |
| Did we retrieve the right evidence? | Retrieval quality — Recall@K, MRR, nDCG@10 |
| Did the answer use that evidence? | Grounding — unsupported-claim rate |
| Do citations point at real, retrieved chunks? | Citation validity and coverage |
| Did it refuse when it should have? | Abstention precision and recall |

A system can retrieve perfectly and still hallucinate. Most RAG demos report
neither.

## Core capabilities

| Capability | What it actually does |
| --- | --- |
| **Hybrid retrieval** | Postgres full-text search + pgvector similarity, fused by Reciprocal Rank Fusion. No second datastore. |
| **Citation validation** | Every cited chunk ID is checked against the retrieved, *selected* evidence set. An invalid citation rejects the answer rather than displaying it. |
| **Honest abstention** | Refuses on insufficient evidence — zero candidates, a calibrated rerank-score threshold, or the model's own `sufficient_evidence=false`. |
| **Document versioning** | Only the active version answers; superseded versions stay queryable for audit. One active version per document, enforced by a partial unique index. |
| **Dual evaluation harnesses** | Separate retrieval and answer suites, run from the CLI, offline, with no API credential. |
| **Server-rendered UI** | Jinja2, no build step: documents, version history, query box, answer page with citations and evidence, evaluation results. |

## How the pipeline works

```
Ingest → Parse → Chunk → Index → Retrieve → Rerank → Generate → Cite → Evaluate
```

| Stage | Detail |
| --- | --- |
| Chunk | Fixed-size by token count (512/64 overlap), paragraph-aware. `chunk_uid = sha256(version:sequence:text)[:32]` makes re-indexing idempotent. |
| Retrieve | Top 50 lexical + top 50 vector → RRF fusion → dedupe → top 20. Metadata filters are SQL `WHERE` clauses, never an instruction to the model to ignore results. |
| Rerank | Local cross-encoder behind an interface, plus a passthrough implementation so reranking's contribution is a measured number rather than an assumption. |
| Generate | Versioned prompt file, structured output (`answer`, `citations`, `sufficient_evidence`), parsed and validated in Python. |
| Cite | Three deterministic rules, then two grounding layers (deterministic and semantic) reported separately. |

## Architecture

```
app/  api · ingestion · parsing · chunking · indexing · retrieval
      reranking · generation · providers · observability · ui
evals/  fixtures/ · retrieval/ · answers/ · run.py
```

**Layering rule, enforced by tests:** `app/retrieval/`, `app/chunking/` and
`app/generation/citations.py` make no provider calls and no I/O beyond the
database — they are the parts that get evaluated, so they must be callable
as pure functions over fixture data.

## Engineering highlights

- **Every provider sits behind an interface.** The Gemini adapter is the only
  file importing a vendor SDK, lazily. Swapping a model means writing an
  adapter, not setting an environment variable.
- **Static tests enforce architecture, not just behaviour** — no application
  module may select a fake provider, the application never imports the
  evaluation package, Postgres FTS is never mislabelled.
- **Invariants live in the database.** One active version per document is a
  partial unique index, and the suite builds its schema by running the real
  migration — a schema that could not enforce it would fail the tests.
- **Nothing is fabricated.** No metric, threshold, price or model ID appears
  unless measured or verified; unknown pricing raises a config error rather
  than silently becoming zero.

## Evaluation approach

Evaluation runs against a committed fixture corpus (10 documents, one with a
materially changed v2) and 52 labelled questions across eight categories:
directly answerable, multi-document, ambiguous, insufficient evidence,
conflicting versions, metadata-filtered, citation-sensitive, and adversarial.

Deterministic checks (citation validity, metadata-filter correctness,
abstention triggering) are kept separate from model-judged checks, and the
reports say which is which. Both suites run offline from the CLI:
`python -m evals.run --suite retrieval | answers`.

**Results.** No retrieval or answer-quality number appears anywhere in this
repository. Both harnesses exist and are proven correct against fixtures with
deterministic test providers, but **neither has been run officially** against
the full evaluation corpus — no `python -m evals.run --suite retrieval |
answers` invocation has produced a committed report or an `eval_runs` row.
(The real BGE and cross-encoder models have since been run directly, once
each, to generate the standalone demo's fixtures — see **Running modes**
below — but
that is fixture generation, not an evaluation suite run.) See
[docs/ENGINEERING.md](docs/ENGINEERING.md) for the full detail.

## Running modes

The repository contains **three** ways to run KnowledgeOS: the real,
credentialed production path, and two independent demo mechanisms that
happen to share the word "demo" but share no code, no fixtures, and no
generation provider. Neither demo replaces or supersedes the other; both
are documented below, distinctly, and both are also covered by their own
static isolation tests that prove they never call Gemini.

| Mode | Credentials | Status |
| --- | --- | --- |
| **Live Mode** | Gemini key + local models | Implemented |
| **Integrated demo mode** (`Settings.demo_mode`) | None | **Implemented** |
| **Standalone Demo Mode** (P3, `demo/`) | None | **Implemented** |

**Live Mode — implemented.**
- Real local `BgeEmbeddingProvider` and `CrossEncoderRerankProvider` —
  require `BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2`
  in the local sentence-transformers cache.
- Real generation through Google Gemini (`GEMINI_API_KEY` or
  `GOOGLE_API_KEY`, plus `KNOWLEDGEOS_LLM_MODEL` — no credential is
  configured in this build environment).
- The production default for every provider dependency; both demo modes
  below only ever override it, at their own respective seams, never
  change what it does by default.

**Integrated demo mode — implemented (`Settings.demo_mode`, inside `app/`).**
- A `Settings.demo_mode` flag that, when true, makes
  `app/api/query.py::llm_provider()` resolve to
  `app.providers.demo_llm.DemoLLMProvider` instead of the real Gemini
  adapter — structurally, not by preference: the demo branch never
  reaches the real Gemini accessor at all.
- A fixed set of curated, hand-verified question/answer scenarios
  (`app/generation/demo_scenarios.py`), matched by exact
  case/whitespace-normalized question text — never a fuzzy or
  embedding-based match, and never a guessed answer for an unmatched
  question.
- Visitor-facing routes `GET /demo` and `POST /demo/query`
  (`app/ui/demo_routes.py`) that render the real `QueryResponse` the
  normal query pipeline returns — not a second implementation of it.
- Real, unmodified retrieval, RRF fusion, citation validation and
  grounding run for every request; only the generation step is
  deterministic and scripted, and only while `Settings.demo_mode` is on.
- Mutation protection: `POST /documents`, `POST /documents/{id}/versions`,
  `POST /documents/{id}/reindex`, `POST /answers/{id}/feedback`, and
  `POST /ui/answers/{id}/feedback` all refuse with `403` while demo mode
  is on, so a visitor cannot alter what a public demo serves. Read
  routes (`GET /documents`, query, retrieval) stay available.
- **What this mode's verification actually covers:** the merged test
  suite (see **Verified project facts** below) exercises this mode's
  routing, curated-question matching, citation validation, grounding,
  and mutation guard end to end — through `FakeEmbeddingProvider`/
  `PassthroughRerankProvider` at that HTTP-test layer specifically, by
  design, so those assertions isolate citation/grounding/matching
  correctness from real-model variance. **Separately, real
  `BgeEmbeddingProvider` and `CrossEncoderRerankProvider` have been
  verified directly**, on a Windows development machine with both
  models in the local sentence-transformers cache: each loads and
  produces real output (384-dimension embeddings; real cross-encoder
  relevance scores) with no network access. What remains unexercised is
  a single live run of this mode with all three real pieces together —
  `KNOWLEDGEOS_DEMO_MODE=true`, real BGE, real cross-encoder, and
  `DemoLLMProvider`, over one actual HTTP request — and this mode's
  retrieval *quality* at scale is still unmeasured.
- No public URL is deployed; run locally with `KNOWLEDGEOS_DEMO_MODE=true`
  and open `/demo`.

**Standalone Demo Mode — implemented (P3, `demo/`).**
- Committed, deterministic fixture providers (`demo/fixtures/`) replay
  real, precomputed BGE embeddings, real precomputed cross-encoder
  reranking scores, and curated answer fixtures — never invented or
  hash-derived data. **The committed fixture values themselves were not
  regenerated or diffed against fresh model output during this
  repository's most recent verification pass** — but that pass did
  independently confirm, on a separate Windows machine with both models
  cached, that `BAAI/bge-small-en-v1.5` and
  `cross-encoder/ms-marco-MiniLM-L-6-v2` are real, loadable, working
  models; see **Current limitations**.
- No Gemini credential; no BGE/cross-encoder download at runtime; and,
  unlike the integrated demo mode above, no `KNOWLEDGEOS_DEMO_MODE=true`
  either — `demo/app.py` overrides its three provider dependencies
  directly, so this mode never depends on that flag and never enters the
  integrated mode's own `Settings.demo_mode`-gated M1-M4 corpus seed in
  `app/main.py`'s lifespan.
- Runs through the exact same `app.main.create_app()` and
  `POST /query`/`/ui/query` routes Live Mode uses — `demo/app.py`
  overrides the three provider dependencies at the same FastAPI seam a
  test already does, **never a second implementation** of retrieval,
  reranking, or generation.
- Five flagship scenarios (normal retrieval, citation-sensitive
  reranking, conflicting versions, multi-document evidence,
  insufficient-evidence abstention), reproducible every run.
- Its own generation provider, `demo.llm.DemoLLMProvider`, is a
  **different class from, and unrelated to,**
  `app.providers.demo_llm.DemoLLMProvider` above — the two share a name
  by coincidence, live in two unrelated packages (`demo/` vs.
  `app/providers/`), and neither imports or calls the other.
- **Mutation protection:** the same `require_mutation_allowed`
  dependency the integrated demo mode uses is also overridden directly
  on `demo/app.py`'s own FastAPI instance (`app.dependency_overrides`),
  so `POST /documents`, `POST /documents/{id}/versions`, `POST
  /documents/{id}/reindex`, `POST /answers/{id}/feedback`, and `POST
  /ui/answers/{id}/feedback` all refuse with `403` here too — without
  ever setting `Settings.demo_mode`.
- **Rate limiting:** `docker-compose.demo.yml` sets
  `KNOWLEDGEOS_DEMO_RATE_LIMIT_ENABLED=true`, which caps `POST /query`
  and `POST /ui/query` at 10 requests per minute per client IP
  (`app/api/rate_limit.py`); an excess request gets `429` with a
  `Retry-After` header. Off by default everywhere else, including Live
  Mode.
- No public demo URL is deployed; run it locally with
  `uvicorn demo.app:app` — full walkthrough in
  [docs/DEMO.md](docs/DEMO.md), which documents this mechanism
  specifically, not the integrated demo mode above.

## Verified project facts

Nothing estimated — but this project has genuinely been verified in two
different environments that produce two different, both-true full-suite
results. Conflating them would be misleading, so they're kept separate
rather than reported as one number.

| Verification | Full suite result |
| --- | --- |
| **CI** (GitHub Actions, `.github/workflows/ci.yml`) | 1139 collected — 1135 passed, 4 skipped, 0 failed |
| **Windows development laptop**, both models cached locally | 1139 collected — 1127 passed, **11 failed**, 1 skipped, 2 warnings |

CI intentionally forces model-dependent tests to skip (no BGE/cross-encoder
download, ever, in CI) — an honest, reproducible skip, not a pass, and not
the same thing as "verified." On the Windows laptop, both models are
cached, so those same tests actually *run* — and 11 of them fail. This is
not an application regression; every one is a known, categorized,
pre-existing environment/assumption mismatch:

- **3** — a Windows `cp1252`-vs-UTF-8 encoding issue in a test reading
  `docs/ENGINEERING.md`, unrelated to KnowledgeOS's own runtime code.
- **1** — a Windows symlink-privilege restriction in a storage
  path-traversal test.
- **7** — tests asserting "the real BGE/reranker are unavailable" (3
  direct provider tests + 4 in the evaluation CLI's own tests); now that
  both models *are* cached on this machine, that assumption no longer
  holds here specifically, which is exactly what these tests are
  (correctly) noticing.

None of the 11 touch retrieval, reranking, generation, citation
validation, or abstention correctness — every test exercising those
passes identically in both environments.

**A real defect was found and fixed in this same verification pass**
(commit `050fe0d`): `tests/demo_fixtures.py` read fixture markdown
without an explicit encoding, which corrupted a chunk's text — and
therefore its `chunk_uid` — on Windows specifically, breaking 10
integrated-demo-mode scenario tests (citation validation was correctly
rejecting a citation that, on this platform, no longer pointed at any
real chunk). Fixed by reading the fixture as UTF-8 explicitly; both
`tests/test_demo_query_api.py` and `tests/test_demo_ui.py` now pass all
scenarios on the Windows laptop, identically to CI.

| Fact | Value |
| --- | --- |
| Alembic migrations / database tables | 7 / 9 |
| Continuous integration | Configured — full suite, no external API key |
| Retrieval / answer-quality metrics | **None produced**, in any environment |

Neither result means real BGE/CrossEncoder retrieval *quality*, a real
Docker build, or full-corpus real-model evaluation have been verified —
see **Current limitations** below for exactly what remains unverified,
and **Running modes** above for which claim belongs to which demo.

## Current limitations

- **The standalone demo's fixtures were generated from real BGE/cross-encoder
  inference, once**, in a different environment that had model access
  (`demo/fixtures/`) — the committed fixture *values* were not
  regenerated or independently re-verified against fresh model output.
  (Both models are, however, independently confirmed loadable and
  working on a Windows development laptop with them cached — see
  **Verified project facts** above.)
- **The integrated demo mode (`Settings.demo_mode`) precomputes nothing** —
  it reaches the real `BgeEmbeddingProvider`/`CrossEncoderRerankProvider`
  seams directly. Both models are now confirmed to load and run for real
  on a Windows development laptop (local sentence-transformers cache, no
  network) — no longer blocked. What remains unverified is a single live
  run of this mode with all three real providers together over one
  actual HTTP request, and **neither demo mode's real-model retrieval
  quality has been independently measured** in any environment — both
  are verified only at the code-path/component level described in
  **Running modes** above.
- **A full Live Mode query — all three real providers together, including
  Gemini — has not been exercised end to end here**: no Gemini credential
  has ever been configured in this environment.
- **No official evaluation numbers have been produced**, and none are claimed.
- **No Docker image has been built and no container run** from the committed
  Docker assets here — the daemon is unavailable, so those files are verified
  statically only.
- Postgres full-text search is **not BM25**; ranking behaviour differs and
  this README does not claim otherwise.
- Semantic grounding is approximate and will miss subtle unsupported claims;
  the deterministic citation checks are the exact layer.
- Fixed-size chunking with overlap; semantic chunking is deferred because it
  cannot be evaluated meaningfully at this corpus size.
- Evaluated on a small fixture corpus; behaviour at 100k+ chunks is untested.
- No authentication or authorization layer, and no claim of enterprise-grade
  security is made anywhere in this project.

## Tech stack

Python 3.13 · FastAPI · SQLAlchemy 2.x · PostgreSQL 16 + pgvector · psycopg 3
· Alembic · sentence-transformers (local embeddings and reranking) · Google
Gemini via `google-genai` behind a provider interface · Jinja2 · pytest ·
Docker · GitHub Actions

> The build specification names the Anthropic SDK for generation; this build
> uses **Google Gemini** — a documented deviation from that wording, not from
> the behavioural requirements, none of which name a vendor. Nothing outside
> the one adapter file is vendor-shaped.

## Running locally

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

# pgvector goes into the PostgreSQL server, not just into Python
sudo apt-get install postgresql-16-pgvector
createuser knowledgeos --pwprompt --createdb
createdb -O knowledgeos knowledgeos
createdb -O knowledgeos knowledgeos_test   # the suite migrates and drops this

alembic upgrade head
uvicorn app.main:app --reload

curl localhost:8000/health   # liveness: touches nothing
curl localhost:8000/ready    # readiness: database, migrations, extension
pytest
```

Then open `http://localhost:8000/ui/`. Tests needing PostgreSQL skip when no
server answers, so `pytest` stays meaningful without one, and no credential
is needed either way. If `CREATE EXTENSION vector` fails the role is not
privileged enough — [docs/ENGINEERING.md](docs/ENGINEERING.md) has the
one-time superuser bootstrap.

## Deployment

`Dockerfile`, `docker-compose.yml` and a CI workflow are committed. Migrations
are always a separate, explicit step — the application never runs them at
startup, and `/ready` reports the drift until they have been applied:

```bash
docker compose up --build -d db
docker compose run --rm app alembic upgrade head
docker compose up --build app
```

**No image has been built or run in this environment** — see Current
limitations. Full detail in [docs/ENGINEERING.md](docs/ENGINEERING.md).

## Scope

**Deliberately not built:** chatbot platform, Slack/Teams bot, agent
framework, SSO. No Pinecone, Weaviate, Qdrant, Elasticsearch, Redis, Celery,
Kubernetes, or React. **Postgres is the only datastore.**

**Not in v1:** Google Drive / Notion / Slack connectors · permission-aware
retrieval · table and image understanding · query decomposition for
multi-hop questions.

## Engineering deep dive

The full milestone-by-milestone record — every decision, every deferral, and
every limitation recorded at the time it was found — is in
**[docs/ENGINEERING.md](docs/ENGINEERING.md)**.
