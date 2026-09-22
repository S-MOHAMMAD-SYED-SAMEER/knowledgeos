# KnowledgeOS

**Retrieval-grounded answers over internal documents, with citation validation and honest abstention.**

Ask "how do I request production database access?" and get an answer built
only from your own documents — the exact chunk behind every claim, citations
checked programmatically before the answer is shown, and an explicit refusal
when the knowledge base does not contain the answer.

**Status: milestone 10 of 10 — v1 complete.**
**GitHub** [S-MOHAMMAD-SYED-SAMEER/knowledgeos](https://github.com/S-MOHAMMAD-SYED-SAMEER/knowledgeos) ·
**Live demo** No public instance is deployed — run the local, keyless [Demo Mode](docs/DEMO.md) instead ·
**[Engineering deep dive](docs/ENGINEERING.md)**

> **Demo Mode.** A deterministic, keyless local demo that exercises the real
> query pipeline — no Gemini credential, no runtime model download. Five
> minutes from a fresh clone to a cited, grounded answer:
> **[docs/DEMO.md](docs/DEMO.md)**.

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
each, to generate Demo Mode's fixtures — see **Running modes** below — but
that is fixture generation, not an evaluation suite run.) See
[docs/ENGINEERING.md](docs/ENGINEERING.md) for the full detail.

## Running modes

| Mode | Credentials | Status |
| --- | --- | --- |
| **Demo Mode** | None | **Implemented** |
| **Live Mode** | Gemini key + local models | Implemented |

**Demo Mode — implemented (P3, `demo/`).**
- Committed, deterministic fixture providers (`demo/fixtures/`) replay real,
  precomputed BGE embeddings, real precomputed cross-encoder reranking
  scores, and curated answer fixtures — never invented or hash-derived data.
- No Gemini credential; no BGE/cross-encoder download at runtime.
- Runs through the exact same `app.main.create_app()` and
  `POST /query`/`/ui/query` routes Live Mode uses — `demo/app.py` overrides
  the three provider dependencies at the same FastAPI seam a test already
  does, **never a second implementation** of retrieval, reranking, or
  generation.
- Five flagship scenarios (normal retrieval, citation-sensitive reranking,
  conflicting versions, multi-document evidence, insufficient-evidence
  abstention), reproducible every run.
- No public demo URL is deployed; run it locally with `uvicorn demo.app:app`
  — full walkthrough in [docs/DEMO.md](docs/DEMO.md).

**Live Mode — implemented.**
- Real local `BgeEmbeddingProvider` and `CrossEncoderRerankProvider` —
  require `BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2`
  in the local sentence-transformers cache.
- Real generation through Google Gemini (`GEMINI_API_KEY` or
  `GOOGLE_API_KEY`, plus `KNOWLEDGEOS_LLM_MODEL` — no credential is
  configured in this build environment).
- The production default for every provider dependency; Demo Mode only ever
  overrides them, at the same seam, never inside `app/` itself.

## Verified project facts

Measured directly against this checkout — nothing estimated.

| Fact | Value |
| --- | --- |
| P3 demo test suite (`tests/test_demo_*.py`) | **88 passed** |
| Full suite, with PostgreSQL (this Windows dev environment) | **1026 passed, 11 failed, 1 skipped** |
| Test suite, without PostgreSQL | **588 passed, 362 skipped** |
| Alembic migrations / database tables | 7 / 9 |
| Continuous integration | Configured — full suite, no external API key |
| Retrieval / answer-quality metrics | **None produced** |

The 11 failures on this Windows dev environment are pre-existing and
environment-dependent, not P3 regressions: three assume the real BGE/reranker
providers are unavailable here, which they no longer are (P1/P3 ran both
directly to generate Demo Mode's fixtures); four more in the evaluation CLI
tests make that same now-outdated assumption; three are Windows
`cp1252`-encoding failures reading a UTF-8 documentation file; one is a
Windows symlink-privilege restriction. See
[docs/ENGINEERING.md](docs/ENGINEERING.md) for the full breakdown. The one
remaining skip is the opt-in Gemini generation smoke test — no credential has
been configured in this environment.

## Current limitations

- **BGE and the cross-encoder have each been run for real, once**, to
  generate Demo Mode's precomputed fixtures (`demo/fixtures/`) — neither is
  "unavailable" in this build environment any more. **A full Live Mode query
  — all three real providers together, including Gemini — has not been
  exercised end to end here**: no Gemini credential has ever been configured
  in this environment.
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
