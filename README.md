# KnowledgeOS

An internal knowledge system that answers from retrieved evidence — and
measures whether it actually did.

**Status: milestone 1 of 10.** The foundation exists: configuration, the two
probes, PostgreSQL with pgvector, Alembic, and the document/version schema.
Nothing is ingested, retrieved, or answered yet. Metrics sections will be
filled with measured numbers only after the evaluation harness runs. **No
benchmark figure appears in this README until it is real.**

## The problem

An organization has 100+ SOPs, policies, and manuals. An employee asks "how do
I request production database access?" and either digs through Confluence for
twenty minutes or asks an LLM that confidently invents a process.

The second failure is worse than the first. A wrong answer that cites nothing
is indistinguishable from a right one.

## The approach

KnowledgeOS answers only from retrieved documents, cites the exact chunk behind
each claim, validates those citations programmatically, and abstains when the
knowledge base doesn't contain the answer.

Four things are measured separately, because they fail separately:

| Question | What it measures |
| --- | --- |
| Did we retrieve the right evidence? | Retrieval quality — Recall@K, MRR, nDCG@K |
| Did the answer use that evidence? | Grounding — unsupported-claim rate |
| Do the citations point at real, retrieved chunks? | Citation validity and coverage |
| Did it refuse when it should have? | Abstention correctness |

A system can retrieve perfectly and still hallucinate. Most RAG demos report
neither.

## Core loop

```
Ingest → Parse → Chunk → Index → Retrieve → Rerank → Generate → Cite → Evaluate
```

## What it does

- **Ingestion** — PDF, DOCX, Markdown, plain text. Idempotent, retry-safe,
  observable job status.
- **Versioning** — documents have explicit versions. Only the active version
  participates in default retrieval; superseded versions stay queryable for
  audit.
- **Hybrid retrieval** — Postgres full-text search and pgvector similarity,
  fused by Reciprocal Rank Fusion.
- **Metadata filtering** — applied in the query, not by asking the model to
  ignore irrelevant results.
- **Reranking** — local cross-encoder behind a provider interface.
- **Grounded generation** — versioned prompt, evidence-only instruction,
  explicit abstention path.
- **Citation validation** — every cited chunk ID is checked against the
  retrieved set. Invalid citations are rejected, not displayed.
- **Evaluation** — separate retrieval and answer suites, runnable offline with
  no API credentials.

## What it deliberately is not

Not a chatbot platform, Slack/Teams bot, agent framework, or SSO-enabled
enterprise product. No Pinecone, Weaviate, Qdrant, Elasticsearch, Redis,
Celery, Kubernetes, or React. **Postgres is the only datastore.**

## Stack

Python 3.13 · FastAPI · SQLAlchemy 2.x · PostgreSQL 16 + pgvector · psycopg 3 ·
Alembic · sentence-transformers (local embeddings and reranking) · Anthropic
SDK behind a provider interface · pytest · Docker

Embeddings and reranking run locally. The only paid dependency is answer
generation, and the entire evaluation suite runs without it.

Dependencies arrive with the milestone that first imports them, so an install
never carries a library the code does not yet use. Milestone 1 installs seven
runtime packages: FastAPI, Uvicorn, Pydantic, pydantic-settings, SQLAlchemy,
Alembic, psycopg. sentence-transformers, the Anthropic SDK, the `pgvector`
Python package and Jinja2 are not installed yet.

## Evaluation methodology

Evaluation runs against a fixture knowledge base with labelled questions across
eight categories: directly answerable, multi-document, ambiguous,
insufficient-evidence, conflicting-version, metadata-filtered,
citation-sensitive, and adversarial.

Deterministic checks (citation validity, metadata filter correctness,
abstention triggering) are separated from model-judged checks (answer
relevance, claim support). The README reports which is which.

These metrics are evidence about this system's behaviour on this evaluation
dataset. They are not a claim about general model quality.

**Results: (to be filled after M9)**

## Known limitations

- Postgres full-text search **is not BM25**; ranking behaviour differs and this
  README does not claim otherwise.
- Grounding verification is imperfect. Deterministic checks catch citation
  errors reliably; semantic entailment checking catches obvious unsupported
  claims and will miss subtle ones.
- Chunking is fixed-size with overlap. Semantic chunking is deferred because it
  cannot be evaluated meaningfully at this corpus size.
- Evaluated on a small fixture corpus. Retrieval behaviour at 100k+ chunks is
  untested.

## Roadmap (not in v1)

Google Drive / Notion / Slack connectors · permission-aware retrieval · table
and image understanding · cross-knowledge-base search · query decomposition for
multi-hop questions

---

# Implementation status

**Milestone 1 of 10 is implemented.** Everything above this line describes the
system as specified; everything below describes only what exists today.

## What milestone 1 built

- FastAPI application (`app/main.py`) with a `create_app()` factory, serving
  `GET /health` and `GET /ready` and nothing else.
- Environment-driven configuration (`app/config.py`, prefix `KNOWLEDGEOS_`) —
  five settings, because that is what milestone 1 reads.
- SQLAlchemy 2.x on psycopg 3 (`app/db/`), with a lazily-created cached engine.
- `documents` and `document_versions` as ORM models under `app/models/`.
- One Alembic migration creating the `vector` extension and both tables,
  verified to upgrade, downgrade and re-upgrade on a real database.
- 74 tests against a really-migrated PostgreSQL, not an in-memory stand-in.

No upload, no parsing, no chunking, no embeddings, no retrieval, no
generation, no UI. Those are milestones 2 to 10.

## The two probes

`GET /health` is **liveness**. It touches nothing — no database, no provider,
no filesystem — and that is the contract, not an optimisation. A liveness probe
that failed because PostgreSQL blinked would have an orchestrator restart a
process that was working perfectly. `tests/test_health.py` asserts it answers
`200` while pointed at a database that does not exist.

`GET /ready` is **readiness**: should this process be given work?

```json
{
  "status": "ready",
  "database":   {"ok": true, "detail": "reachable"},
  "migrations": {"ok": true, "detail": "at head (70c2f52e54ba)"},
  "extension":  {"ok": true, "detail": "vector 0.6.0"}
}
```

It returns `503` when the database is unreachable, when the stamped Alembic
revision is not the head this build expects, or when the `vector` extension is
missing. With the database down the other two checks report `not checked`
rather than guessing — there is nothing to read a revision or an extension list
out of.

The extension check is here because milestone 1 exists to prove that
foundation is real. Nothing declares a vector column yet; a database missing
the extension would otherwise fail much later and far less clearly.

The connection string never appears in a response or a log line — it carries a
password. Failures are reported by exception class name only.

## The schema

**`documents`** — what people refer to. It does not change.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | `uuid` PK | `gen_random_uuid()`; appears in URLs, and feeds the deterministic chunk ID later |
| `title` | `varchar(512)` | |
| `source_type` | `varchar(32)` | **Provenance**, not format. `CHECK IN ('upload')` — connectors are not in v1 |
| `department`, `category` | `varchar(128)`, nullable | Filter columns, indexed. The customer's vocabulary, so not constrained to a fixed set |
| `tags` | `jsonb` NOT NULL `'[]'` | Array of strings, `CHECK jsonb_typeof = 'array'`, GIN index |
| `created_at`, `updated_at` | `timestamptz` | |

**`document_versions`** — one concrete artefact, with a file behind it. Versions
are what retrieval reaches.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | `uuid` PK | |
| `document_id` | `uuid` FK | `ON DELETE CASCADE`, explicitly indexed |
| `version_number` | `integer` | `CHECK >= 1`; `UNIQUE (document_id, version_number)` |
| `status` | `varchar(16)` | `draft` \| `active` \| `superseded`, by `CHECK` |
| `checksum` | `varchar(64)`, nullable | sha256 of normalized text — `CHECK` on the hex shape when present |
| `original_filename` | `varchar(512)` | Kept as metadata only |
| `storage_path` | `varchar(1024)` | |
| `effective_date` | `date`, nullable | A calendar day, not an instant |
| `page_count` | `integer`, nullable | `CHECK > 0` when present |
| `created_at` | `timestamptz` | |

Three decisions in that table are worth the sentence they cost.

**`status` is a `VARCHAR` with a `CHECK`, not a native PostgreSQL `ENUM`** —
with a Python `StrEnum` on the application side. A native enum makes the
partial index predicate clumsier and is awkward to extend: a new value needs
`ALTER TYPE`, which cannot be used in the same transaction that adds it. A
check constraint is one line of a migration.

**`checksum` is nullable**, and has to be. It is defined as the sha256 of the
*normalized* text, and normalization happens at parsing — a later milestone. A
version row exists before that. It is null until then and never invented in the
meantime.

**`source_type` is provenance, not file format.** Format belongs to the
version, because v2 of a policy may arrive as DOCX when v1 was a PDF. The
column that records it arrives with the milestone that can read a file.

## One active version per document

```sql
CREATE UNIQUE INDEX uq_document_versions_one_active
    ON document_versions (document_id)
 WHERE status = 'active';
```

This is the rule the whole versioning model rests on, and it is enforced by
PostgreSQL rather than by whichever code path happens to promote a version. The
alternative is two uploads racing and a knowledge base that answers from two
versions of the same policy at once.

It is **partial**, so any number of `draft` and `superseded` versions may sit
beside the active one — which is what keeps history queryable for the audit
question about last quarter.

It lives in the migration, so the test suite builds its schema by **running the
real migration** rather than `Base.metadata.create_all`. A schema built any
other way would not have this index, and the tests that matter most would pass
against a database that could not enforce anything.

## Running locally

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

# pgvector must be installed into the PostgreSQL server, not just into Python
sudo apt-get install postgresql-16-pgvector

createuser knowledgeos --pwprompt --createdb
createdb -O knowledgeos knowledgeos
createdb -O knowledgeos knowledgeos_test   # the suite migrates and drops this

alembic upgrade head
uvicorn app.main:app --reload
curl localhost:8000/health      # liveness: touches nothing
curl localhost:8000/ready       # readiness: database, migrations, extension
pytest
```

**The pgvector bootstrap.** pgvector 0.6.0 is not a *trusted* extension, so an
unprivileged role cannot run `CREATE EXTENSION vector`. Migration 0001 issues
`CREATE EXTENSION IF NOT EXISTS vector`, which works where the role has the
privilege and is a harmless no-op where a superuser has already done it. Where
it has not been done and the role cannot, run it once per database as a
superuser:

```bash
sudo -u postgres psql -d knowledgeos      -c 'CREATE EXTENSION IF NOT EXISTS vector;'
sudo -u postgres psql -d knowledgeos_test -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

The migration's `downgrade()` deliberately does **not** drop the extension.
Downgrading means "remove KnowledgeOS's two tables", not "uninstall a shared
PostgreSQL feature" — and `DROP EXTENSION` cascades to anything using its type.

Alembic reads the database URL from `KNOWLEDGEOS_DATABASE_URL` through
`app/config.py`; `alembic.ini` deliberately holds no URL, so migrations and the
application cannot disagree about which database they are using. A caller may
set one explicitly in code, which is how the test fixtures guarantee they never
touch the development database — they refuse any name not ending in `_test`.

Tests that need PostgreSQL skip when no server answers, so `pytest` still runs
without one (36 pass, 38 skip). With a database: **74 pass**. No credential is
needed either way.

KnowledgeOS is a separate application from DocIntel and VoiceDesk in this
repository: its own package, dependencies, virtualenv, configuration prefix and
database. Nothing is shared between them, and `tests/test_isolation.py` asserts
it.

## Implementation specification

The authoritative build specification is a separate document, worked milestone
by milestone: inspect, propose, implement, test, commit, stop. It is not
committed to this repository.
