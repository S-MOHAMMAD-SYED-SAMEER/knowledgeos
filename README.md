# KnowledgeOS

An internal knowledge system that answers from retrieved evidence — and
measures whether it actually did.

**Status: milestone 2 of 10.** Documents can be uploaded, validated, stored
and versioned, and each upload queues an ingestion job. **Nothing reads what a
document says yet** — parsing, chunking, embeddings, retrieval and answering
are milestones 3 onwards. Metrics sections will be filled with measured
numbers only after the evaluation harness runs. **No benchmark figure appears
in this README until it is real.**

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

**Milestones 1 and 2 of 10 are implemented.** Everything above this line
describes the system as specified; everything below describes only what exists
today.

## What milestone 1 built

- FastAPI application (`app/main.py`) with a `create_app()` factory, serving
  `GET /health` and `GET /ready` and nothing else.
- Environment-driven configuration (`app/config.py`, prefix `KNOWLEDGEOS_`) —
  five settings, because that is what milestone 1 reads.
- SQLAlchemy 2.x on psycopg 3 (`app/db/`), with a lazily-created cached engine.
- `documents` and `document_versions` as ORM models under `app/models/`.
- One Alembic migration creating the `vector` extension and both tables,
  verified to upgrade, downgrade and re-upgrade on a real database.
- Tests against a really-migrated PostgreSQL, not an in-memory stand-in.

No parsing, no chunking, no embeddings, no retrieval, no generation, no UI.
Those are milestones 3 to 10.

## What milestone 2 built

Upload. A file arrives, is checked, is stored under a name this system chose,
and becomes a document, a version and a queued ingestion job — in one database
transaction.

```
POST /documents          →  validate → store → document + version 1 + job
POST /documents/{id}/versions  →  validate → store → version n+1 + job
GET  /documents          GET /documents/{id}          GET /ingestion/{job_id}
```

- `app/storage/` — a four-method `Storage` protocol and `LocalStorage`, the
  only implementation and the only place in the application that touches the
  filesystem.
- `app/ingestion/validation.py` — extension, declared type, signature, UTF-8
  and size.
- `app/ingestion/service.py` — version numbering, the queued job, and
  `promote_version`.
- `app/api/documents.py`, `app/api/ingestion.py` — the five endpoints.
- `app/models/ingestion_job.py` and migration 0002.

**No parsing.** Nothing here opens a PDF, reads a DOCX archive, renders
Markdown or normalises text. That is milestone 3, and the boundary is enforced
by tests rather than by intention.

### What validation may and may not conclude

Validation looks at bytes; it does not interpret them.

| Format | Extension | Checked |
| --- | --- | --- |
| PDF | `.pdf` | extension, declared type, `%PDF-` signature |
| DOCX | `.docx` | extension, declared type, `PK\x03\x04` (ZIP) signature |
| Markdown | `.md`, `.markdown` | extension, declared type, decodes as UTF-8 |
| Plain text | `.txt` | extension, declared type, decodes as UTF-8 |

Plus: an empty file is refused, and the size limit is enforced **while
reading, in bounded chunks** — never from `Content-Length`, which the client
chose.

The declared `Content-Type` is evidence, not authority: a client that declares
nothing useful is not refused for that alone, but one that declares a type
belonging to a different format is contradicting itself and is.

`PK\x03\x04` proves a ZIP and nothing more. Proving a file is really a Word
document means opening the archive, which is parsing — so a renamed `.zip` is
accepted here and fails in milestone 3's parser, recorded as a job error. That
is the right place for it to fail.

### Storage

```
<storage_root>/documents/<document_id>/<version_id><ext>
```

Every component is a UUID this application generated. **The uploaded filename
never appears in a path** — it is kept as metadata and nothing else, which is
what makes path traversal impossible rather than merely blocked. It is
sanitised anyway (POSIX and Windows directory parts stripped, control
characters and bidirectional overrides removed, trimmed to the column) so it
cannot carry a surprise into a log line or a page.

`storage_path` is stored relative to the root, so moving the root does not
invalidate a row.

A write goes to a temporary file in the destination directory, is `fsync`ed,
and is then `os.replace`d into position — same filesystem, so the rename is
atomic. A reader never sees a partial file at a real key, and a failed write
leaves neither a partial file nor a temporary one.

`LocalStorage` refuses any key that is absolute, contains `..`, is not already
normalised, or resolves outside the root — checked before and after
resolution, so a symlinked directory cannot lead out either.

### Version lifecycle

The invariant this project depends on:

> **A version is `active` only if its content has been indexed.**

So an uploaded version is a `draft`, and **uploading a new version does not
supersede the current active one**. Superseding a version that answers
questions in favour of one that has not been indexed would take a working
answer away and put nothing in its place — default retrieval filters on
`status = 'active'`, and the new version has no chunks behind it.

`promote_version()` is the transition, and it is fully implemented and tested
here: it demotes the incumbent, flushes, then promotes the target, so the
partial unique index never sees two active versions. **Nothing in milestone 2
calls it.** The caller arrives with the milestone that can index a version,
and the docstring says so.

### Version numbering under concurrency

`max(version_number) + 1`, read inside the transaction and **not trusted**.
Two requests can read the same maximum; the `UNIQUE (document_id,
version_number)` constraint is what actually decides.

When a request loses that race the **database transaction alone** is retried:
the file has already been stored under a key built from the version UUID,
which does not change, so a retry re-reads the maximum and re-inserts the rows.
It never re-reads the request body, never writes a second file and never
re-runs validation. `tests/test_ingestion_service.py` proves this with a real
second connection committing the contested number first, and asserts the file
was written exactly once.

### Ingestion jobs

Created `queued`, with `attempts = 0`, `started_at` and `completed_at` null.
**Milestone 2 writes no other state.** The runner that advances a job through
`parsing → chunking → indexing → ready` is milestone 3; there is no
`BackgroundTasks`, no scheduler and no worker here. A queued job that nothing
picks up is the correct state of the system today, and `GET /ingestion/{id}`
reports it honestly.

`stage_error` records where a job broke, never document content. There is
deliberately no unique constraint on `document_version_id`, because
re-indexing creates a second job for the same version.

### Failure, and the one residue that is not swept

The file is written first and the rows last, and a database failure deletes
the file it had already written.

The ordering is deliberate, because the two possible residues are not equally
bad. A file with no row is invisible, costs disk, and can be removed. A row
with no file is a version this system believes it has, and it breaks the
moment a parser opens it.

| Failure | Result |
| --- | --- |
| Validation | Nothing written to disk or database |
| Storage write | No rows; opaque `500` |
| Any database failure | Rows rolled back, file deleted, opaque `500` |
| Job creation | Document and version roll back with it — all three are one transaction |

**Accepted limitation.** A process that dies between the storage write and the
database commit leaves an orphan file, because the compensating delete never
runs. Milestone 2 ships no reaper: an unreferenced file costs disk and nothing
else, which is the cheaper of the two failures. A test asserts this window
exists rather than pretending it is closed.

Error responses carry a short reason and nothing else — no path, no storage
root, no exception class, no traceback. `storage_path` appears in no response
body at all.

### What milestone 2 did not touch

`/health` and `/ready` are unchanged. Storage reachability is deliberately
**not** a readiness condition: a full disk should fail an upload, not take the
process out of rotation.

`checksum` and `page_count` stay null. The checksum is defined over normalized
text and the page count comes from parsing; neither exists until milestone 3,
and a value invented here would be a plausible-looking lie.

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
without one (117 pass, 103 skip). With a database: **220 pass**. No
credential is needed either way.

KnowledgeOS is a separate application from DocIntel and VoiceDesk in this
repository: its own package, dependencies, virtualenv, configuration prefix and
database. Nothing is shared between them, and `tests/test_isolation.py` asserts
it.

## Implementation specification

The authoritative build specification is a separate document, worked milestone
by milestone: inspect, propose, implement, test, commit, stop. It is not
committed to this repository.
