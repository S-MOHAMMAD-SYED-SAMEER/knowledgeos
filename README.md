# KnowledgeOS

An internal knowledge system that answers from retrieved evidence — and
measures whether it actually did.

**Status: milestone 3 of 10.** Documents can be uploaded, validated, stored,
versioned, parsed and chunked. **Nothing is searchable yet** — embeddings,
full-text indexing, retrieval and answering are milestones 4 onwards. Metrics
sections will be filled with measured numbers only after the evaluation
harness runs. **No benchmark figure appears in this README until it is real.**

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
never carries a library the code does not yet use. Today that is FastAPI,
Uvicorn, Pydantic, pydantic-settings, SQLAlchemy, Alembic, psycopg,
python-multipart, `pypdf` and APScheduler. sentence-transformers, the
Anthropic SDK, the `pgvector` Python package and Jinja2 are not installed yet;
DOCX is read with the standard library, so `python-docx` is not installed at
all.

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

**Milestones 1 to 4 of 10 are implemented.** Everything above this line
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

No embeddings, no retrieval, no generation, no UI. Those are milestones 4
to 10.

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

## What milestone 3 built

Parsing and chunking. A queued job is picked up, the stored file is read, its
text is normalized once, and that text becomes deterministic chunks in the
database.

```
queued → parsing → chunking → indexing        ← milestone 3 stops here
              ↓ (either stage)
            failed, with stage_error
```

- `app/parsing/` — one parser per format, plus the normalizer.
- `app/chunking/` — the deterministic chunker and `chunk_uid`. Pure functions:
  no database, no storage, no provider, which is what the specification
  requires of this package.
- `app/ingestion/pipeline.py` — the stages, and the one place normalized text
  is produced.
- `app/ingestion/runner.py` — APScheduler polling with `FOR UPDATE SKIP LOCKED`.
- `app/models/chunk.py` and migration 0003.

**No embeddings, no tsvector, no retrieval.** Making chunks findable is
milestone 4.

### The parsers

| Format | Library | Pages | Sections |
| --- | --- | --- | --- |
| PDF | `pypdf` | Real, one block per page | None — a PDF structurally provides none |
| DOCX | **standard library** (`zipfile` + `ElementTree`) | None — pagination is a rendering property, not in the file | Nearest preceding paragraph styled as a heading |
| Markdown | standard library | None | Nearest preceding ATX or setext heading |
| Plain text | standard library | None | None |

`python-docx` was deliberately not added: extracting text from Office Open XML
needs a ZIP reader and an XML parser, both of which are in the standard
library, and the package would have pulled `lxml` with it.

**A renamed archive fails here, and that is by design.** Milestone 2 validates
a `.docx` only as far as its `PK\x03\x04` signature, because proving a file is
really a Word document means opening it — and opening it is parsing. So a
`.zip` renamed to `.docx` is accepted at upload and fails at ingestion with a
recorded stage error. This is the other half of that hand-off.

Missing metadata stays missing. A `.txt` file has no pages, so its page is
null — not 1. Nothing is invented.

### Normalization

One function, twelve rules, each with its own test, because every `checksum`
and every `chunk_uid` is computed over its output:

1. CRLF and CR become LF.
2. Unicode NFC.
3. Categories `Cc` and `Cf` removed, except LF and TAB — which takes out NUL,
   zero-width joiners and bidirectional overrides.
4. Trailing whitespace stripped per line.
5. Runs of blank lines collapse to one.
6. Outer whitespace stripped.

And what it does not do: no lowercasing, no punctuation stripping, no
collapsing of internal spacing, no semantic rewriting. The text stored beside
a citation should be a quotation from the document, not a paraphrase of it.
`normalize(normalize(x)) == normalize(x)`, asserted.

### Chunking

512 units with 64 of overlap, both configurable, with a paragraph boundary
preferred whenever one falls within 20% of the target.

**What a unit is, stated plainly: a whitespace-delimited word.** The
specification asks for a count of "tokens" and names no tokenizer anywhere.
The embedding model's own tokenizer would mean downloading a model asset,
which this milestone has no business doing and which would break the
offline-by-default posture. The settings keep the specification's `*_tokens`
names.

> **Known risk, recorded rather than solved.** A whitespace word is longer
> than a subword token, so 512 words may exceed the embedding model's input
> window in milestone 4 and be silently truncated. This milestone does not
> attempt to fix that. Changing the unit later changes every chunk boundary
> and therefore every `chunk_uid`, which is a full re-index.

Offsets are half-open `[char_start, char_end)` into the **complete normalized
document text**, so `document_text[char_start:char_end]` is exactly the chunk.
Not the raw parser output, and not chunk-relative.

**A chunk that spans pages records the page it started on.** The column holds
one page, the specification asks for no more, and an array would be inventing
schema. The consequence is real and worth knowing before citations are built
on it in milestone 8: a citation may name where a chunk begins rather than
every page it touches.

### `chunk_uid`

```python
sha256(f"{document_version_id}:{sequence}:{normalized_text}".encode("utf-8")).hexdigest()[:32]
```

Lower-case hyphenated UUID, plain decimal sequence, literal `:`, UTF-8,
lower-case hex, first 32 characters. Page, section, document id and offsets
are **not** inputs — a corrected page leaves a chunk's identity alone.

This is what makes re-indexing idempotent: the identifier is derived from
content rather than allocated, so a re-run writes rows identical to the ones
it replaced instead of duplicates. A golden-value test pins the exact string,
computed outside Python with `sha256sum` so it checks the implementation
rather than restating it.

### The runner

APScheduler polling, started and stopped by the application's lifespan and
held on the application rather than in a module-level variable.

Polling rather than a background task attached to the upload request, because
a job table exists so that work survives: a background task cannot retry after
a restart and cannot touch jobs queued before the process started — and
milestone 2 left real queued rows behind.

A job is claimed with `SELECT … FOR UPDATE SKIP LOCKED`: a row another worker
has already taken is skipped rather than waited for, so the design is safe if
it is ever run in more than one process, with no broker and no lock table. A
test proves it with two genuine database connections.

**The claim commits before the work begins.** `attempts` is what bounds
retrying, so it has to survive the rollback a failure performs — counted
inside the work's own transaction it would be undone by every failure, and a
job that always failed would be retried for ever.

### Retries

Every try increments `attempts`. Below `max_attempts` a failed job returns to
`queued` and is picked up again; at the bound it stops at `failed`. No
backoff and no retryable/non-retryable classification — the specification asks
for retries bounded by `attempts` and nothing more. The bound is configuration,
not a constant.

A stage error names the stage and the failure class. It never carries document
content, a storage path or a traceback, and a test asserts each.

### Where milestone 3 stops

A successful job ends at **`indexing`**, and the version stays a `draft`.

Parsing and chunking are complete; writing embeddings and a full-text vector
is milestone 4's work, so the job rests at the stage that owns it — exactly as
milestone 2's jobs rested at `queued` before this runner existed. `ready` is
not claimed before indexing has happened, and the invariant **active ⇒
indexed** holds because nothing is promoted at all.

### Known limitations

- **Chunk size is measured in whitespace words**, with the milestone-4
  truncation risk described above.
- **Nothing bounds DOCX decompression.** Milestone 2 caps the size of an
  upload, not the size of what it expands to, so a crafted archive could
  expand further than intended. A decompression limit was deliberately left
  out of this milestone's scope.
- **No OCR.** A scanned PDF with no text layer produces no text, and its job
  fails rather than succeeding with nothing.
- **A chunk that spans pages records only its first page.**
- **Re-running a job was available in code, not over HTTP.** The endpoint
  (`POST /documents/{id}/reindex`) arrived with milestone 4, below.

## What milestone 4 built

Embeddings and full-text vectors — the specification's *index writes*. A
chunked version becomes searchable, and only then becomes `active`.

```
queued → parsing → chunking → indexing → ready     ← milestone 4 finishes it
              ↓ (any stage)                          draft → active
            failed, with stage_error
```

- `app/providers/embeddings.py` — the `EmbeddingProvider` interface, the fixed
  `DIMENSIONS = 384`, and the shape check every implementation runs.
- `app/providers/bge.py` — `BAAI/bge-small-en-v1.5`, loaded locally through
  sentence-transformers.
- `app/providers/fake_embeddings.py` — deterministic vectors from a hash, for
  unit tests only.
- `app/indexing/writer.py` — generating the vectors, writing them, and
  promoting the version.
- `app/ingestion/pipeline.py` — `run_indexing`, the stage after chunking.
- `POST /documents/{id}/reindex` in `app/api/documents.py`.
- `chunks.embedding`, `chunks.tsv` and migration 0004.

**No retrieval.** Nothing here searches: hybrid search, reranking and citations
are milestone 5 onwards. This milestone only makes the columns a search would
read.

### The two columns

| Column | Type | Written by |
| --- | --- | --- |
| `embedding` | `vector(384)`, nullable | The indexing stage |
| `tsv` | `tsvector`, `GENERATED ALWAYS AS (to_tsvector('english', text)) STORED` | **PostgreSQL** |

`tsv` is maintained by the database, not by this application, and that is the
whole reason for the choice: it cannot drift from the text it describes and it
cannot fail independently of the row write. There is no trigger to forget and
no backfill to run. Writing to it is an error, and a test asserts the database
refuses.

The two-argument `to_tsvector('english', text)` is used because a generated
column requires an `IMMUTABLE` expression; the one-argument form depends on a
session setting and is only `STABLE`, so PostgreSQL rejects it there.

It is a **PostgreSQL full-text vector, not BM25.** The specification is
explicit that Postgres FTS must never be labelled BM25 in code, comments or
docs — it names `ts_rank_cd` as the ranking function retrieval must use, a
correction this document got wrong until milestone 5 re-checked it against
the specification's own text.

`embedding` is **nullable**, because a chunk exists from the moment it is cut
and is embedded a stage later.

There is deliberately **no index on `embedding`**. The specification says exact
vector search is fast enough at v1 corpus size and that no HNSW index may be
added until a measured latency number justifies one — and no such number can
exist before retrieval and evaluation do. A test asserts no vector index has
crept in. `tsv` has a GIN index, which is a physical decision about a tsvector
rather than a behavioural one.

### Where the transaction starts

The ordering is the design, and it is asserted rather than described:

1. the job is claimed and **the claim is committed** — so `attempts` survives
   what follows;
2. the version's chunk texts are read, and **that read transaction is ended**;
3. the vectors are generated with no transaction open — a test records
   `session.in_transaction()` from inside the provider and asserts `False`;
4. one transaction writes every vector, moves the job to `ready` and promotes
   the version;
5. the caller commits.

Inference is not transactional work. Holding a row lock and a pooled connection
open across a model run would buy nothing and cost both. Index *writes* are
transactional, which is what the specification actually requires: a failure
leaves no vectors, no `ready` job and no promotion — none of it, rather than
some.

`generate_embeddings` takes a list of strings, not rows, so it has nothing it
could reach the database with. That is the signature doing the work an
instruction would otherwise have to.

### Promotion

A version becomes `active` in exactly one place: the indexing transaction, once
its vectors are written. The previous active version becomes `superseded` in
the same transaction, so default retrieval never sees two and never sees none —
the partial unique index would refuse the alternative anyway.

The invariant milestone 3 held vacuously now holds for real: **active implies
indexed.**

### Retries, and what an attempt costs

Milestone 3's whole-job retry is reused unchanged. A failure during indexing
rolls the work back and returns the job to `queued` below `max_attempts`, or
leaves it `failed` at the bound.

A document therefore takes **two attempts** to reach `ready`: one claim parses
and chunks it, a second indexes it. `attempts` counts job attempts rather than
stage attempts, which is milestone 3's locked behaviour, so `max_attempts` must
be at least 2. It is 3 by default.

Two consequences, both observed on a live run rather than reasoned about:

- A job that fails at indexing returns to **`queued`**, so its next attempt
  re-parses and re-chunks before reaching indexing again. Re-chunking is
  deterministic and rewrites the same rows, so this is wasteful rather than
  wrong.
- Because of that, `attempts` can finish one past `max_attempts` — the bound
  is checked when an attempt *fails*, and the intervening parse attempt
  succeeds. With `max_attempts=3` and an embedding provider that cannot load,
  a job fails at attempts 2 and 4 and stops at `failed`. Each failure applies
  the rule exactly as milestone 3 defines it.

### `POST /documents/{id}/reindex`

Queues the **active** version to be run again. It answers `202` with the
document, the version and the new job.

| Case | Answer |
| --- | --- |
| The document has an active version | `202`, a `queued` job |
| No such document | `404` |
| The document has no active version | `409` |
| A job on that version is already outstanding | `409` |

It creates **no new version**. Re-running rewrites the same chunks: because
`chunk_uid` is derived from the version, the sequence and the text, the rows
written are identical to the ones they replaced. Same identifiers, same count,
no duplicates — asserted end to end.

A draft that never finished indexing is the upload path's business, and a
superseded version is history, so neither is a target and a document with
nothing active is a conflict rather than a silent no-op. Two jobs racing on one
version would delete and rewrite each other's chunks, so that is a conflict
too.

### The model

`BAAI/bge-small-en-v1.5`, 384 dimensions, run **locally** through
sentence-transformers. No embedding API, and never Anthropic — which has no
embeddings API and is used for generation only.

The adapter loads with `local_files_only=True` and raises `EmbeddingError`
rather than downloading a model at runtime. Nothing in the application selects
the fake provider; a test asserts that by reading the application's source.

> **The real model has not been exercised in this environment.** The
> sentence-transformers cache does not contain `BAAI/bge-small-en-v1.5`, and
> `huggingface.co` is blocked by this network's proxy (a `403` to `CONNECT`,
> confirmed directly), so the weights could not be fetched. The test that
> loads the real model and checks it produces 384-dimensional vectors
> **skips**, and says why in its skip reason. Every other test in this
> milestone runs against `FakeEmbeddingProvider`, which the specification
> permits for unit tests. Transactions, state transitions, idempotency and
> shapes are therefore proven; **the real model's output is not**. Run
> `pytest tests/test_embeddings.py` on a machine that can reach the model
> cache and that skip becomes a pass.

### Known limitations

- **The real embedding model has not been run here.** See the box above.
- **Keyword search is `ts_rank_cd` over PostgreSQL full-text search, not
  BM25.** The column this milestone builds is a PostgreSQL `tsvector`, and
  the specification is explicit that this must never be called BM25 — an
  earlier version of this document got that backwards; milestone 5 uses
  `ts_rank_cd` exactly as specified.
- **English only.** The generated column names the `english` text search
  configuration, and changing it means a migration.
- **No vector index**, by specification, until a measured latency number
  justifies one.
- **A 512-word chunk is not a 512-token chunk.** BGE truncates at its own
  limit, so the tail of a long chunk may not be represented in its vector. The
  chunker counts whitespace words because no tokenizer is specified anywhere;
  this is what that costs, and it is measurable only once the real model runs.
- **Re-indexing re-parses.** The endpoint queues an ordinary job, which starts
  at `queued` and goes through parsing and chunking again. That is correct
  after a parser fix and wasteful after a model change; a model-only re-embed
  is not in this milestone.

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
  "migrations": {"ok": true, "detail": "at head (3fdbad950293)"},
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
without one (222 pass, 197 skip). With a database: **418 pass, 1 skip** — the
one skip is the real-embedding-model test described above, which cannot fetch
its weights in this environment. No credential is needed either way.

KnowledgeOS is a separate application from DocIntel and VoiceDesk in this
repository: its own package, dependencies, virtualenv, configuration prefix and
database. Nothing is shared between them, and `tests/test_isolation.py` asserts
it.

## Implementation specification

The authoritative build specification is a separate document, worked milestone
by milestone: inspect, propose, implement, test, commit, stop. It is not
committed to this repository.
