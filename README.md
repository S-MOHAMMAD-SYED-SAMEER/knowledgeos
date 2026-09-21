# KnowledgeOS

An internal knowledge system that answers from retrieved evidence — and
measures whether it actually did.

**Status: milestone 10 of 10 — v1 complete.** Documents can be uploaded,
validated, stored, versioned, parsed, chunked, indexed, retrieved, reranked
and answered, with citations, grounding, and abstention, a feedback
endpoint, and a minimal server-rendered UI over all of it. Both evaluation
harnesses (retrieval, milestone 7; answers, milestone 9) exist and are
proven correct against fixtures — **neither has been run officially in this
environment**, because the required real local models are absent from its
model cache, and no Docker image has been built here either (the Docker
daemon itself is unavailable in this sandbox). **No benchmark figure
appears in this README until it is real, and no Docker build/runtime claim
appears until one has actually run** — see "What milestone 9 built" and
"What milestone 10 built" below for the exact blocking reasons.

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
- **Server-rendered UI** — Jinja2, no build step: document list, document
  detail with version history, a query box, an answer page with citations and
  retrieved evidence side by side, and an evaluation results page.
- **Feedback** — `POST /answers/{id}/feedback` records a helpful/not-helpful
  rating and an optional reason against a generated answer.
- **Deployment** — a Dockerfile, `docker-compose.yml`, and a CI workflow that
  runs the whole suite with no external credential.

## What it deliberately is not

Not a chatbot platform, Slack/Teams bot, agent framework, or SSO-enabled
enterprise product. No Pinecone, Weaviate, Qdrant, Elasticsearch, Redis,
Celery, Kubernetes, or React. **Postgres is the only datastore.**

## Stack

Python 3.13 · FastAPI · SQLAlchemy 2.x · PostgreSQL 16 + pgvector · psycopg 3 ·
Alembic · sentence-transformers (local embeddings and reranking) · Google
Gemini (`google-genai`) behind a provider interface · Jinja2 (server-rendered
UI) · pytest · Docker

> **Generation provider deviation, stated plainly.** The specification's own
> stack list and milestone 8 scope line name "Anthropic SDK" / "Anthropic
> generation". This build uses **Google Gemini** instead — an explicit,
> authorized, documented deviation from that wording, not from the
> specification's behavioural requirements for generation (§9), none of
> which name a vendor. See "What milestone 8 built" below for the full
> explanation and what stayed provider-agnostic as a result.

Embeddings and reranking run locally. The only paid dependency is answer
generation, and the entire evaluation suite runs without it.

Dependencies arrive with the milestone that first imports them, so an install
never carries a library the code does not yet use. Today that is FastAPI,
Uvicorn, Pydantic, pydantic-settings, SQLAlchemy, Alembic, psycopg,
python-multipart, `pypdf`, APScheduler, sentence-transformers, `PyYAML`,
`pgvector`, `google-genai` and, as of milestone 10, `jinja2` — already
present transitively since milestone 4 (torch depends on it), now finally
declared on purpose because `app/ui/` actually imports it. The Anthropic
SDK is not installed at all — see the deviation note above; DOCX is read with
the standard library, so `python-docx` is not installed either.

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

**Results.** No retrieval or answer evaluation number appears anywhere in
this document. Both harnesses exist and are proven correct against
fixtures with fake providers (a pytest-only allowance); neither has run
officially, because the real local models both require are absent from
this environment's model cache. See "What milestone 9 built" below for
the exact blocking reason and what "proven correct" does and does not
mean here.

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

**Milestones 1 to 6 of 10 are implemented.** Everything above this line
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

## What milestone 6 built

Reranking. Milestone 5's top-20 fused candidates get one more pass — a
cross-encoder scores each `(query, chunk text)` pair directly, rather than
comparing two independently-computed vectors the way retrieval does — and
`/query` returns the reranked order instead of the fusion order.

```
POST /query
  → normalize → embed → vector retrieval
                       + lexical retrieval  → RRF → dedupe → top 20
  → rerank (this milestone)
  → evidence, reranked                                  ← no answer, still
```

- `app/providers/reranker.py` — the `RerankProvider` interface: `rerank(query,
  candidates) -> list[ScoredChunk]`, exactly as specified, built around plain
  `Candidate(id, text)` pairs rather than an ORM row or `app.retrieval`'s own
  types.
- `app/providers/cross_encoder.py` — `CrossEncoderRerankProvider`,
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, loaded locally through
  `sentence_transformers.CrossEncoder`. The specification's default.
- `app/providers/passthrough_reranker.py` — `PassthroughRerankProvider`,
  preserving whatever order it is given. Named explicitly by the
  specification as a shipped configuration — "so the pipeline is testable
  without the model and so evals can measure what reranking actually adds"
  — not a test-only convenience.
- `app/providers/fake_reranker.py` — a deterministic, content-derived scorer
  for tests only. Passthrough provably never reorders anything by
  construction, which makes it useless for proving the reranking stage
  *can* reorder; this exists for exactly that.
- `app/reranking/` — the orchestration. `pipeline.py::rerank()` maps
  milestone 5's candidates to `Candidate` pairs, calls whichever provider is
  configured, sorts by score descending with `chunk_uid` ascending as the
  tie-break, and assigns `final_rank` 1..N. `provider.py` holds the cached
  accessor (`get_rerank_provider()`, mirroring
  `app.ingestion.runner.get_embedding_provider()` exactly), always resolving
  to the real cross-encoder in production.

**No new dependency.** `sentence-transformers` already provides
`CrossEncoder` alongside `SentenceTransformer`; both were declared in
milestone 4.

**No migration, no new table, no new endpoint.** `queries` and
`retrieved_chunks` remain deferred — the same locked decision milestone 5
made, extended rather than revisited. `/query` is still the only route this
adds to; nothing this milestone does needed a fifth migration.

### `app/retrieval/` stayed provider-free

The specification's layering rule names `app/retrieval/`, `app/chunking/`
and `app/generation/citations.py` as the packages that may hold no provider
calls. It does not name `app/reranking/` — which is exactly what lets
`app/reranking/pipeline.py` call a `RerankProvider` while every file under
`app/retrieval/` stays exactly as milestone 5 left it. Nothing in this
milestone touches `vector.py`, `lexical.py`, `fusion.py`, `filters.py`, or
`dedupe.py`; the full milestone 5 test suite passes unmodified, and a test
asserts `app/retrieval/` still imports no reranking provider.

### `final_rank` changed meaning; `fusion_rank` is what it used to mean

Milestone 5's `final_rank` was the position after RRF fusion — the last
stage that existed. Once a reranking stage exists, "final" has to mean the
rank a caller actually sees last, so `final_rank` in the API response is now
the **post-rerank** position, and the pre-rerank position that milestone 5
called `final_rank` is exposed as **`fusion_rank`** — never silently
overloaded onto the same field name. `RetrievedChunk.final_rank` (milestone
5's own internal type, in `app/retrieval/pipeline.py`) is untouched; the
rename happens only where `app/reranking/pipeline.py` reads it into
`RerankedChunk.fusion_rank`.

### Score direction and ties

Higher `rerank_score` is more relevant — the standard convention for this
model family and for `CrossEncoder.predict()` generally, though the
specification itself does not state a direction. Candidates are sorted by
score descending; equal scores (the passthrough's synthetic scores, or two
genuinely tied real scores) fall back to `chunk_uid` ascending, the same
tie-break rule milestone 5 already uses in both its own channels.

### Evidence selection

The specification names "evidence selection" as part of this milestone's
scope but gives no rule anywhere for it — no threshold, no top-N. The one
threshold the specification does describe (§9, abstention) is explicitly
calibrated on evaluation data that will not exist until milestone 7 has run.
Rather than invent a number the specification withholds, milestone 6 does
not add a `selected` field or a selection cutoff at all: **the reranked
candidate set itself is the returned evidence.** A real selection rule,
grounded in calibration data, is milestone 8's problem when it exists to
solve.

### Failure is loud, never silent

If the real cross-encoder cannot be loaded from the local cache, or
inference fails, `/query` answers `503` — the same convention milestone 5
already established for the embedding provider. It never falls back to the
passthrough, to milestone 5's unreranked order, or to another model; a
silently degraded reranker would be indistinguishable, from the outside,
from a working one. Reranking is skipped — not attempted and not faked —
when retrieval found nothing to score.

> **The real cross-encoder has not been exercised in this environment.**
> `cross-encoder/ms-marco-MiniLM-L-6-v2` is not in the local
> sentence-transformers cache, and `huggingface.co` is blocked by this
> network's proxy — the same, unchanged condition milestone 4 and 5 already
> documented for the embedding model. `tests/test_reranking.py`'s one
> real-model test skips, and says why. Every other reranking test uses the
> passthrough or the deterministic fake, which the specification permits.
> The live smoke test below confirms the unmodified endpoint reaches for
> the real provider and answers `503` — never a silent substitution — and
> separately proves, through explicit dependency injection, that the
> reranking stage itself can reorder candidates.

### Known limitations

- **The real reranking model has not been run here.** See the box above.
- **No `selected` field, and no persisted `retrieved_chunks`.** Both remain
  deferred; see "Evidence selection" above.
- **A long chunk may be truncated by the cross-encoder's own tokenizer**,
  for the same reason a long chunk may be truncated by BGE's: chunk size is
  measured in whitespace words, not the model's real subword tokens.

## What milestone 7 built

The retrieval evaluation harness. Everything before this milestone answers
"does retrieval work at all?" by reading code and a handful of hand-picked
examples; this milestone answers it with a fixed corpus, a frozen question
set, and six numbers, on demand rather than by feel.

```
python -m evals.run --suite retrieval
```

is the only way this ever runs. There is no `/evals/run` HTTP endpoint —
evaluation is an offline, operator-run process, never something the running
application exposes to a caller — and it never touches generation, answers,
citations, or abstention: those are milestone 8/9's evaluation, once
milestone 8/9 exist to have something to evaluate.

**What it measures.** Six retrieval metrics, computed over the specification's
eight question categories (directly answerable, multi-document, ambiguous,
insufficient evidence, conflicting versions, metadata-filtered,
citation-sensitive, adversarial):

- **Recall@5**, **Recall@10** — `|expected ∩ returned@k| / |expected|`.
- **Precision@5** — `|expected ∩ returned@5| / 5`, a fixed denominator.
- **MRR** — `1 / rank` of the first relevant result over the full returned
  top-20 list, `0` if none.
- **nDCG@10** — binary relevance, the standard `DCG@10 / IDCG@10` formula.
- **Metadata-filter correctness** — the fraction of filtered questions for
  which zero returned candidates violate the requested filter.

A question with an empty `expected_chunk_uids` (the insufficient-evidence
category) is excluded from the first four means rather than counted as a
zero — `evals/retrieval/metrics.py` refuses to compute any of them over an
empty expected set — and the excluded count is reported alongside the
aggregates, never folded silently into them.

**Milestone 5 vs. milestone 6, compared honestly.** Every question is
retrieved exactly once — `retrieve()` runs a single time per question — and
that identical pre-reranking top-20 candidate list is then reranked twice:
once by `PassthroughRerankProvider` (milestone 5's own fusion order,
preserved) and once by `CrossEncoderRerankProvider` (milestone 6's real
model). Reranking only ever reorders that one list; it cannot add or remove
a candidate, so the two conditions' metrics differ because of reordering
alone, never because of a different retrieval. Metadata-filter correctness
is reported once, not per condition, since it depends only on which chunks
were retrieved, not on the order reranking put them in.

**The fixture corpus** (`evals/fixtures/knowledge_base/`) is ten policy and
runbook documents (one of them, the Production Database Access SOP, with a
second version that materially changes the answer — three-day grants and
two approvers, instead of the first version's seven days and one approver),
seeded through the real upload → parse → chunk → embed pipeline, never a
shortcut. `evals/fixtures/knowledge_base/manifest.yaml` pins every document
and version UUID, and the chunk configuration (512 tokens, 64 overlap) that
the frozen question set's `chunk_uid`s were generated under — `chunk_uid`
is derived from the version id and sequence (`app/chunking/uid.py`), so an
unpinned id would silently invalidate every expected answer the next time
the corpus was seeded.

**The question set** (`evals/fixtures/questions/*.yaml`, 52 questions
across the eight categories) carries exactly six fields per question — `id`,
`text`, `category`, `expected_chunk_uids`, `expected_abstain`, `filters` —
enforced by `evals/retrieval/questions.py`. Every `expected_chunk_uid` was
read back from the real corpus after seeding it, never invented by hand;
several questions list more than one expected chunk where the fixture
corpus's own 64-token chunk overlap, or a fact genuinely stated in two
different documents, means more than one chunk honestly contains the
answer.

**Persistence.** A successful run writes one row to `eval_runs` (`id`,
`suite`, `prompt_version` — null for this suite, which has no prompt —
`config`, `metrics`, `created_at`; migration `0005`) in the same configured
KnowledgeOS database every other part of the application uses, never a
second `knowledgeos_evals`-style database, plus a timestamped JSON artifact
under `var/eval_runs/` and a console summary — all three built from the same
result, so they can never disagree.

**Official evaluation requires the real local models — no exceptions.**
`BAAI/bge-small-en-v1.5` for embedding and
`cross-encoder/ms-marco-MiniLM-L-6-v2` for the cross-encoder condition, both
read from the local sentence-transformers cache exactly as milestones 4 and
6 already require. If either is unavailable, `python -m evals.run` exits
non-zero with a structural message and writes **nothing** — no `eval_runs`
row, no JSON artifact, no printed metrics — checked with a canary call
against each model before the fixture corpus is touched at all, so a failed
run's database footprint is exactly zero. It never substitutes a fake
provider to produce a number anyway; the specification's "fakes only inside
pytest tests" allowance is exactly that, and `evals/run.py` never reaches
for one.

> **Official evaluation has not been run in this environment.** Both
> required local models are absent from the sentence-transformers cache —
> the same, unchanged condition milestones 4 and 6 already documented for
> the embedding and reranking models individually — so `python -m evals.run
> --suite retrieval` fails at its canary check, exactly as designed. No
> retrieval metric numbers appear anywhere in this document, and no gate
> from the specification has been claimed as passed: the harness has been
> proven correct (`tests/test_evals_*.py`, run against the real fixture
> corpus and database with deterministic fakes standing in for the two
> models, per the specification's own pytest-only allowance), not run
> officially.

### Known limitations

- **The real embedding and cross-encoder models have not been run
  officially here.** See the box above.
- **No answer, citation, or abstention evaluation.** Milestone 7 evaluates
  retrieval only; those require milestone 8/9's generation to exist first.
- **Metadata-filter correctness only checks presence, not recall.** A
  question whose retriever returned zero candidates trivially violates
  nothing and scores as "correct" — the metric measures whether filtering
  leaked the wrong department or category into the result, not whether
  filtering found the right one.

## What milestone 8 built

Generation. `POST /query` now answers the question, with inline citations,
deterministic validation, and an honest abstention when the evidence does
not support one — finishing what milestone 5 started ("`/query` returning
evidence only, no answer") rather than adding a second endpoint next to it.

```
POST /query
  → normalize → embed → vector retrieval
                       + lexical retrieval  → RRF → dedupe → top 20
  → rerank (milestone 6)
  → select top 8 by final_rank              ← this milestone
  → abstain (zero candidates, or a calibrated rerank-score threshold)
  → generate → parse → validate citations → ground
  → persist queries / retrieved_chunks / answers, one transaction
  → evidence + grounded, cited answer (or an abstention) — still 200
```

### The generation provider is Google Gemini, not Anthropic — stated plainly

The specification's stack list (§3) and its milestone 8 scope line (§17)
name "Anthropic SDK" / "Anthropic generation". This build uses **Google
Gemini** (`google-genai`) instead. This is an explicit, authorized,
documented **deviation from that wording** — not from the specification's
*behavioural* requirements for generation. §9, the section that actually
specifies what generation must do (a versioned prompt, structured output,
three citation rules, two grounding layers, abstention on two triggers),
names no vendor anywhere. Every one of those requirements is implemented
exactly as specified, unchanged by which vendor answers the request.

What stayed provider-agnostic as a result:

- `app/providers/llm.py` — `LLMProvider.complete(*, system, user, max_tokens)
  -> LLMResult(text, input_tokens, output_tokens)`. Nothing in this
  interface, or in anything that calls it, is Anthropic- or Gemini-shaped.
  It returns **raw text plus token usage**, never a parsed answer — parsing
  and validating the model's structured output happens in
  `app/generation/generator.py`, in Python, exactly as the specification's
  own words require ("Parse and validate in Python"), which is also what
  makes the mandatory malformed/invalid-citation test fixtures testable at
  all.
- `app/generation/` — the entire orchestration package (evidence selection,
  prompt loading, citation validation, grounding, abstention, persistence)
  is written against `LLMProvider` alone and does not import
  `google.genai` anywhere.
- `app/providers/gemini_llm.py` — the **only** file in this codebase that
  imports the Gemini SDK, and it does so lazily, inside the method that
  actually calls it, the same discipline `bge.py` and `cross_encoder.py`
  already follow for their own local models.

An Anthropic adapter is not implemented. It is **deferred**, the same way
the specification's own §1 defers a Voyage embeddings adapter without
building one — a second `LLMProvider` implementation could be added later
without touching `app/generation/` at all.

### Why Gemini, specifically

No `ANTHROPIC_API_KEY` is available to this project, while a Gemini
credential is available to whoever operates it. Nothing about the
provider choice is architectural: `LLMProvider` was designed so that
whichever credential is actually available decides which adapter is live,
without changing anything in `app/generation/`.

### The model ID was never chosen from memory

`Settings.llm_model` (env `KNOWLEDGEOS_LLM_MODEL`) has **no default**. At
the time `app/providers/gemini_llm.py` was written, no Gemini credential
was available in the build environment to verify a model against the live
API (`client.models.list()`), so none was guessed. An operator who has
verified one sets it; `GeminiLLMProvider.complete()` raises a structural
`LLMError` — never a fabricated model name — if it is unset when generation
is attempted.

### Evidence selection: top 8, fixed, no threshold

The specification names no selection rule — no top-N, no score threshold,
no token budget — for which of milestone 6's up-to-20 reranked candidates
reach the model. This project's own, locked, documented choice
(`app/generation/evidence.py::SELECTION_LIMIT`): the **top 8** by
`final_rank`, fixed rather than configurable, the same reasoning the
specification's own top-50/top-20 retrieval limits are facts rather than
settings. The other 12 are still retrieved, still recorded in
`retrieved_chunks.selected = false`, and a citation to one of them is
invalid (citation rule 3) — evaluated, not merely omitted from the prompt
by convenience.

### Citations: inline markers, a flat declared list, and three exact rules

The specification's output format gives `citations` as one flat list of
chunk_uid for the whole answer, with no per-sentence structure. That alone
cannot support rule 2 ("every sentence containing a factual claim carries
at least one citation") or the semantic layer's "per-sentence entailment
against *its* cited chunks" — there is no *its* without a sentence-level
mapping. This project's locked resolution: the model is instructed to place
inline `[<chunk_uid>]` markers on every factual sentence, and the declared
`citations` list must exactly equal the union of markers actually present
in the answer text. A mismatch is itself an invalid citation.

The three specification rules, plus that agreement check, are all
deterministic and live in `app/generation/citations.py` — which makes **no
provider call and performs no I/O beyond nothing at all** (not even the
database), exactly matching the specification's layering rule (§4), which
names this file specifically alongside `app/retrieval/` and
`app/chunking/`:

1. Every cited chunk_uid exists in the retrieved evidence set. If not,
   invalid citation.
2. Every sentence carries at least one citation marker — checked by
   requiring every sentence in a non-abstained answer to carry one; the
   specification introduces "substantive factual sentence" without
   defining it, and this project's own choice is exactness over an
   undefined judgment call (see the module's own docstring for the full
   reasoning).
3. Citations pointing to non-selected chunks are invalid.

**"Answer rejected" means exactly that.** A citation violation raises
`CitationError`, which `POST /query` maps to `502` — nothing is persisted.
Every `answers` row this milestone ever writes has `citation_valid = true`
by construction, because an invalid answer never reaches persistence at
all. `app/generation/generator.py`'s own docstring states this plainly,
including the consequence for what the column means in practice.

### Grounding: two layers, and the semantic one is honestly unexercised

Deterministic — citation validity, citation coverage, selected-chunk
validity, abstention-flag consistency — runs on every request and is
exact by construction: it is a second, independent recomputation of the
same checks the citation gate already enforced, the same defense-in-depth
instinct `evals/retrieval/metrics.py` applies to its own
structurally-unreachable nDCG guard.

Semantic — per-sentence entailment against cited chunks, using the local
cross-encoder as an NLI-style scorer — is **not implemented in this
milestone**. `cross-encoder/ms-marco-MiniLM-L-6-v2` is absent from the
local model cache in this environment, the same, unchanged condition
milestones 4, 6 and 7 already documented. `grounding_detail.semantic` is
always `{"status": "unavailable", "reason": ...}` — never a number, and
never the reranker's own relevance score reused as if it were an entailment
signal, which the specification permits as a *third* signal but never as
*the* signal and never authorizes substituting for the real one.

### Abstention: both triggers, and the threshold was never invented

Two independent triggers, per the specification: the top rerank score
falling below a threshold, or the model's own `sufficient_evidence=false`.
A third, structurally necessary case this project adds: **zero retrieved
candidates never reach the model at all** — there is nothing to send.

`KNOWLEDGEOS_ABSTENTION_RERANK_THRESHOLD` has **no default** —
`app/generation/abstention.py::DEFAULT_ABSTENTION_RERANK_THRESHOLD` is
`None`, meaning the score-based trigger is inactive. The specification is
explicit that this threshold "is calibrated on a dev split of the eval
questions, not chosen by taste," and calibrating one honestly needs the
real local cross-encoder's scores over a genuine dev/test split of
milestone 7's question set — neither of which exists in this environment
(the model is absent; milestone 7's 52 questions were never split). Rather
than choose a number "by taste," which the specification forbids by name,
this milestone ships the mechanism with the gate inactive and states so
here, plainly, rather than quietly.

### Persistence: `queries`, `retrieved_chunks`, `answers` — migration 0006

The specification's own three tables (§5), created together in one
migration since none of them had anything to write until now. `queries`
carries what this milestone genuinely produces — `model`, `prompt_version`,
`input_tokens`, `output_tokens` — and creates the four `*_ms` timing
columns and `cost_usd` **nullable**, left `NULL` in every row this
milestone writes: the specification requires unknown pricing to raise a
config error rather than silently become zero (§14), and this milestone
measures no stage latency at all. `retrieved_chunks` has a composite
primary key `(query_id, chunk_id)` — the specification gives it no
surrogate id, and a query's chunk is already a natural key. `answers` is
unique on `query_id`: the specification states no explicit rule, but "one
answer belongs to the query" is this project's own locked invariant, and a
retried request writing a second row would silently turn "the answer" into
an arbitrary one.

`chunk_id` is resolved from `chunk_uid` by one indexed lookup at
persistence time (`app/generation/persistence.py`) rather than carried on
milestone 5's own `ChunkEvidence`, which stays locked and untouched.

### Transaction boundaries

Retrieval and reranking read the database under ordinary `READ COMMITTED`,
same as milestone 5 left it. The Gemini call happens with **no database
transaction open** — a slow external call has no business holding a
connection. Persistence is a single transaction at the very end, after
generation has already succeeded and citations have already validated: a
failed generation, a parsing failure, or a rejected citation leaves **no**
`queries`, `retrieved_chunks`, or `answers` row behind — there is nothing
partial to clean up, because nothing was ever written.

### Errors

`503` for a provider that could not be reached — embedding, reranking, or
generation, the same convention milestones 5 and 6 already established.
`502` for a model response that parsed as something other than the
required structure, or whose citations failed validation — distinct from
`503` so "the vendor is down" and "the vendor answered with something we
cannot use" are distinguishable in logs. `200` for an abstention, always —
insufficient evidence is a real, valid answer at this layer, never a
failure. No response or log line ever carries an API key, the prompt, the
retrieved chunk text, the model's raw output, a traceback, or a file path.

> **The real Gemini model has not been exercised in this environment
> either.** No `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) was set in the
> environment this milestone was built in — the same, unchanged condition
> milestones 4, 6 and 7 already documented for their own real models, now
> true of this one too. `tests/test_llm_provider.py`'s one real-model
> test is opt-in (`KNOWLEDGEOS_RUN_GENERATION_SMOKE_TEST=1`, plus a
> genuine credential and a `KNOWLEDGEOS_LLM_MODEL` already verified
> against `client.models.list()`) and skips here, honestly, rather than
> being silently bypassed. Every other generation test uses the scripted
> fake or a deterministic, content-derived test double, which the
> specification permits. No retrieval or answer **evaluation** number
> appears anywhere in this document: milestone 8 implements generation, it
> does not run milestone 9's answer eval suite.

### Known limitations

- **Semantic grounding has not been implemented.** See "Grounding" above —
  the local cross-encoder NLI scorer this needs is absent from this
  environment's model cache.
- **The abstention rerank-score threshold has not been calibrated.** See
  "Abstention" above; only `sufficient_evidence=false` and the
  zero-candidate case can trigger an abstention here.
- **No answer evaluation.** Grounded-answer rate, unsupported-claim rate,
  citation precision/recall, and the specification's answer-side
  acceptance gates are milestone 9's `evals/answers/` — not built here.
- **No cost or latency tracking.** `cost_usd` and the four `*_ms` columns
  exist in the schema (§5) and are `NULL` in every row; computing them is
  milestone 9's observability work.
- **An Anthropic adapter is not implemented.** Deferred, not missing by
  oversight — see "The generation provider is Google Gemini" above.

## What milestone 9 built

The answer evaluation harness. Milestone 7 measured whether retrieval found
the right evidence; this milestone measures whether milestone 8's generation
actually used it — grounded-answer rate, citation validity and coverage,
abstention precision/recall, semantic grounding, latency, tokens, and cost —
over the same frozen corpus and the same 52 questions, run through the real
retrieve → rerank → generate pipeline, never a second implementation of any
of the three.

```
python -m evals.run --suite answers
```

is the only way this ever runs, extending milestone 7's own CLI rather than
adding a second entry point — there is still no `/evals/run` HTTP endpoint.
Every question in the frozen set is evaluated, not just one half of the
split; the dev/test split exists specifically for calibration (see below),
because nothing in the specification scopes the answer *metrics* themselves
to a split, and milestone 7's own retrieval suite already evaluates its
whole frozen set the same way.

**Official evaluation requires three real providers, not two.** Embedding
and reranking, the same two milestone 7 already requires, plus Google
Gemini for generation. All three are canary-checked before anything is
written — the corpus is never even seeded if one is missing — the same
zero-footprint-on-failure discipline milestone 7 established.

### The frozen dev/test split — milestone 7's questions, never modified

`evals/fixtures/splits.yaml` assigns each of the 52 existing question ids to
`dev` (29) or `test` (23), stratified by `(category, expected_abstain)` so
both halves see every category and both classes of question, and generated
once, checked in, and never re-stratified afterwards. No question's `id`,
`text`, `category`, `expected_chunk_uids`, `expected_abstain`, or fixture
corpus changed to produce it. `evals/answers/splits.py` refuses to run at
all — raising, not silently drifting — the moment the live question set and
the frozen split disagree about which ids exist.

**Stated plainly: this split is too small for statistical confidence.**
Only 7 of the 52 questions have `expected_abstain: true` (4 land in `dev`,
3 in `test`). A recall figure computed over 3-4 positives is real evidence,
not proof of generalization — `evals/answers/calibration.py::SAMPLE_SIZE_LIMITATION`
carries this exact caveat into every calibration result and every report,
rather than letting a clean-looking recall number imply more precision than
the sample supports.

### Abstention calibration — a documented method, never a number chosen by taste

The specification's own words: the abstention threshold "is calibrated on a
dev split of the eval questions, not chosen by taste." `evals/answers/calibration.py`
implements that literally: `collect_dev_observations` runs the real
retrieve → rerank pipeline for every `dev`-split question and records its
top rerank score (or `None`, for zero retrieved candidates, which always
abstains regardless of threshold); `calibrate_threshold` then sweeps every
observed score as a candidate cutoff and picks the one that **maximizes
dev-split abstention recall**, ties broken by precision, ties broken by the
lowest surviving threshold, for a result that never depends on iteration
order. The sweep objective itself — maximize recall — is this project's own
documented choice: the specification states the *procedure* ("calibrated on
a dev split") but gives no formula for what "calibrated" optimizes, and
recall is what the specification's own abstention gate (§18: "≥ 90% recall
on expected-abstain questions") measures directly.

`KNOWLEDGEOS_ABSTENTION_RERANK_THRESHOLD` still has no default. Calibration
only ever runs as part of an official `--suite answers` run, against the
real embedding and reranking models this environment does not have — so no
calibrated threshold exists here, and none was invented to make the gate
read green. `Settings.abstention_rerank_threshold` stays `None`.

### Semantic grounding — the local cross-encoder as an NLI-style scorer, and the threshold that was never invented

Milestone 8 shipped deterministic grounding only, with
`grounding_detail.semantic` permanently `{"status": "unavailable"}` in the
serving path — that stays true forever; semantic grounding is an
**evaluation-time** computation, never added to `POST /query` itself.
`evals/answers/semantic.py` implements the specification's own words: "per-
sentence entailment against its cited chunks, using the local cross-encoder
as an NLI-style scorer. This is approximate." It reuses
`cross-encoder/ms-marco-MiniLM-L-6-v2` — the identical model
`app/providers/cross_encoder.py` already wraps for reranking — for a
different purpose, entailment-style scoring of one sentence against its own
cited chunk text, never the reranker's own relevance score treated as if it
were an entailment signal.

**No unsupported-claim threshold is invented.** `cross-encoder/ms-marco-MiniLM-L-6-v2`
produces an unbounded relevance logit, not a probability — there is no
principled absolute cutoff ("this score means supported") without
empirically calibrating one against real model output, which this build
cannot do (the model is unavailable here). The authoritative specification
was searched in full for a formula, threshold, or worked example for
"unsupported claim" and states none anywhere — this is a genuine
specification gap, not an oversight. Rather than pick a number "by taste"
(the same discipline the specification states explicitly for the
abstention threshold, applied here to an equally undefined one),
`DEFAULT_UNSUPPORTED_CLAIM_THRESHOLD` is `None`: every cited sentence's
**raw** entailment score is still computed and reported whenever the
scorer is available — real, honest data — but the binary supported/
unsupported classification, and therefore `unsupported_claim_rate`, stays
`None` until a threshold has actually been calibrated and documented. This
is an explicit, project-level decision, recorded here because the
specification does not make it.

Deterministic grounding and semantic grounding are kept as two separate
reported layers, per the specification's own "kept distinct and reported
separately" — one never substitutes for the other, and neither is averaged
into a single combined number.

### Observability — stage timing, token accounting, configurable pricing

`app/observability/timing.py` times three stages independently
(`retrieval_ms` — embed plus retrieve, as one measured stage — `rerank_ms`,
`llm_ms`) with a context manager whose `finally` block records elapsed time
even when the stage raised, so a rejected answer's real retrieval and
reranking time is still reported rather than left blank. `total_ms` is
their sum. `app/api/query.py` (`POST /query`) and `evals/answers/suite.py`
(the evaluation suite) both instrument the identical three stages the same
way and write through the identical `persist_query` parameters — one
measurement discipline, not two.

`app/observability/pricing.py` implements the specification's own words
literally: "unknown pricing raises a config error — it must never silently
become zero." `Settings.llm_pricing_usd_per_million_tokens`
(`KNOWLEDGEOS_LLM_PRICING_USD_PER_MILLION_TOKENS`) defaults to an **empty**
mapping — no model's rate is pre-entered, because none has been verified
against a real vendor pricing page from this environment. Computing a cost
for an unconfigured model raises `PricingError`; `POST /query` catches that
one error and leaves `cost_usd` `NULL` for that query rather than failing
the request over unset pricing, while the evaluation suite's own cost
report refuses to publish a cost figure at all when pricing is unconfigured
(see the integrity box below) — the same fact, handled two different ways
for two different audiences, one live request and one official report.

### Persistence and reproducibility

`app/models/query.py`'s five observability columns
(`retrieval_ms`/`rerank_ms`/`llm_ms`/`total_ms`/`cost_usd`) already existed,
nullable, from milestone 8's own migration — **no new migration was needed
for this milestone**. An official answer-evaluation run writes one
`eval_runs` row (`suite="answers"`, `prompt_version` from the versioned
prompt file, `config` carrying every pinned setting plus the calibration
result, `metrics` carrying every computed figure) to the same configured
KnowledgeOS database every other part of the application uses, plus a
timestamped JSON artifact under `var/eval_runs/` and a console summary —
all three built from the same result object, so they can never disagree —
and persists every successful attempt through milestone 8's own,
unmodified `persist_query`, leaving the identical `queries`/
`retrieved_chunks`/`answers` trail a live request would, readable back
through `GET /queries/{id}`.

> **Official answer evaluation has not been run in this environment.**
> `python -m evals.run --suite answers` was attempted here and failed at
> its canary check with: *"the embedding model (BAAI/bge-small-en-v1.5) is
> unavailable: the embedding model BAAI/bge-small-en-v1.5 could not be
> loaded from the local cache (OSError)"* — the same, unchanged condition
> milestones 4, 6, 7 and 8 already documented, now blocking this milestone's
> generation-side evaluation too, before the Gemini credential (also unset
> here) is ever reached. The run wrote **nothing**: no `eval_runs` row, no
> JSON artifact, no printed metrics, verified directly against the
> database and the filesystem after the attempt.
>
> What *has* been done, and what it does and does not prove:
> - **Structural / unit tests** (`tests/test_observability_timing.py`,
>   `tests/test_pricing.py`, `tests/test_semantic_grounding.py`,
>   `tests/test_answer_metrics.py`, `tests/test_split_integrity.py`) prove
>   the pure functions — percentiles, pricing, semantic scoring shape,
>   metric arithmetic, split integrity — are correct in isolation. No
>   model, no database.
> - **Deterministic harness verification** (`tests/test_calibration.py`,
>   `tests/test_answers_suite.py`, most of `tests/test_answers_cli.py`)
>   runs the real database, the real fixture corpus, and the real
>   retrieve → rerank → generate wiring, with `FakeEmbeddingProvider`,
>   `FakeRerankProvider`, and a reactive citing test double standing in for
>   the three real models — the specification's own pytest-only allowance.
>   This proves the harness's *plumbing* is correct: every question
>   produces one attempt, rejected attempts are excluded from persistence,
>   timing is threaded through, the CLI writes one `eval_runs` row and one
>   report. **None of the numbers these tests produce are grounded-answer
>   rate, unsupported-claim rate, abstention recall, semantic grounding, or
>   cost in any official sense** — a fake reranker's score carries no
>   relevance information, exactly as milestone 7 already states for its
>   own retrieval metrics.
> - **Official real-provider evaluation** — the only source this project
>   ever treats as a real result — has not run, for the reason stated
>   above.
> - No number from either of the first two categories appears in this
>   README's "Results" section, and no `eval_runs` row from a fake-provider
>   run has ever been written outside of a test's own isolated,
>   transaction-scoped database.

### Known limitations

- **No official answer, grounding, abstention, or cost evaluation has been
  run.** See the box above for the exact blocking reason.
- **The calibrated abstention threshold does not exist in this
  environment.** Calibration is implemented and unit-tested; it has never
  run against the real models, so `abstention_rerank_threshold` stays
  `None`, the same as milestone 8 left it.
- **The dev/test split is too small for statistical confidence in any
  abstention recall figure it eventually produces.** 7 expected-abstain
  questions total, split 4/3 — see "The frozen dev/test split" above.
- **The unsupported-claim rate has no calibrated threshold.** Raw
  per-sentence entailment scores are computed whenever the scorer is
  available; the binary unsupported/supported classification is not, by
  documented project-level decision — see "Semantic grounding" above.
- **No LLM judge.** The specification's optional third evaluation signal is
  explicitly deferred; core semantic evaluation is the local cross-encoder
  layer only.
- **`POST /query`'s live `cost_usd` is `NULL` for every request in this
  environment**, because `KNOWLEDGEOS_LLM_PRICING_USD_PER_MILLION_TOKENS`
  has no entry for any model here — not because pricing computation is
  unimplemented (see "Observability" above).

## What milestone 10 built

The server-rendered UI, the feedback endpoint, and deployment: a Dockerfile,
`docker-compose.yml`, and a CI workflow. The last milestone in the
specification's own table (§17) — nothing here changes retrieval, reranking,
generation, citation validation, grounding, abstention, or either evaluation
harness; every one of those is reused exactly as milestone 8/9 left it.

### Five pages, mounted under `/ui`, never in the JSON API's contract

The specification (§12): "Jinja2, server-rendered, minimal: document list,
document detail with version history, query box, answer page showing
citations and the retrieved evidence side by side, eval results page. No
React, no build step." All five exist, at exactly those routes
(`app/ui/routes.py`), and nowhere else:

```
GET  /ui/                          document list
GET  /ui/documents/{id}            document detail + version history
GET  /ui/query                     the query box
POST /ui/query                     runs the pipeline, 303 → the answer page
GET  /ui/answers/{query_id}        answer, citations, evidence, feedback form
POST /ui/answers/{answer_id}/feedback   the feedback form's own target
GET  /ui/evals                     evaluation results
```

Every one of them is `include_in_schema=False` — `/ui/*` never appears in
`/openapi.json`, and `tests/test_health.py`'s own exact-path guard (unchanged
in spirit since milestone 1, extended each milestone a new endpoint
legitimately arrived) proves the JSON API's contract is unaffected by this
package's existence at all. The UI is a separate, purely additive layer, not
a second copy of the API with a different response format.

### Reuse, not a second implementation

`POST /ui/query` calls `app.api.query.run_query` **directly** — the identical
function `POST /query` calls, with providers sourced through the identical
`Depends(embedding_provider)` / `Depends(rerank_provider)` /
`Depends(llm_provider)` dependencies `run_query` itself declares. Declaring
them as dependencies on the UI route, rather than calling the accessor
functions directly, is what makes them overridable in tests the same way
`tests/test_query_api.py` already overrides them for the JSON API — calling
`embedding_provider()` directly would silently always reach for the real,
cached provider, in a test or in production alike, and no test could ever
prove the UI's query flow works against anything else. Nothing about
retrieval, reranking, generation, citation validation, grounding, or
abstention is reimplemented anywhere in `app/ui/`.

The document list and detail pages call `app.api.documents.list_documents`
and `get_document` directly for the same reason — one query, one place it is
written.

**The answer page's evidence view is a new read, not new retrieval.**
`GET /queries/{id}` (the JSON API) deliberately does not carry chunk text or
document/version metadata — it is a ranking record, not an evidence viewer.
The answer page needs both, so `app/ui/routes.py` reads `retrieved_chunks`
joined to `chunks`, `document_versions` and `documents` — a read-only join
over rows milestone 8's own `persist_query` already wrote, never a second
call into `app.retrieval` or `app.reranking`.

### POST/redirect/GET, and reused validation for feedback

`POST /ui/query` ends in a `303` redirect to `/ui/answers/{query_id}`, so
reloading the answer page never resubmits the query. The feedback form on
that page posts to `/ui/answers/{answer_id}/feedback`, which shares one
write path (`app.api.feedback.create_feedback`) with the JSON
`POST /answers/{id}/feedback` endpoint below — an existence check, a rating
check, an insert, written once and called from both places.

### Escaping: autoescaped by default, `|safe` used nowhere

Every template extends `app/ui/templates/base.html` through Starlette's
`Jinja2Templates`, whose environment autoescapes `.html` templates by
default (`autoescape=jinja2.select_autoescape()` — verified against the
installed library, not assumed). Model answers, document titles, chunk
text, and feedback reasons are interpolated as plain Python strings and
escaped by the template engine — **no template in this package uses `|safe`
anywhere**, and `tests/test_ui.py::test_no_template_uses_the_safe_filter`
checks the committed template source directly rather than trusting a
one-time read. A citation is linked to its evidence block only by an
already-validated `chunk_uid` (32 lower-case hex characters,
`app/generation/citations.py`'s own validated shape) used as an HTML anchor
fragment — never by injecting raw model output as a link or as markup.
`tests/test_ui.py` seeds a document title and chunk text each containing
`<script>...</script>` and asserts the raw tag never appears in the
rendered response, only its escaped form.

Every UI error path renders `error.html` with the same sanitized
`HTTPException.detail` string the JSON API already returns to its own
callers — never a traceback, a file path, a raw provider error, or a
secret. `tests/test_ui.py` asserts this for a 404 (unknown document/query),
a 422 (an empty query, an invalid feedback rating), and a 503 (the real,
unavailable embedding model, reached with no dependency override at all).

### The evaluation results page never fabricates a number

`GET /ui/evals` reads `eval_runs` through `app.models.EvalRun` — the ORM
model, never `evals/` itself. `tests/test_retrieval_scope.py`'s existing
guard (no `app` module may import `evals`) is unchanged and still passes;
`tests/test_ui.py` adds a second, static check of `app/ui/routes.py`'s own
imports for the same property. With zero recorded rows it renders one
honest sentence — no placeholder number, no estimate, no percentage sign
anywhere on the page. With a recorded row, it renders that row's `config`
and `metrics` JSON verbatim, exactly as `evals/run.py` wrote it — reading a
persisted fact, never recomputing or approximating one.

### Feedback: the specification's last deferred table

`feedback` (`app/models/feedback.py`, migration `9176dca9861f`) is the
specification's own last column list (§5) — `id`, `answer_id` FK, `rating`
(`helpful`/`not_helpful`), `reason`, `created_at` — append-only, since the
specification states no uniqueness rule and a second opinion on the same
answer is not an error. `rating` is validated twice: the pydantic `Literal`
type at `POST /answers/{id}/feedback` (`app/api/schemas.py::FeedbackIn`),
and a database `CHECK` constraint in the migration itself, so a write that
bypassed the API still cannot store a third value. `reason` is bounded to
2000 characters (`app/models/feedback.py::MAX_REASON_LENGTH`) at both
layers — the specification names no limit; an unbounded column is this
project's own choice to avoid, the same reasoning `query_max_length`
already applies to `POST /query`.

### `DELETE /documents/{id}` — deliberately still not built

The specification's v1 API list (§11) includes it, and no milestone's scope
line ever assigns it (locked decision D1). It is destructive — cascading to
every version, chunk, query and answer belonging to a document — and its
ownership is genuinely ambiguous rather than merely deferred. Milestone 10
does not resolve that ambiguity unilaterally; it is recorded here, not
implemented.

### Docker

`Dockerfile`, `docker-compose.yml`, `.dockerignore`. A non-root user (uid
10001), `/health` as the `HEALTHCHECK` target — deliberately not `/ready`,
which would have an orchestrator restart a perfectly healthy process merely
because migrations have not been applied yet — an absolute
`KNOWLEDGEOS_STORAGE_ROOT` (`/var/lib/knowledgeos/storage`) mounted as a
volume in compose, and no secret of any kind as a literal value in either
file: `KNOWLEDGEOS_DATABASE_URL` inside compose points at the compose
network's own `db` service using the same local-development credentials the
database container itself defines (not a real secret); `KNOWLEDGEOS_LLM_MODEL`,
`GEMINI_API_KEY` and `GOOGLE_API_KEY` are read from the shell environment at
`docker compose up` time and are never written as literal values anywhere in
either file.

**The database image is `pgvector/pgvector:pg16`, not plain `postgres:16`**
(locked decision D7) — the extension migration 0001 already requires needs
to actually be installable in the server image, which the sibling
`docintel`/`voicedesk` projects in this monorepo never needed and plain
`postgres:16` does not provide.

**Migrations are never run automatically** (locked decision D8, and
`app/main.py`'s own decision, unchanged since milestone 1): "Migrations are
not run from here." `docker compose run --rm app alembic upgrade head` is
a separate, documented command; `/ready` is what notices when it has not
been run yet, exactly as it already does outside a container.

**CPU-only PyTorch, investigated and documented, not build-verified here**
(locked decision D6). `sentence-transformers` (`pyproject.toml`) declares
`torch>=2.2` and does not care which build satisfies it; the local
embedding and cross-encoder models this image serves are CPU inference
only, so the CUDA build a plain `pip install .` would otherwise resolve
buys the image several gigabytes of unused GPU libraries for nothing. The
Dockerfile installs `torch` from `https://download.pytorch.org/whl/cpu`
*before* `pip install .`, so the later, unconstrained `torch` requirement
is already satisfied and is never silently upgraded to a different build.
This environment's own network policy blocks `download.pytorch.org` with
the identical CONNECT-tunnel refusal that blocks the HuggingFace weights
below — so this line has been checked against sentence-transformers' own
declared constraint (confirmed compatible) but never actually executed
here. It is the standard, widely-documented method for a CPU-only PyTorch
install; if the index were unreachable in a real build environment, `pip
install` would fail loudly and the build would stop, never silently fall
back to a different, unintended build.

> **No Docker image has been built, and no container has been run, in this
> environment.** `docker info` fails outright: "Cannot connect to the
> Docker daemon at unix:///var/run/docker.sock" — confirmed directly, not
> assumed, and `sudo service docker start` itself fails
> (`ulimit: error setting limit (Operation not permitted)`, an unprivileged
> nested-container restriction of this sandbox). Every claim made about
> `Dockerfile`, `docker-compose.yml`, and `.dockerignore` in this section is
> a **static** one — the committed file's own content, read and checked by
> `tests/test_docker_assets.py` (non-root `USER`, the `/health` healthcheck
> target, no secret literal, a pgvector-capable database image, persistent
> volumes for both the database and document storage) — never a claim that
> an image was built or a container actually ran, here or anywhere else.

### CI

`.github/workflows/ci.yml` (locked decision D9): one focused job — install,
start a `pgvector/pgvector:pg16` service container, run `pytest`. No lint
step, no build step, no deployment step. `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
`GOOGLE_API_KEY` and `OPENAI_API_KEY` are never set anywhere in the workflow
— the specification's own words, "CI must pass with `ANTHROPIC_API_KEY`
unset" (§16), extended to every provider credential this project has ever
named, none of which it uses to pass. `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` are set explicitly so the real-model tests **skip**
in CI exactly as they already do in this development sandbox
(`tests/test_embeddings.py`, `tests/test_reranking.py`) — proven directly:
with those two variables set, `BgeEmbeddingProvider().embed(...)` fails
cleanly and immediately, the same `EmbeddingError` this environment's own
missing cache already produces, rather than CI silently downloading several
gigabytes of model weights on a runner that happens to have internet
access, which would make "CI passes" mean a different, nondeterministic
thing on every run.

### Configuration

`.env.example` now documents the full milestone 1-10 surface, including
milestone 8's `KNOWLEDGEOS_LLM_MODEL` (no default, required before
generation works), milestone 9's `KNOWLEDGEOS_ABSTENTION_RERANK_THRESHOLD`
and `KNOWLEDGEOS_LLM_PRICING_USD_PER_MILLION_TOKENS`, and milestone 10's own
deployment notes (the absolute storage root and the compose network's
database host, both already set for you inside `docker-compose.yml`) — every
entry names a configuration key, never a real value, and the Gemini
credential is documented as read directly from the process environment,
never as a `KNOWLEDGEOS_`-prefixed setting, because `app/providers/gemini_llm.py`
itself never reads one.

### Known limitations

- **No Docker image has been built or run in this environment.** See the
  box above for the exact, confirmed reason. Every Docker-related claim in
  this section is static file verification, not a build or runtime result.
- **CPU-only PyTorch has been investigated and documented, not
  build-verified.** See "Docker" above.
- **The real embedding and cross-encoder models remain unavailable here**,
  the same, unchanged condition every milestone since 4 has documented —
  the UI's query flow inherits this exactly: `POST /ui/query` against the
  real, un-overridden providers returns the identical `503` `POST /query`
  already does, proven by `tests/test_ui.py`.
- **No official evaluation numbers appear on the evaluation results page**,
  because none have been recorded in this environment's database — see
  milestone 9's own section above for why, unchanged by this milestone.
- **`DELETE /documents/{id}` remains unbuilt.** See its own section above;
  the ambiguity is recorded, not resolved.
- **No enterprise-grade security is claimed anywhere in this document.**
  The UI has no authentication; anyone who can reach `/ui/query` can ask a
  question and anyone who can reach an answer's URL can leave feedback on
  it — the same trust boundary every other v1 endpoint already has, stated
  plainly rather than implied away.

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
without one (588 pass, 362 skip). With a database: **947 pass, 3 skip** — the
three skips are the real-embedding-model and real-reranking-model tests
described above and below, plus the generation smoke test
(`tests/test_llm_provider.py`, opt-in only), none of which can reach their
real model or credential in this environment. No credential is needed
either way. Both figures were measured directly against this checkout, not
carried over from an earlier milestone.

KnowledgeOS is a separate application from DocIntel and VoiceDesk in this
repository: its own package, dependencies, virtualenv, configuration prefix and
database. Nothing is shared between them, and `tests/test_isolation.py` asserts
it.

## Deployment

```bash
cp .env.example .env   # then set KNOWLEDGEOS_LLM_MODEL and a Gemini credential

docker compose up --build -d db
docker compose run --rm app alembic upgrade head    # migrations: a separate
                                                      # step, always — see
                                                      # "What milestone 10
                                                      # built" above
docker compose up --build app
curl localhost:8000/health
curl localhost:8000/ready
```

`docker-compose.yml` builds the application image from the `Dockerfile` in
this repository and starts it alongside `pgvector/pgvector:pg16` (locked
decision D7 — plain `postgres:16` has no `vector` extension to offer).
`KNOWLEDGEOS_LLM_MODEL`, `GEMINI_API_KEY` and `GOOGLE_API_KEY` are read from
the shell environment `docker compose` runs in, never written into either
file as literal values — see `.env.example`.

**No image has been built and no container has been run from these files in
this environment** — the Docker daemon itself is unavailable here (`docker
info`: "Cannot connect to the Docker daemon"), a sandbox limitation
confirmed directly rather than assumed. `tests/test_docker_assets.py`
verifies every claim this section and "What milestone 10 built" make about
`Dockerfile`/`docker-compose.yml`/`.dockerignore` **statically** — reading
the committed file, never a build or run result — and this README makes no
claim beyond what that static check actually proves.

The local sentence-transformers model cache is not baked into the image
(see "What milestone 10 built" for why) and is not populated by anything in
this repository either; an operator with real HuggingFace access mounts a
pre-populated cache volume at `/home/knowledgeos/.cache/huggingface` (see
`docker-compose.yml`'s own `hf-cache` volume) or lets the first `/query`
request download it, exactly as running `uvicorn` directly would.

## Implementation specification

The authoritative build specification is a separate document, worked milestone
by milestone: inspect, propose, implement, test, commit, stop. It is not
committed to this repository.
