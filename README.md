# KnowledgeOS

**Retrieval-grounded answers over internal documents, with citation validation and honest abstention.**

Ask "how do I request production database access?" and get an answer built
only from your own documents — a citation to a retrieved chunk after every sentence, the citations
checked programmatically before the answer is shown (that they point at a chunk the model was given, not that the
chunk supports the sentence), and an explicit refusal
when the knowledge base does not contain the answer.

**Status:** v1 is built and tested, and measured with a real model on a small fixture corpus (see **Results** below, with its limits). MIT licensed.
**GitHub** [S-MOHAMMAD-SYED-SAMEER/knowledgeos](https://github.com/S-MOHAMMAD-SYED-SAMEER/knowledgeos) ·
**Live demo** No public instance is deployed — two independent, local, keyless demo modes exist instead, see **Running modes** below ·
**[Engineering deep dive](docs/ENGINEERING.md)** ·
**[Setup](docs/SETUP.md)** · **[Testing](docs/TESTING.md)** · **[Evaluation](docs/EVAL.md)** · **[Prompts](docs/PROMPTS.md)**

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

KnowledgeOS answers only from retrieved documents, cites a retrieved chunk
after each sentence, validates those citations in Python before returning the
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
| **Citation validation** | Every cited chunk ID is checked against the retrieved, *selected* evidence set. An invalid citation rejects the answer rather than displaying it: the model is asked once more, and if that reply fails too the person sees "I couldn't verify this answer", never the rejected text. |
| **Honest abstention** | Refuses on insufficient evidence — zero candidates, a rerank-score threshold (calibrated on the dev split of the same 52 questions it is scored on, so in-sample; see the results), or the model's own `sufficient_evidence=false`. |
| **Document versioning** | Only the active version answers; superseded versions stay queryable for audit. One active version per document, enforced by a partial unique index. |
| **Dual evaluation harnesses** | Separate retrieval and answer suites, run from the CLI. The retrieval suite is offline and needs no credential; the answers suite calls a model and needs one. |
| **Server-rendered UI** | Jinja2, no build step: documents, version history, query box, answer page with citations and evidence, evaluation results. |

## How the pipeline works

```
Ingest → Parse → Chunk → Index → Retrieve → Rerank → Generate → Cite → Evaluate
```

| Stage | Detail |
| --- | --- |
| Chunk | Fixed-size by word count (512/64 overlap; a "token" in the settings is a whitespace-delimited word, not a model token), paragraph-aware. `chunk_uid = sha256(version:sequence:text)[:32]` makes re-indexing idempotent. |
| Retrieve | Top 50 lexical + top 50 vector → RRF fusion → dedupe → top 20. Metadata filters are SQL `WHERE` clauses, never an instruction to the model to ignore results. |
| Rerank | Local cross-encoder behind an interface, plus a passthrough implementation so reranking's contribution is a measured number rather than an assumption. |
| Generate | Versioned prompt file, structured output (`answer`, `citations`, `sufficient_evidence`), parsed and validated in Python. |
| Cite | Three deterministic rules plus a check that the declared citations match the markers in the text, then a deterministic grounding check. A semantic grounding layer exists only in the answers evaluation; in a live query it is always reported as unavailable. |

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
  unless measured or verified; unknown pricing is never silently priced at
  zero: the cost function raises an error, a live query stores its cost as unknown (null), and the evaluation
  reports "cost unavailable".

## Evaluation approach

Evaluation runs against a committed fixture corpus (10 documents, one with a
materially changed v2) and 52 labelled questions across eight categories:
directly answerable, multi-document, ambiguous, insufficient evidence,
conflicting versions, metadata-filtered, citation-sensitive, and adversarial.

Deterministic checks (citation validity, metadata-filter correctness,
abstention triggering) are kept separate from model-judged checks, and the
reports say which is which. Both suites run from the CLI:
`python -m evals.run --suite retrieval | answers`.

The retrieval suite needs no key. The answers suite calls a model at most about 53 times
(one availability check, one call per question that reaches the model, a second call for any reply that
has to be retried); the abstention threshold stops 19 of the 52 questions before the model is asked, so the recorded
runs used 41 requests (prompt v1) and 35 (v2). It prints that budget before it starts, spaces the calls
(`--requests-per-minute`, default 10, under a free tier's 15), stops after two
provider failures in a row (`--max-consecutive-failures`), and saves its progress
after every question, so `--resume <file>` continues without paying for the
questions already answered. A stopped run writes a checkpoint and nothing
official: no `eval_runs` row, no report. A reply the model could not make
verifiable ("I couldn't verify this answer") is counted on its own, never as an
abstention.

**Results.** Both suites have been run, against AI-written gold, on a 30-chunk corpus.

### Retrieval

The retrieval suite (`python -m evals.run --suite retrieval`, offline, no API
key, real BGE and cross-encoder) was run twice against the fixture corpus: on
2026-10-14 against the original question set (gold-v1) and on 2026-10-20 against
the frozen gold-v2 (see [docs/EVAL.md](docs/EVAL.md)), which changed the expected
chunks or wording of four questions. The corpus is 30 chunks from 10 documents,
and 45 questions out of 52 are scored (the other 7 are "the documents don't say"
questions with no chunk to find). Both runs are published:

| Metric | Without reranking, gold-v1 | With the cross-encoder, gold-v1 | Without reranking, gold-v2 | With the cross-encoder, gold-v2 |
| --- | --- | --- | --- | --- |
| Recall@5 | 0.8667 | **0.9333** | 0.8889 | **0.9444** |
| Recall@10 | 0.9519 | **0.9926** | 0.9630 | **0.9926** |
| MRR | 0.7375 | **0.9081** | 0.7375 | **0.9081** |
| nDCG@10 | 0.7823 | **0.9076** | 0.7927 | **0.9142** |
| Precision@5 | 0.2311 | 0.2533 | 0.2311 | 0.2489 |

Metadata-filter correctness is 1.0000 over the 9 filtered questions in both runs. The
reranker lifts MRR from 0.74 to 0.91 on the same candidates. A change in the
question set that raises a score is not an improvement in the retriever.

**Read this with the corpus in mind:** 30 chunks, with top 10 covering a third of
the corpus, so Recall@10 is easy and is not a prediction for a corpus of
thousands of documents; the questions and their expected chunks were written by
the same author against the same fixtures; and the retriever is not compared
with BM25 or any other baseline, only with and without the reranker. Full records:
[gold-v1, 2026-10-14](docs/results/2026-10-14-retrieval-eval.md) and
[gold-v2, 2026-10-20](docs/results/2026-10-20-retrieval-eval.md).

### Answers (real model)

The answers suite (`python -m evals.run --suite answers`) was run on **2026-10-20** against the frozen **gold-v2** with
**`gemini-3.5-flash-lite`** (Google AI Studio free tier, temperature 1.0, adapter retries off), once per prompt version. Both
runs are published; the better one by the rule written in [docs/EVAL.md](docs/EVAL.md) before either was made is the default.

| Measure | prompt v1 | prompt v2 (default) |
| --- | --- | --- |
| Questions that reached the model | 33 of 52 | 33 of 52 |
| Grounded answers (citation check passed) | 29 of 33 (87.9%) | **32 of 33** (97.0%) |
| Rejected by the citation check ("couldn't verify") | 4 of 33 | **1 of 33** |
| Abstention recall (questions with no answer in the documents) | 7 of 7 | 7 of 7 |
| Abstention precision | 7 of 20 | 7 of 19 |
| Answerable questions abstained on | 13 of 45 | 12 of 45 |

**Limits, next to the numbers.**
- The gold is **Claude-drafted, reviewed by ChatGPT and Claude, and approved by Sameer**; it is one author's question set, not
  an independent annotation. The 52 questions and 10 documents (11 versions, 30 active chunks) are fictional and labelled as such.
- The abstention threshold (1.4451) was calibrated on the dev split of **the same 52 questions** it is scored on: the abstention
  figures are **in-sample**. Recall of 7 of 7 rests on 7 questions, and one flip would move it by 14 points.
- **Twelve of the 45 answerable questions get no answer** because the threshold stops them before the model is asked (the same
  twelve with both prompts). That is a known cost of this threshold; no prompt changes it.
- "Grounded" means the citation check passed (every marker points at a chunk the model was given, and every sentence has one).
  It does not check that the cited passage supports the sentence, and **no unsupported-claim rate is claimed.**
- **One model, one machine, one day, one run per prompt**, at temperature 1.0. The gap between the prompts (4 rejected against 1)
  is small in absolute terms and could change on a second run; on 2026-10-16 the same v1 prompt had 7 of 34 rejected.
- The first v2 attempt stopped on an HTTP 503 from the provider at its second question and was restarted fresh after more than
  30 minutes; the checkpoint is kept. 78 of the 100-request cap were used in total.

Full records, with the question-by-question differences:
[docs/results/2026-10-20-answers-eval.md](docs/results/2026-10-20-answers-eval.md); the earlier run of 2026-10-16 (gold-v1, prompt
v1) is in [docs/results/2026-10-16-answers-eval.md](docs/results/2026-10-16-answers-eval.md).

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
| **Standalone Demo Mode** (`demo/`) | None | **Implemented** |

**Live Mode — implemented.**
- Real local `BgeEmbeddingProvider` and `CrossEncoderRerankProvider` —
  require `BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2`
  in the local sentence-transformers cache.
- Real generation through Google Gemini (`GEMINI_API_KEY` or
  `GOOGLE_API_KEY`). The model defaults to `gemini-3.5-flash-lite`
  (`KNOWLEDGEOS_LLM_MODEL`) at temperature 1.0 (`KNOWLEDGEOS_LLM_TEMPERATURE`),
  the model's own default and what Google's prompting guide says to keep for
  Gemini 3 models.
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

**Standalone Demo Mode — implemented (`demo/`).**
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
  integrated mode's own `Settings.demo_mode`-gated corpus seed in
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

Nothing estimated. Test results, by where they were run (2026-10-20, the Phase A
pull request; no API key anywhere):

| Where | Result |
| --- | --- |
| **CI, Linux** (`test`: real PostgreSQL + pgvector, no models in the cache) | 1,318 collected: **1,314 passed**, 4 skipped, 0 failed |
| **CI, Windows** (`windows`: everything that needs neither PostgreSQL nor the models) | 840 passed, 478 skipped, 0 failed |
| **Windows laptop**, both models cached | 1,318 collected: **1,316 passed**, 2 skipped, 0 failed |

(The earlier "models not in the cache" laptop row was not re-measured for this
commit, so it is not repeated here.)

What is skipped, and why: on Linux CI, the three tests that load the real BGE
or cross-encoder (CI never downloads weights) and the opt-in Gemini smoke test
(`KNOWLEDGEOS_RUN_GENERATION_SMOKE_TEST=1`, needs a key); on a laptop, the
symlink test when the account may not create symlinks, and the smoke test. A
skip is not a pass and is not the same as "verified": the real-model tests only
ran in the last row.

CI also lints with ruff, audits the installed dependencies with `pip-audit`
(clean in that run), and in the `docker-build` job builds the image, starts the
documented one-command demo (`docker compose -f docker-compose.demo.yml up
--build`), waits for `/ready` and asks it a question, asserting a verified,
cited, non-abstained answer.

Until 2026-10-14 the same suite showed 11 failures on a laptop with the models
cached, and 4 without. None were application bugs: three read a file as cp1252,
one needed symlink privilege, and seven assumed the models are absent. They are
fixed (the tests now simulate a missing model, read files as UTF-8, and skip the
symlink test where it cannot run). An earlier real defect found by the same
exercise (commit `050fe0d`): `tests/demo_fixtures.py` read fixture markdown
without an encoding, which corrupted a chunk's text, and so its `chunk_uid`, on
Windows.

| Fact | Value |
| --- | --- |
| Alembic migrations / database tables | 7 / 9 |
| Continuous integration | Four jobs: test (Linux), windows, docker-build, audit; no API key set anywhere |
| Retrieval metrics | Produced, 2026-10-14 ([results](docs/results/2026-10-14-retrieval-eval.md)) |
| Answer-quality metrics | Produced, 2026-10-20, `gemini-3.5-flash-lite`, against AI-written gold ([results](docs/results/2026-10-20-answers-eval.md)) |

These are results about the engineering: the code does what its tests say, the
image builds and the demo answers. The answer-quality figures are in **Results**
above, with their limits. See **Current limitations** below for exactly what
remains unverified, and **Running modes** above for which claim belongs to which
demo.

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
- **Live Mode has been exercised end to end with a real model**: real BGE embedding, real retrieval, real CrossEncoder
  reranking and real Gemini generation (`gemini-3.5-flash-lite`). Earlier attempts failed with a `ServerError` from the
  provider (cause not determined); generation worked on 2026-10-16, and the answers evaluation of 2026-10-20 is
  the current record ([docs/results/2026-10-20-answers-eval.md](docs/results/2026-10-20-answers-eval.md)). Google's API returns
  intermittent HTTP 503s ("high demand"): one stopped the first v2 attempt, and 8 of 9 live-check questions answered on
  2026-10-16. The integrated `Settings.demo_mode` and the demos use no model.
- **The answer-quality figures are narrow.** They are scored against AI-written gold on a 30-chunk corpus, with an in-sample
  abstention threshold, one model, one day, one run per prompt (see **Results** for each limit). The retrieval numbers come from
  the same small corpus and say little about scale.
- **The standalone demo stack (`docker-compose.demo.yml`) has been built and
  run successfully**, on a Windows Docker Desktop 29.8.1 / Compose 5.5.1
  laptop: all three images built, the full `db → migrate → seed → app` chain
  started, all 7 migrations applied, the demo seed produced the expected
  corpus, `/health`/`/ready`/`/ui/query` all returned 200, all five
  deterministic scenarios matched their fixtures exactly through the real
  HTTP API, the UI query flow was verified end to end, all five
  mutation-guarded routes returned 403 with no resulting database change,
  and the rate limiter enforced its configured 10/minute limit — see
  [docs/ENGINEERING.md](docs/ENGINEERING.md) for the full evidence.
  **`docker-compose.yml` (Live Mode) has since been built and run
  successfully too**, on the same laptop: the database, migrations, corpus
  seeding, health/readiness, real BGE embedding, real retrieval, and real
  CrossEncoder reranking all passed inside that container. **The integrated
  `Settings.demo_mode` demo running in Docker has still not been attempted
  in any environment**, and no public instance of anything in this
  repository is deployed.
- Postgres full-text search is **not BM25**; ranking behaviour differs and
  this README does not claim otherwise.
- In a live query there is no semantic grounding: only the deterministic layer runs (the semantic scorer lives in the
  answers evaluation). A passed check means every sentence has a well-formed citation to a retrieved chunk, not that
  the chunk supports the sentence; the deterministic citation checks are the exact layer.
- A "token" in chunking is a whitespace-delimited word, so 512 of them can exceed the embedding model's input window
  (recorded as a known risk in [docs/ENGINEERING.md](docs/ENGINEERING.md)); it is not solved.
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

> The original project brief named the Anthropic SDK for generation; this build
> uses **Google Gemini** — a deliberate deviation from that wording, not from
> the behavioural requirements, none of which name a vendor. An Anthropic
> adapter is not implemented. Nothing outside
> the one adapter file is vendor-shaped.

## Running locally

The full procedure, with the Windows notes and the evaluation commands, is in [docs/SETUP.md](docs/SETUP.md); the tests are described in [docs/TESTING.md](docs/TESTING.md). The short version:

**What you need first**

| Need | Why |
| --- | --- |
| Python 3.13 | `requires-python = ">=3.13"`. |
| PostgreSQL 16 **with the pgvector extension** | The only datastore. A plain `postgres` image has no `vector` type. `pgvector/pgvector:pg16` is the quickest source (below). |
| About 2 GB of disk for `pip install` | Local embeddings and reranking pull in torch. |
| The two local models in your Hugging Face cache (about 200 MB) | The application loads them with `local_files_only=True` and never downloads while it runs. Without them every ingest and query fails with "the embedding model is unavailable". `python scripts/download_models.py` fetches them once. |
| A Gemini key | Only for generating answers in Live Mode (the model defaults to `gemini-3.5-flash-lite`). Retrieval, the evaluation of retrieval, and both demos need neither. |

**Steps** (Linux, macOS and Windows; the database line works everywhere Docker does)

```bash
python3.13 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                 # Windows: copy .env.example .env
python scripts/download_models.py    # once; --dry-run shows what it would fetch

# PostgreSQL 16 + pgvector, as a throwaway container (or install pgvector natively:
# on Debian/Ubuntu, `sudo apt-get install postgresql-16-pgvector`)
docker run -d --name knowledgeos-db -p 5432:5432 \
  -e POSTGRES_USER=knowledgeos -e POSTGRES_PASSWORD=knowledgeos -e POSTGRES_DB=knowledgeos \
  pgvector/pgvector:pg16
docker exec knowledgeos-db createdb -U knowledgeos knowledgeos_test   # the suite migrates and drops this

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

To try it with no Postgres install, no models and no key, use the standalone
demo: `docker compose -f docker-compose.demo.yml up --build`, then open
`http://localhost:8000/ui/query` ([docs/DEMO.md](docs/DEMO.md)). If a Docker
build fails on `pip install` with `CERTIFICATE_VERIFY_FAILED`, some antivirus
products (AVG's web shield, for one) re-sign HTTPS traffic with a root the
container does not trust; turn that scanning off while you build.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| A Gemini call from Python hangs for a minute or more with no error (the first request after start-up, or every one). `curl` to the same host works | The machine has an IPv6 address but no working IPv6 route to Google. Python's HTTP client tries each IPv6 address in turn, each waiting for its connect timeout, before it tries IPv4 | Prefer IPv4 for the process: on Windows, turn off IPv6 for the network adapter, or start the app through a small launcher that filters `socket.getaddrinfo` to IPv4 results. Check with `curl -6 -m 5 https://generativelanguage.googleapis.com/` (it hangs) against `curl -4 ...` (it answers) |
| `[SSL: CERTIFICATE_VERIFY_FAILED] ... unable to get local issuer certificate` on a Gemini call, or on `pip install` / a Docker build | An antivirus or proxy that scans HTTPS (AVG's web shield is one) re-signs the traffic with its own root, which is in the Windows certificate store but not in Python's `certifi` bundle | Either turn HTTPS scanning off while you run it, or give Python a bundle that includes the Windows roots: export them to a PEM file next to `certifi`'s bundle and point `SSL_CERT_FILE` (and `REQUESTS_CA_BUNDLE`) at the combined file. Do not turn certificate verification off |
| `503` from `POST /query` with "the language model is unavailable" | The adapter could not get an answer. The server log line `Generation failed (LLMError): generation failed (HTTP nnn)` says which status | Read the status in the log. `503 "high demand"` from Google passes: wait and try again. `404` is a wrong model ID; `400/403` is a rejected key or request |
| `429`, or "the language model is busy right now" | Free-tier rate limit (about 15 requests a minute when measured elsewhere on this model family) | Wait; the answers evaluation paces itself (`--requests-per-minute`) |

## License

MIT. See [LICENSE](LICENSE).

## Deployment

`Dockerfile`, `docker-compose.yml` and a CI workflow are committed. Migrations
are always a separate, explicit step — the application never runs them at
startup, and `/ready` reports the drift until they have been applied:

```bash
docker compose up --build -d db
docker compose run --rm app alembic upgrade head
docker compose up --build app
```

**This image has since been built and run from `docker-compose.yml` (Live
Mode, shown above)**, on a Windows Docker Desktop laptop: the database,
migrations, corpus seeding, health/readiness, real BGE embedding, real
retrieval, and real CrossEncoder reranking all passed. Real Gemini
generation has since worked (see **Results**), run from a local Python
environment, not from this container. The separate standalone demo stack (`docker-compose.demo.yml`)
has also been built and run successfully — see
**Current limitations** above and full detail in
[docs/ENGINEERING.md](docs/ENGINEERING.md).

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
