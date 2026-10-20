# Setting KnowledgeOS up

How to get from a clean machine to a running app, to the test suite, and to the evaluations. The README has the overview and
the results; this file is the procedure. Commands are for a POSIX shell; the Windows differences are marked.

KnowledgeOS has three ways to run (see the README, "Running modes"). They need different amounts:

| You want | You need |
|---|---|
| The standalone demo (five scripted questions, nothing live) | Docker, nothing else. No database install, no models, no key. |
| The test suite | Python 3.13, PostgreSQL 16 with pgvector for the tests that use it. No key. |
| The real app (Live Mode) and the answers evaluation | All of: Python 3.13, PostgreSQL 16 + pgvector, the two local models, and a Gemini key. |

## 1. The quickest look: the standalone demo

```bash
docker compose -f docker-compose.demo.yml up --build
```

Then open `http://localhost:8000/ui/query`. This starts PostgreSQL with pgvector, applies the migrations, seeds the demo corpus
and starts the app, each step waiting for the previous one. No credential is read anywhere in that file. It answers five
scripted questions from committed fixtures; it does not run a model. [docs/DEMO.md](DEMO.md) describes what it does and does not
show.

## 2. Python environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Python 3.13 is required (`requires-python = ">=3.13"`). The install pulls `torch` through `sentence-transformers`, so allow about
2 GB of disk. `ruff` and `pytest` come with the `[dev]` extra.

## 3. The two local models

Embedding (`BAAI/bge-small-en-v1.5`, 384 dimensions) and reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) run in the app's
own process, from your Hugging Face cache. The app loads them with `local_files_only=True` and never downloads at run time, so
without them every ingest and every query fails with "the embedding model is unavailable".

```bash
python scripts/download_models.py            # once, about 200 MB
python scripts/download_models.py --dry-run  # prints what it would fetch (it does not check the cache first)
```

Only the hybrid retrieval and the answer path need them. The test suite does not: tests that need the real weights skip when they
are absent.

## 4. PostgreSQL with pgvector

A plain `postgres` image has no `vector` type. The quickest source is the image the project's own CI uses:

```bash
docker run -d --name knowledgeos-db -p 5432:5432 \
  -e POSTGRES_USER=knowledgeos -e POSTGRES_PASSWORD=knowledgeos -e POSTGRES_DB=knowledgeos \
  pgvector/pgvector:pg16
docker exec knowledgeos-db createdb -U knowledgeos knowledgeos_test    # the test suite migrates and drops this
```

Installing pgvector natively also works (on Debian or Ubuntu, `sudo apt-get install postgresql-16-pgvector`). If
`CREATE EXTENSION vector` fails, the role is not privileged enough; [ENGINEERING.md](ENGINEERING.md) has the one-time superuser
bootstrap.

**Point every command at the database you mean.** `KNOWLEDGEOS_DATABASE_URL` is the application's database,
`KNOWLEDGEOS_TEST_DATABASE_URL` the test suite's. If either is already set in your environment (a user-level Windows variable
persists across terminals), it silently redirects every command. The test fixtures refuse to run against a database whose name
does not end in `_test`.

## 5. Configuration

```bash
cp .env.example .env        # Windows: copy .env.example .env
alembic upgrade head        # migrations are always an explicit step; the app never runs them at start-up
```

`.env` is git-ignored. `.env.example` documents every setting and each has a working default; nothing in it is needed just to
run the tests.

Two things are not defaults:

- **The model**: `KNOWLEDGEOS_LLM_MODEL=gemini-3.5-flash-lite` (the value in `.env.example`). Check it after editing; a typo
  here fails at the first call, not at start-up, and it overrides the code default.
- **The Gemini key**: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`, which the SDK prefers if both are set). It is deliberately not a
  `KNOWLEDGEOS_` setting. The app does not read it from `.env` itself; the Google SDK reads it from the **process environment**.
  Put it there however you normally do (your shell profile, a secrets manager, or `docker compose`, which passes it through from
  the shell). Never commit it, never paste it into an issue or a log, and keep it out of the files in `docs/results/`
  (the run records contain model replies and token counts, not credentials).

The other settings worth knowing: `KNOWLEDGEOS_LLM_TEMPERATURE` (1.0, sent explicitly), `KNOWLEDGEOS_LLM_MAX_RETRIES` (2 for
the app; 0 for any counted evaluation run), `KNOWLEDGEOS_ABSTENTION_RERANK_THRESHOLD` (default 1.4451, **calibrated on the same
52 questions it is scored on, so in-sample**; see [EVAL.md](EVAL.md)), and `KNOWLEDGEOS_PROMPT_VERSION` (default
`knowledge_answer_v2`; v1 is kept, see [PROMPTS.md](PROMPTS.md)).

## 6. Run the app

```bash
uvicorn app.main:app --reload
curl localhost:8000/health        # liveness: touches nothing
curl localhost:8000/ready         # readiness: database, migrations, extension
```

Open `http://localhost:8000/ui/`. Upload documents (PDF, DOCX, Markdown or plain text) through the UI or `POST /documents`; the ingestion job
parses, chunks and embeds them. A question then goes through retrieval, reranking, the abstention threshold, generation and the
citation check.

**With no key configured, a question in the UI shows "The model is busy — try again" (HTTP 503; the API returns 503 JSON).** The server log says why. Nothing
leaves the machine in that state.

## 7. Run the tests

See [TESTING.md](TESTING.md). In short: `ruff check .` then `pytest`, with a PostgreSQL available for the tests that use one.

## 8. Run the evaluations

The retrieval evaluation needs the two local models and a database, and no key:

```bash
python -m evals.run --suite retrieval     # writes var/eval_runs/retrieval-<timestamp>.json
```

The answers evaluation calls the model. It is capped and stops on the first provider error:

```bash
export KNOWLEDGEOS_LLM_MAX_RETRIES=0
python -m evals.run --suite answers --request-cap 100 --strict --max-consecutive-failures 1
KNOWLEDGEOS_PROMPT_VERSION=knowledge_answer_v1 python -m evals.run --suite answers --request-cap 100 --strict --max-consecutive-failures 1
```

- `--request-cap N` is a hard cap on model requests for the whole piece of work, counted in `var/request_ledger.json` before each
  request is sent. Delete that file only to start a new piece of work, on purpose.
- `--strict` with `--max-consecutive-failures 1` stops at the first provider error (401, 403, 429, 503, 529, a quota error, a
  failed availability check). Nothing is retried. A stopped run writes a checkpoint, never a result.
- The run prints its request budget first and paces itself (`--requests-per-minute`, default 10).
- The gold set under `evals/fixtures/` is frozen. Changing any file there changes its digest and fails
  `tests/test_gold_frozen.py`; a change after a model run is a new gold version ([EVAL.md](EVAL.md)).
- The saved results are in `docs/results/`; the Evaluation page of the UI (`/ui/evals`) reads them.

## 9. Docker (Live Mode)

```bash
docker compose up --build -d db
docker compose run --rm app alembic upgrade head
docker compose up --build app
```

`docker-compose.yml` bind-mounts your Hugging Face cache for the two models (`KNOWLEDGEOS_HF_HUB` if it is not in the default
place) and passes `KNOWLEDGEOS_LLM_MODEL` and the Gemini key through from the shell.

## 10. Problems you may meet

| Symptom | Cause | Fix |
|---|---|---|
| A Gemini call from Python hangs for a minute or more, while `curl` works | The machine has an IPv6 address but no working IPv6 route to Google | Prefer IPv4 for the process (turn off IPv6 for the adapter, or start through a launcher that filters `socket.getaddrinfo` to IPv4 results) |
| `CERTIFICATE_VERIFY_FAILED ... unable to get local issuer certificate` | Antivirus or a proxy re-signs HTTPS with a root that is in the OS store but not in Python's `certifi` | Build a bundle that includes your OS roots and point `SSL_CERT_FILE` at it. Do not turn certificate verification off |
| HTTP 503 from Google ("high demand") | Intermittent on this model family, seen on the free tier | Wait and try again. An evaluation run stops on it by design and is reported, not retried |
| "the embedding model is unavailable" | The two models are not in the Hugging Face cache | `python scripts/download_models.py` |
| Commands act on the wrong database | `KNOWLEDGEOS_DATABASE_URL` is set in your user environment | Set it explicitly in the command, or unset it |

The first two cost a lot of time on the development laptop. They are environment issues, not application ones.
