# Testing KnowledgeOS

What the tests are, how to run them, what each environment skips and why, and what a green run does and does not tell you.
Setup is in [SETUP.md](SETUP.md); the evaluations (which measure retrieval and answers, and are not tests) are in
[EVAL.md](EVAL.md).

## Run them

```bash
ruff check .          # lint; CI runs it first
pytest                # the whole suite, from the repository root
pytest tests/test_citations.py -q          # one file
pytest -k "gold" -q                        # by name
```

The suite needs **no API key and makes no network call**. CI sets none; the one test that would call Gemini is opt-in (below).
Tests that need PostgreSQL **skip rather than fail** when no server answers, so `pytest` is meaningful without a database, but a
skipped test is not a passed one: for a real run start a pgvector database and name it:

```bash
docker run -d --name knowledgeos-test-db -p 5433:5432 \
  -e POSTGRES_USER=knowledgeos -e POSTGRES_PASSWORD=knowledgeos -e POSTGRES_DB=knowledgeos_test \
  pgvector/pgvector:pg16
export KNOWLEDGEOS_TEST_DATABASE_URL=postgresql+psycopg://knowledgeos:knowledgeos@localhost:5433/knowledgeos_test
pytest
```

Two rules the fixtures enforce: the test database's name must end in `_test` (the suite migrates and drops it, and refuses any
other name), and the schema is built by running the **real Alembic migrations**, not a shortcut, so the tables and the
one-active-version-per-document index the tests see are the ones that exist in production. If `KNOWLEDGEOS_DATABASE_URL` is set
in your environment, set `KNOWLEDGEOS_TEST_DATABASE_URL` explicitly as above so the two are never confused. A full run takes
about 8 to 15 minutes on a laptop with the models cached and a local database (the real-model tests dominate).

## What is covered

About 80 test files, in groups:

| Group (file prefixes) | What it checks |
|---|---|
| Ingestion: `test_parsing`, `test_chunking`, `test_ingestion_*`, `test_upload_*`, `test_indexing`, `test_reindex`, `test_normalization`, `test_storage` | PDF, DOCX, Markdown and text parsing, chunking with overlap, upload validation, the durable job runner, failure handling, re-indexing |
| Retrieval: `test_retrieval_*`, `test_embeddings`, `test_reranking*` | Full-text and vector search, Reciprocal Rank Fusion, metadata filters, version scope (only the active version answers), dedupe, the cross-encoder |
| Generation and citations: `test_generation*`, `test_citations`, `test_grounding`, `test_semantic_grounding`, `test_evidence_selection`, `test_prompt*`, `test_gemini_adapter`, `test_llm_provider` | The prompt files and their hashes, the strict citation validator, the retry-then-"could not verify" path, evidence escaping, the Gemini adapter's timeout, retry and error handling (all against fakes) |
| Abstention and answers: `test_abstention`, `test_calibration`, `test_answer*`, `test_split_integrity` | The threshold trigger and its guards, calibration, the answer metrics, the dev/test split |
| Evaluation harness: `test_evals_*`, `test_answers_*`, `test_request_budget`, `test_gold_frozen`, `test_published_results` | Metric arithmetic, the checkpoint and stop rules, the request cap, that the gold set hashes to the digest recorded in EVAL.md, and that the README and the results pages quote exactly the figures in the committed run files |
| API and UI: `test_query_api`, `test_feedback_api`, `test_health`, `test_ready`, `test_ui`, `test_demo_ui` | Endpoints, readiness, the answer page (numbered citations, collapsed evidence), the Evaluation page reading the committed results, the "model is busy" page |
| Demos: `test_demo_*` | Both demo mechanisms, the mutation guard that makes them read-only, the rate limiter, the five scripted scenarios |
| Architecture: `test_isolation`, `test_docker_assets`, `test_config`, `test_migration`, `test_db`, `test_models` | No application module may select a fake provider, the application never imports the evaluation package, full-text search is never labelled BM25, the compose files and Dockerfile are consistent, the migrations apply |

Three kinds of check are worth singling out, because they are the ones that stop a claim drifting from the evidence:

- **`test_gold_frozen.py`**: hashes everything under `evals/fixtures/` and compares it with `gold-sha256:` in `docs/EVAL.md`. A
  change to a question, a chunk, the split or a fixture document fails it. A deliberate change is a new gold version.
- **`test_published_results.py`**: reads the committed `docs/results/*.json` and fails if the README, `docs/EVAL.md` or a results
  page prints a different number, or if the README drops the limits (in-sample threshold, who wrote the gold, one run per prompt).
- **Static architecture tests** (`test_isolation.py`): the dependency rules are enforced by reading the code, so they cannot be
  broken by accident.

## What each environment runs

The counts are from the runs recorded in the README (2026-10-20 unless the README says otherwise).

| Environment | Result | Why the rest is skipped |
|---|---|---|
| **CI, Linux** (`test` job: real PostgreSQL + pgvector, no models in the cache, offline flags set) | 1,318 collected: 1,314 passed, 4 skipped | The tests that load the real BGE and cross-encoder weights (CI never downloads them), and the opt-in Gemini test |
| **CI, Windows** (`windows` job) | 840 passed, 478 skipped | A Windows runner has no PostgreSQL service container, so every database test skips; it exists to catch what only Windows shows (cp1252 reads, symlink privilege) |
| **Development laptop** (Windows, both models cached, local PostgreSQL) | 1,318 collected: 1,316 passed, 2 skipped | The opt-in Gemini test and the symlink test (this account cannot create symlinks) |

CI has two more jobs: `docker-build` (builds the image, starts the standalone demo stack and asks it a question) and `audit`
(`pip-audit` over the installed dependencies). No job sets a model key.

## The skipped tests, and why none is faked

A skip prints its reason, for example "the real model has NOT been exercised". A skipped test is not counted as passed, and the
project does not replace a real-model test with a fake one that passes. What the skips mean in practice:

- Where the models are absent, retrieval quality against the real BGE and cross-encoder is **not** checked by that run. It is
  checked by the retrieval evaluation (`python -m evals.run --suite retrieval`), which needs them.
- The one test that touches a real model, `test_the_real_gemini_model_can_generate`, runs only with a key in the environment,
  `KNOWLEDGEOS_RUN_GENERATION_SMOKE_TEST=1` and a verified `KNOWLEDGEOS_LLM_MODEL`. It makes one request.
  `scripts/gemini_smoke.py` is the same idea as a script: exactly one request, no retries, `--record` to save the facts.

## What a green run does not show

Passing tests show that the code does what the tests say. They do not show that the answers are good: that is what the
answers evaluation measures, against a gold set written by Claude and reviewed by ChatGPT and Claude, with a threshold calibrated
on the same questions (see the README, "Results", for the numbers and their limits). A model is never called in the suite, so a
change in the model's behaviour, or a provider outage, cannot turn a test red; it shows up only in an evaluation run.

Other limits: the Starlette `httpx` deprecation warning appears in every run and is not hidden; the suite does not load-test
anything; and the corpus used in the evaluations is 30 chunks, so nothing here says how retrieval behaves at scale.

## Adding a test

Put it in `tests/` beside the code's group, use the fakes in `tests/fixtures.py`, `tests/generation_fixtures.py` or
`tests/retrieval_fixtures.py` rather than a real model, and use the `session` fixture in `tests/conftest.py` for database work (it skips
cleanly without PostgreSQL). If a test quotes a figure from a results file, read it from the file; do not retype it. If the
change touches anything under `evals/fixtures/`, expect `test_gold_frozen.py` to fail and treat that as a decision, not a nuisance.
