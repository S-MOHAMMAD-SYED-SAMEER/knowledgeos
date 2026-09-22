# KnowledgeOS — Demo Mode

A self-contained, keyless walkthrough of the real KnowledgeOS query
pipeline, using a committed fixture corpus and deterministic providers
instead of a live embedding model, a live reranking model, or a live LLM
credential. For the project overview, capabilities and the normal
(Live Mode) run instructions, see [README.md](../README.md).

Demo Mode was built in P3 (`demo/`), entirely outside `app/`. Every
package under `app/` — retrieval, reranking, generation, the API, the
UI — is exactly what Live Mode runs; Demo Mode never forks, reimplements,
or weakens it.

## 1. What Demo Mode is

- **Keyless.** No Gemini/Google credential, no `KNOWLEDGEOS_LLM_MODEL`.
- **Deterministic.** The same five questions produce the same retrieved
  evidence, the same reranked order, and the same answer every run.
- **Backed by committed fixture data**, not live inference:
  - `demo/fixtures/embeddings.json` — real `BAAI/bge-small-en-v1.5`
    vectors, computed once and replayed.
  - `demo/fixtures/reranker_scores.json` — real
    `cross-encoder/ms-marco-MiniLM-L-6-v2` scores, computed once and
    replayed.
  - `demo/fixtures/answers.yaml` — real, hand-verified answers, replayed
    as citation-marker-compliant text through the real citation
    validator.
- **Does not call Gemini.** `demo/llm.py::DemoLLMProvider` never imports
  or constructs `GeminiLLMProvider`.
- **Does not download BGE or the cross-encoder at runtime.** Both
  fixtures were generated once, in an environment with the models
  available (`demo/generate_embeddings.py`,
  `demo/generate_reranker_scores.py`); nothing in the demo request path
  loads either model.
- **Exercises the real KnowledgeOS query pipeline.** A demo request runs
  through the identical, unmodified `POST /query` route
  (`app/api/query.py::run_query`) — real retrieval and RRF fusion
  (`app/retrieval/`), real reranking orchestration
  (`app/reranking/pipeline.py`), real generation orchestration, citation
  validation, and grounding (`app/generation/`). Only the three provider
  *dependencies* are swapped, at the same FastAPI `dependency_overrides`
  seam the test suite already uses — nothing in `app/` changes selection
  logic or knows Demo Mode exists.

## 2. Quick start (Docker) — the recommended path

The fastest way to see Demo Mode: clone, then one command. Nothing else
to install — no local PostgreSQL, no Python environment, no manual
migration or seeding step. This uses only the existing image build
(`Dockerfile`) and the existing `demo/` package; `docker-compose.demo.yml`
adds no new seeding logic and no new demo code.

```bash
git clone https://github.com/S-MOHAMMAD-SYED-SAMEER/knowledgeos.git
cd knowledgeos
docker compose -f docker-compose.demo.yml up --build
```

That one command, in order:

1. Starts PostgreSQL 16 with pgvector and waits for it to report healthy.
2. Runs the existing Alembic migrations (`alembic upgrade head`) against a
   dedicated `knowledgeos_demo` database.
3. Seeds the existing demo corpus through the existing
   `demo.seed.seed_demo_corpus` — the same function §5 (below) describes,
   invoked here as `python -m demo.seed`.
4. Only then starts `demo.app:app` — each step above is gated on the
   previous one's success, so the app is never reachable before its data
   is.

Wait for a line like `Uvicorn running on http://0.0.0.0:8000`, then open
**<http://localhost:8000/ui/query>** and run any of the five flagship
scenarios in §6. No credential is requested anywhere in this path —
`docker-compose.demo.yml` declares none, the same way `demo/llm.py`
itself never reads one.

To stop and remove the demo's containers (its PostgreSQL data and seeded
corpus live in Docker volumes local to this compose file, separate from
Live Mode's own `docker-compose.yml` volumes):

```bash
docker compose -f docker-compose.demo.yml down
```

Prefer to run it without Docker, or already have a local PostgreSQL you'd
rather reuse? The manual path in §3–§5 below does exactly the same thing
by hand.

## 3. Manual setup (fallback, no Docker)

Same as Live Mode (see the README's own **Running locally** section for
the full walkthrough) — Demo Mode adds no new prerequisite and removes
one:

- Python 3.13 (`requires-python = ">=3.13"`, `pyproject.toml`).
- PostgreSQL 16 with the `pgvector` extension — the only datastore this
  project has.
- `pip install -e ".[dev]"` (installs `demo/`'s own dependencies too —
  it adds none beyond what `app/` already declares).
- A dedicated database, migrated to head. Reusing the README's own
  `createdb` pattern, a **separate** database keeps demo fixture
  documents out of whatever real documents you may have uploaded to your
  normal `knowledgeos` development database:

  ```bash
  createdb -O knowledgeos knowledgeos_demo
  KNOWLEDGEOS_DATABASE_URL=postgresql+psycopg://knowledgeos:knowledgeos@localhost:5432/knowledgeos_demo \
    alembic upgrade head
  ```

- **No Gemini/Google credential is requested or read.** `GEMINI_API_KEY`,
  `GOOGLE_API_KEY`, and `KNOWLEDGEOS_LLM_MODEL` can all stay unset —
  `demo.app.create_demo_app()` never resolves them
  (`tests/test_demo_app.py::test_no_credential_or_network_setting_is_required_to_build_the_demo_app`
  proves this directly).

## 4. Demo startup (manual)

The entrypoint is `demo/app.py`, built by P3 Step 4:
`demo.app.create_demo_app()` — the real `app.main.create_app()`, with
`app.dependency_overrides` set for `embedding_provider`,
`rerank_provider`, and `llm_provider`. The module also exposes a
ready-to-serve instance at `demo.app.app`, the same
`uvicorn <module>:<attribute>` shape `app/main.py` already uses:

```bash
KNOWLEDGEOS_DATABASE_URL=postgresql+psycopg://knowledgeos:knowledgeos@localhost:5432/knowledgeos_demo \
  uvicorn demo.app:app --reload
```

```bash
curl localhost:8000/health   # liveness: touches nothing
curl localhost:8000/ready    # readiness: database, migrations, extension
```

## 5. Demo data initialization (manual)

The demo corpus is seeded through the exact same function P1 built and
P3's own tests already exercise — `demo.seed.seed_demo_corpus`, which
calls the frozen `evals.retrieval.corpus.ensure_corpus_seeded` with
`demo.providers.DemoEmbeddingProvider` in place of the real embedding
model. No new seeding mechanism exists or is needed; run it once against
the demo database, using the application's own real, cached
engine/storage accessors:

```bash
KNOWLEDGEOS_DATABASE_URL=postgresql+psycopg://knowledgeos:knowledgeos@localhost:5432/knowledgeos_demo \
python -c "
from sqlalchemy.orm import Session
from app.db.session import get_engine
from app.storage import get_storage
from demo.seed import seed_demo_corpus

with Session(get_engine()) as session:
    result = seed_demo_corpus(session, get_storage())
    print(f'seeded {result.total_active_chunks} active chunks '
          f'across {len(result.documents)} documents '
          f'({result.newly_seeded} newly seeded, {result.skipped_existing} already present)')
"
```

Idempotent: re-running it against an already-seeded database skips every
version already present (`ensure_corpus_seeded`'s own contract) rather
than duplicating it.

The inline snippet above and `python -m demo.seed` (what
`docker-compose.demo.yml`'s `seed` step in §2 runs) call the exact same
`seed_demo_corpus` function — `demo/seed.py`'s own `__main__` block is
nothing but that snippet, so either works identically against a local,
non-Docker PostgreSQL too:

```bash
KNOWLEDGEOS_DATABASE_URL=postgresql+psycopg://knowledgeos:knowledgeos@localhost:5432/knowledgeos_demo \
  python -m demo.seed
```

## 6. Demo scenarios

Five flagship questions, pinned in
`demo/generate_embeddings.py::FLAGSHIP_QUERIES` and answered in
`demo/fixtures/answers.yaml` — the same five PROJECT_PLAN.md names and
the same five `tests/test_demo_e2e.py` verifies end to end:

| id | scenario | question |
| --- | --- | --- |
| `da001` | retrieved correctly / grounded answer | "How many days per week may an employee work remotely?" |
| `cs001` | citation-sensitive reranking | "Within how many minutes of declaring a severity-one incident must an executive be notified?" |
| `cv001` | conflicting versions / current version only | "How many days does standard production database access last for?" |
| `md001` | multi-document evidence | "If I'm on call and need production database access during an active incident, what's the process, and does it require the same approval as a normal request?" |
| `ie001` | insufficient evidence / abstention | "What is the company's policy on using generative AI tools for writing code?" |

## 7. Expected behavior

For each of `da001`, `cs001`, `cv001`, `md001`, `POST /query` (or the UI
query box) returns:

- **Retrieved evidence** — `candidates`, the real RRF-fused top-20 from
  `app/retrieval/`, each carrying `chunk_uid`, the source document and
  version, and its ranks.
- **Reranked evidence** — the same candidates re-ordered by
  `rerank_score`, replayed from the real, precomputed cross-encoder
  fixture. For `cs001` specifically, this is the scenario reranking
  exists to demonstrate: the golden chunk (evidence of the "within
  thirty minutes" answer) ranks *behind* a same-document neighbor after
  fusion alone, and *ahead* of it once reranking runs — proven by
  `tests/test_demo_e2e.py::test_cs001_cited_correctly_after_fixture_driven_reranking`.
- **Citations** — `citations`, a real `chunk_uid` list, checked by the
  real, unmodified `validate_citations()` before ever being returned
  (`citation_valid: true`).
- **A grounded answer** — `answer`, the fixture's own curated text
  (word for word — `demo/llm.py` only ever adds citation markers, never
  changes a claim), `grounded: true`, and a non-fabricated
  `grounding_detail` computed by the real, unmodified
  `deterministic_grounding()`.

For `ie001`, the response instead carries `abstained: true`,
`citations: []`, and `answer` equal to the fixture's own abstention
text — no substantive claim is fabricated in its place.

`cv001` additionally demonstrates that the superseded `access-sop` v1
chunk never appears in `candidates` at all — milestone 5's own
current-version-only default, exercised unchanged.

## 8. UI/API access

Demo Mode serves the **existing, unmodified** UI and JSON API — nothing
new was built for either. With the server running (either the Docker
quick start in §2 or the manual startup in §4):

- **UI:** `http://localhost:8000/ui/query` — the same query form P2
  polished, POSTing to the same `/ui/query` route, landing on the same
  `/ui/answers/{id}` answer page with citations, evidence, and the
  grounded/citation-valid badges.
- **JSON API:** `POST http://localhost:8000/query` with
  `{"query": "<one of the five questions above>"}`; `GET
  http://localhost:8000/queries/{query_id}` to re-read a persisted
  result.

```bash
curl -s localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"query": "How many days per week may an employee work remotely?"}' | python -m json.tool
```

No separate demo UI exists or was built — this is the production
template set, rendering whatever the demo-provider-backed pipeline
returns.

## 9. Runtime guarantees

- No Gemini/Google API key is required or read.
- No external LLM request is ever made — `demo.llm.DemoLLMProvider`
  never imports `google`, `genai`, `requests`, or `httpx`
  (`tests/test_demo_llm.py::test_the_demo_provider_never_imports_a_real_or_network_provider`).
- No HuggingFace model download happens at demo runtime — the embedding
  and reranking fixtures are replayed from committed JSON, and neither
  `demo/providers.py` nor `demo/reranking.py` imports
  `sentence_transformers` or `torch`
  (`tests/test_demo_reranking.py::test_the_demo_provider_never_imports_the_real_model_loading_code`).
- The three deterministic fixture providers are the only ones a demo
  request can reach — proven directly in `tests/test_demo_e2e.py` and
  `tests/test_demo_app.py` by patching the real `BgeEmbeddingProvider`,
  `CrossEncoderRerankProvider`, and `GeminiLLMProvider` classes to raise
  if ever invoked, then running every one of the five flagship requests
  through them successfully.
- The production pipeline is unchanged: `app/retrieval/`,
  `app/reranking/`, `app/generation/`, `app/api/`, and every real
  provider are exactly what Live Mode runs — Demo Mode only ever
  supplies different *instances* at the existing dependency-injection
  seam.

## 10. Troubleshooting

- **`docker compose -f docker-compose.demo.yml up` exits with `migrate`
  or `seed` showing a non-zero exit code** — run `docker compose -f
  docker-compose.demo.yml logs migrate` (or `logs seed`) for the actual
  error. The `app` service never starts in this case (it depends on
  `seed` completing successfully), so there is no half-seeded state to
  clean up — fix the reported error and `up` again; both steps are
  idempotent.
- **`curl: (7) Failed to connect`** — PostgreSQL is not reachable, or the
  demo server did not start. Under Docker, `docker compose -f
  docker-compose.demo.yml ps` shows which service is not `Up`/healthy.
  Manually, confirm PostgreSQL is running (`pg_isready`) before starting
  `uvicorn`.
- **`/ready` returns a non-200 status** — the database exists but has not
  been migrated (or the `vector` extension is missing). Manually, run
  `alembic upgrade head` against `knowledgeos_demo` (§3) before seeding.
  This is the same readiness check and the same failure mode the main
  README documents for Live Mode — Demo Mode does not change it.
- **`CREATE EXTENSION vector` fails during migration** — the database
  role is not privileged enough to create the extension; see the main
  README's **Running locally** section and
  [docs/ENGINEERING.md](ENGINEERING.md) for the one-time superuser
  bootstrap. Identical in Demo Mode. (The Docker path's `db` service
  always has this privilege — this applies to the manual path only.)
- **Address already in use** — another process is already listening on
  port 8000. Under Docker, change the host-side port in
  `docker-compose.demo.yml`'s `app.ports` (e.g. `"8001:8000"`). Manually,
  pass `--port 8001` (or any free port) to the `uvicorn` command in §4.
- **Every flagship question abstains, including `da001`** — the demo
  corpus was never seeded (or was seeded against a different database
  than `KNOWLEDGEOS_DATABASE_URL` points the server at). Under Docker
  this should not happen (`seed` is a required, gated step); manually,
  re-run §5 against the exact same `KNOWLEDGEOS_DATABASE_URL` the server
  in §4 uses — `result.total_active_chunks` printed by that command
  should read `30`.
- **A question other than the five above always fails** — expected and
  by design: `DemoEmbeddingProvider`, `DemoRerankProvider`, and
  `DemoLLMProvider` each raise rather than silently improvising for any
  text outside the pinned fixture set (the same "never a general-purpose
  provider" discipline every demo provider documents in its own
  module). Demo Mode is five specific, verifiable scenarios, not a
  general-purpose keyless deployment.

## 11. Production vs. Demo

| | Production (Live Mode) | Demo Mode |
| --- | --- | --- |
| Embedding provider | `BgeEmbeddingProvider` — real `BAAI/bge-small-en-v1.5`, loaded from the local model cache | `DemoEmbeddingProvider` — the same model's real output, replayed from `demo/fixtures/embeddings.json` |
| Reranking provider | `CrossEncoderRerankProvider` — real `cross-encoder/ms-marco-MiniLM-L-6-v2`, loaded from the local model cache | `DemoRerankProvider` — the same model's real output, replayed from `demo/fixtures/reranker_scores.json` |
| Generation provider | `GeminiLLMProvider` — live Google Gemini, requires `GEMINI_API_KEY`/`GOOGLE_API_KEY` and `KNOWLEDGEOS_LLM_MODEL` | `DemoLLMProvider` — `demo/fixtures/answers.yaml`'s real, curated answers, replayed |
| Retrieval, reranking orchestration, generation orchestration, citation validation, grounding | `app/retrieval/`, `app/reranking/`, `app/generation/` — unmodified | identical, unmodified — the same code, the same call graph |
| Entrypoint | `uvicorn app.main:app` | `uvicorn demo.app:app` |
| Credential required | Gemini/Google API key | none |

Neither provider-selection path changes for the other's sake: production
selection (`app/api/query.py`'s `embedding_provider`/`rerank_provider`/
`llm_provider`, and each real provider's own cached accessor) is exactly
what it was before P3, and Demo Mode reaches it only by overriding those
same three dependencies on its own, separately-built application
instance.
