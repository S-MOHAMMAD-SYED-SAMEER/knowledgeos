# KnowledgeOS — Project Plan

The authoritative source for KnowledgeOS's remaining work. Future work
starts from a short instruction against this document (for example, "Start
P1") rather than re-deriving scope from scratch each time.

## Project Rules

- M1–M10 are frozen.
- Remaining work must not rewrite or weaken the existing engineering core.
- No fabricated metrics, embeddings, evaluation results, or demo claims.
- Demo code must live outside `app/`.
- Existing fake-provider safety guards must remain intact and unmodified
  unless a future milestone explicitly requires otherwise and is
  separately approved.
- Every milestone follows:
  inspect → implement → test → audit → commit → report → STOP.
- Never automatically continue to the next milestone.
- Only the explicitly requested milestone may be implemented.

## Current State

- Standalone repository: `S-MOHAMMAD-SYED-SAMEER/knowledgeos`
- Current HEAD: `050fe0d03453274f5492da96ad93429d5d427e9b` — pushed to
  `origin/main`, working tree clean.
- M1–M10 complete/frozen.
- **Since P1–P3 (below), two further phases completed and merged:**
  - **M1–M4 integrated demo mode** (`Settings.demo_mode`, `GET /demo`,
    `app/api/demo_guard.py`, `app/api/rate_limit.py`,
    `app/ingestion/corpus_seed.py`, `app/generation/demo_scenarios.py`,
    `app/providers/demo_llm.py`) — a second, independent, keyless demo
    mechanism built directly into `app/`, distinct from the standalone
    P3 demo below; merged via the `migrate-m1-m4` branch (commits
    `da7d9c6`..`94fbb40`). See the README's **Running modes** section
    for the full distinction between the two demo mechanisms.
  - **M5-A through M5-E** — a from-scratch baseline/verification pass on
    a Windows development laptop, run after the M1–M4 merge: environment
    inspection, sync to `origin/main`, full-suite verification (1139
    collected / 1127 passed / 11 failed / 1 skipped, all 11 failures
    known and categorized — see README's **Verified project facts**),
    root-cause investigation of 10 of those failures (a Windows
    `cp1252`-vs-UTF-8 encoding bug in `tests/demo_fixtures.py`
    corrupting a fixture chunk's `chunk_uid`), and the fix itself
    (`050fe0d`, `tests/demo_fixtures.py` only — no application code
    changed). Full detail in README's **Verified project facts**.
- **P1 (demo embeddings), P2 (UI polish), and P3 (demo wiring &
  verification) are all complete.** A deterministic, keyless Demo Mode
  exists (`demo/`), commit `3b8b105` (`feat(demo): add deterministic
  keyless demo`).
  - Committed embedding fixtures (`demo/fixtures/embeddings.json`, real
    precomputed BGE output), reranker fixtures
    (`demo/fixtures/reranker_scores.json`, real precomputed cross-encoder
    output), and answer fixtures (`demo/fixtures/answers.yaml`) all exist
    and are committed.
  - The five flagship scenarios (`da001`, `cs001`, `cv001`, `md001`,
    `ie001`) have end-to-end HTTP tests (`tests/test_demo_e2e.py`),
    driven through the real `POST /query` route on
    `demo.app.create_demo_app()`.
  - Demo documentation exists: [docs/DEMO.md](docs/DEMO.md) (run
    instructions) and `docs/ENGINEERING.md`'s "P3 — Deterministic Keyless
    Demo" section (architecture detail).
  - P3 demo-focused test suite: originally reported as 88 passed, at the
    time P3 was implemented; re-run in a later merged environment without
    the CrossEncoder model cached, the same suite produced 87 passed, 1
    skipped (a model-cache/network limitation of that environment, not a
    regression). **Both figures are historical and describe environments
    without the model cached — on the Windows laptop used for M5-A
    through M5-E, both models are cached and this skip does not occur;
    see README's Verified project facts for that environment's actual
    current numbers.** All figures refer to the standalone P3 demo
    (`demo/`) test files only, not the separate M1–M4 integrated demo
    mode.
- BGE and the cross-encoder have each been run for real, once, during P1/P3
  fixture generation — neither is "unavailable" in this build environment
  any more. A full Live Mode query (all three real providers together,
  including Gemini) has not been exercised end to end here.
- Gemini credential still unavailable in the development sandbox.
- No screenshots are currently tracked (P4 remains optional — see below).
- `FakeEmbeddingProvider` is **not** acceptable as the public demo retrieval
  foundation: a verified five-scenario probe against the real fixture
  corpus, through the real `retrieve()` pipeline, failed 3 of 5 flagship
  scenarios when fake hash-derived vectors were used (the vector channel
  returns the full 30-chunk corpus on every query, since it is smaller
  than the top-50 retrieval limit, injecting a random permutation into RRF
  fusion that the lexical channel cannot reliably correct). This was
  reproduced independently twice, in two separate checkouts, with
  identical results both times. This finding is why P1 built real,
  precomputed fixture providers instead.

## Product / Engineering Goal

The core KnowledgeOS demonstration story:

```
Retrieved correctly
→ ranked correctly
→ answered from evidence
→ cited correctly
→ abstained when evidence is insufficient.
```

The demo exists to **demonstrate this existing engineering behavior**, not
to replace or weaken it. Every remaining milestone is presentation and
packaging work around a production retrieval/generation architecture that
is already complete — none of it changes what that architecture does.

## Remaining Milestones

### P1 — Demo Foundation: Real Embeddings & Deterministic Providers

**Objective.** Create a reliable, offline-capable demonstration foundation
using real `BAAI/bge-small-en-v1.5` embeddings generated once in an
environment where the model is available, plus deterministic demo
answer/provider fixtures.

**Scope:**
- `demo/` package outside `app/`
- precomputed real embeddings for exactly the existing 33 demo chunks
- 384-dimensional embeddings
- `demo/fixtures/embeddings.json`
- deterministic curated answer fixture
- deterministic precomputed embedding provider
- demo seeding/wiring required to exercise the existing retrieval pipeline
- tests validating fixture shape and flagship retrieval scenarios

**Acceptance criteria:**
- exactly 33 expected demo chunk embeddings
- 384 dimensions
- chunk IDs match the demo corpus
- five flagship scenarios are verified through the real retrieval pipeline
- no `FakeEmbeddingProvider` used as the public retrieval foundation
- existing M1–M10 behavior remains unchanged
- full regression remains green

**Important external dependency.** The initial BGE embedding generation
requires an environment with HuggingFace/model access. Once generated, the
committed fixture must allow the demo to consume embeddings offline.

**Explicitly out of scope:**
- modifying `app/retrieval`
- modifying `app/reranking`
- modifying `app/generation`
- modifying `app/observability`
- modifying the existing evaluation harness
- changing production provider selection
- changing M1–M10 behavior

### P2 — Demo UI Polish

**Objective.** Make the existing UI understandable to a non-engineering
reviewer.

**Scope:**
- human-readable citation labels
- remove raw chunk UID presentation
- humanize grounded/citation-valid status
- hide raw grounding JSON behind an appropriate details/advanced section
- provide example questions
- make the core query experience the primary entry point

**Acceptance criteria.** A non-engineering reviewer can understand the
answer, evidence, citations, and abstention state without seeing raw
developer-oriented representations.

**Explicitly out of scope:**
- retrieval algorithm changes
- generation architecture changes
- new production providers
- unrelated UI features
- changing the evaluation methodology

### P3 — Demo Wiring & End-to-End Verification

**Objective.** Provide a genuinely runnable keyless demo using the P1
fixtures and P2 presentation.

**Scope:**
- demo entrypoint/wiring outside `app/`
- dependency overrides for demo providers
- offline deterministic demo behavior
- demo compose/run instructions
- end-to-end tests for the flagship scenarios

**Acceptance criteria:**
- demo runs without external API credentials
- demo does not require live Gemini
- demo does not require downloading BGE at runtime
- all five flagship scenarios can be demonstrated
- existing application provider-selection boundaries remain intact
- existing M1–M10 tests remain green

**Explicitly out of scope:**
- changing production provider selection
- modifying app-level fake-provider restrictions
- changing production retrieval/reranking/generation behavior

### P4 — Screenshots & Presentation Assets (Optional)

**Objective.** Capture portfolio-quality screenshots only after P2/P3 are
complete.

**Scope:**
- screenshots of the working demo
- key query/answer/evidence states
- abstention state
- conflicting-version state
- evaluation/result presentation if available

**Explicitly out of scope:**
- portfolio repository integration
- changing application behavior
- adding fake screenshots
- representing unavailable functionality as implemented

## Testing Requirements

- Full regression must remain green.
- Report exact pass/skip counts.
- No existing test may be weakened merely to make a milestone pass.
- New milestone-specific tests are required where specified.
- Demo tests must verify actual fixture correctness rather than only file
  existence.
- P1 retrieval scenarios must use real precomputed BGE embeddings, not
  hash/random fake embeddings.
- Demo tests must preserve the existing fake-provider guard behavior.

## Commit Requirements

- One focused commit per milestone unless explicitly approved otherwise.
- Conventional Commit style.
- Never rewrite/squash the M1–M10 history.
- Inspect the diff before committing.
- Verify working tree after committing.
- Push only the explicitly completed milestone.
- Report commit SHA.

## Stop Rule

After completing the requested milestone, STOP. Do not automatically begin
the next milestone. The next milestone requires an explicit user
instruction such as "Start P2."

## Known Risks / Blockers

- BGE availability for generating the initial fixture.
- Gemini credential availability is irrelevant to the keyless demo once
  deterministic demo answers are used, but real Gemini generation remains
  outside the demo scope.
- `FakeEmbeddingProvider` must not be used as the public retrieval
  foundation.
- Demo wiring must remain outside `app/`.

## Current File Impact

Expected P1 files, from inspection:

**New:**
- `demo/__init__.py`
- `demo/seed.py`
- `demo/providers.py`
- `demo/fixtures/embeddings.json`
- `demo/fixtures/answers.yaml`
- `tests/test_demo_fixtures.py`

P1 should not modify existing production application modules unless the
approved implementation later proves that a minimal integration change is
unavoidable; any such change must be reported before proceeding.
