# Evaluation: what is measured, against what, and what it does not show

This page is the record of how KnowledgeOS is evaluated. The numbers themselves are in `docs/results/` (one JSON file per
run, with the raw model replies) and are summarised under **Results** below once a run exists. Every figure here is a
count with its denominator, and every limit is stated next to the number it limits.

## The gold set, and that it is frozen

The "gold" is everything under `evals/fixtures/`: 52 questions in eight categories, each with the chunks that answer it
and whether it should be abstained on; the dev/test split (29 dev, 23 test); the provenance record; and the fixture corpus the
chunk ids refer to (10 documents, 11 versions, 30 active chunks).

> **Gold written by Claude; reviewed with ChatGPT assistance; final decisions by Sameer; facts checked against fixtures.**

That is the whole of its provenance. It is not an independent human review, it is 52 questions written by one author
against one small corpus, and only **7 of the 52** expect an abstention, so any abstention rate rests on 7 cases.

**Status: frozen.** The set is `gold-v2`. No model output was seen before it was fixed, and it is not to be changed after
a model has been run against it. `tests/test_gold_frozen.py` recomputes the digest below and fails if any file under
`evals/fixtures/` has moved. A deliberate change means a new gold version, a new digest here, and a note saying why.

```
gold-version: gold-v2
gold-sha256: 2db847142d903c53c08cf7c8f9d781c204e98e1578c0a2a2919537eba3a6b037
```

The digest is the sha256 over the sorted `path<NUL>sha256(file)` lines of every file under `evals/fixtures/`, with line
endings normalised so a Windows and a Linux checkout agree. `python -m evals.gold` prints it, with the per-file hashes.

### Gold Review 2 (decided 2026-10-16, applied 2026-10-20)

Applied exactly as recorded in the review decisions, and nothing else:

| Question | Decision | What changed |
|---|---|---|
| `da005` | Reword to the typical contractual window | "What is the typical contractual window for a vendor to notify the company of a security incident on the vendor's side?" The source says the window is typically seventy-two hours. Expected chunks unchanged. |
| `md003` | Expected chunk is Data Retention Policy chunk 0 only | The Production Database Access SOP v2 chunk 0 was removed from the expected chunks. |
| `md004` | Expected chunk is Employee Handbook chunk 1 only | The Security Policy chunk 0 was removed from the expected chunks. |
| `am001`–`am005` | Keep as deliberate ambiguity tests | No label changed; the intended behaviour is written in `ambiguous.yaml`. |
| `mf002` | Name a specific data category | "How long are database backups retained before older backups are automatically deleted?" Filter stays `category: Policy`; expected chunk stays Data Retention Policy chunk 0. |
| `adv004`, `adv005` | Keep as deliberate false-premise tests | No label changed; the intended behaviour is written in `adversarial.yaml`. |

Facts and filter metadata were checked against the fixtures and recorded per question in `evals/fixtures/PROVENANCE.yaml`:
the five fact checks (`cs001`, `cs003`, `cs004`, `cs005`, `mf003`) against the document text, and the filter metadata of
`mf001`–`mf007` against `knowledge_base/manifest.yaml` (every filter admits the document that holds its expected chunk).
No fact check required a label to change.

Known overlap: `da004` and the reworded `mf002` now ask nearly the same question (database backup retention). It is
kept because the decision was to keep it; it means one fact is counted twice.

## The abstention threshold

The application abstains, before the model is called, when the top cross-encoder rerank score is below **1.4451**. It is
the application default (`app/abstention_constants.py`, `Settings.abstention_rerank_threshold`).

- **Where it came from:** `evals/answers/calibration.py` run on the **dev split** (29 questions, 4 of which expect an
  abstention) of the 2026-10-16 answers evaluation, against gold-v1.
- **It is in-sample.** It was calibrated on questions drawn from the same 52 it is then scored on. The test split was held
  out from the calibration, but it is a split of the same small, author-written set. Do not read any abstention figure
  scored with it as how the threshold would do on questions it has not seen.
- It was kept at 1.4451 after the gold-v2 edits (a decision, not a re-calibration). Each answers run records the value it used
  and where it came from, and also records what a fresh calibration on gold-v2's dev split gives, for reference only.
- It is on the cross-encoder's score scale, so it is applied only when that cross-encoder did the reranking (not with the
  passthrough reranker or a test double), and not in the integrated demo mode, which answers fixed scenarios.

## Prompts

`app/generation/prompts/knowledge_answer_v1.md` is the original. `knowledge_answer_v2.md` was written from the
2026-10-16 failure analysis; `docs/PROMPTS.md` says what changed and why. It was committed before any run that uses it.
Which one is the default is decided by the runs below.

### How the default prompt will be chosen (written before any run)

Two runs, one per prompt, on the frozen gold-v2, same model, same settings, same threshold. Both are published whatever they
show. The default becomes `knowledge_answer_v2` only if, **in these two runs**, all of the following hold; otherwise it
stays `knowledge_answer_v1`:

1. v2 has **fewer** answers rejected by the citation check than v1 (the failure v2 was written to fix);
2. v2 has **no fewer** grounded answers than v1 (so it does not trade verified answers for something else);
3. the abstention counts (recall of the expected-abstain questions, and the answerable questions abstained on) are
   **not worse** than v1's.

The threshold is fixed, so the same questions reach the model in both runs and the comparison is like for like. With one
run each and about 34 questions, a difference of one or two answers is within what a rerun could change; the decision rule
is deliberately about the direction on the failure v2 targets, and the report says how large the gap is. There will be no
third run to break a tie.

## How a run is made

```bash
# Real model: Gemini. The key comes from the git-ignored .env and is never printed.
export KNOWLEDGEOS_LLM_MAX_RETRIES=0
python -m evals.run --suite answers --request-cap 100 --strict --max-consecutive-failures 1
KNOWLEDGEOS_PROMPT_VERSION=knowledge_answer_v2 python -m evals.run --suite answers --request-cap 100 --strict --max-consecutive-failures 1
```

- `--request-cap N` is a hard cap on model requests for the whole piece of work. It is counted in `var/request_ledger.json`
  before each request is sent, so a run that is interrupted cannot hide a request.
- `--strict` with `--max-consecutive-failures 1` stops at the first provider error of any kind (401, 403, 429, 503, 529, a
  quota or credit error, or a failed availability check). Nothing is retried.
- A run that stops writes a checkpoint and no result. It is reported, not re-run quietly.
- The model is `gemini-3.5-flash-lite` at temperature 1.0, adapter retries off.

## Results

See the end of this file once the runs have been made.
