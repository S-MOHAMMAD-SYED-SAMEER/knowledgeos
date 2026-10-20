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
gold-sha256: 643dc9cecb0de518a1ed03cf3886bfbad04a81515ccd63e5ab04b4620e34eb86
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
Which one is the default was decided by the runs below: **`knowledge_answer_v2`**.

### How the default prompt was chosen (rule written before any run)

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

Full write-up, including the question-by-question differences: [docs/results/2026-10-20-answers-eval.md](results/2026-10-20-answers-eval.md).
Both runs are published; neither was selected from several.

Scored against **gold-v2**: gold written by Claude; reviewed with ChatGPT assistance; final decisions by Sameer; facts checked
against fixtures. Model `gemini-3.5-flash-lite`, temperature 1.0, adapter retries 0, threshold 1.4451 (**in-sample**: calibrated
on the dev split of the same 52 questions it is scored on), 2026-10-20, one machine, **one run per prompt**.

| Measure | prompt v1 | prompt v2 |
|---|---|---|
| Reached the model | 33 of 52 | 33 of 52 |
| Grounded answers (citation check passed) | 29 of 33 (87.9%) | 32 of 33 (97.0%) |
| Rejected by the citation check | 4 of 33 | 1 of 33 |
| Replies with a comma-grouped marker | 11 of 40 | 0 of 34 |
| Abstention recall | 7 of 7 | 7 of 7 |
| Abstention precision | 7 of 20 (0.35) | 7 of 19 (0.37) |
| Answerable questions abstained on | 13 of 45 | 12 of 45 |
| Model requests | 41 | 35 |

**Default prompt: `knowledge_answer_v2`.** All three conditions of the rule above were met in these two runs: fewer rejections
(1 against 4), no fewer grounded answers (32 against 29), abstention not worse (recall 7 of 7 in both; 12 against 13 answerable
questions abstained on). The gap is small in absolute terms and comes from one run each at temperature 1.0; on 2026-10-16 the
same v1 prompt had 7 of 34 rejected (against gold-v1), so run-to-run spread on an unchanged prompt is itself about this size.

**Known limit: the answerable questions abstained on.** Twelve of the 45 answerable questions are stopped by the threshold before
the model is asked, the same twelve in both runs (the model abstained on a thirteenth in v1). The threshold is in-sample, so
recall of 7 of 7 is partly the threshold having been fitted to these questions, and about a quarter of the answerable questions
get no answer as the price. No prompt changes that. Five of the twelve are ambiguous or adversarial questions that Gold Review 2
deliberately kept as "answerable".

What does not show: an unsupported-claim rate (no threshold is calibrated), any other model, a larger corpus, an independent
annotator, or whether v2's gain holds on a second run. "Grounded" means the citation check passed, not that a person checked the
cited passage supports the sentence.

### The first v2 attempt stopped on a 503

The first v2 attempt (09:28 UTC) stopped at its second question, `adv002`, on an HTTP 503 from the provider. By the stopping rule
nothing was retried and it was not resumed. After at least 30 minutes v2 was run again **fresh** from the start with the same
settings (one attempt), and that is the v2 run above. The first attempt's checkpoint is kept in
`docs/results/2026-10-20-answers-eval-v2-attempt1-stopped-503.partial.json`. Model requests for the whole piece of work: **78 of
the 100-request cap** (41 for v1, 2 for the stopped attempt, 35 for v2).

### Notes on the saved run files

The `gold_provenance` text inside the two JSON files still says no person has reviewed the gold; that was stale by then, the
harness now writes a corrected text, and the files were left as they were. The `gold_label`, `gold_version` and `gold_sha256`
beside it are the accurate statements. The model name in the local `.env` was mistyped at the start (`gemini-3.5-flash-liten`);
both runs set `gemini-3.5-flash-lite` explicitly and record it, and `.env` has been corrected.

Earlier runs: the retrieval runs of 2026-10-14 (gold-v1) and 2026-10-20 (gold-v2) are in `docs/results/`, and the answers run of
2026-10-16 (gold-v1, prompt v1, 52 questions, 27 of 34 verified) is in `docs/results/2026-10-16-answers-eval.md`.
