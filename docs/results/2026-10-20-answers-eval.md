# Answers evaluation, prompt v1 against prompt v2, 2026-10-20 (gold-v2)

**Read this first.** Every score here is scored against **gold-v2**: gold written by Claude; reviewed with ChatGPT
assistance; final decisions by Sameer; facts checked against fixtures. It is one author's question set, not an independent
annotation. The abstention threshold (1.4451) was calibrated on the dev split of the same 52 questions it is scored on, so
the abstention figures are **in-sample**. One model, one machine, one day, **one run per prompt** at temperature 1.0: a
difference of one or two answers is inside what a second run could change.

Raw records, exactly as `python -m evals.run --suite answers` wrote them (raw model replies included):

- [`2026-10-20-answers-eval-v1-gemini-3.5-flash-lite.json`](2026-10-20-answers-eval-v1-gemini-3.5-flash-lite.json)
- [`2026-10-20-answers-eval-v2-gemini-3.5-flash-lite.json`](2026-10-20-answers-eval-v2-gemini-3.5-flash-lite.json)
- [`2026-10-20-answers-eval-v2-attempt1-stopped-503.partial.json`](2026-10-20-answers-eval-v2-attempt1-stopped-503.partial.json),
  the checkpoint of the first v2 attempt (see "What happened to the first v2 attempt")

## What was run

| | prompt v1 | prompt v2 |
|---|---|---|
| Prompt file | `knowledge_answer_v1` (content hash `b41fcc32cc976a92`) | `knowledge_answer_v2` (`1266e624604464ad`) |
| Run id | `90c642ae-3fc5-4d57-b8f2-03bea6e694fc` | `399a194e-7d16-466c-8a6a-bbbf971c96d3` |
| Started (UTC) | 2026-10-20 09:24 | 2026-10-20 10:05 (fresh run) |
| Model | `gemini-3.5-flash-lite` (Google AI Studio free tier) | same |
| Temperature, adapter retries, max output tokens | 1.0, 0, 4096 | same |
| Abstention threshold | 1.4451, the application default (in-sample) | same |
| Pacing, stopping | 10 requests a minute; `--strict --max-consecutive-failures 1` | same |
| Gold | gold-v2, sha256 `643dc9cecb0de518a1ed03cf3886bfbad04a81515ccd63e5ab04b4620e34eb86`, unchanged between the runs | same |
| Embedding, reranker | `BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, 30 active chunks | same |
| Model requests | 41 (1 availability check, 33 questions, 7 asked a second time) | 35 (1 availability check, 33 questions, 1 asked a second time) |

Only the prompt differs. The same 19 questions were stopped by the abstention threshold before the model was asked in both
runs (the threshold is deterministic), so the same 33 questions reached the model.

## Results

| Measure | v1 | v2 |
|---|---|---|
| Questions that reached the model | 33 of 52 | 33 of 52 |
| **Grounded answers** (verified by the citation check; an unverified answer counts as not grounded) | **29 of 33** (87.9%) | **32 of 33** (97.0%) |
| **Rejected by the citation check** ("I couldn't verify this answer" shown) | **4 of 33** (12.1%) | **1 of 33** (3.0%) |
| Asked a second time after a rejected first reply | 7 of 33 | 1 of 33 |
| Passed only on the second try | 3 | 0 |
| Model replies containing a comma-grouped marker such as `[a, b]` | 11 of 40 replies | 0 of 34 replies |
| Abstention recall (questions the gold says have no answer) | 7 of 7 | 7 of 7 |
| Abstention precision (abstentions that were right) | 7 of 20 (0.35) | 7 of 19 (0.37) |
| Answerable questions abstained on | 13 of 45 | 12 of 45 |
| Citation coverage of answered questions | 100% (guaranteed by the gate) | 100% (guaranteed by the gate) |
| Dev split (29 questions): asked / unverified | 18 / 3 | 18 / 1 |
| Test split (23 questions): asked / unverified | 15 / 1 | 15 / 0 |
| Input and output tokens in total | 224,935 and 8,123 | 205,839 and 6,958 |
| Cost | Not reported: no verified pricing is configured | same |

"Grounded" here means the citation check passed: every `[chunk_uid]` marker points at a chunk the model was given, and every
sentence carries one. It does not check that the cited passage supports the sentence. A semantic scorer ran, but no
"unsupported claim" threshold has been calibrated, so **no unsupported-claim rate is claimed.**

### What changed between the runs, question by question

Seven questions differ in a way that matters:

| Question | v1 | v2 |
|---|---|---|
| `da011` (on-call acknowledgement window) | rejected twice, shown as "couldn't verify" | verified on the first reply |
| `mf007` (what the access SOP covers) | rejected twice | verified on the first reply |
| `md006` (training required for access) | rejected twice | verified on the first reply |
| `cv003`, `md001`, `md003` | verified on the second reply | verified on the first reply |
| `adv004` (mileage reimbursement; the gold says answerable) | the model abstained | the model answered, citing the Expense Policy |

`adv003` (the "is that still the current rule?" question about the Security Policy) is **rejected in both runs**. In v2 it is a
different failure from the one v2 was written for: the reply began with a sentence of its own ("No, that is no longer the
current rule.") that carried no marker, and the checker requires one on every sentence. v2 was not changed for it; there is no
third prompt version.

The comma-grouped marker is the failure v2 targets, and it went from 11 of 40 replies to 0 of 34. That is a large movement on
a specific, countable failure, but it is one run each, so the exact counts (4 against 1) should not be read as stable. The
same v1 prompt gave 7 of 34 unverified on 2026-10-16 (against gold-v1) and 4 of 33 on this day: run-to-run spread on the prompt that did not
change is itself about that size.

### The 13 answerable questions abstained on: a known limit

**Twelve of the 45 answerable questions are stopped by the abstention threshold before the model is asked, in both runs: the
same twelve** (`adv001`, `adv005`, `am001`, `am002`, `am004`, `am005`, `da003`, `da006`, `da007`, `da010`, `mf003`, `md004`).
In v1 the model abstained on a thirteenth, `adv004`. That is the cost of a threshold that was set to catch all 7 expected
abstentions: it was calibrated on the dev split of these same 52 questions, so recall of 7 of 7 is partly the threshold
having been fitted to them, and the price is that about a quarter of the answerable questions get no answer. Neither prompt
can change that, because the model is never asked. Five of the twelve (`am001`, `am002`, `am004`, `am005`, `adv005`) are
questions Gold Review 2 deliberately kept as "answerable" test cases (ambiguous questions and one adversarial one), so part of
the figure reflects how the gold was decided. A threshold calibrated on other questions, or a larger corpus, may behave very
differently. Abstention recall rests on 7 questions; one flip would move it by 14 points.

### Latency is not compared

Latency is in the JSON files, but it is not a comparison of the prompts: the two runs were made on a laptop at different
times and the reranker time alone differs 2.6-fold between them (p50 3,699 ms against 1,406 ms) for the same work.

## Which prompt is the default, and why

The rule was written in `docs/EVAL.md` before either run: v2 becomes the default only if it has fewer citation-check
rejections, no fewer grounded answers, and abstention counts that are not worse.

1. Fewer rejections: 1 against 4. **Met.**
2. No fewer grounded answers: 32 against 29. **Met.**
3. Abstention not worse: recall 7 of 7 in both; answerable questions abstained on 12 against 13. **Met.**

**`knowledge_answer_v2` is the default** (`Settings.prompt_version`). v1 stays in the repository and can be selected with
`KNOWLEDGEOS_PROMPT_VERSION=knowledge_answer_v1`. This is the decision the pre-written rule gives for these two runs; it is
not a claim that v2 is robustly better.

## What happened to the first v2 attempt

The first attempt to run v2 started at 09:28 UTC, directly after the v1 run. It stopped at the second question, `adv002`
(the first, `adv001`, had been stopped by the threshold), because the provider returned **HTTP 503**. By the stopping rule, nothing was retried and the attempt was not
resumed. It wrote a checkpoint and no result. After waiting at least 30 minutes, v2 was run again **from the start** (not from
the checkpoint), with the same settings, as the single second attempt. The checkpoint is kept as
`2026-10-20-answers-eval-v2-attempt1-stopped-503.partial.json`. That attempt used 2 model requests, so the whole piece of work
used **78 of the 100-request cap** (41 + 2 + 35).

## Things the saved JSON says that are no longer true

- Both run files carry a `gold_provenance` text from the harness that still reads "No person has reviewed them yet." That
  sentence was stale by the time of these runs (Gold Review 2 had been decided and applied). The `gold_label` beside it, the
  `gold_version` and the `gold_sha256` are the correct statements, and the harness now writes a corrected `gold_provenance`.
  The files were left untouched.
- `.env` held a mistyped model name (`gemini-3.5-flash-liten`) at the start of this work. Both runs set the model explicitly
  to `gemini-3.5-flash-lite`, which is what their `llm_model` records; `.env` has since been corrected.

## What this does not show

Answer quality against anyone's judgement but one author's gold; an unsupported-claim rate; behaviour of other models;
behaviour at a corpus size beyond 30 chunks; whether v2's gain holds on a second run; and an abstention threshold that has
been tested on questions it was not calibrated on.

## Reproducing it

```bash
export KNOWLEDGEOS_LLM_MAX_RETRIES=0
python -m evals.run --suite answers --request-cap 100 --strict --max-consecutive-failures 1
KNOWLEDGEOS_PROMPT_VERSION=knowledge_answer_v1 python -m evals.run --suite answers --request-cap 100 --strict --max-consecutive-failures 1
```

See `docs/EVAL.md` ("How a run is made").
