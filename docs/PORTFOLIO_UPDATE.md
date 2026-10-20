# Portfolio update for KnowledgeOS: exact old → new text

**Not applied.** This file prepares the text changes for the normal portfolio (`portfolio`) and the 3D site
(`sameer-3d-portfolio`); neither repository was edited. Each block is the exact current text (read from the clones on
2026-10-20) and the text to replace it with. Nothing here is posted anywhere.

Every figure below is in the KnowledgeOS repository and matches its README and results files exactly:

| Figure | Value | Where it comes from |
|---|---|---|
| Tests, CI (Linux) | **1,318 collected: 1,314 passed, 4 skipped, 0 failed** | CI `test` job on `main` after PR #18 (`ab9a4c8`) |
| Tests, Windows CI | 840 passed, 478 skipped (no database or models there) | CI `windows` job, same commit |
| Tests, laptop, models cached | 1,318 collected: 1,316 passed, 2 skipped, 0 failed | run on 2026-10-20 |
| Demo-focused suite | 98 tests (`test_demo_e2e`, `_fixtures`, `_reranking`, `_llm`, `_app`), all passed | run on 2026-10-20 |
| Retrieval, gold-v2, 2026-10-20 (45 of 52 scored) | with the reranker / without: Recall@5 0.9444 / 0.8889, Recall@10 0.9926 / 0.9630, MRR 0.9081 / 0.7375, nDCG@10 0.9142 / 0.7927 | `docs/results/2026-10-20-retrieval-eval.md` |
| Answers, prompt v2 (default) | 32 of 33 grounded; 1 of 33 rejected by the citation check; abstention recall 7 of 7, precision 7 of 19; 12 of 45 answerable questions abstained on | `docs/results/2026-10-20-answers-eval.md` |
| Answers, prompt v1 | 29 of 33 grounded; 4 of 33 rejected; recall 7 of 7, precision 7 of 20; 13 of 45 abstained on | same |

**Limits that travel with the answer figures** (put them next to the numbers, as below): the 52 questions and the expected
answers were written by Claude, reviewed by ChatGPT and Claude, and approved by Sameer; the abstention threshold was calibrated
on the same 52 questions it is scored on (in-sample); one model (`gemini-3.5-flash-lite`), one day (2026-10-20), one run per
prompt; "grounded" means the citation check passed, not that a person confirmed the cited passage supports the sentence; the
corpus is 30 chunks from 10 fictional documents; 12 of the 45 answerable questions get no answer because the threshold stops
them before the model is asked.

**Do not add:** a claim that the system is accurate, safe, production-ready or "hallucination-free"; the retrieval Recall@10 of
0.99 without its "a third of the corpus" caveat; any cost figure (no pricing is configured in the repository); the v2
improvement as established (it is one run each at temperature 1.0).

Apply order: normal portfolio first, then the 3D site (its test figure and wording should match). Both sites' test suites assert
some of these strings, so run their tests after editing.

---

## A. Normal portfolio (`portfolio`)

### A1. Project card: `src/data/projects.ts` (the `p4` entry, around line 169)

OLD:
```ts
    proof: ["1,259 passing (CI)", "deterministic, credential-free demo"],
```
NEW:
```ts
    proof: ["1,314 passing (CI)", "32 of 33 answers passed the citation check (52 AI-written test questions)"],
```
`testCountOf(p4)` reads the first proof entry, so the card and the "RAG / Knowledge Systems" service (A3) both pick up 1,314.
The demo line stays in the description ("A deterministic, credential-free demo replays…") so nothing is lost. If the second
entry is too long for the card, use `"32 of 33 answers passed the citation check"` and leave the caveat to the case study.

The card `description` and `tags` do not need to change.

### A2. Case study: `src/pages/KnowledgeOsCaseStudy.tsx`

**A2.1. Header comment of the engineering figures (line 36 to 38)**

OLD:
```tsx
/**
 * Engineering test-suite figures, verified directly against the repository and its CI (commit
 * 94e1cd7, 2026-10-14) — not answer-quality results. Retrieval is measured separately below.
 */
```
NEW:
```tsx
/**
 * Engineering test-suite figures, verified directly against the repository and its CI (commit
 * ab9a4c8, 2026-10-20). Retrieval and answer quality are measured separately below.
 */
```

**A2.2. `RESULTS` (line 39 to 64)**

OLD (first card):
```tsx
    figure: "95",
    label: "Demo-suite tests passing",
```
NEW:
```tsx
    figure: "98",
    label: "Demo-suite tests passing",
```
(the detail text beneath it, which names the five test files, is unchanged)

OLD (second card):
```tsx
    figure: "1,259",
    label: "Full suite passing, CI (GitHub Actions)",
    detail:
      "1,263 collected, 4 skipped, 0 failed. The skips are the three tests that load the real BGE and cross-encoder weights (CI never downloads them) and the opt-in Gemini smoke test: skipped, not faked, and not counted as passing.",
```
NEW:
```tsx
    figure: "1,314",
    label: "Full suite passing, CI (GitHub Actions)",
    detail:
      "1,318 collected, 4 skipped, 0 failed. The skips are the three tests that load the real BGE and cross-encoder weights (CI never downloads them) and the opt-in Gemini smoke test: skipped, not faked, and not counted as passing.",
```

OLD (third card):
```tsx
    figure: "1,261",
    label: "Full suite passing, Windows laptop (models cached)",
    detail:
      "1,263 collected, 2 skipped, 0 failed. This is the run where the real BGE and cross-encoder tests execute. Earlier in the project this machine showed 11 failures; all were environmental (a file read as cp1252, symlink privilege, tests that assumed the models were absent), none were application bugs, and they are fixed.",
```
NEW:
```tsx
    figure: "1,316",
    label: "Full suite passing, Windows laptop (models cached)",
    detail:
      "1,318 collected, 2 skipped, 0 failed. This is the run where the real BGE and cross-encoder tests execute. Earlier in the project this machine showed 11 failures; all were environmental (a file read as cp1252, symlink privilege, tests that assumed the models were absent), none were application bugs, and they are fixed.",
```
(the "7 / 9" migrations card is unchanged)

**A2.3. `RESULTS_CAPTION` (line 66 to 67)**

OLD:
```tsx
const RESULTS_CAPTION =
  "These are engineering figures, verified from the repository and its CI: the code does what its tests say, and the image builds and the demo answers. They are not answer-quality results — no Gemini call has succeeded yet. Retrieval was measured separately; see Evaluation above.";
```
NEW:
```tsx
const RESULTS_CAPTION =
  "These are engineering figures, verified from the repository and its CI: the code does what its tests say, and the image builds and the demo answers. They are not answer-quality results; those are in Evaluation above, with their limits.";
```

**A2.4. `RETRIEVAL_RESULTS` and its comment (line 69 to 100)**

OLD (comment):
```tsx
/**
 * Retrieval evaluation, from the committed run docs/results/2026-10-14-retrieval-eval.md in the
 * KnowledgeOS repository (real BGE and cross-encoder, no API key). Exact four-decimal figures are
 * in each detail; the card figure is the rounded value.
 */
```
NEW:
```tsx
/**
 * Retrieval evaluation, from the committed run docs/results/2026-10-20-retrieval-eval.md in the
 * KnowledgeOS repository (real BGE and cross-encoder, no API key, the frozen gold-v2 question set).
 * Exact four-decimal figures are in each detail; the card figure is the rounded value.
 */
```

OLD (the first three cards):
```tsx
    figure: "0.91",
    label: "MRR with the reranker (0.74 without)",
    detail: "0.9081 with the cross-encoder, 0.7375 with the fused order kept, over the top 20 of 45 scored questions.",
  },
  {
    figure: "0.93",
    label: "Recall@5 with the reranker (0.87 without)",
    detail: "0.9333 with the cross-encoder, 0.8667 without.",
  },
  {
    figure: "0.99",
    label: "Recall@10 with the reranker (0.95 without)",
    detail: "0.9926 with the cross-encoder, 0.9519 without. Top 10 is a third of this corpus, so this one is easy.",
  },
```
NEW:
```tsx
    figure: "0.91",
    label: "MRR with the reranker (0.74 without)",
    detail: "0.9081 with the cross-encoder, 0.7375 with the fused order kept, over the top 20 of 45 scored questions.",
  },
  {
    figure: "0.94",
    label: "Recall@5 with the reranker (0.89 without)",
    detail: "0.9444 with the cross-encoder, 0.8889 without.",
  },
  {
    figure: "0.99",
    label: "Recall@10 with the reranker (0.96 without)",
    detail: "0.9926 with the cross-encoder, 0.9630 without. Top 10 is a third of this corpus, so this one is easy.",
  },
```
(the fourth card, "45 of 52 / Questions scored", is unchanged)

**A2.5. `RETRIEVAL_CAPTION` (line 102 to 103)**

OLD:
```tsx
const RETRIEVAL_CAPTION =
  "Read this with the corpus in mind: 30 chunks, so these are not a prediction for thousands of documents; the questions and their expected chunks were written by the same author against the same fixtures; and the retriever is not compared with BM25 or any other baseline, only with and without the reranker. One machine, one day, but repeatable: the same figures to the last digit every time the suite has run.";
```
NEW:
```tsx
const RETRIEVAL_CAPTION =
  "Read this with the corpus in mind: 30 chunks, so these are not a prediction for thousands of documents; the questions and their expected chunks were written by the same author against the same fixtures (Claude, reviewed by ChatGPT and Claude, approved by Sameer); and the retriever is not compared with BM25 or any other baseline, only with and without the reranker. One machine. The suite has been run on two versions of the question set (gold-v1 on 2026-10-14, gold-v2 on 2026-10-20, which changed four questions); both are published in the repository, and the gold-v2 figures are shown here.";
```

**A2.6. New block: the answers results.** Add after `RETRIEVAL_CAPTION`:

NEW (no old text):
```tsx
/**
 * Answers evaluation, from the committed runs docs/results/2026-10-20-answers-eval*.json in the
 * KnowledgeOS repository (gemini-3.5-flash-lite, temperature 1.0, the frozen gold-v2). Prompt v2 is the
 * repository default; the v1 figures are in each detail.
 */
const ANSWER_RESULTS: Result[] = [
  {
    figure: "32 of 33",
    label: "Answers that passed the citation check",
    detail: "Prompt v2, the default. Prompt v1: 29 of 33. 'Passed' means every citation points at evidence the model was given and every sentence has one; it does not check that the passage supports the sentence.",
  },
  {
    figure: "1 of 33",
    label: "Answers rejected by the citation check",
    detail: "Prompt v2. Prompt v1: 4 of 33. The person sees \"I couldn't verify this answer\" instead. One run of each at temperature 1.0, so the gap could change on another run.",
  },
  {
    figure: "7 of 7",
    label: "Questions with no answer in the documents, abstained on",
    detail: "Both prompts. The abstention threshold was calibrated on the same 52 questions it is scored on, so this is in-sample, and 7 questions is a small sample.",
  },
  {
    figure: "12 of 45",
    label: "Answerable questions that got no answer",
    detail: "Stopped by the same threshold before the model was asked (the same twelve with both prompts; 13 under v1). A known cost of the threshold.",
  },
];

const ANSWER_CAPTION =
  "Read this with its limits: the 52 questions and expected answers were written by Claude, reviewed by ChatGPT and Claude, and approved by Sameer, not independently annotated; one model, one day (2026-10-20), one run per prompt; a 30-chunk corpus of fictional documents; and no unsupported-claim rate is claimed.";
```

**A2.7. "Evaluation" section, last two paragraphs (line 335 to 355)**

OLD:
```tsx
          <p className="mt-4 max-w-3xl text-small text-ink-muted">
            The answer suite adds a 29/23 dev/test split for
            abstention-threshold calibration. The calibration method is
            implemented and unit-tested, but only 7 of the 52 questions are
            expected-abstain cases — a sample the repository itself documents
            as too small for statistical confidence.
          </p>
          <p className="mt-4 max-w-3xl text-small text-ink-muted">
            <strong className="text-ink">
              Retrieval has been measured. Answer quality has not.
            </strong>{" "}
            The retrieval suite ran on 2026-10-14 with the real BGE and
            cross-encoder models and no API key; the run is committed in the
            repository. The answers suite needs a Gemini model and has not
            been run: no Gemini call has succeeded yet. A run that cannot
            reach its models writes nothing — no report, no recorded metric —
            and an answers run that stops part-way saves its progress but
            reports nothing.
          </p>
          <div className="mt-6">
            <ResultsPanel results={RETRIEVAL_RESULTS} caption={RETRIEVAL_CAPTION} />
          </div>
```
NEW:
```tsx
          <p className="mt-4 max-w-3xl text-small text-ink-muted">
            The answer suite adds a 29/23 dev/test split for
            abstention-threshold calibration. The threshold the application
            ships with (1.4451) was calibrated on the dev split of the same 52
            questions it is scored on, so the abstention figures are
            in-sample, and only 7 of the 52 questions are expected-abstain
            cases — a sample the repository itself documents as too small for
            statistical confidence.
          </p>
          <p className="mt-4 max-w-3xl text-small text-ink-muted">
            <strong className="text-ink">
              Both suites have been run, against AI-written gold, on a small
              corpus.
            </strong>{" "}
            The retrieval suite ran with the real BGE and cross-encoder models
            and no API key. The answers suite ran on 2026-10-20 with
            gemini-3.5-flash-lite, once per prompt version; both runs are
            committed in the repository, and the prompt that did better under
            a rule written beforehand (v2) is the default. A run that cannot
            reach its models writes nothing, and a run that stops part-way
            saves its progress but reports nothing: the first attempt at the
            v2 run stopped on a provider error and was restarted from the
            start.
          </p>
          <div className="mt-6">
            <ResultsPanel results={RETRIEVAL_RESULTS} caption={RETRIEVAL_CAPTION} />
          </div>
          <div className="mt-6">
            <ResultsPanel results={ANSWER_RESULTS} caption={ANSWER_CAPTION} />
          </div>
```

**A2.8. "Production boundary" section (line 391 to 409)**

OLD:
```tsx
          <p className="max-w-3xl text-body text-ink-muted">
            Live Mode is implemented: real embedding, real reranking and
            real Gemini generation are the production default for every
            provider dependency. Real BGE and cross-encoder inference has run
            end to end (the retrieval evaluation above, and the demo
            fixtures). Real Gemini generation was attempted twice and both
            attempts failed with a server error whose cause was never
            determined; since then the adapter has gained a timeout, retries
            that honour the service&apos;s Retry-After, JSON mode, and a
            safe &ldquo;could not verify&rdquo; reply instead of an error
            page.
          </p>
          <p className="mt-4 max-w-3xl text-small text-ink-muted">
            Two things have not been exercised in any environment to date:
            a full Live Mode query end to end with all three real providers
            together, including Gemini, and an answers evaluation. None of
            this is a claim that the architecture is invalid — it is the
            current, honestly-stated verification boundary.
          </p>
```
NEW:
```tsx
          <p className="max-w-3xl text-body text-ink-muted">
            Live Mode is implemented: real embedding, real reranking and
            real Gemini generation are the production default for every
            provider dependency. Real BGE and cross-encoder inference has run
            end to end (the retrieval evaluation above, and the demo
            fixtures), and so has real Gemini generation
            (gemini-3.5-flash-lite): 52 evaluation questions on 2026-10-16
            and again on 2026-10-20, and a nine-question live check over
            HTTP. Earlier attempts failed with a server error whose cause was
            never determined; since then the adapter has gained a timeout,
            retries that honour the service&apos;s Retry-After, JSON mode,
            and a safe &ldquo;could not verify&rdquo; reply instead of an
            error page. Google&apos;s API still returns an occasional 503
            (&ldquo;high demand&rdquo;), which stopped the first v2
            evaluation attempt; the evaluation stops on it by design.
          </p>
          <p className="mt-4 max-w-3xl text-small text-ink-muted">
            What has not been exercised: the Docker Live Mode image with a
            model key (the model runs were made from a local Python
            environment), any model other than gemini-3.5-flash-lite, and any
            corpus larger than 30 chunks. This is the current, honestly-stated
            verification boundary, not a claim that the architecture is
            invalid.
          </p>
```

**A2.9. "What this is not, yet": three cards (line 413 to 440)**

OLD:
```tsx
              {
                title: "No answer-quality numbers",
                detail:
                  "Retrieval is measured, on a 30-chunk corpus. The answers harness is implemented and unit-tested but has not been run against a real model, so there is no grounded-answer rate, citation-validity or abstention figure to report.",
              },
              {
                title: "No real Gemini answer yet",
                detail:
                  "Two real attempts failed with a server error before the adapter was hardened, and none has succeeded since, so Live Mode's generation stage has never produced an answer. Retrieval, reranking and citation checking have run for real.",
              },
```
NEW:
```tsx
              {
                title: "The answer figures are narrow",
                detail:
                  "Scored against 52 questions written by Claude (reviewed by ChatGPT and Claude, approved by Sameer), on a 30-chunk corpus of fictional documents, with one model on one day and one run per prompt. No unsupported-claim rate is claimed, and \"grounded\" means the citation check passed.",
              },
              {
                title: "About a quarter of answerable questions get no answer",
                detail:
                  "12 of the 45 answerable questions are stopped by the abstention threshold before the model is asked. That is what a threshold fitted to catch all 7 unanswerable questions costs here, and no prompt changes it.",
              },
```

OLD:
```tsx
              {
                title: "The abstention threshold has never been calibrated",
                detail:
                  "The score-based trigger stays inactive. Only an explicit insufficient-evidence signal from the model, or zero retrieved candidates, cause an abstention today.",
              },
```
NEW:
```tsx
              {
                title: "The abstention threshold is in-sample",
                detail:
                  "The shipped threshold (1.4451) was calibrated on the dev split of the same 52 questions it is scored on, using only 4 expected abstentions, so it has not been tested on questions it was not fitted to.",
              },
```
(These two cards replace the two that said there were no answer-quality numbers and no real Gemini answer, so the grid keeps its card count.)

**A2.10. Check nothing else says it:** after editing, `grep -n -i "no Gemini call\|has not been run\|never been calibrated\|answer quality has not" src` must return nothing.

### A3. Services: `src/data/services.ts` (the `rag-knowledge` entry, around line 128)

No text change is required. `evidence[0]` is `testCountOf(p4)`, which reads the project's proof entry (A1) and becomes
"1,314 passing (CI)" on its own. The other two entries ("deterministic, credential-free demo", "citations checked in Python
before display") remain true. Optional single replacement if you want the real-model result on the service:

OLD:
```ts
    evidence: [
      testCountOf(p4),
      "deterministic, credential-free demo",
      "citations checked in Python before display",
    ],
```
NEW:
```ts
    evidence: [
      testCountOf(p4),
      "32 of 33 answers passed the citation check (small test set, limits published)",
      "citations checked in Python before display",
    ],
```

### A4. Tests that may need the new strings

Search the portfolio's tests for `1,259`, `1259`, `1,263`, `1,261`, `0.9333`, `0.8667` and "no Gemini call". Any assertion on the
old figures changes to the new ones above.

---

## B. 3D site (`sameer-3d-portfolio`)

### B1. Project monitor: `src/data/projects.ts` (the `p4` entry, around line 419)

**B1.1. `proof`**

OLD:
```ts
    proof: {
      tests: 1259,
      evaluation: 'retrieval MRR 0.91 with reranking (0.74 without)',
      properties: [
        'Hybrid retrieval (PostgreSQL full-text + pgvector) fused by Reciprocal Rank Fusion, reranked by a local CrossEncoder',
        'Retrieval measured offline with the real models and no API key, on a 30-chunk fixture corpus (45 of 52 questions scored): Recall@5 0.9333, Recall@10 0.9926, MRR 0.9081 and nDCG@10 0.9076 with the reranker, against 0.8667, 0.9519, 0.7375 and 0.7823 without it. Answer quality has not been measured: no Gemini call has succeeded yet',
```
NEW:
```ts
    proof: {
      tests: 1314,
      evaluation: 'retrieval MRR 0.91 with reranking (0.74 without); 32 of 33 answers passed the citation check',
      properties: [
        'Hybrid retrieval (PostgreSQL full-text + pgvector) fused by Reciprocal Rank Fusion, reranked by a local CrossEncoder',
        'Retrieval measured offline with the real models and no API key, on a 30-chunk fixture corpus (45 of 52 questions scored): Recall@5 0.9444, Recall@10 0.9926, MRR 0.9081 and nDCG@10 0.9142 with the reranker, against 0.8889, 0.9630, 0.7375 and 0.7927 without it. Answers measured on gemini-3.5-flash-lite on 2026-10-20 against 52 questions written by Claude, reviewed by ChatGPT and Claude, and approved by Sameer: 32 of 33 answers passed the citation check with the default prompt (29 of 33 with the original), 7 of 7 unanswerable questions abstained on, one run per prompt, and an abstention threshold calibrated on the same questions',
```
(the next two `properties` strings are unchanged). The `evaluation` string is shown in the monitor: check it fits; the shorter
fallback is `'retrieval MRR 0.91 with reranking (0.74 without)'` as it is today.

**B1.2. `caseStudy.result`** (a single string; replace the whole value)

OLD (the value begins `'The full merged test suite` and ends `no call has succeeded since.'`):
```ts
      result:
        'The full merged test suite — covering both demo mechanisms, the core retrieval/reranking/generation pipeline, and the citation and grounding checks — passes 1,259 of 1,263 collected tests in CI, with 4 skipped and 0 failed. The skips are the three tests that load the real BGE and CrossEncoder weights (CI never downloads them) and the opt-in Gemini smoke test: skipped rather than faked or counted as passing. On a laptop with both models cached, 1,261 of the same 1,263 pass, with 2 skipped and 0 failed; an earlier 11 failures there were all environmental, none were application bugs, and they are fixed. CI also builds the image, starts the demo stack and asks it a question. The retrieval evaluation ran on 2026-10-14 with the real models and no API key: on a 30-chunk corpus, the reranker lifts MRR from 0.7375 to 0.9081 and Recall@5 from 0.8667 to 0.9333. The questions were written by the same author against the same fixtures, and nothing has been measured at scale. Answer quality has not been measured: Gemini was tried twice and failed with a server error, and no call has succeeded since.',
```
NEW:
```ts
      result:
        'The full merged test suite — covering both demo mechanisms, the core retrieval/reranking/generation pipeline, and the citation and grounding checks — passes 1,314 of 1,318 collected tests in CI, with 4 skipped and 0 failed. The skips are the three tests that load the real BGE and CrossEncoder weights (CI never downloads them) and the opt-in Gemini smoke test: skipped rather than faked or counted as passing. On a laptop with both models cached, 1,316 of the same 1,318 pass, with 2 skipped and 0 failed; an earlier 11 failures there were all environmental, none were application bugs, and they are fixed. CI also builds the image, starts the demo stack and asks it a question. The retrieval evaluation ran on 2026-10-20 with the real models and no API key: on a 30-chunk corpus, the reranker lifts MRR from 0.7375 to 0.9081 and Recall@5 from 0.8889 to 0.9444. The answers evaluation ran the same day on gemini-3.5-flash-lite, once per prompt version: with the default prompt 32 of the 33 questions that reached the model passed the citation check (1 rejected), against 29 of 33 (4 rejected) with the original prompt, and all 7 questions with no answer in the documents were abstained on. Those figures are small and narrow: the 52 questions were written by Claude, reviewed by ChatGPT and Claude, and approved by Sameer; the abstention threshold was calibrated on the same 52 questions; it is one model, one day and one run per prompt; and 12 of the 45 answerable questions got no answer because the threshold stopped them before the model was asked. Nothing has been measured at scale.',
```

**B1.3. `architecture` / `engineering` bullets:** one bullet mentions "Retrieval and answer-quality evaluation each run as a separate
offline harness"; it remains true. No change.

### B2. Services board: `src/data/services.ts` (the `rag-knowledge` entry, around line 59)

No text change required. `delivers[0]` is `testsPassingInCiLabel('p4')`, built from `proof.tests`, so it becomes
"1,314 tests passing (CI)" after B1.1. Optional, matching A3:

OLD:
```ts
    delivers: [
      testsPassingInCiLabel('p4'),
      'deterministic, credential-free demo',
      'citations checked in Python before display',
    ],
```
NEW:
```ts
    delivers: [
      testsPassingInCiLabel('p4'),
      '32 of 33 answers passed the citation check (small test set, limits published)',
      'citations checked in Python before display',
    ],
```

### B3. Tests that may need the new strings

Search the 3D site's tests for `1259`, `1,259`, `0.9333`, `0.8667`, `0.9076`, `0.7823`, "no Gemini call" and the
`evaluation:` string. Update any assertion to the values above. Re-run the site's build and test commands before merging, as with
the earlier KnowledgeOS-adjacent changes.

---

## C. What to say, and not say, wherever this appears

- Lead with a figure that is true and carries its denominator: "32 of 33", not "97%".
- Name the model and the date once: gemini-3.5-flash-lite, 2026-10-20.
- Keep "written by Claude, reviewed by ChatGPT and Claude, approved by Sameer" within one screen of any answers figure.
- Keep "in-sample" next to any abstention figure and "12 of 45 answerable questions get no answer" wherever 7 of 7 appears on a
  page long enough to hold it.
- Do not say v2 "fixed" or "solved" anything. Say it is the default, chosen by a rule written before the runs, from one run each.
