# Retrieval evaluation, 2026-10-14

The retrieval suite, run for real: real BGE embeddings, the real cross-encoder, a real
PostgreSQL with pgvector. No API key, no network call, no model API involved.

Raw record: [`2026-10-14-retrieval-eval.json`](2026-10-14-retrieval-eval.json), exactly as
`python -m evals.run --suite retrieval` wrote it.

## What was run

| | |
|---|---|
| Command | `python -m evals.run --suite retrieval` |
| Commit | `cd05c3b` (`main`, 2026-10-14) |
| Run ID | `6942906c-1d9b-4d2f-a733-a2c2f26c0a5a` |
| Started / finished | 2026-10-14 10:10 UTC / 10:14 UTC (about four minutes, including seeding the corpus) |
| Embedding model | `BAAI/bge-small-en-v1.5` (384 dimensions, local) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local), compared with a passthrough that keeps the fused order |
| Chunking | 512 words, 64 overlap (`chunk_size_tokens` counts whitespace-delimited words) |
| Fusion | Reciprocal Rank Fusion, k = 60; top 50 lexical + top 50 vector, then top 20 |
| Machine | Windows 11, Python 3.13.5, torch 2.14.1, sentence-transformers 6.1.0, transformers 5.19.0 |
| Database | PostgreSQL 16 (`pgvector/pgvector:pg16`) with pgvector 0.8.6 |

## The data

- **Corpus:** 10 documents (one, the Production Database Access SOP, has a second version that
  changes the answer), 30 active chunks.
- **Questions:** 52, in 8 categories: directly answerable 12, multi-document 6, metadata-filtered 7,
  insufficient evidence 6, adversarial 6, ambiguous 5, conflicting versions 5, citation-sensitive 5.
- **Scored:** 45. The other 7 are questions whose right answer is "the documents don't say", so there
  is no expected chunk to find; they are excluded from retrieval metrics and reported as excluded.
- **Filtered:** 9 of the scored questions carry a metadata filter (department, category, tags,
  document, or superseded versions).

## Results

| Metric | Without reranking (passthrough) | With the cross-encoder |
|---|---|---|
| Recall@5 | 0.8667 | **0.9333** |
| Recall@10 | 0.9519 | **0.9926** |
| MRR | 0.7375 | **0.9081** |
| nDCG@10 | 0.7823 | **0.9076** |
| Precision@5 | 0.2311 | 0.2533 |
| Metadata-filter correctness | 1.0000 (condition-invariant, over 9 filtered questions) | |

What the columns mean: **Recall@k** is the share of a question's expected chunks found in the top k
(averaged over questions); **MRR** is 1 over the rank of the first expected chunk, over the whole top 20;
**nDCG@10** rewards expected chunks that appear higher; **Precision@5** divides by a fixed 5, so with
usually one expected chunk per question its ceiling is well below 1 and is not a quality signal on its
own; **metadata-filter correctness** is the share of filtered questions where no returned chunk
violates the filter.

**The result worth reading is the reranker's effect:** with the same candidates, the cross-encoder
raises MRR from 0.7375 to 0.9081 and Recall@5 from 0.8667 to 0.9333. Hybrid retrieval alone already
puts nearly everything in the top 10 (Recall@10 0.9519); the reranker moves it to the top.

## Read these numbers with the corpus in mind

- **The corpus is small.** 30 chunks. With top-10 covering a third of the whole corpus, Recall@10 is
  easy; do not read 0.99 as a prediction for a corpus of thousands of documents. Nothing here was
  measured at scale.
- **The questions and their expected chunks were written by the same author against the same
  fixtures**, not by an independent annotator.
- **45 of 52 questions are scored.** The 7 abstention questions are measured by the answers suite, which
  needs a language model and has not been run.
- **Postgres full-text search is not BM25**, and the hybrid retriever is not compared with BM25 or any
  other baseline here. The only comparison is with and without the reranker.
- **One machine.** Every run, including the 14 compared below, was on the same Windows laptop. The figures
  have been identical every time on that machine, so the result is deterministic there; it has not been
  reproduced on a different machine.

## What this does not show

Answer quality. No generated answer, citation validity under a real model, abstention accuracy or
unsupported-claim rate is measured here: the answers suite (`python -m evals.run --suite answers`)
needs a Gemini model and has not been run. No Gemini call has yet succeeded for this project.

## Reproducing it

```bash
python scripts/download_models.py        # once: the two local models into the Hugging Face cache
# a PostgreSQL 16 + pgvector database, migrated:
alembic upgrade head
python -m evals.run --suite retrieval    # writes var/eval_runs/retrieval-<timestamp>.json
```

The suite seeds the fixture corpus itself and refuses to run, writing nothing, if either model cannot
be loaded.

## About the earlier files

Before this record, a development laptop had 12 untracked `var/eval_runs/retrieval-*.json` files
(2026-09-22, 09-29 and 10-06) with the same figures to the last digit. They were not deliberate runs:
four tests that expect the harness to fail hard when the models are missing ran it for real on a machine
where the models were present, and wrote a report each time. Those tests now simulate the missing
models (K3a-6), so a test run no longer writes into `var/eval_runs/`. Those 12, a deliberate run on
2026-10-13 and this one agree to the last digit (14 in all, all on one laptop): evidence that the result
is deterministic, not that it was run 14 times on purpose.
