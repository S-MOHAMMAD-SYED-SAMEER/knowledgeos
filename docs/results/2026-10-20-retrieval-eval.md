# Retrieval evaluation, 2026-10-20 (gold-v2)

The retrieval suite, run for real: real BGE embeddings, the real cross-encoder, a real PostgreSQL with pgvector.
No API key, no network call, no model API involved. This is the same suite as `2026-10-14-retrieval-eval.md`, run
again on the frozen **gold-v2** question set (see `docs/EVAL.md`), because Gold Review 2 changed the expected chunks
of two questions and so can change retrieval scores.

Raw record: [`2026-10-20-retrieval-eval.json`](2026-10-20-retrieval-eval.json), exactly as
`python -m evals.run --suite retrieval` wrote it. It records `gold_version: gold-v2` and the gold's sha256.

## What was run

| | |
|---|---|
| Command | `python -m evals.run --suite retrieval` |
| Run ID | `6e790012-1118-4f26-ae99-f6d54bc09986` |
| Gold | gold-v2, sha256 `2db847142d903c53c08cf7c8f9d781c204e98e1578c0a2a2919537eba3a6b037` |
| Embedding model | `BAAI/bge-small-en-v1.5` (384 dimensions, local) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local), compared with a passthrough that keeps the fused order |
| Fusion | Reciprocal Rank Fusion, k = 60; top 50 lexical + top 50 vector, then top 20 |
| Machine | Windows 11, Python 3.13.5, one machine |
| Database | PostgreSQL 16 (`pgvector/pgvector:pg16`) |

## Results

45 of the 52 questions are scored (the other 7 expect an abstention, so there is no chunk to find), 9 of them
carrying a metadata filter.

| Metric | Without reranking (passthrough) | With the cross-encoder |
|---|---|---|
| Recall@5 | 0.8889 | **0.9444** |
| Recall@10 | 0.9630 | **0.9926** |
| MRR | 0.7375 | **0.9081** |
| nDCG@10 | 0.7927 | **0.9142** |
| Precision@5 | 0.2311 | 0.2489 |
| Metadata-filter correctness | 1.0000 (condition-invariant, over 9 filtered questions) | |

### Against the 2026-10-14 run (gold-v1)

Only the gold changed between the two runs; the code that retrieves and reranks did not. Gold Review 2 changed four items:
two were reworded (`da005`, `mf002`, so the query text itself differs) and two had their expected chunks narrowed
(`md003`, `md004`). The Recall and nDCG figures moved and MRR did not (0.7375 without the reranker and 0.9081 with it, in
both runs). The change has not been attributed to individual questions.

| Metric (with the cross-encoder) | gold-v1, 2026-10-14 | gold-v2, 2026-10-20 |
|---|---|---|
| Recall@5 | 0.9333 | 0.9444 |
| Recall@10 | 0.9926 | 0.9926 |
| MRR | 0.9081 | 0.9081 |
| nDCG@10 | 0.9076 | 0.9142 |

A change in the question set that raises the score is not an improvement in the retriever; both runs are published
so that nobody has to take the newer figure as the only one.

## Read these numbers with the corpus in mind

- **The corpus is small.** 30 chunks. With top-10 covering a third of the whole corpus, Recall@10 is easy; do not
  read 0.99 as a prediction for a corpus of thousands of documents.
- **The questions and their expected chunks were written by one author against the same fixtures** (Claude; reviewed with
  ChatGPT assistance; final decisions by Sameer), not by an independent annotator.
- **Postgres full-text search is not BM25**, and the hybrid retriever is not compared with any other baseline here.
  The only comparison is with and without the reranker.
- **One machine.** It has not been reproduced on a different machine.

## What this does not show

Answer quality: nothing generated, no citation validity under a real model, no abstention accuracy. Those are in the
answers evaluation (`docs/EVAL.md`).

## Reproducing it

```bash
python scripts/download_models.py        # once: the two local models into the Hugging Face cache
alembic upgrade head                     # a PostgreSQL 16 + pgvector database
python -m evals.run --suite retrieval    # writes var/eval_runs/retrieval-<timestamp>.json
```
