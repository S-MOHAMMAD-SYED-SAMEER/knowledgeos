"""Milestone 9: the answer evaluation suite.

Measures what milestone 8's generation pipeline actually produces —
grounded-answer rate, citation validity and coverage, abstention
precision/recall, latency, tokens, and cost — over the frozen milestone 7
corpus and question set, through the real retrieve → rerank → generate
pipeline. Nothing here reimplements retrieval, reranking, generation, or
citation validation; every one of those is milestone 5/6/8's own function,
called unmodified.

`evals/answers/semantic.py` is the one module that calls a provider
directly (the local cross-encoder, reused as an NLI-style scorer) —
everything else here is orchestration and pure metric computation, the
same split `evals/retrieval/` already keeps between `metrics.py` (pure)
and `corpus.py`/`suite.py` (provider-calling orchestration).

Run from the command line, never from HTTP: `python -m evals.run --suite
answers`. There is no `/evals/run` endpoint, by specification.
"""

__all__: list[str] = []
