"""KnowledgeOS evaluation harness.

Milestone 7: measures retrieval quality — the actual `app.retrieval` and
`app.reranking` pipelines, over a fixture corpus and a labelled question
set — before generation exists to paper over a weak retriever.

Run from the command line, never from HTTP: `python -m evals.run --suite
retrieval`. There is no `/evals/run` endpoint, by specification.

Answer quality, citations, abstention, cost and latency are explicitly out
of scope here — `evals/answers/` and everything it would measure belongs to
milestone 9, once generation exists.
"""

__all__: list[str] = []
