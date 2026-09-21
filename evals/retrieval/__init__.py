"""Retrieval evaluation: metrics, the fixture corpus, the question set, and
the suite that runs both the M5 baseline and the M6 reranked condition over
them.

`metrics.py` is pure — no database, no provider, no filesystem — per the
specification's rule that the parts of this system that get evaluated must
be callable as functions over fixture data. `corpus.py` and `suite.py` are
where provider calls and database I/O actually happen; the layering rule
that forbids them names `app/retrieval/`, `app/chunking/` and
`app/generation/citations.py` specifically, and does not extend to this
package.
"""
