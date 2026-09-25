"""Demo mode: a self-contained, credential-free public demonstration of the
real KnowledgeOS pipeline against a fixed, synthetic document corpus.

See `app/demo/seed.py` for how the corpus becomes available, and
`app/generation/demo_scenarios.py` / `app/providers/demo_llm.py` for how
generation is answered without a real model.
"""

from app.demo.seed import ensure_demo_corpus_seeded

__all__ = ["ensure_demo_corpus_seeded"]
