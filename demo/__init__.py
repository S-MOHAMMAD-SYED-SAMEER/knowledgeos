"""The KnowledgeOS demo package.

Presentation and packaging around the existing, frozen M1-10 retrieval and
generation architecture -- not a replacement for it. Every module here
lives outside `app/`, uses real precomputed BAAI/bge-small-en-v1.5
embeddings (never `FakeEmbeddingProvider`), and exists to demonstrate the
product goal chain offline:

    retrieved correctly -> ranked correctly -> answered from evidence ->
    cited correctly -> abstained when evidence is insufficient.

See `PROJECT_PLAN.md` (P1-P4) for the full scope and acceptance criteria.
"""

from demo.providers import DemoEmbeddingProvider
from demo.seed import seed_demo_corpus

__all__ = ["DemoEmbeddingProvider", "seed_demo_corpus"]
