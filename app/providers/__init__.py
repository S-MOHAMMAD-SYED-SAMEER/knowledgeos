"""Interfaces to the things this system does not implement itself.

The specification's rule is that every provider gets an interface. Milestone 4
added the first one: embeddings. Milestone 6 adds the second: reranking. The
language model arrives with the milestone that calls it.

Two provider families share this package, and their like-named exports
(`MODEL_NAME`, `check_shape`) are re-exported here under distinct names —
`RERANK_MODEL_NAME`/`check_rerank_shape` — so importing both from
`app.providers` cannot silently shadow one with the other. Code that only
needs one family can still import its plain name directly from the specific
submodule, as `app/ingestion/runner.py` and `app/reranking/provider.py` do.
"""

from app.providers.bge import BgeEmbeddingProvider
from app.providers.cross_encoder import CrossEncoderRerankProvider
from app.providers.cross_encoder import MODEL_NAME as RERANK_MODEL_NAME
from app.providers.embeddings import (
    DIMENSIONS,
    MODEL_NAME,
    EmbeddingError,
    EmbeddingProvider,
    check_shape,
)
from app.providers.fake_embeddings import FakeEmbeddingProvider
from app.providers.fake_llm import FAKE_LLM_MODEL_NAME, FakeLLMProvider
from app.providers.fake_reranker import FakeRerankProvider
from app.providers.gemini_llm import GeminiLLMProvider
from app.providers.llm import LLMError, LLMProvider, LLMResult
from app.providers.passthrough_reranker import PassthroughRerankProvider
from app.providers.reranker import Candidate, RerankError, RerankProvider, ScoredChunk
from app.providers.reranker import check_shape as check_rerank_shape

__all__ = [
    "DIMENSIONS",
    "FAKE_LLM_MODEL_NAME",
    "MODEL_NAME",
    "RERANK_MODEL_NAME",
    "BgeEmbeddingProvider",
    "Candidate",
    "CrossEncoderRerankProvider",
    "EmbeddingError",
    "EmbeddingProvider",
    "FakeEmbeddingProvider",
    "FakeLLMProvider",
    "FakeRerankProvider",
    "GeminiLLMProvider",
    "LLMError",
    "LLMProvider",
    "LLMResult",
    "PassthroughRerankProvider",
    "RerankError",
    "RerankProvider",
    "ScoredChunk",
    "check_rerank_shape",
    "check_shape",
]
