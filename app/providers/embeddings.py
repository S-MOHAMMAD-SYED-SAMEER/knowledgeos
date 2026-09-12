"""The embedding boundary.

The specification is explicit about two things here: embeddings come from a
**local** model, and they come through an `EmbeddingProvider` interface — not
from an API, and never from Anthropic, which has no embeddings API and is used
for generation only. A Voyage adapter is named as something that *may* be added
later; it is not this milestone's work.

The interface exists so that the thing which produces vectors can be swapped
without anything above it noticing. Two implementations ship here:

* `BgeEmbeddingProvider` — `BAAI/bge-small-en-v1.5`, 384 dimensions, run
  locally through sentence-transformers.
* `FakeEmbeddingProvider` — deterministic vectors from a hash, for tests.

The fake is permitted by the specification for **unit tests only**: retrieval
metrics computed over fake vectors are meaningless, and the evaluation
milestone must use the real model. Nothing in the application chooses the fake;
only a test does.
"""

from typing import Protocol, runtime_checkable

# The specification fixes both the model and the width of what it produces.
DIMENSIONS = 384
MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingError(RuntimeError):
    """An embedding could not be produced.

    The message names what failed structurally — a model that would not load,
    a vector of the wrong width — and never the text that was being embedded.
    """


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors."""

    @property
    def dimensions(self) -> int:
        """How wide the vectors are. Must match the `chunks.embedding` column."""
        ...

    @property
    def model_name(self) -> str:
        """Which model produced them, for the record."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in the order given.

        Raises `EmbeddingError` rather than returning something the caller
        would have to guess about.
        """
        ...


def check_shape(vectors: list[list[float]], texts: list[str], dimensions: int) -> None:
    """Refuse a result that does not match what was asked for.

    Checked here rather than left to PostgreSQL: a width mismatch would
    otherwise surface as an opaque error from a column definition, several
    layers from the provider that caused it.
    """
    if len(vectors) != len(texts):
        raise EmbeddingError(
            f"the provider returned {len(vectors)} vectors for {len(texts)} texts"
        )
    for vector in vectors:
        if len(vector) != dimensions:
            raise EmbeddingError(
                f"the provider returned a {len(vector)}-dimensional vector, "
                f"but {dimensions} are required"
            )


__all__ = [
    "DIMENSIONS",
    "MODEL_NAME",
    "EmbeddingError",
    "EmbeddingProvider",
    "check_shape",
]
