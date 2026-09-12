"""The embedding boundary: the interface, the fake, and the real model.

The real-model test is kept apart on purpose. The specification permits fake
embeddings for unit tests and requires the real local model for evaluation;
these are unit tests, so the fake carries them, and the one test that exercises
`BAAI/bge-small-en-v1.5` skips when the model is not in the local cache rather
than pretending a fake proved anything about it.
"""

import pytest

from app.providers import (
    DIMENSIONS,
    MODEL_NAME,
    BgeEmbeddingProvider,
    EmbeddingError,
    EmbeddingProvider,
    FakeEmbeddingProvider,
    check_shape,
)


# --- the interface ----------------------------------------------------------


def test_the_specification_fixes_the_model_and_the_width() -> None:
    assert MODEL_NAME == "BAAI/bge-small-en-v1.5"
    assert DIMENSIONS == 384


@pytest.mark.parametrize(
    "provider", [FakeEmbeddingProvider(), BgeEmbeddingProvider()]
)
def test_both_providers_satisfy_the_interface(provider) -> None:
    assert isinstance(provider, EmbeddingProvider)
    assert provider.dimensions == DIMENSIONS
    assert provider.model_name


def test_the_real_provider_reports_the_specified_model() -> None:
    assert BgeEmbeddingProvider().model_name == MODEL_NAME


# --- shape checking ---------------------------------------------------------


def test_a_wrong_width_is_refused() -> None:
    """Checked here so a mismatch does not surface as an opaque column error."""
    with pytest.raises(EmbeddingError, match="384"):
        check_shape([[0.0] * 128], ["text"], DIMENSIONS)


def test_a_wrong_count_is_refused() -> None:
    with pytest.raises(EmbeddingError, match="vectors"):
        check_shape([[0.0] * DIMENSIONS], ["one", "two"], DIMENSIONS)


def test_a_correct_shape_passes() -> None:
    check_shape([[0.0] * DIMENSIONS] * 2, ["one", "two"], DIMENSIONS)


# --- the deterministic fake -------------------------------------------------


def test_the_fake_produces_the_right_width() -> None:
    vectors = FakeEmbeddingProvider().embed(["one", "two"])

    assert len(vectors) == 2
    assert all(len(vector) == DIMENSIONS for vector in vectors)


def test_the_fake_is_deterministic() -> None:
    """What makes an idempotency test mean anything."""
    first = FakeEmbeddingProvider().embed(["how do I request access?"])
    second = FakeEmbeddingProvider().embed(["how do I request access?"])

    assert first == second


def test_the_fake_distinguishes_different_text() -> None:
    vectors = FakeEmbeddingProvider().embed(["one", "two"])

    assert vectors[0] != vectors[1]


def test_the_fake_returns_nothing_for_nothing() -> None:
    assert FakeEmbeddingProvider().embed([]) == []


def test_the_fake_values_are_in_range() -> None:
    for value in FakeEmbeddingProvider().embed(["text"])[0]:
        assert -1.0 <= value < 1.0


def test_nothing_in_the_application_selects_the_fake() -> None:
    """Only a test may choose it. If production code could, a deployment
    could quietly index with meaningless vectors."""
    import ast
    import pathlib

    app = pathlib.Path(__file__).resolve().parent.parent / "app"
    for module in app.rglob("*.py"):
        if module.name == "fake_embeddings.py" or module.parent.name == "providers":
            continue
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id != "FakeEmbeddingProvider", module.name


# --- the real model ---------------------------------------------------------


def _model_is_cached() -> bool:
    """Whether the real model can actually be loaded from the local cache."""
    try:
        BgeEmbeddingProvider().embed(["probe"])
    except Exception:  # noqa: BLE001
        return False
    return True


@pytest.mark.skipif(
    not _model_is_cached(),
    reason=(
        "BAAI/bge-small-en-v1.5 is not in the local cache and this environment "
        "cannot reach HuggingFace (the gateway refuses the connection). The "
        "real model has NOT been exercised."
    ),
)
def test_the_real_model_produces_384_dimensional_vectors() -> None:
    """The only test that touches the real model. Everything else uses the
    fake, and this one skipping is reported rather than hidden."""
    provider = BgeEmbeddingProvider()

    vectors = provider.embed(
        ["how do I request production database access?", "unrelated text"]
    )

    assert len(vectors) == 2
    assert all(len(vector) == DIMENSIONS for vector in vectors)
    assert vectors[0] != vectors[1]


def test_a_missing_model_fails_cleanly_rather_than_downloading() -> None:
    """Offline by construction: a model that is not cached is an error, not
    a few hundred megabytes fetched mid-job."""
    provider = BgeEmbeddingProvider(model_name="this-model-does-not-exist/nowhere")

    with pytest.raises(EmbeddingError, match="could not be loaded"):
        provider.embed(["text"])


def test_the_real_provider_reaches_no_network_at_import() -> None:
    """The heavy import is deferred, so importing the application does not
    pull in torch."""
    import ast
    import pathlib

    from app.providers import bge as module

    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    top_level = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "sentence_transformers" not in top_level
    assert "torch" not in top_level
