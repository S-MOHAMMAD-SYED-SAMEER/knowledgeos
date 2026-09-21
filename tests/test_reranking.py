"""The reranking boundary: the interface, the providers, and the orchestration.

Mirrors `tests/test_embeddings.py`'s structure. The real-model test is kept
apart on purpose: the specification requires the local cross-encoder for
production and permits a passthrough/fake for tests, so these unit tests are
carried by the passthrough and the deterministic fake, and the one test that
exercises `cross-encoder/ms-marco-MiniLM-L-6-v2` skips when the model is not
in the local cache rather than pretending a fake proved anything about it.
"""

import uuid

import pytest

from app.providers import (
    RERANK_MODEL_NAME,
    Candidate,
    CrossEncoderRerankProvider,
    FakeRerankProvider,
    PassthroughRerankProvider,
    RerankError,
    RerankProvider,
    ScoredChunk,
    check_rerank_shape,
)
from app.reranking import RerankedChunk, get_rerank_provider, reset_rerank_provider, rerank
from app.retrieval.pipeline import RetrievedChunk
from app.retrieval.vector import ChunkEvidence


def _evidence(chunk_uid: str, text: str) -> ChunkEvidence:
    return ChunkEvidence(
        chunk_uid=chunk_uid,
        text=text,
        sequence=0,
        page=None,
        section=None,
        char_start=0,
        char_end=len(text),
        token_count=1,
        document_id=uuid.uuid4(),
        document_title="doc",
        department=None,
        category=None,
        tags=(),
        version_id=uuid.uuid4(),
        version_number=1,
        version_status="active",
    )


def _candidate(chunk_uid: str, text: str, *, final_rank: int = 1, **overrides) -> RetrievedChunk:
    fields = {
        "evidence": _evidence(chunk_uid, text),
        "lexical_rank": 1,
        "vector_rank": 1,
        "rrf_score": 0.1,
        "final_rank": final_rank,
    }
    fields.update(overrides)
    return RetrievedChunk(**fields)


# --- the interface ----------------------------------------------------------


def test_the_specification_fixes_the_model() -> None:
    assert RERANK_MODEL_NAME == "cross-encoder/ms-marco-MiniLM-L-6-v2"


@pytest.mark.parametrize(
    "provider",
    [FakeRerankProvider(), PassthroughRerankProvider(), CrossEncoderRerankProvider()],
)
def test_every_provider_satisfies_the_interface(provider) -> None:
    assert isinstance(provider, RerankProvider)
    assert provider.model_name


def test_the_real_provider_reports_the_specified_model() -> None:
    assert CrossEncoderRerankProvider().model_name == RERANK_MODEL_NAME


def test_the_interface_is_decoupled_from_orm_and_retrieval_types() -> None:
    """D3: the provider boundary is built around plain identity/text data,
    never a SQLAlchemy model, a Session, or `app.retrieval`'s own types."""
    import ast
    import pathlib

    module = pathlib.Path("app/providers/reranker.py")
    tree = ast.parse(module.read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    forbidden = {"sqlalchemy", "app.retrieval", "app.models", "fastapi"}
    for name in imported:
        assert not any(name == f or name.startswith(f + ".") for f in forbidden)


# --- shape checking -----------------------------------------------------


def test_a_wrong_count_is_refused() -> None:
    with pytest.raises(RerankError, match="scores"):
        check_rerank_shape(
            [ScoredChunk(id="a", score=1.0)],
            [Candidate(id="a", text="x"), Candidate(id="b", text="y")],
        )


def test_a_mismatched_identity_is_refused() -> None:
    with pytest.raises(RerankError, match="different candidates"):
        check_rerank_shape(
            [ScoredChunk(id="wrong-id", score=1.0)],
            [Candidate(id="a", text="x")],
        )


def test_a_correct_shape_passes() -> None:
    check_rerank_shape(
        [ScoredChunk(id="a", score=1.0), ScoredChunk(id="b", score=2.0)],
        [Candidate(id="a", text="x"), Candidate(id="b", text="y")],
    )


# --- one output per candidate, identity preserved ---------------------------


@pytest.mark.parametrize("provider", [FakeRerankProvider(), PassthroughRerankProvider()])
def test_one_score_per_candidate(provider) -> None:
    candidates = [Candidate(id="a", text="alpha"), Candidate(id="b", text="beta")]
    scored = provider.rerank("query", candidates)

    assert len(scored) == len(candidates)
    assert {item.id for item in scored} == {"a", "b"}


@pytest.mark.parametrize("provider", [FakeRerankProvider(), PassthroughRerankProvider()])
def test_scores_are_floats(provider) -> None:
    scored = provider.rerank("query", [Candidate(id="a", text="alpha")])
    assert isinstance(scored[0].score, float)


@pytest.mark.parametrize("provider", [FakeRerankProvider(), PassthroughRerankProvider()])
def test_empty_input_returns_empty_output(provider) -> None:
    assert provider.rerank("query", []) == []


# --- passthrough: order-preserving, D3/§3 --------------------------------


def test_passthrough_preserves_input_order() -> None:
    candidates = [
        Candidate(id="third", text="z"),
        Candidate(id="first", text="a"),
        Candidate(id="second", text="m"),
    ]
    scored = PassthroughRerankProvider().rerank("query", candidates)

    ordered = sorted(scored, key=lambda item: -item.score)
    assert [item.id for item in ordered] == ["third", "first", "second"]


def test_passthrough_never_changes_ordering_regardless_of_query() -> None:
    candidates = [Candidate(id="a", text="x"), Candidate(id="b", text="y")]
    one = PassthroughRerankProvider().rerank("query one", candidates)
    two = PassthroughRerankProvider().rerank("an entirely different query", candidates)

    assert [item.id for item in sorted(one, key=lambda i: -i.score)] == [
        item.id for item in sorted(two, key=lambda i: -i.score)
    ]


def test_passthrough_has_no_load_failure_mode() -> None:
    """No model, so nothing to fail to load."""
    PassthroughRerankProvider().rerank("query", [Candidate(id="a", text="x")])


# --- the deterministic test reranker: proves reordering is possible, D8 --


def test_the_fake_reranker_is_deterministic() -> None:
    candidates = [Candidate(id="a", text="alpha"), Candidate(id="b", text="beta")]
    first = FakeRerankProvider().rerank("query", candidates)
    second = FakeRerankProvider().rerank("query", candidates)

    assert sorted(first) == sorted(second)


def test_the_fake_reranker_can_reorder_candidates() -> None:
    """Unlike passthrough, this one's score depends on content, not
    position — so it can, and here does, put the second input candidate
    ahead of the first."""
    candidates = [Candidate(id="a", text="text-a-0"), Candidate(id="b", text="text-b-0")]
    scored = FakeRerankProvider().rerank("query", candidates)

    ordered = sorted(scored, key=lambda item: -item.score)
    assert [item.id for item in ordered] == ["b", "a"]


def test_higher_score_sorts_first() -> None:
    candidates = [Candidate(id="low", text="x"), Candidate(id="high", text="y")]
    scored = [ScoredChunk(id="low", score=0.1), ScoredChunk(id="high", score=0.9)]
    ordered = sorted(scored, key=lambda item: -item.score)
    assert [item.id for item in ordered] == ["high", "low"]


def test_equal_scores_break_ties_on_id_ascending() -> None:
    scored = [ScoredChunk(id="b", score=1.0), ScoredChunk(id="a", score=1.0)]
    ordered = sorted(scored, key=lambda item: (-item.score, item.id))
    assert [item.id for item in ordered] == ["a", "b"]


def test_nothing_in_the_application_selects_the_fake_or_passthrough() -> None:
    """Only a test may choose either. The application's own dependency
    (`app.reranking.get_rerank_provider`) always resolves to the real
    cross-encoder."""
    import ast
    import pathlib

    app = pathlib.Path(__file__).resolve().parent.parent / "app"
    for module in app.rglob("*.py"):
        if module.parent.name == "providers" or module.name in (
            "fake_reranker.py",
            "passthrough_reranker.py",
        ):
            continue
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id not in ("FakeRerankProvider", "PassthroughRerankProvider"), (
                    module.name
                )


# --- provider caching ----------------------------------------------------


def test_the_cached_accessor_returns_the_same_instance() -> None:
    reset_rerank_provider()
    try:
        first = get_rerank_provider()
        second = get_rerank_provider()
        assert first is second
    finally:
        reset_rerank_provider()


def test_reset_clears_the_cache() -> None:
    reset_rerank_provider()
    try:
        first = get_rerank_provider()
        reset_rerank_provider()
        second = get_rerank_provider()
        assert first is not second
    finally:
        reset_rerank_provider()


def test_the_cached_accessor_is_the_real_cross_encoder() -> None:
    reset_rerank_provider()
    try:
        assert isinstance(get_rerank_provider(), CrossEncoderRerankProvider)
    finally:
        reset_rerank_provider()


# --- the real model -------------------------------------------------------


def _model_is_cached() -> bool:
    """Whether the real model can actually be loaded from the local cache."""
    try:
        CrossEncoderRerankProvider().rerank("probe", [Candidate(id="a", text="probe text")])
    except Exception:  # noqa: BLE001
        return False
    return True


@pytest.mark.skipif(
    not _model_is_cached(),
    reason=(
        "cross-encoder/ms-marco-MiniLM-L-6-v2 is not in the local cache and "
        "this environment cannot reach HuggingFace (the gateway refuses the "
        "connection). The real model has NOT been exercised."
    ),
)
def test_the_real_model_scores_a_relevant_pair_higher() -> None:
    """The only test that touches the real model. Everything else uses the
    passthrough or the fake, and this one skipping is reported rather than
    hidden."""
    provider = CrossEncoderRerankProvider()

    scored = provider.rerank(
        "how do I get production database access",
        [
            Candidate(id="relevant", text="Production database access requires manager approval."),
            Candidate(id="irrelevant", text="The office coffee machine is on the third floor."),
        ],
    )

    by_id = {item.id: item.score for item in scored}
    assert by_id["relevant"] > by_id["irrelevant"]


def test_a_missing_model_fails_cleanly_rather_than_downloading() -> None:
    """Offline by construction: a model that is not cached is an error, not
    a few hundred megabytes fetched mid-query."""
    provider = CrossEncoderRerankProvider(model_name="this-model-does-not-exist/nowhere")

    with pytest.raises(RerankError, match="could not be loaded"):
        provider.rerank("query", [Candidate(id="a", text="text")])


def test_the_real_provider_reaches_no_network_at_import() -> None:
    """The heavy import is deferred, so importing the application does not
    pull in torch."""
    import ast
    import pathlib

    from app.providers import cross_encoder as module

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


def test_the_error_never_carries_document_text() -> None:
    """A structural message only -- the specification forbids logging or
    surfacing full chunk text."""
    provider = CrossEncoderRerankProvider(model_name="this-model-does-not-exist/nowhere")
    secret_text = "SUPERSECRET production access policy details"

    with pytest.raises(RerankError) as raised:
        provider.rerank("query", [Candidate(id="a", text=secret_text)])

    assert secret_text not in str(raised.value)
    assert "Traceback" not in str(raised.value)


# --- orchestration: app/reranking/pipeline.py ------------------------------


def test_rerank_maps_scores_back_to_the_correct_candidates() -> None:
    candidates = [_candidate("a", "alpha", final_rank=1), _candidate("b", "beta", final_rank=2)]
    provider = FakeRerankProvider()

    result = rerank("query", candidates, provider)

    expected = {item.id: item.score for item in provider.rerank("query", [
        Candidate(id=c.evidence.chunk_uid, text=c.evidence.text) for c in candidates
    ])}
    for chunk in result:
        assert chunk.rerank_score == expected[chunk.evidence.chunk_uid]


class _InvertingProvider:
    """Scores each candidate by its position, last scored highest -- so
    sorting by score descending reverses whatever order it was given."""

    model_name = "inverting"

    def rerank(self, query, provider_candidates):
        del query
        return [
            ScoredChunk(id=c.id, score=float(index))
            for index, c in enumerate(provider_candidates)
        ]


def test_rerank_sorts_by_score_descending() -> None:
    candidates = [_candidate("a", "alpha", final_rank=1), _candidate("b", "beta", final_rank=2)]

    result = rerank("query", candidates, _InvertingProvider())

    scores = [chunk.rerank_score for chunk in result]
    assert scores == sorted(scores, reverse=True)


def test_rerank_can_change_order_relative_to_fusion_order() -> None:
    candidates = [_candidate("a", "alpha", final_rank=1), _candidate("b", "beta", final_rank=2)]

    result = rerank("query", candidates, _InvertingProvider())

    assert [chunk.evidence.chunk_uid for chunk in result] == ["b", "a"]
    # Fusion order is preserved as data, even though final order changed.
    assert result[0].fusion_rank == 2
    assert result[1].fusion_rank == 1


def test_rerank_preserves_order_with_passthrough() -> None:
    candidates = [
        _candidate("a", "alpha", final_rank=1),
        _candidate("b", "beta", final_rank=2),
        _candidate("c", "gamma", final_rank=3),
    ]
    result = rerank("query", candidates, PassthroughRerankProvider())

    assert [chunk.evidence.chunk_uid for chunk in result] == ["a", "b", "c"]
    assert [chunk.final_rank for chunk in result] == [1, 2, 3]
    assert [chunk.fusion_rank for chunk in result] == [1, 2, 3]


def test_final_rank_is_contiguous_from_one() -> None:
    candidates = [_candidate(f"uid-{i}", f"text {i}", final_rank=i + 1) for i in range(7)]
    result = rerank("query", candidates, FakeRerankProvider())

    assert [chunk.final_rank for chunk in result] == list(range(1, 8))


def test_fewer_than_twenty_candidates_works() -> None:
    candidates = [_candidate("a", "alpha", final_rank=1)]
    result = rerank("query", candidates, PassthroughRerankProvider())
    assert len(result) == 1


def test_twenty_candidates_works() -> None:
    candidates = [_candidate(f"uid-{i}", f"text {i}", final_rank=i + 1) for i in range(20)]
    result = rerank("query", candidates, PassthroughRerankProvider())
    assert len(result) == 20


def test_zero_candidates_returns_empty_and_never_calls_the_provider() -> None:
    calls = []

    class Spy:
        model_name = "spy"

        def rerank(self, query, provider_candidates):
            calls.append(provider_candidates)
            return []

    result = rerank("query", [], Spy())

    assert result == []
    assert calls == []  # the provider was never invoked


def test_deterministic_tie_break_on_chunk_uid_ascending() -> None:
    """Two candidates the provider scores identically come back in
    chunk_uid order."""
    candidates = [_candidate("zzz", "z", final_rank=1), _candidate("aaa", "a", final_rank=2)]

    class Tied:
        model_name = "tied"

        def rerank(self, query, provider_candidates):
            return [ScoredChunk(id=c.id, score=1.0) for c in provider_candidates]

    result = rerank("query", candidates, Tied())
    assert [chunk.evidence.chunk_uid for chunk in result] == ["aaa", "zzz"]


def test_lexical_and_vector_ranks_are_preserved_through_reranking() -> None:
    candidate = _candidate(
        "a", "alpha", final_rank=1, lexical_rank=3, vector_rank=7, rrf_score=0.042
    )
    result = rerank("query", [candidate], PassthroughRerankProvider())

    assert result[0].lexical_rank == 3
    assert result[0].vector_rank == 7
    assert result[0].rrf_score == 0.042


def test_fusion_rank_equals_m5s_own_final_rank() -> None:
    """D2: `RetrievedChunk.final_rank` (M5's meaning, unchanged) becomes
    `RerankedChunk.fusion_rank` -- never silently reused as the new
    `final_rank`."""
    candidate = _candidate("a", "alpha", final_rank=5)
    result = rerank("query", [candidate], PassthroughRerankProvider())

    assert result[0].fusion_rank == 5
    assert isinstance(result[0], RerankedChunk)
