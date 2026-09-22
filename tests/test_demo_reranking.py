"""P3 (PROJECT_PLAN.md): the demo reranker fixture
(`demo/fixtures/reranker_scores.json`) and its replay provider
(`demo.reranking.DemoRerankProvider`).

No database needed: real chunk text is reconstructed from the pinned
manifest the same way `tests/test_demo_fixtures.py` already does
(`demo.generate_embeddings.iter_manifest_chunks`), never invented.
"""

import ast
import json
import pathlib
import re

import pytest

from app.providers.cross_encoder import MODEL_NAME
from app.providers.reranker import Candidate, RerankError
from demo.generate_embeddings import FLAGSHIP_QUERIES, iter_manifest_chunks
from demo.reranking import (
    DEMO_RERANK_MODEL_NAME,
    RERANKER_SCORES_FIXTURE_PATH,
    DemoRerankProvider,
)

FLAGSHIP_QUERY_IDS = [query_id for query_id, _ in FLAGSHIP_QUERIES]
FLAGSHIP_QUERY_TEXT = dict(FLAGSHIP_QUERIES)


@pytest.fixture(scope="module")
def fixture_payload() -> dict:
    return json.loads(RERANKER_SCORES_FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chunk_text_by_uid() -> dict[str, str]:
    return {chunk.chunk_uid: chunk.text for chunk in iter_manifest_chunks()}


# --- demo/fixtures/reranker_scores.json: shape and integrity --------------


def test_fixture_reports_the_real_cross_encoder_model_name(fixture_payload) -> None:
    assert fixture_payload["model_name"] == MODEL_NAME == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_fixture_covers_exactly_the_five_flagship_queries(fixture_payload) -> None:
    fixture_ids = {entry["id"] for entry in fixture_payload["queries"]}
    assert fixture_ids == set(FLAGSHIP_QUERY_IDS)
    assert len(fixture_payload["queries"]) == 5


def test_fixture_question_text_matches_the_flagship_query_text(fixture_payload) -> None:
    for entry in fixture_payload["queries"]:
        assert entry["text"] == FLAGSHIP_QUERY_TEXT[entry["id"]]


def test_fixture_has_exactly_20_candidates_per_query(fixture_payload) -> None:
    assert fixture_payload["candidates_per_query"] == 20
    for entry in fixture_payload["queries"]:
        assert len(entry["candidates"]) == 20


def test_fixture_has_exactly_100_scores_total(fixture_payload) -> None:
    total = sum(len(entry["candidates"]) for entry in fixture_payload["queries"])
    assert total == 100


def test_fixture_has_no_duplicate_chunk_within_one_query(fixture_payload) -> None:
    for entry in fixture_payload["queries"]:
        uids = [c["chunk_uid"] for c in entry["candidates"]]
        assert len(uids) == len(set(uids)), entry["id"]


def test_fixture_chunk_uids_are_real_demo_corpus_chunks(fixture_payload, chunk_text_by_uid) -> None:
    """Not merely 32-hex-char-shaped -- chunk_uids the real demo corpus
    (the same one `demo/fixtures/embeddings.json` pins) actually has."""
    uid_pattern = re.compile(r"^[0-9a-f]{32}$")
    for entry in fixture_payload["queries"]:
        for candidate in entry["candidates"]:
            assert uid_pattern.match(candidate["chunk_uid"])
            assert candidate["chunk_uid"] in chunk_text_by_uid


def test_fixture_scores_are_real_floats_never_a_placeholder(fixture_payload) -> None:
    """A cheap but real sanity check that these are model output, not a
    constant or a repeated placeholder: no two candidates within the same
    query share a score."""
    for entry in fixture_payload["queries"]:
        scores = [c["rerank_score"] for c in entry["candidates"]]
        assert all(isinstance(s, float) for s in scores)
        assert len(set(scores)) == len(scores), entry["id"]


def test_cs001_golden_chunk_outranks_its_same_document_neighbor(fixture_payload) -> None:
    """The P1 triage's own finding: raw RRF fusion ranks
    `49446816...` (seq 1) ahead of the golden `f9cbee4e...` (seq 0) for
    cs001. This is the real cross-encoder score the whole P3 reranker
    fixture exists to prove corrects that."""
    entry = next(e for e in fixture_payload["queries"] if e["id"] == "cs001")
    by_uid = {c["chunk_uid"]: c["rerank_score"] for c in entry["candidates"]}
    golden = "f9cbee4e0939d52e95eebe93a063a85c"
    neighbor = "49446816bd2ba01147d70c4fcfeaa7fc"
    assert by_uid[golden] > by_uid[neighbor]
    assert max(by_uid, key=by_uid.get) == golden


def test_cv001_active_version_chunk_is_present_and_scored(fixture_payload) -> None:
    """The superseded v1 chunk never reaches reranking at all -- milestone
    5's default retrieval already excludes it -- so cv001's guarantee is
    that the active v2 chunk has a real score, not a ranking comparison
    against a chunk that was never a candidate."""
    entry = next(e for e in fixture_payload["queries"] if e["id"] == "cv001")
    uids = {c["chunk_uid"] for c in entry["candidates"]}
    assert "da1cdb5e0e87ad69728e9d2781239d13" in uids  # active v2
    assert "eef29ba299e4f0dd72c16e1741ac6928" not in uids  # superseded v1


# --- DemoRerankProvider: replay correctness --------------------------------


def test_model_name_is_distinct_from_the_live_model() -> None:
    provider = DemoRerankProvider()
    assert provider.model_name == DEMO_RERANK_MODEL_NAME
    assert provider.model_name != MODEL_NAME
    assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in provider.model_name


@pytest.mark.parametrize("query_id", FLAGSHIP_QUERY_IDS)
def test_all_20_stored_candidates_resolve_for_each_flagship_query(
    query_id, fixture_payload, chunk_text_by_uid
) -> None:
    provider = DemoRerankProvider()
    entry = next(e for e in fixture_payload["queries"] if e["id"] == query_id)
    candidates = [
        Candidate(id=c["chunk_uid"], text=chunk_text_by_uid[c["chunk_uid"]])
        for c in entry["candidates"]
    ]

    scored = provider.rerank(FLAGSHIP_QUERY_TEXT[query_id], candidates)

    assert len(scored) == 20
    expected = {c["chunk_uid"]: c["rerank_score"] for c in entry["candidates"]}
    for item in scored:
        assert item.score == expected[item.id]


def test_scores_are_deterministic_across_repeated_calls(fixture_payload, chunk_text_by_uid) -> None:
    provider = DemoRerankProvider()
    entry = next(e for e in fixture_payload["queries"] if e["id"] == "da001")
    candidates = [
        Candidate(id=c["chunk_uid"], text=chunk_text_by_uid[c["chunk_uid"]])
        for c in entry["candidates"]
    ]

    first = provider.rerank(FLAGSHIP_QUERY_TEXT["da001"], candidates)
    second = provider.rerank(FLAGSHIP_QUERY_TEXT["da001"], candidates)

    assert first == second


def test_an_unrecognized_query_raises(chunk_text_by_uid) -> None:
    provider = DemoRerankProvider()
    known_uid, known_text = next(iter(chunk_text_by_uid.items()))

    with pytest.raises(RerankError):
        provider.rerank(
            "a question nobody ever precomputed a demo rerank score for",
            [Candidate(id=known_uid, text=known_text)],
        )


def test_an_unrecognized_chunk_raises() -> None:
    provider = DemoRerankProvider()

    with pytest.raises(RerankError):
        provider.rerank(
            FLAGSHIP_QUERY_TEXT["da001"],
            [Candidate(id="0" * 32, text="text nobody ever precomputed a score for")],
        )


def test_the_error_never_carries_chunk_text() -> None:
    provider = DemoRerankProvider()
    secret_text = "SUPERSECRET production access policy details"

    with pytest.raises(RerankError) as raised:
        provider.rerank(FLAGSHIP_QUERY_TEXT["da001"], [Candidate(id="0" * 32, text=secret_text)])

    assert secret_text not in str(raised.value)


def test_the_demo_provider_never_imports_the_real_model_loading_code() -> None:
    """Static guard, mirroring `tests/test_reranking.py`'s own
    `test_the_real_provider_reaches_no_network_at_import`: this module
    must never import `sentence_transformers` or `torch` -- only the
    frozen `MODEL_NAME` string constant, which triggers no model load."""
    import demo.reranking as module

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


def test_the_demo_provider_never_imports_the_real_cross_encoder_provider() -> None:
    """`CrossEncoderRerankProvider` itself (not just the heavy libraries
    it lazily imports) must never appear here either -- the whole point
    of this module is to never construct or call it."""
    import demo.reranking as module

    source = pathlib.Path(module.__file__).read_text()
    assert "CrossEncoderRerankProvider" not in source
