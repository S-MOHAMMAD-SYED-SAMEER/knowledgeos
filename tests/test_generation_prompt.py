"""The versioned prompt file and evidence serialization."""

import dataclasses
import uuid

import pytest

from app.generation.evidence import SelectedEvidence
from app.generation.prompt import (
    DEFAULT_PROMPT_PATH,
    PromptError,
    load_prompt,
    serialize_evidence,
)
from app.reranking.pipeline import RerankedChunk
from app.retrieval.vector import ChunkEvidence


def _evidence(uid: str, text: str = "some text") -> ChunkEvidence:
    return ChunkEvidence(
        chunk_uid=uid,
        text=text,
        sequence=0,
        page=None,
        section=None,
        char_start=0,
        char_end=len(text),
        token_count=len(text.split()),
        document_id=uuid.uuid4(),
        document_title="Some Document",
        department=None,
        category=None,
        tags=(),
        version_id=uuid.uuid4(),
        version_number=1,
        version_status="active",
    )


def _reranked(uid: str, final_rank: int, text: str = "some text") -> RerankedChunk:
    return RerankedChunk(
        evidence=_evidence(uid, text),
        lexical_rank=1,
        vector_rank=1,
        rrf_score=1.0,
        fusion_rank=final_rank,
        rerank_score=1.0,
        final_rank=final_rank,
    )


# --- loading the file ---------------------------------------------------


def test_the_real_prompt_file_loads() -> None:
    prompt = load_prompt()
    assert prompt.text


def test_the_default_path_points_at_the_specified_filename() -> None:
    """The specification's own filename: `knowledge_answer_v1.md`."""
    assert DEFAULT_PROMPT_PATH.name == "knowledge_answer_v1.md"


def test_the_version_is_the_filename_stem() -> None:
    prompt = load_prompt()
    assert prompt.version == "knowledge_answer_v1"


def test_the_content_hash_is_stable_across_loads() -> None:
    a = load_prompt()
    b = load_prompt()
    assert a.content_hash == b.content_hash


def test_the_content_hash_changes_if_the_file_changes(tmp_path) -> None:
    path = tmp_path / "knowledge_answer_v1.md"
    path.write_text("version one content")
    first = load_prompt(path)

    path.write_text("version one content, edited")
    second = load_prompt(path)

    assert first.content_hash != second.content_hash
    assert first.version == second.version  # filename unchanged


def test_a_missing_prompt_file_raises_prompt_error(tmp_path) -> None:
    with pytest.raises(PromptError, match="not found"):
        load_prompt(tmp_path / "does_not_exist.md")


def test_the_prompt_is_never_inlined_in_python() -> None:
    """The specification: 'Never inline, never unversioned.' Checked by
    proving the loader reads from disk rather than returning a Python
    string literal that happens to match."""
    import inspect

    source = inspect.getsource(load_prompt)
    assert "read_bytes" in source or "read_text" in source


# --- the prompt's required instructions ---------------------------------


@pytest.mark.parametrize(
    "required_phrase",
    [
        "DATA",
        "chunk_uid",
        "sufficient_evidence",
        "outside knowledge",
    ],
)
def test_the_prompt_carries_the_specifications_required_instructions(
    required_phrase: str,
) -> None:
    prompt = load_prompt()
    assert required_phrase in prompt.text


def test_the_prompt_states_evidence_is_data_not_instructions() -> None:
    prompt = load_prompt()
    lowered = prompt.text.lower()
    assert "data" in lowered
    assert "not" in lowered and "instruction" in lowered


# --- evidence serialization -----------------------------------------------


def test_serialize_evidence_includes_only_selected_chunks() -> None:
    selected = [
        SelectedEvidence(candidate=_reranked("a" * 32, 1), selected=True),
        SelectedEvidence(candidate=_reranked("b" * 32, 2), selected=False),
    ]

    block = serialize_evidence(selected)

    assert "a" * 32 in block
    assert "b" * 32 not in block


def test_serialize_evidence_carries_the_chunk_uid_as_an_attribute() -> None:
    uid = "c" * 32
    selected = [SelectedEvidence(candidate=_reranked(uid, 1), selected=True)]

    block = serialize_evidence(selected)

    assert f'chunk_uid="{uid}"' in block


def test_serialize_evidence_is_ordered_by_final_rank() -> None:
    selected = [
        SelectedEvidence(candidate=_reranked("b" * 32, 2, text="second"), selected=True),
        SelectedEvidence(candidate=_reranked("a" * 32, 1, text="first"), selected=True),
    ]

    block = serialize_evidence(selected)

    assert block.index("first") < block.index("second")


def test_serialize_evidence_of_an_empty_selection_is_empty_string() -> None:
    assert serialize_evidence([]) == ""


def test_serialize_evidence_escapes_document_titles() -> None:
    candidate = _reranked("d" * 32, 1)
    poisoned_evidence = dataclasses.replace(
        candidate.evidence, document_title='a "quoted" & title'
    )
    poisoned_candidate = dataclasses.replace(candidate, evidence=poisoned_evidence)
    selected = [SelectedEvidence(candidate=poisoned_candidate, selected=True)]

    block = serialize_evidence(selected)

    assert '"quoted"' not in block
    assert "&quot;" in block
    assert "&amp;" in block
