"""Citation validation: the specification's three deterministic rules,
plus this project's declared/inline agreement — exact, and provider-free.
"""

import ast
import pathlib

import pytest

from app.generation.citations import (
    CitationError,
    abstention_flag_consistent,
    citation_coverage,
    citation_validity,
    extract_all_inline_citations,
    extract_inline_citations,
    is_fully_covered,
    split_sentences,
    validate_citations,
)

UID_A = "a" * 32
UID_B = "b" * 32
UID_C = "c" * 32
UID_UNKNOWN = "f" * 32


# --- purity: no provider call, no I/O -----------------------------------


def test_citations_module_makes_no_provider_call_and_no_io() -> None:
    """The specification's layering rule (§4) names
    `app/generation/citations.py` explicitly: no provider calls, no I/O
    beyond the database -- and this module touches no database either."""
    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "app"
        / "generation"
        / "citations.py"
    )
    tree = ast.parse(path.read_text())
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    for forbidden in ("app.providers", "app.db", "sqlalchemy", "httpx", "requests"):
        assert not any(m.startswith(forbidden) for m in imported), imported


# --- sentence splitting --------------------------------------------------


def test_split_sentences_splits_on_terminal_punctuation() -> None:
    text = f"Access requires approval [{UID_A}]. It expires in three days [{UID_A}]."
    sentences = split_sentences(text)
    assert len(sentences) == 2


def test_split_sentences_of_empty_text_is_empty_list() -> None:
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_of_one_sentence_is_one_element() -> None:
    assert split_sentences(f"Just one claim [{UID_A}].") == [f"Just one claim [{UID_A}]."]


# --- marker extraction -----------------------------------------------------


def test_extract_inline_citations_finds_markers() -> None:
    assert extract_inline_citations(f"A fact [{UID_A}] and another [{UID_B}].") == {
        UID_A,
        UID_B,
    }


def test_extract_inline_citations_of_no_markers_is_empty() -> None:
    assert extract_inline_citations("no markers here.") == frozenset()


def test_extract_all_inline_citations_spans_the_whole_answer() -> None:
    text = f"First [{UID_A}]. Second [{UID_B}]. Third [{UID_A}]."
    assert extract_all_inline_citations(text) == {UID_A, UID_B}


def test_marker_regex_requires_the_exact_chunk_uid_shape() -> None:
    """Not 'a valid-looking bracket' -- exactly 32 lowercase hex chars,
    the same shape `app.chunking.uid.chunk_uid` produces."""
    assert extract_inline_citations("[not-a-uid]") == frozenset()
    assert extract_inline_citations("[ABCDEF0123456789ABCDEF0123456789]") == frozenset()  # uppercase
    assert extract_inline_citations(f"[{UID_A[:31]}]") == frozenset()  # too short


# --- coverage --------------------------------------------------------------


def test_coverage_is_one_when_every_sentence_is_cited() -> None:
    text = f"First [{UID_A}]. Second [{UID_B}]."
    assert citation_coverage(text) == 1.0
    assert is_fully_covered(text)


def test_coverage_is_partial_when_one_sentence_lacks_a_citation() -> None:
    text = f"First [{UID_A}]. Second has none."
    assert citation_coverage(text) == 0.5
    assert not is_fully_covered(text)


def test_coverage_of_empty_answer_is_one() -> None:
    """Nothing to cover -- vacuously covered, the same reasoning an
    empty-set precision metric would use."""
    assert citation_coverage("") == 1.0
    assert is_fully_covered("")


def test_coverage_is_zero_when_no_sentence_is_cited() -> None:
    assert citation_coverage("No citations anywhere in this answer.") == 0.0


# --- citation_validity (rules 1 and 3, plus declared/inline agreement) ----


def test_citation_validity_true_when_declared_matches_retrieved_selected_and_inline() -> None:
    text = f"Fact [{UID_A}]."
    assert citation_validity(
        answer_text=text,
        declared_citations=[UID_A],
        retrieved_chunk_uids=frozenset({UID_A, UID_B}),
        selected_chunk_uids=frozenset({UID_A}),
    )


def test_citation_validity_false_for_a_chunk_uid_not_retrieved() -> None:
    """'Nonexistent citation' scenario -- a chunk_uid never in the
    retrieved evidence set at all."""
    text = f"Fact [{UID_UNKNOWN}]."
    assert not citation_validity(
        answer_text=text,
        declared_citations=[UID_UNKNOWN],
        retrieved_chunk_uids=frozenset({UID_A, UID_B}),
        selected_chunk_uids=frozenset({UID_A}),
    )


def test_citation_validity_false_for_a_retrieved_but_unselected_chunk() -> None:
    """Rule 3 -- retrieved, but not among the chunks sent to the model."""
    text = f"Fact [{UID_B}]."
    assert not citation_validity(
        answer_text=text,
        declared_citations=[UID_B],
        retrieved_chunk_uids=frozenset({UID_A, UID_B}),
        selected_chunk_uids=frozenset({UID_A}),  # UID_B retrieved but not selected
    )


def test_citation_validity_false_when_declared_and_inline_disagree() -> None:
    text = f"Fact [{UID_A}]."  # inline only cites A
    assert not citation_validity(
        answer_text=text,
        declared_citations=[UID_A, UID_B],  # declares A and B
        retrieved_chunk_uids=frozenset({UID_A, UID_B}),
        selected_chunk_uids=frozenset({UID_A, UID_B}),
    )


def test_citation_validity_false_for_a_malformed_chunk_uid_shape() -> None:
    assert not citation_validity(
        answer_text="Fact [not-a-real-uid].",
        declared_citations=["not-a-real-uid"],
        retrieved_chunk_uids=frozenset({UID_A}),
        selected_chunk_uids=frozenset({UID_A}),
    )


# --- abstention-flag consistency -------------------------------------------


def test_abstention_consistent_when_abstained_with_no_citations() -> None:
    assert abstention_flag_consistent(
        abstained=True, declared_citations=[], answer_text="No evidence found."
    )


def test_abstention_inconsistent_when_abstained_but_citations_declared() -> None:
    assert not abstention_flag_consistent(
        abstained=True, declared_citations=[UID_A], answer_text="No evidence found."
    )


def test_abstention_inconsistent_when_abstained_but_inline_markers_present() -> None:
    assert not abstention_flag_consistent(
        abstained=True, declared_citations=[], answer_text=f"No evidence [{UID_A}]."
    )


def test_abstention_consistency_places_no_constraint_when_not_abstained() -> None:
    assert abstention_flag_consistent(
        abstained=False, declared_citations=[], answer_text="anything"
    )


# --- validate_citations: the hard, raising gate ----------------------------


def test_validate_citations_passes_a_fully_valid_non_abstained_answer() -> None:
    text = f"Access requires approval [{UID_A}]. It lasts three days [{UID_B}]."
    validate_citations(
        answer_text=text,
        declared_citations=[UID_A, UID_B],
        retrieved_chunk_uids=frozenset({UID_A, UID_B, UID_C}),
        selected_chunk_uids=frozenset({UID_A, UID_B}),
        abstained=False,
    )  # must not raise


def test_validate_citations_passes_a_valid_abstention() -> None:
    validate_citations(
        answer_text="There is insufficient evidence to answer this question.",
        declared_citations=[],
        retrieved_chunk_uids=frozenset(),
        selected_chunk_uids=frozenset(),
        abstained=True,
    )  # must not raise


def test_validate_citations_rejects_a_nonexistent_citation() -> None:
    with pytest.raises(CitationError, match="not in the retrieved evidence"):
        validate_citations(
            answer_text=f"Fact [{UID_UNKNOWN}].",
            declared_citations=[UID_UNKNOWN],
            retrieved_chunk_uids=frozenset({UID_A}),
            selected_chunk_uids=frozenset({UID_A}),
            abstained=False,
        )


def test_validate_citations_rejects_a_citation_to_an_unselected_chunk() -> None:
    with pytest.raises(CitationError, match="not selected"):
        validate_citations(
            answer_text=f"Fact [{UID_B}].",
            declared_citations=[UID_B],
            retrieved_chunk_uids=frozenset({UID_A, UID_B}),
            selected_chunk_uids=frozenset({UID_A}),
            abstained=False,
        )


def test_validate_citations_rejects_a_declared_inline_mismatch() -> None:
    with pytest.raises(CitationError, match="do not match"):
        validate_citations(
            answer_text=f"Fact [{UID_A}].",
            declared_citations=[UID_A, UID_B],
            retrieved_chunk_uids=frozenset({UID_A, UID_B}),
            selected_chunk_uids=frozenset({UID_A, UID_B}),
            abstained=False,
        )


def test_validate_citations_rejects_incomplete_coverage() -> None:
    """'Citation coverage failure' scenario -- one sentence with no
    citation at all in a non-abstained answer."""
    with pytest.raises(CitationError, match="no citation"):
        validate_citations(
            answer_text=f"First fact [{UID_A}]. Second fact has none.",
            declared_citations=[UID_A],
            retrieved_chunk_uids=frozenset({UID_A}),
            selected_chunk_uids=frozenset({UID_A}),
            abstained=False,
        )


def test_validate_citations_rejects_an_abstention_carrying_citations() -> None:
    with pytest.raises(CitationError, match="must carry no citations"):
        validate_citations(
            answer_text=f"Some fact [{UID_A}].",
            declared_citations=[UID_A],
            retrieved_chunk_uids=frozenset({UID_A}),
            selected_chunk_uids=frozenset({UID_A}),
            abstained=True,
        )


def test_validate_citations_rejects_a_malformed_declared_uid_shape() -> None:
    with pytest.raises(CitationError, match="not a chunk_uid shape"):
        validate_citations(
            answer_text="Fact [not-shaped-right].",
            declared_citations=["not-shaped-right"],
            retrieved_chunk_uids=frozenset({UID_A}),
            selected_chunk_uids=frozenset({UID_A}),
            abstained=False,
        )


def test_validate_citations_allows_multiple_citations_on_one_sentence() -> None:
    text = f"A fact supported by two chunks [{UID_A}][{UID_B}]."
    validate_citations(
        answer_text=text,
        declared_citations=[UID_A, UID_B],
        retrieved_chunk_uids=frozenset({UID_A, UID_B}),
        selected_chunk_uids=frozenset({UID_A, UID_B}),
        abstained=False,
    )  # must not raise


def test_validate_citations_error_never_contains_the_full_answer_text() -> None:
    """Errors name the offending chunk_uid or sentence, never a
    document's worth of evidence."""
    long_evidence_flavoured_text = "SECRET " * 500 + f"[{UID_UNKNOWN}]."
    with pytest.raises(CitationError) as exc_info:
        validate_citations(
            answer_text=long_evidence_flavoured_text,
            declared_citations=[UID_UNKNOWN],
            retrieved_chunk_uids=frozenset({UID_A}),
            selected_chunk_uids=frozenset({UID_A}),
            abstained=False,
        )
    assert "SECRET" not in str(exc_info.value)
