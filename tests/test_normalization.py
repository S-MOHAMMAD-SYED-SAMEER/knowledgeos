"""The twelve normalization rules, one test each, plus idempotence.

Every checksum and every chunk_uid in the corpus is computed over this
function's output, so each rule is asserted on its own rather than through a
single omnibus example. If one changes, exactly one test says so.
"""

import unicodedata

from app.parsing import normalize


# --- rule 1: line endings ---------------------------------------------------


def test_crlf_becomes_lf() -> None:
    assert normalize("a\r\nb") == "a\nb"


def test_bare_cr_becomes_lf() -> None:
    assert normalize("a\rb") == "a\nb"


def test_mixed_line_endings_all_become_lf() -> None:
    assert normalize("a\r\nb\rc\nd") == "a\nb\nc\nd"
    assert "\r" not in normalize("a\r\nb\rc")


# --- rule 2: NFC ------------------------------------------------------------


def test_decomposed_characters_are_composed() -> None:
    """Two spellings of the same character must hash to the same value."""
    decomposed = "e" + "́"  # e + combining acute
    composed = "é"  # é

    assert normalize(decomposed) == normalize(composed) == composed


def test_the_result_is_in_composed_form() -> None:
    assert unicodedata.is_normalized("NFC", normalize("café résumé"))


# --- rule 3: control and format characters ----------------------------------


def test_nul_is_removed() -> None:
    assert normalize("a\x00b") == "ab"


def test_a_zero_width_joiner_is_removed() -> None:
    assert normalize("a‍b") == "ab"


def test_a_bidirectional_override_is_removed() -> None:
    """The trick used to make text display as something else."""
    assert "‮" not in normalize("invoice‮gnp.exe")


def test_newline_and_tab_survive() -> None:
    """Both are control characters, and both carry structure a reader needs."""
    assert normalize("a\tb\nc") == "a\tb\nc"


# --- rule 4: trailing whitespace --------------------------------------------


def test_trailing_whitespace_is_stripped_from_every_line() -> None:
    assert normalize("a   \nb\t\nc") == "a\nb\nc"


def test_leading_whitespace_on_a_line_is_kept() -> None:
    """Indentation is meaningful — in a code block, in a list, in a quote."""
    assert normalize("a\n    indented") == "a\n    indented"


# --- rule 5: blank-line runs ------------------------------------------------


def test_a_run_of_blank_lines_collapses_to_one() -> None:
    assert normalize("a\n\n\n\n\nb") == "a\n\nb"


def test_a_single_blank_line_is_left_alone() -> None:
    """It separates paragraphs, which is what the chunker looks for."""
    assert normalize("a\n\nb") == "a\n\nb"


def test_blank_lines_of_spaces_also_collapse() -> None:
    assert normalize("a\n   \n   \nb") == "a\n\nb"


# --- rule 6: outer whitespace -----------------------------------------------


def test_outer_whitespace_is_stripped() -> None:
    assert normalize("\n\n  hello  \n\n") == "hello"


# --- rules 7 to 11: what must survive ---------------------------------------


def test_internal_spacing_between_words_is_preserved() -> None:
    """Collapsing internal runs would change offsets and quoted text."""
    assert normalize("a  b") == "a  b"


def test_punctuation_is_preserved() -> None:
    assert normalize("Don't — really; do.") == "Don't — really; do."


def test_casing_is_preserved() -> None:
    assert normalize("Production Database Access") == "Production Database Access"


def test_nothing_is_lowercased() -> None:
    assert normalize("SHOUTING") == "SHOUTING"


def test_no_semantic_rewriting_happens() -> None:
    original = "The approval chain is: manager, then the DBA on call."

    assert normalize(original) == original


# --- rule 12: idempotence ---------------------------------------------------


def test_normalization_is_idempotent() -> None:
    messy = "  Title \r\n\r\n\r\n\r\n  Body\x00 with‍ junk \t\r\n\r\n  "
    once = normalize(messy)

    assert normalize(once) == once


def test_empty_input_normalizes_to_empty() -> None:
    assert normalize("") == ""
    assert normalize("   \n\n \t ") == ""
