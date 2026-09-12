"""Deterministic chunking, and the identifier that makes re-indexing idempotent.

The golden-value test is the important one here. `chunk_uid` is derived from
content rather than allocated, so its exact formula is immutable once any
document exists — pinning a literal value is what stops it drifting by
accident.
"""

import uuid

import pytest

from app.chunking import Chunk, chunk_document, chunk_uid, count_tokens
from app.chunking.chunker import PARAGRAPH_TOLERANCE
from app.parsing.base import Block

from .fixtures import words

VERSION = uuid.UUID("11111111-2222-3333-4444-555555555555")
OTHER_VERSION = uuid.UUID("99999999-8888-7777-6666-555555555555")


def _chunks(text: str, size: int = 512, overlap: int = 64, **extra) -> list[Chunk]:
    return chunk_document(
        document_version_id=VERSION, text=text, size=size, overlap=overlap, **extra
    )


# --- counting ---------------------------------------------------------------


def test_tokens_are_whitespace_delimited_words() -> None:
    """The unit this milestone counts. Recorded explicitly because the
    specification says "tokens" and names no tokenizer."""
    assert count_tokens("one two three") == 3
    assert count_tokens("one   two\n\nthree\tfour") == 4
    assert count_tokens("") == 0
    assert count_tokens("   ") == 0


def test_punctuation_travels_with_its_word() -> None:
    assert count_tokens("Hello, world! Really?") == 3


# --- size and overlap -------------------------------------------------------


def test_a_document_shorter_than_the_target_is_one_chunk() -> None:
    chunks = _chunks("just a few words here")

    assert len(chunks) == 1
    assert chunks[0].sequence == 0
    assert chunks[0].token_count == 5


def test_a_chunk_holds_the_target_number_of_words() -> None:
    chunks = _chunks(words(1200), size=100, overlap=0)

    assert chunks[0].token_count == 100


def test_a_document_of_exactly_the_target_is_one_chunk() -> None:
    chunks = _chunks(words(100), size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0].token_count == 100


def test_overlap_repeats_the_tail_of_the_previous_chunk() -> None:
    chunks = _chunks(words(300), size=100, overlap=20)

    first = chunks[0].text.split()
    second = chunks[1].text.split()
    assert first[-20:] == second[:20]


def test_no_overlap_means_no_repetition() -> None:
    chunks = _chunks(words(300), size=100, overlap=0)

    assert chunks[0].text.split()[-1] != chunks[1].text.split()[0]
    assert len(chunks) == 3


def test_sequences_start_at_zero_and_are_contiguous() -> None:
    chunks = _chunks(words(1000), size=100, overlap=10)

    assert [chunk.sequence for chunk in chunks] == list(range(len(chunks)))


def test_every_word_appears_in_some_chunk() -> None:
    """Nothing is dropped between chunks."""
    text = words(543)
    covered: set[str] = set()
    for chunk in _chunks(text, size=100, overlap=10):
        covered.update(chunk.text.split())

    assert covered == set(text.split())


# --- edge cases that must not loop forever ----------------------------------


def test_an_empty_document_produces_no_chunks() -> None:
    assert _chunks("") == []
    assert _chunks("   \n\n  ") == []


def test_a_single_word_produces_one_chunk() -> None:
    chunks = _chunks("word")

    assert len(chunks) == 1
    assert chunks[0].token_count == 1


def test_an_overlap_as_large_as_the_size_is_refused() -> None:
    """Otherwise each chunk starts where the last one did, for ever."""
    with pytest.raises(ValueError, match="overlap"):
        _chunks(words(500), size=100, overlap=100)


def test_a_non_positive_size_is_refused() -> None:
    with pytest.raises(ValueError, match="size"):
        _chunks(words(10), size=0, overlap=0)


def test_a_large_overlap_still_terminates() -> None:
    chunks = _chunks(words(400), size=100, overlap=99)

    assert len(chunks) < 500
    assert chunks[-1].char_end == len(words(400))


# --- paragraph boundaries ---------------------------------------------------


def test_a_paragraph_break_near_the_target_is_preferred() -> None:
    """A break within 20% of the target wins over cutting mid-paragraph."""
    text = words(95) + "\n\n" + words(200, prefix="z")
    chunks = _chunks(text, size=100, overlap=0)

    assert chunks[0].token_count == 95
    assert chunks[0].text.endswith("w94")


def test_a_paragraph_break_far_from_the_target_is_ignored() -> None:
    text = words(40) + "\n\n" + words(300, prefix="z")
    chunks = _chunks(text, size=100, overlap=0)

    assert chunks[0].token_count == 100


def test_the_tolerance_is_twenty_percent_of_the_target() -> None:
    assert PARAGRAPH_TOLERANCE == 0.20


# --- offsets ----------------------------------------------------------------


def test_offsets_index_the_normalized_document_text() -> None:
    """Half-open `[start, end)`, so slicing the document gives the chunk."""
    text = words(600)

    for chunk in _chunks(text, size=100, overlap=10):
        assert text[chunk.char_start : chunk.char_end] == chunk.text


def test_offsets_are_document_relative_not_chunk_relative() -> None:
    chunks = _chunks(words(600), size=100, overlap=0)

    assert chunks[0].char_start == 0
    assert chunks[1].char_start > 0


def test_the_last_chunk_reaches_the_end_of_the_document() -> None:
    text = words(250)
    chunks = _chunks(text, size=100, overlap=10)

    assert chunks[-1].char_end == len(text)


# --- metadata ---------------------------------------------------------------


def test_page_and_section_come_from_the_block_a_chunk_starts_in() -> None:
    first = words(50)
    second = words(50, prefix="z")
    text = f"{first}\n\n{second}"
    blocks = (
        Block(text=first, page=1, section="Intro"),
        Block(text=second, page=2, section="Detail"),
    )
    offsets = ((0, len(first)), (len(first) + 2, len(text)))

    chunks = chunk_document(
        document_version_id=VERSION,
        text=text,
        blocks=blocks,
        block_offsets=offsets,
        size=40,
        overlap=0,
    )

    assert (chunks[0].page, chunks[0].section) == (1, "Intro")
    assert chunks[-1].page == 2


def test_a_chunk_spanning_two_pages_records_where_it_started() -> None:
    """The locked behaviour: one page column, and it names the start."""
    first = words(30)
    second = words(30, prefix="z")
    text = f"{first}\n\n{second}"
    blocks = (Block(text=first, page=4), Block(text=second, page=5))
    offsets = ((0, len(first)), (len(first) + 2, len(text)))

    chunks = chunk_document(
        document_version_id=VERSION,
        text=text,
        blocks=blocks,
        block_offsets=offsets,
        size=512,
        overlap=0,
    )

    assert len(chunks) == 1
    assert chunks[0].page == 4


def test_missing_page_information_stays_null() -> None:
    """Never invented — not page 1, not zero."""
    chunks = _chunks(words(50))

    assert chunks[0].page is None
    assert chunks[0].section is None


# --- determinism ------------------------------------------------------------


def test_the_same_input_produces_identical_chunks() -> None:
    text = words(900)

    first = _chunks(text, size=100, overlap=10)
    second = _chunks(text, size=100, overlap=10)

    assert [chunk.chunk_uid for chunk in first] == [chunk.chunk_uid for chunk in second]
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]
    assert [chunk.char_start for chunk in first] == [c.char_start for c in second]


def test_a_different_configuration_produces_different_chunks() -> None:
    text = words(900)

    assert _chunks(text, size=100, overlap=10) != _chunks(text, size=200, overlap=10)


# --- chunk_uid --------------------------------------------------------------


def test_the_uid_is_a_pinned_golden_value() -> None:
    """The formula is immutable once documents exist. This is the pin.

    The literal below was computed outside Python, with
    `printf '11111111-2222-3333-4444-555555555555:0:hello world' | sha256sum`,
    so it checks the implementation rather than restating it.
    """
    assert chunk_uid(VERSION, 0, "hello world") == "eb40bb7741165afa76f77c3bd4c3083d"


def test_the_uid_matches_the_specification_formula() -> None:
    from hashlib import sha256

    expected = sha256(f"{VERSION}:0:hello world".encode("utf-8")).hexdigest()[:32]

    assert chunk_uid(VERSION, 0, "hello world") == expected


def test_the_uid_is_thirty_two_lowercase_hex_characters() -> None:
    value = chunk_uid(VERSION, 7, "some text")

    assert len(value) == 32
    assert value == value.lower()
    assert all(character in "0123456789abcdef" for character in value)


def test_a_different_sequence_changes_the_uid() -> None:
    assert chunk_uid(VERSION, 0, "same") != chunk_uid(VERSION, 1, "same")


def test_different_text_changes_the_uid() -> None:
    assert chunk_uid(VERSION, 0, "one") != chunk_uid(VERSION, 0, "two")


def test_a_different_version_changes_the_uid() -> None:
    assert chunk_uid(VERSION, 0, "same") != chunk_uid(OTHER_VERSION, 0, "same")


def test_page_and_section_do_not_change_the_uid() -> None:
    """Neither is an input. A corrected page keeps the chunk's identity."""
    first = words(20)
    second = words(20, prefix="z")
    text = f"{first}\n\n{second}"

    plain = chunk_document(
        document_version_id=VERSION, text=text, size=512, overlap=0
    )
    with_metadata = chunk_document(
        document_version_id=VERSION,
        text=text,
        blocks=(Block(text=first, page=9, section="Nine"),),
        block_offsets=((0, len(text)),),
        size=512,
        overlap=0,
    )

    assert plain[0].chunk_uid == with_metadata[0].chunk_uid
    assert with_metadata[0].page == 9


def test_the_uid_accepts_the_uuid_as_a_string_identically() -> None:
    assert chunk_uid(VERSION, 3, "text") == chunk_uid(str(VERSION), 3, "text")
