"""Deterministic chunking.

The contract, from the specification: fixed size by token count, 512 by
default with 64 of overlap, both configurable; paragraph boundaries respected
where a break falls within 20% of the target; page and section metadata
carried through from parsing; missing page information left null, never
invented. Same input and same configuration must produce identical chunks —
and therefore identical `chunk_uid`s — every time.

**What a "token" is here.** A whitespace-delimited word. The specification
names a count and no tokenizer, and the embedding model's own tokenizer would
mean downloading a model asset, which this milestone has no business doing.
The configuration keeps the specification's `*_tokens` names. The consequence
is recorded rather than hidden: a whitespace word is longer than a subword
token, so 512 words may exceed the embedding model's input window in milestone
4. That is a known risk, written down in the README, and not solved here.

This module is pure. It takes normalized text and block metadata and returns
chunks: no database, no storage, no provider, nothing to mock. The
specification requires exactly that of `app/chunking/`, because this is one of
the parts that later gets evaluated over fixture data.
"""

import re
import uuid
from dataclasses import dataclass

from app.chunking.uid import chunk_uid
from app.parsing.base import Block

# A "token" for the purposes of this milestone: a run of non-whitespace.
_WORD = re.compile(r"\S+")

# How near a paragraph break has to be, as a fraction of the target, before it
# is preferred over cutting mid-paragraph.
PARAGRAPH_TOLERANCE = 0.20

# A blank line separates paragraphs, which is exactly what normalization
# guarantees survives.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    """One chunk, ready to be persisted.

    `char_start` and `char_end` are offsets into the **complete normalized
    document text**, as a half-open interval `[start, end)`, so
    `document_text[char_start:char_end]` is exactly `text`.
    """

    sequence: int
    chunk_uid: str
    text: str
    page: int | None
    section: str | None
    char_start: int
    char_end: int
    token_count: int


def count_tokens(text: str) -> int:
    """How many whitespace-delimited words a piece of text contains."""
    return len(_WORD.findall(text))


@dataclass(frozen=True)
class _Word:
    """One word, and where it sits in the normalized document."""

    start: int
    end: int


def chunk_document(
    *,
    document_version_id: uuid.UUID,
    text: str,
    blocks: tuple[Block, ...] = (),
    block_offsets: tuple[tuple[int, int], ...] = (),
    size: int,
    overlap: int,
) -> list[Chunk]:
    """Split normalized document text into overlapping chunks.

    `blocks` and `block_offsets` are parallel: the offsets say where each
    parsed block landed in the normalized document, which is how a chunk
    learns the page and section it starts in.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0:
        raise ValueError("chunk overlap cannot be negative")
    if overlap >= size:
        # Otherwise each chunk would start at or before the previous one and
        # the loop below would never finish.
        raise ValueError("chunk overlap must be smaller than chunk size")

    words = [_Word(match.start(), match.end()) for match in _WORD.finditer(text)]
    if not words:
        return []

    breaks = _paragraph_breaks(text)
    tolerance = int(size * PARAGRAPH_TOLERANCE)

    chunks: list[Chunk] = []
    start_index = 0
    sequence = 0

    while start_index < len(words):
        end_index = _end_of_chunk(words, start_index, size, breaks, tolerance)

        char_start = words[start_index].start
        char_end = words[end_index - 1].end
        body = text[char_start:char_end]

        page, section = _metadata_at(char_start, blocks, block_offsets)
        chunks.append(
            Chunk(
                sequence=sequence,
                chunk_uid=chunk_uid(document_version_id, sequence, body),
                text=body,
                page=page,
                section=section,
                char_start=char_start,
                char_end=char_end,
                token_count=end_index - start_index,
            )
        )
        sequence += 1

        if end_index >= len(words):
            break
        # Step forward by at least one word however the overlap is configured,
        # so a pathological configuration cannot produce an endless loop.
        start_index = max(start_index + 1, end_index - overlap)

    return chunks


# --- internals --------------------------------------------------------------


def _end_of_chunk(
    words: list[_Word],
    start_index: int,
    size: int,
    breaks: list[int],
    tolerance: int,
) -> int:
    """Where this chunk ends, preferring a nearby paragraph boundary.

    The exclusive index of the last word. A paragraph break is used when one
    falls within the tolerance of the target, which keeps a chunk from ending
    mid-sentence when the document offered a clean place to stop a few words
    earlier or later.
    """
    target = start_index + size
    if target >= len(words):
        return len(words)

    lowest = max(start_index + 1, target - tolerance)
    highest = min(len(words), target + tolerance)

    # The candidate boundaries are word indices at which a paragraph break has
    # just happened. Closest to the target wins; ties go to the earlier one,
    # so the choice does not depend on iteration order.
    candidates = [
        index
        for index in breaks
        if lowest <= index <= highest
    ]
    if candidates:
        return min(candidates, key=lambda index: (abs(index - target), index))
    return target


def _paragraph_breaks(text: str) -> list[int]:
    """Word indices that begin a new paragraph.

    Computed once per document: for each blank-line break, how many words
    precede it. That count is the index of the first word of the next
    paragraph, and therefore an exclusive end index for a chunk that stops
    there.
    """
    breaks: list[int] = []
    for match in _PARAGRAPH_BREAK.finditer(text):
        breaks.append(len(_WORD.findall(text[: match.start()])))
    return sorted(set(breaks))


def _metadata_at(
    char_start: int,
    blocks: tuple[Block, ...],
    block_offsets: tuple[tuple[int, int], ...],
) -> tuple[int | None, str | None]:
    """The page and section of the block a chunk starts in.

    **A chunk that spans pages records the page it starts on.** The column
    holds one page and the specification asks for no more; the consequence —
    that a citation may name the page where a chunk begins rather than every
    page it touches — is documented in the README rather than papered over
    with an array.
    """
    for block, (start, end) in zip(blocks, block_offsets, strict=False):
        if start <= char_start < end:
            return block.page, block.section
    return None, None


__all__ = ["PARAGRAPH_TOLERANCE", "Chunk", "chunk_document", "count_tokens"]
