"""What a parser is, and what it is allowed to say.

A parser turns an opened file into **ordered blocks of text**, each carrying
whatever page and section information the format genuinely provides. Blocks
rather than one string, because page and section are properties of a region of
a document and are lost the moment everything is concatenated — and chunking
has to carry them through.

The rule that governs every field here is the specification's: *missing page
info is null, never invented*. A `.txt` file has no pages, so its blocks have
`page = None`; it is not page 1. The same goes for sections. A parser that
guessed would be manufacturing evidence that a citation later points at.

Parsers read **through the storage abstraction**. They are handed an open
binary handle and never a path, so nothing below this line can address the
filesystem.
"""

from dataclasses import dataclass
from typing import BinaryIO, Protocol, runtime_checkable


class ParseError(Exception):
    """A file could not be parsed.

    The message names the format and what went wrong structurally. It never
    carries document content — the specification requires stage errors to be
    stored "without raw document content", and this is what ends up in one.
    """


@dataclass(frozen=True)
class Block:
    """One contiguous run of text, with whatever the format actually knew."""

    text: str
    # 1-based, and only where the format has real pages. None everywhere else.
    page: int | None = None
    # The nearest preceding heading, where the format has headings.
    section: str | None = None


@dataclass(frozen=True)
class ParsedDocument:
    """Everything a parser extracted.

    `page_count` is the real number of pages for a paginated format and
    `None` for everything else. A Markdown file does not have one page; it
    has no pages at all, and recording `1` would be inventing one.
    """

    blocks: tuple[Block, ...]
    page_count: int | None = None


@runtime_checkable
class Parser(Protocol):
    """Reads one format."""

    def parse(self, handle: BinaryIO) -> ParsedDocument:
        """Extract ordered blocks from an open file.

        The caller owns the handle and closes it. A parser that cannot make
        sense of the bytes raises `ParseError` rather than returning
        something empty and plausible.
        """
        ...


__all__ = ["Block", "ParseError", "ParsedDocument", "Parser"]
