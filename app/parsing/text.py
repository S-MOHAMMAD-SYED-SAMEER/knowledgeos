"""Plain text.

The simplest parser there is, and the one that most clearly shows what a
parser must not do: a `.txt` file has no pages and no sections, so `page`,
`section` and `page_count` are all None. Not 1, not "Document" — nothing.
"""

from typing import BinaryIO

from app.parsing.base import Block, ParsedDocument, ParseError


class TextParser:
    """Reads a UTF-8 text file as a single block."""

    name = "text"

    def parse(self, handle: BinaryIO) -> ParsedDocument:
        raw = handle.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            # Milestone 2 already checked this at upload; a file that fails
            # here has changed underneath us, or arrived another way.
            raise ParseError("the file is not valid UTF-8 text") from exc

        if not text.strip():
            return ParsedDocument(blocks=())
        return ParsedDocument(blocks=(Block(text=text),))


__all__ = ["TextParser"]
