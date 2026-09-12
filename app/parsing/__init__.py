"""Reading documents: one parser per format, and one normalizer for all of them.

Nothing here interprets meaning. A parser extracts the text a file contains
and whatever page or section structure the format genuinely provides; the
normalizer puts that text into the single canonical form that the checksum and
every chunk_uid are computed over.
"""

from app.parsing.base import Block, ParsedDocument, ParseError, Parser
from app.parsing.docx import DocxParser
from app.parsing.markdown_ import MarkdownParser
from app.parsing.normalize import normalize
from app.parsing.pdf import PdfParser
from app.parsing.text import TextParser

# Which parser reads which extension. The keys match milestone 2's upload
# whitelist, so a file that was accepted at the door always has something to
# read it — an extension accepted for upload with no parser here would be a
# job that fails after the caller was told the upload succeeded.
PARSERS: dict[str, Parser] = {
    ".pdf": PdfParser(),
    ".docx": DocxParser(),
    ".md": MarkdownParser(),
    ".markdown": MarkdownParser(),
    ".txt": TextParser(),
}


def parser_for(extension: str) -> Parser:
    """The parser for a file extension."""
    try:
        return PARSERS[extension.lower()]
    except KeyError:
        raise ParseError(f"no parser for {extension!r} files") from None


__all__ = [
    "PARSERS",
    "Block",
    "DocxParser",
    "MarkdownParser",
    "ParseError",
    "ParsedDocument",
    "Parser",
    "PdfParser",
    "TextParser",
    "normalize",
    "parser_for",
]
