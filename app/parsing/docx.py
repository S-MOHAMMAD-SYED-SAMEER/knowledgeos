"""DOCX, read with the standard library.

A `.docx` is a ZIP holding Office Open XML, and the part that carries the
prose is `word/document.xml`. `zipfile` opens the container and
`xml.etree.ElementTree` reads the part, so no third-party library is needed
for what this milestone does — extract text in document order and recognise
headings well enough to fill in `section`.

**This is where a renamed archive is finally caught.** Milestone 2 validates
an upload only as far as the `PK\\x03\\x04` signature, because proving a file
is really a Word document means opening it, and opening it is parsing. So a
`.zip` renamed to `.docx` is accepted at upload and fails here, as a job with
a recorded stage error. That hand-off was designed; this is the other half of
it.

DOCX has **no pages**. Pagination is decided when a document is rendered, by
the renderer, and is not in the file — so `page` and `page_count` are None.

Known limitation, recorded rather than solved: nothing here bounds how much a
crafted archive expands to. Milestone 2 caps the size of an upload, not the
size of what it decompresses into. A decompression limit was deliberately
left out of this milestone's scope.
"""

import xml.etree.ElementTree as ElementTree
import zipfile
from typing import BinaryIO

from app.parsing.base import Block, ParsedDocument, ParseError

DOCUMENT_PART = "word/document.xml"

# The WordprocessingML namespace. Every element below lives in it.
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

PARAGRAPH = f"{_W}p"
RUN_TEXT = f"{_W}t"
TAB = f"{_W}tab"
BREAK = f"{_W}br"
PARAGRAPH_PROPERTIES = f"{_W}pPr"
STYLE = f"{_W}pStyle"
VALUE = f"{_W}val"


class DocxParser:
    """Paragraph text in document order, tagged with the heading above it."""

    name = "docx"

    def parse(self, handle: BinaryIO) -> ParsedDocument:
        try:
            archive = zipfile.ZipFile(handle)
        except zipfile.BadZipFile as exc:
            raise ParseError("the file is not a readable DOCX container") from exc

        with archive:
            if DOCUMENT_PART not in archive.namelist():
                # A ZIP, but not a Word document. This is the renamed-archive
                # case, and it must fail rather than be read as text.
                raise ParseError(
                    "the file is a ZIP archive but not a DOCX: it has no "
                    f"{DOCUMENT_PART}"
                )
            try:
                with archive.open(DOCUMENT_PART) as part:
                    root = ElementTree.parse(part).getroot()
            except (ElementTree.ParseError, zipfile.BadZipFile, OSError) as exc:
                raise ParseError(
                    f"the DOCX body could not be read ({type(exc).__name__})"
                ) from exc

        blocks: list[Block] = []
        section: str | None = None

        for paragraph in root.iter(PARAGRAPH):
            text = _paragraph_text(paragraph)
            if not text.strip():
                continue
            if _is_heading(paragraph):
                # The heading names what follows, and is itself part of the
                # text: dropping it would lose a line the document contains.
                section = text.strip()
            blocks.append(Block(text=text, section=section))

        return ParsedDocument(blocks=tuple(blocks))


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    """Everything a paragraph says, in order.

    Tabs and line breaks are structural elements rather than characters in a
    text run, so they are turned back into the characters they represent.
    """
    pieces: list[str] = []
    for node in paragraph.iter():
        if node.tag == RUN_TEXT:
            pieces.append(node.text or "")
        elif node.tag == TAB:
            pieces.append("\t")
        elif node.tag == BREAK:
            pieces.append("\n")
    return "".join(pieces)


def _is_heading(paragraph: ElementTree.Element) -> bool:
    """Whether Word calls this paragraph a heading.

    Read from the paragraph's named style. Style names are a convention
    rather than a guarantee, so this recognises the built-in ones and does
    not try to infer a heading from formatting — inferring would be inventing
    section metadata.
    """
    properties = paragraph.find(PARAGRAPH_PROPERTIES)
    if properties is None:
        return False
    style = properties.find(STYLE)
    if style is None:
        return False
    name = (style.get(VALUE) or "").lower()
    return name.startswith("heading") or name in ("title", "subtitle")


__all__ = ["DOCUMENT_PART", "DocxParser"]
