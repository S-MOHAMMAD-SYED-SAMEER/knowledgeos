"""PDF text, page by page.

The only format in this system with real pages, so it is the only one that
fills in `page` and `page_count`. Each page becomes its own block, which is
what lets a chunk record the page it started on.

Text extraction only. Nothing here renders, executes, or follows anything
inside the file — a PDF's scripting and embedded-file features are simply
never reached by asking `pypdf` for the text of a page.
"""

from typing import BinaryIO

from app.parsing.base import Block, ParsedDocument, ParseError


class PdfParser:
    """Extracts the text layer, one block per page."""

    name = "pdf"

    def parse(self, handle: BinaryIO) -> ParsedDocument:
        import pypdf

        try:
            reader = pypdf.PdfReader(handle)
            pages = list(reader.pages)
        except Exception as exc:  # noqa: BLE001 - any malformed file is one case
            raise ParseError(f"the PDF could not be read ({type(exc).__name__})") from exc

        blocks: list[Block] = []
        for number, page in enumerate(pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                raise ParseError(
                    f"page {number} of the PDF could not be read "
                    f"({type(exc).__name__})"
                ) from exc
            # A page with no text layer contributes nothing rather than an
            # empty block: a scanned page is not a blank page, and this
            # system does no OCR.
            if text.strip():
                blocks.append(Block(text=text, page=number))

        # Sections are not something a PDF structurally provides, so `section`
        # stays None throughout. Guessing from font sizes would be inventing
        # metadata a citation might later point at.
        return ParsedDocument(blocks=tuple(blocks), page_count=len(pages))


__all__ = ["PdfParser"]
