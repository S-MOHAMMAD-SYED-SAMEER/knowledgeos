"""Small real documents, built in memory.

Real files rather than mocks: a PDF that `pypdf` genuinely reads, a DOCX that
is genuinely a ZIP of Office Open XML. Built here rather than committed as
binaries so what they contain is readable in the diff.
"""

import io
import zipfile

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def pdf_bytes(pages: list[str]) -> bytes:
    """A minimal but genuine PDF: one page per string, each with a text layer.

    Written by hand rather than with a library, so the fixture is a real file
    `pypdf` has to parse — including the font resource without which there is
    nothing to extract.
    """

    def escape(text: str) -> str:
        return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids: list[int] = []
    content_ids: list[int] = []
    for body in pages:
        stream = f"BT /F1 12 Tf 72 700 Td ({escape(body)}) Tj ET".encode("latin-1")
        content_ids.append(
            add(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        )
        page_ids.append(add(b"PLACEHOLDER"))

    pages_id = add(b"PLACEHOLDER")
    for index, page_id in enumerate(page_ids):
        objects[page_id - 1] = (
            b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
            % (pages_id, font, content_ids[index])
        )
    kids = b" ".join(b"%d 0 R" % page_id for page_id in page_ids)
    objects[pages_id - 1] = (
        b"<< /Type /Pages /Count %d /Kids [%s] >>" % (len(page_ids), kids)
    )
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"

    start = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += b"%010d 00000 n \n" % offset
    out += (
        b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objects) + 1, catalog, start)
    )
    return bytes(out)


def docx_bytes(paragraphs: list[tuple[str, str | None]]) -> bytes:
    """A DOCX containing these paragraphs, each with an optional style name."""
    body = []
    for text, style in paragraphs:
        properties = (
            f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        )
        escaped = (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        body.append(f"<w:p>{properties}<w:r><w:t>{escaped}</w:t></w:r></w:p>")

    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{WORD_NS}"><w:body>{"".join(body)}</w:body></w:document>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        "</Types>"
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def plain_zip_bytes() -> bytes:
    """A ZIP that is not a DOCX — the renamed-archive case."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", "this is not a word document")
    return buffer.getvalue()


def words(count: int, prefix: str = "w") -> str:
    """A deterministic run of distinct words."""
    return " ".join(f"{prefix}{index}" for index in range(count))
