"""The four parsers, and the line they must not cross.

A parser extracts what a file contains and whatever structure the format
genuinely provides. It never invents: a `.txt` has no pages, so its page is
null rather than 1, and a PDF has no headings this system can see, so its
section is null rather than a guess from font size.
"""

import io

import pytest

from app.parsing import (
    DocxParser,
    MarkdownParser,
    ParseError,
    PdfParser,
    TextParser,
    parser_for,
)

from .fixtures import docx_bytes, pdf_bytes, plain_zip_bytes


# --- dispatch ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("extension", "expected"),
    [
        (".pdf", "pdf"),
        (".docx", "docx"),
        (".md", "markdown"),
        (".markdown", "markdown"),
        (".txt", "text"),
        (".PDF", "pdf"),
    ],
)
def test_each_accepted_extension_has_a_parser(extension: str, expected: str) -> None:
    """Every extension milestone 2 accepts at upload can be read here. An
    accepted extension with no parser would be an upload that succeeds and an
    ingestion that always fails."""
    assert parser_for(extension).name == expected


def test_an_extension_with_no_parser_is_refused() -> None:
    with pytest.raises(ParseError):
        parser_for(".xlsx")


# --- PDF --------------------------------------------------------------------


def test_a_pdf_yields_one_block_per_page() -> None:
    parsed = PdfParser().parse(io.BytesIO(pdf_bytes(["Alpha text", "Beta text"])))

    assert [block.page for block in parsed.blocks] == [1, 2]
    assert "Alpha" in parsed.blocks[0].text
    assert "Beta" in parsed.blocks[1].text


def test_a_pdf_reports_its_real_page_count() -> None:
    parsed = PdfParser().parse(io.BytesIO(pdf_bytes(["a", "b", "c", "d"])))

    assert parsed.page_count == 4


def test_a_pdf_has_no_sections() -> None:
    """Nothing structural in a PDF says "this is a heading". Inferring one
    from formatting would be inventing metadata a citation points at."""
    parsed = PdfParser().parse(io.BytesIO(pdf_bytes(["Some text"])))

    assert all(block.section is None for block in parsed.blocks)


def test_a_page_with_no_text_layer_contributes_no_block() -> None:
    """A scanned page is not a blank page. This system does no OCR, so the
    page counts but produces nothing."""
    parsed = PdfParser().parse(io.BytesIO(pdf_bytes(["Real text", "", "More text"])))

    assert parsed.page_count == 3
    assert [block.page for block in parsed.blocks] == [1, 3]


def test_a_corrupt_pdf_fails_cleanly() -> None:
    with pytest.raises(ParseError):
        PdfParser().parse(io.BytesIO(b"%PDF-1.4\nthis is not really a pdf"))


# --- DOCX -------------------------------------------------------------------


def test_a_docx_yields_its_paragraphs_in_order() -> None:
    content = docx_bytes([("First para", None), ("Second para", None)])
    parsed = DocxParser().parse(io.BytesIO(content))

    assert [block.text for block in parsed.blocks] == ["First para", "Second para"]


def test_a_docx_carries_the_nearest_heading_as_its_section() -> None:
    content = docx_bytes(
        [
            ("Access SOP", "Heading1"),
            ("How to request access.", None),
            ("Appendix", "Heading2"),
            ("Further detail.", None),
        ]
    )
    parsed = DocxParser().parse(io.BytesIO(content))

    assert [block.section for block in parsed.blocks] == [
        "Access SOP",
        "Access SOP",
        "Appendix",
        "Appendix",
    ]


def test_text_before_any_heading_has_no_section() -> None:
    content = docx_bytes([("Preamble", None), ("Title", "Heading1")])
    parsed = DocxParser().parse(io.BytesIO(content))

    assert parsed.blocks[0].section is None


def test_a_docx_has_no_pages() -> None:
    """Pagination is decided by a renderer and is not in the file."""
    parsed = DocxParser().parse(io.BytesIO(docx_bytes([("Body", None)])))

    assert parsed.page_count is None
    assert all(block.page is None for block in parsed.blocks)


def test_a_zip_that_is_not_a_docx_fails_cleanly() -> None:
    """Milestone 2 validated only the ZIP signature, on purpose: proving a
    file is really a Word document means opening it, which is parsing. This
    is the other half of that hand-off."""
    with pytest.raises(ParseError, match="not a DOCX"):
        DocxParser().parse(io.BytesIO(plain_zip_bytes()))


def test_something_that_is_not_an_archive_fails_cleanly() -> None:
    with pytest.raises(ParseError, match="container"):
        DocxParser().parse(io.BytesIO(b"PK\x03\x04 but truncated"))


def test_a_docx_error_quotes_no_document_content() -> None:
    """Stage errors are stored and read back through the API."""
    try:
        DocxParser().parse(io.BytesIO(plain_zip_bytes()))
    except ParseError as exc:
        assert "not a word document" not in str(exc).lower()


# --- Markdown ---------------------------------------------------------------


def _markdown(text: str):
    return MarkdownParser().parse(io.BytesIO(text.encode("utf-8")))


def test_markdown_headings_become_sections() -> None:
    """A heading starts a block and names it; its body travels with it."""
    parsed = _markdown("# Access SOP\n\nAsk your manager.\n\n## Appendix\n\nDetail.\n")

    assert [block.section for block in parsed.blocks] == ["Access SOP", "Appendix"]
    assert "Ask your manager." in parsed.blocks[0].text
    assert "Detail." in parsed.blocks[1].text


def test_markdown_text_before_the_first_heading_has_no_section() -> None:
    parsed = _markdown("Intro line.\n\n# Later\n\nBody.\n")

    assert parsed.blocks[0].section is None
    assert parsed.blocks[1].section == "Later"


def test_the_underlined_heading_form_is_recognised() -> None:
    parsed = _markdown("Access SOP\n==========\n\nBody text.\n")

    assert parsed.blocks[-1].section == "Access SOP"


def test_a_hash_inside_a_code_fence_is_not_a_heading() -> None:
    parsed = _markdown("# Real\n\n```\n# not a heading\n```\n\nAfter.\n")

    assert all(block.section == "Real" for block in parsed.blocks)


def test_markdown_markup_is_preserved_not_rendered() -> None:
    """The text stored beside a citation should be what the document says."""
    parsed = _markdown("# Title\n\nSome **bold** and `code`.\n")

    assert "**bold**" in parsed.blocks[-1].text
    assert "`code`" in parsed.blocks[-1].text


def test_markdown_has_no_pages() -> None:
    parsed = _markdown("# Title\n\nBody.\n")

    assert parsed.page_count is None
    assert all(block.page is None for block in parsed.blocks)


# --- plain text -------------------------------------------------------------


def test_text_is_read_as_one_block() -> None:
    parsed = TextParser().parse(io.BytesIO(b"How do I request access?\n"))

    assert len(parsed.blocks) == 1
    assert parsed.blocks[0].text.strip() == "How do I request access?"


def test_text_invents_no_page_or_section() -> None:
    parsed = TextParser().parse(io.BytesIO(b"Plain words."))

    assert parsed.page_count is None
    assert parsed.blocks[0].page is None
    assert parsed.blocks[0].section is None


def test_text_that_is_not_utf8_fails_cleanly() -> None:
    with pytest.raises(ParseError, match="UTF-8"):
        TextParser().parse(io.BytesIO(b"\xff\xfe not text"))


# --- empty documents --------------------------------------------------------


@pytest.mark.parametrize(
    ("parser", "content"),
    [
        (TextParser(), b"   \n\n  "),
        (MarkdownParser(), b"\n\n"),
        (DocxParser(), docx_bytes([("   ", None)])),
    ],
)
def test_a_document_with_no_usable_text_yields_no_blocks(parser, content) -> None:
    assert parser.parse(io.BytesIO(content)).blocks == ()
