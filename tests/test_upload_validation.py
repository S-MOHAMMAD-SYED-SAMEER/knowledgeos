"""What may be uploaded, and what the validator refuses to conclude.

The second half matters as much as the first. Validation proves a file is the
kind of thing it claims to be; it never works out what the file says. A
renamed `.zip` passes here and fails in milestone 3's parser, which is the
right place for it to fail.
"""

import io

import pytest

from app.ingestion.validation import (
    FALLBACK_FILENAME,
    MAX_FILENAME_LENGTH,
    EmptyFile,
    FileTooLarge,
    UnsupportedFormat,
    sanitize_filename,
    validate,
)

ALLOWED = (".pdf", ".docx", ".md", ".markdown", ".txt")
PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n"
DOCX = b"PK\x03\x04" + b"\x00" * 60
TEXT = "How do I request production database access?\n".encode()


def _validate(content: bytes, filename: str, declared: str | None = None, **overrides):
    fields = {
        "filename": filename,
        "declared_type": declared,
        "source": io.BytesIO(content),
        "destination": io.BytesIO(),
        "allowed_extensions": ALLOWED,
        "max_bytes": 1024 * 1024,
    }
    fields.update(overrides)
    return validate(**fields)


# --- the four accepted formats ---------------------------------------------


@pytest.mark.parametrize(
    ("content", "filename", "expected"),
    [
        (PDF, "access-sop.pdf", "pdf"),
        (DOCX, "handbook.docx", "docx"),
        (TEXT, "notes.md", "markdown"),
        (TEXT, "notes.markdown", "markdown"),
        (TEXT, "notes.txt", "text"),
    ],
)
def test_a_supported_format_is_accepted(content, filename, expected) -> None:
    result = _validate(content, filename)

    assert result.format_name == expected
    assert result.size == len(content)


def test_the_bytes_are_copied_to_the_destination() -> None:
    destination = io.BytesIO()
    _validate(PDF, "a.pdf", destination=destination)

    assert destination.getvalue() == PDF


def test_the_extension_check_is_case_insensitive() -> None:
    assert _validate(PDF, "SHOUTING.PDF").extension == ".pdf"


# --- formats that are not accepted -----------------------------------------


@pytest.mark.parametrize("filename", ["payload.exe", "archive.zip", "sheet.xlsx"])
def test_an_unsupported_extension_is_refused(filename: str) -> None:
    with pytest.raises(UnsupportedFormat):
        _validate(PDF, filename)


def test_a_file_with_no_extension_is_refused() -> None:
    with pytest.raises(UnsupportedFormat, match="no extension"):
        _validate(TEXT, "README")


def test_an_extension_outside_the_configured_whitelist_is_refused() -> None:
    """The whitelist is configuration, so narrowing it takes effect."""
    with pytest.raises(UnsupportedFormat):
        _validate(PDF, "a.pdf", allowed_extensions=(".txt",))


# --- spoofing ---------------------------------------------------------------


def test_an_executable_renamed_as_a_pdf_is_refused() -> None:
    """The extension says PDF; the bytes say otherwise, and the bytes win."""
    with pytest.raises(UnsupportedFormat, match="does not look like"):
        _validate(b"\x7fELF\x02\x01\x01\x00 and so on", "payload.pdf")


def test_something_that_is_not_a_zip_renamed_as_docx_is_refused() -> None:
    with pytest.raises(UnsupportedFormat, match="does not look like"):
        _validate(b"just some words", "handbook.docx")


def test_a_declared_type_belonging_to_another_format_is_refused() -> None:
    with pytest.raises(UnsupportedFormat, match="declared content type"):
        _validate(PDF, "a.pdf", declared="text/plain")


def test_a_missing_declared_type_is_not_held_against_the_upload() -> None:
    """Plenty of clients send nothing useful; the signature still decides."""
    assert _validate(PDF, "a.pdf", declared=None).format_name == "pdf"


def test_a_generic_declared_type_is_accepted_for_text() -> None:
    assert _validate(TEXT, "a.txt", declared="application/octet-stream")


def test_the_declared_type_may_carry_a_charset() -> None:
    assert _validate(TEXT, "a.md", declared="text/markdown; charset=utf-8")


def test_a_zip_is_still_accepted_as_docx_on_its_signature_alone() -> None:
    """Deliberate. Proving it is really a Word document means opening the
    archive, which is parsing — milestone 3's job. A renamed zip is accepted
    here and fails there, recorded as a job error."""
    assert _validate(DOCX, "not-really-word.docx").format_name == "docx"


# --- text must be text ------------------------------------------------------


@pytest.mark.parametrize("filename", ["notes.txt", "notes.md"])
def test_bytes_that_are_not_utf8_are_refused_for_a_text_format(filename) -> None:
    with pytest.raises(UnsupportedFormat, match="UTF-8"):
        _validate(b"\xff\xfe\x00 invalid", filename)


def test_multibyte_characters_split_across_a_chunk_boundary_are_fine() -> None:
    """Decoded incrementally, so a character straddling a read is not
    mistaken for invalid input."""
    from app.ingestion.validation import READ_CHUNK_BYTES

    payload = ("a" * (READ_CHUNK_BYTES - 1) + "é" + "b" * 10).encode()

    assert _validate(payload, "notes.txt").size == len(payload)


def test_a_pdf_is_not_required_to_be_utf8() -> None:
    assert _validate(PDF + b"\xff\xfe binary tail", "a.pdf").format_name == "pdf"


# --- size --------------------------------------------------------------------


def test_an_empty_file_is_refused() -> None:
    with pytest.raises(EmptyFile):
        _validate(b"", "a.txt")


def test_a_file_over_the_limit_is_refused() -> None:
    with pytest.raises(FileTooLarge):
        _validate(PDF + b"x" * 500, "a.pdf", max_bytes=100)


def test_the_limit_is_enforced_while_reading_not_from_a_header() -> None:
    """A source that never ends must be refused rather than absorbed. If the
    limit were checked after reading, this would not return."""

    class _Endless(io.RawIOBase):
        def read(self, size: int = -1) -> bytes:
            return b"%PDF-" + b"x" * (size or 1024)

    with pytest.raises(FileTooLarge):
        validate(
            filename="a.pdf",
            declared_type=None,
            source=_Endless(),
            destination=io.BytesIO(),
            allowed_extensions=ALLOWED,
            max_bytes=1024 * 1024,
        )


def test_a_file_exactly_at_the_limit_is_accepted() -> None:
    payload = PDF + b"x" * (200 - len(PDF))

    assert _validate(payload, "a.pdf", max_bytes=200).size == 200


# --- filenames are metadata, and are cleaned anyway -------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("../../etc/passwd.txt", "passwd.txt"),
        ("/absolute/path/policy.pdf", "policy.pdf"),
        ("C:\\Users\\ada\\policy.pdf", "policy.pdf"),
        ("..\\..\\windows\\system32\\a.txt", "a.txt"),
        ("plain.pdf", "plain.pdf"),
        ("with space.pdf", "with space.pdf"),
    ],
)
def test_directory_components_are_stripped(given: str, expected: str) -> None:
    assert sanitize_filename(given) == expected


def test_control_characters_and_nul_are_removed() -> None:
    assert sanitize_filename("pol\x00icy\n\t.pdf") == "policy.pdf"


def test_a_bidirectional_override_is_removed() -> None:
    """The trick that makes `exe.pdf` display as `fdp.exe`."""
    assert "\u202e" not in sanitize_filename("invoice\u202egnp.exe")


@pytest.mark.parametrize("given", ["", None, "...", "   ", "/", "\x00"])
def test_a_filename_that_sanitizes_to_nothing_gets_a_fallback(given) -> None:
    assert sanitize_filename(given) == FALLBACK_FILENAME


def test_an_over_long_filename_is_trimmed_to_the_column() -> None:
    name = sanitize_filename("a" * 900 + ".pdf")

    assert len(name) <= MAX_FILENAME_LENGTH
    assert name.endswith(".pdf")


def test_a_traversal_filename_cannot_smuggle_an_extension() -> None:
    """After stripping, what is left must still be an accepted format."""
    with pytest.raises(UnsupportedFormat):
        _validate(PDF, "../../etc/passwd")
