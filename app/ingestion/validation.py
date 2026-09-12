"""Deciding whether an upload is the kind of file it says it is.

The boundary this module lives on is the important thing about it, so it is
worth stating plainly: **validation may look at bytes; it may not interpret
them.**

What that permits here:

* comparing the first few bytes with a signature,
* attempting a UTF-8 decode of a text format,
* counting bytes.

What it forbids, and what milestone 3's parsers exist for:

* extracting text or page counts from a PDF,
* opening the DOCX archive to look for `word/document.xml`,
* rendering or normalising Markdown.

The DOCX case is the one that tempts. `PK\\x03\\x04` proves only "a ZIP", and
proving it is really a Word document means opening the archive — which is
parsing. So validation stops at the signature, and a renamed `.zip` fails
later, in the parser, as a job with a recorded stage error. That is what the
job record is for; guessing here would be worse than failing there.

The size limit is enforced **while reading**, in bounded chunks. `Content-
Length` is a number the client chose and is never the check.
"""

import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import BinaryIO

# `document_versions.original_filename` is varchar(512).
MAX_FILENAME_LENGTH = 512

FALLBACK_FILENAME = "upload"

# How much is read at a time while measuring and copying an upload. The point
# is that memory does not follow the size of the request.
READ_CHUNK_BYTES = 64 * 1024

# Enough for every signature below.
SIGNATURE_BYTES = 8

PDF_SIGNATURE = b"%PDF-"
# DOCX is a ZIP container. This proves the container, and nothing about what
# is inside it — deliberately.
ZIP_SIGNATURE = b"PK\x03\x04"

# What each extension is, what its bytes must start with, and which MIME types
# a client may plausibly declare for it. The declared type is evidence; the
# signature decides.
_PDF_TYPES = frozenset({"application/pdf", "application/x-pdf"})
_DOCX_TYPES = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
    }
)
_TEXT_TYPES = frozenset(
    {"text/plain", "text/markdown", "text/x-markdown", "application/octet-stream"}
)


@dataclass(frozen=True)
class Format:
    """One accepted format."""

    name: str
    signature: bytes | None
    declared_types: frozenset[str]
    is_text: bool


FORMATS: dict[str, Format] = {
    ".pdf": Format("pdf", PDF_SIGNATURE, _PDF_TYPES, is_text=False),
    ".docx": Format("docx", ZIP_SIGNATURE, _DOCX_TYPES, is_text=False),
    ".md": Format("markdown", None, _TEXT_TYPES, is_text=True),
    ".markdown": Format("markdown", None, _TEXT_TYPES, is_text=True),
    ".txt": Format("text", None, _TEXT_TYPES, is_text=True),
}


class UploadRejected(Exception):
    """An upload this system will not accept.

    `reason` is safe to return to the client: it names what was wrong with
    the request and never a path, an exception class or anything about how
    this system stores files.
    """

    status_code = 400

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class UnsupportedFormat(UploadRejected):
    status_code = 415


class FileTooLarge(UploadRejected):
    status_code = 413


class EmptyFile(UploadRejected):
    status_code = 400


def sanitize_filename(name: str | None) -> str:
    """A filename safe to store as metadata and show to a person.

    It is only ever metadata: nothing built from this reaches the filesystem,
    because the storage key comes from server-generated UUIDs. Sanitising it
    anyway means a name cannot carry a surprise into a log line, a response
    body or a page that displays it later.

    Directory parts go under both conventions, because a browser may send
    `C:\\Users\\ada\\policy.pdf` for a file chosen on Windows.
    """
    if not name:
        return FALLBACK_FILENAME

    # NUL truncates in C APIs; the rest are invisible and have no business in
    # a filename. Unicode "format" characters include the bidirectional
    # overrides used to disguise an extension.
    cleaned = "".join(
        character
        for character in name
        if character != "\x00" and unicodedata.category(character) not in ("Cc", "Cf")
    )

    cleaned = PurePosixPath(cleaned).name
    cleaned = PureWindowsPath(cleaned).name
    cleaned = cleaned.strip().strip(".")

    if not cleaned:
        return FALLBACK_FILENAME

    if len(cleaned) > MAX_FILENAME_LENGTH:
        # Trim the stem, keep the extension: the extension is the part a
        # person reads to know what the file is.
        suffix = PurePosixPath(cleaned).suffix[:MAX_FILENAME_LENGTH]
        stem = cleaned[: MAX_FILENAME_LENGTH - len(suffix)]
        cleaned = f"{stem}{suffix}"

    return cleaned


def extension_of(filename: str) -> str:
    return PurePosixPath(filename).suffix.lower()


def resolve_format(filename: str, allowed: tuple[str, ...]) -> tuple[str, Format]:
    """The format this filename claims, if it is one we accept."""
    extension = extension_of(filename)
    if not extension:
        raise UnsupportedFormat("the file has no extension")
    if extension not in allowed or extension not in FORMATS:
        raise UnsupportedFormat(f"{extension} files are not accepted")
    return extension, FORMATS[extension]


def check_declared_type(declared: str | None, fmt: Format) -> None:
    """The client's `Content-Type`, checked for agreement with the extension.

    Evidence, not authority. A client that declares nothing useful is not
    refused for that alone — browsers send `application/octet-stream` for
    plenty of things — but one that declares a type belonging to a different
    format is contradicting itself, and that contradiction is worth refusing.
    """
    if not declared:
        return
    media_type = declared.split(";")[0].strip().lower()
    if not media_type:
        return
    if media_type not in fmt.declared_types:
        raise UnsupportedFormat(
            f"the declared content type does not match a {fmt.name} file"
        )


@dataclass(frozen=True)
class ValidatedUpload:
    """An upload that may be stored."""

    filename: str
    extension: str
    format_name: str
    size: int


def validate(
    *,
    filename: str | None,
    declared_type: str | None,
    source: BinaryIO,
    destination: BinaryIO,
    allowed_extensions: tuple[str, ...],
    max_bytes: int,
) -> ValidatedUpload:
    """Check an upload, copying it to `destination` as it is checked.

    Reads in bounded chunks, so a request larger than memory is refused
    rather than absorbed. The caller supplies `destination` — a temporary
    file — because the bytes have to go somewhere while being counted, and
    they must not go to their final location until they are known to be
    acceptable.
    """
    name = sanitize_filename(filename)
    extension, fmt = resolve_format(name, allowed_extensions)
    check_declared_type(declared_type, fmt)

    head = b""
    size = 0
    while chunk := source.read(READ_CHUNK_BYTES):
        size += len(chunk)
        if size > max_bytes:
            # Refused here rather than from Content-Length, which the client
            # chose and may have lied about in either direction.
            raise FileTooLarge(f"the file is larger than {max_bytes} bytes")
        if len(head) < SIGNATURE_BYTES:
            head += chunk[: SIGNATURE_BYTES - len(head)]
        destination.write(chunk)

    if size == 0:
        raise EmptyFile("the file is empty")

    if fmt.signature is not None and not head.startswith(fmt.signature):
        raise UnsupportedFormat(f"the file does not look like a {fmt.name} file")

    if fmt.is_text:
        destination.flush()
        destination.seek(0)
        _require_utf8(destination, fmt)
        destination.seek(0, 2)

    return ValidatedUpload(
        filename=name, extension=extension, format_name=fmt.name, size=size
    )


def _require_utf8(handle: BinaryIO, fmt: Format) -> None:
    """Whether the bytes are text at all — not what the text says.

    Decoded incrementally so a large file is not held in memory twice, and so
    a multi-byte character split across a chunk boundary is not mistaken for
    invalid input.
    """
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        while chunk := handle.read(READ_CHUNK_BYTES):
            decoder.decode(chunk)
        decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise UnsupportedFormat(
            f"the file is not valid UTF-8 text, so it is not a {fmt.name} file"
        ) from exc


__all__ = [
    "FORMATS",
    "MAX_FILENAME_LENGTH",
    "READ_CHUNK_BYTES",
    "EmptyFile",
    "FileTooLarge",
    "Format",
    "UnsupportedFormat",
    "UploadRejected",
    "ValidatedUpload",
    "check_declared_type",
    "extension_of",
    "resolve_format",
    "sanitize_filename",
    "validate",
]
