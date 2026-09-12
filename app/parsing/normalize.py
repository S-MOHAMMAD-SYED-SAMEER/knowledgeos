"""Turning parsed text into the one canonical form everything downstream uses.

This function is the foundation of two things that must never drift: the
version `checksum` is the SHA-256 of its output, and every `chunk_uid` hashes
a slice of its output. Change a rule here and every checksum and every uid in
the corpus changes with it, which is why the rules are enumerated rather than
left to taste, and why each one has its own test.

The rules, in the order they are applied:

1. CRLF and CR line endings become LF.
2. Unicode NFC.
3. Unicode categories `Cc` (control) and `Cf` (format) are removed, except
   LF and TAB. This is what strips NUL, zero-width joiners, and the
   bidirectional overrides used to disguise text.
4. Trailing whitespace is stripped from every line.
5. Runs of blank lines collapse to a single blank line.
6. Outer whitespace is stripped from the whole document.

And what it deliberately does **not** do: no lowercasing, no punctuation
removal, no stemming, no collapsing of internal spaces, no semantic rewriting
of any kind. A normalizer that changed meaning would make the chunk text
stored beside a citation a paraphrase of the document rather than a quotation
from it.

The whole thing is idempotent: `normalize(normalize(x)) == normalize(x)`, and
a test proves it.
"""

import re
import unicodedata

# Everything in Cc and Cf is removed except these two, which carry structure a
# reader depends on.
_KEPT_CONTROLS = frozenset({"\n", "\t"})

# Three or more newlines — a blank line is two — collapse to exactly two, so
# at most one blank line survives between paragraphs.
_BLANK_LINE_RUN = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """The canonical form of a piece of document text."""
    # 1. Line endings. Done before anything else so every later rule sees one
    #    kind of line break.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Composed form, so two spellings of the same accented character hash
    #    to the same value.
    text = unicodedata.normalize("NFC", text)

    # 3. Control and format characters, except the two that mean something.
    text = "".join(
        character
        for character in text
        if character in _KEPT_CONTROLS
        or unicodedata.category(character) not in ("Cc", "Cf")
    )

    # 4. Trailing whitespace per line. Internal whitespace is left alone.
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # 5. At most one blank line in a row.
    text = _BLANK_LINE_RUN.sub("\n\n", text)

    # 6. And nothing hanging off either end of the document.
    return text.strip()


__all__ = ["normalize"]
