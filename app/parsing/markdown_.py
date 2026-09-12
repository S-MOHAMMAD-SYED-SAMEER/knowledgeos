"""Markdown, split at its headings.

Markdown is the one text format that structurally announces its own sections,
so this is where `section` comes from honestly: each block records the nearest
preceding heading, and text before the first heading has none.

Nothing is rendered. The markup is left exactly as written — the text stored
beside a citation should be what the document says, not a de-formatted
paraphrase of it. The only thing read here is which lines are headings, and
that is structure rather than content.

The module is `markdown_.py` because the specification's repository layout
says so, and because `markdown.py` would shadow a package name.
"""

import re
from typing import BinaryIO

from app.parsing.base import Block, ParsedDocument, ParseError

# `# Heading` through `###### Heading`. Up to three leading spaces is still a
# heading in CommonMark; four makes it an indented code block.
_ATX = re.compile(r"^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$")

# The underlined form:
#     Heading
#     =======
_SETEXT_UNDERLINE = re.compile(r"^ {0,3}(=+|-+)\s*$")

# Fenced code, where a `#` is a comment rather than a heading.
_FENCE = re.compile(r"^ {0,3}(```|~~~)")


class MarkdownParser:
    """Blocks of text, each tagged with the heading it sits under."""

    name = "markdown"

    def parse(self, handle: BinaryIO) -> ParsedDocument:
        raw = handle.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ParseError("the file is not valid UTF-8 text") from exc

        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        blocks: list[Block] = []
        current: list[str] = []
        section: str | None = None
        in_fence = False

        def flush() -> None:
            body = "\n".join(current)
            if body.strip():
                blocks.append(Block(text=body, section=section))
            current.clear()

        for index, line in enumerate(lines):
            if _FENCE.match(line):
                in_fence = not in_fence
                current.append(line)
                continue

            if in_fence:
                current.append(line)
                continue

            heading = _heading_of(line, lines[index + 1] if index + 1 < len(lines) else None)
            if heading is not None:
                # A heading ends the block before it and names the next one.
                flush()
                section = heading
                current.append(line)
                continue

            if _is_setext_underline(line, current):
                # Belongs to the heading line already consumed above.
                current.append(line)
                continue

            current.append(line)

        flush()

        # No pages: Markdown has none, so `page` and `page_count` stay None.
        return ParsedDocument(blocks=tuple(blocks))


def _heading_of(line: str, following: str | None) -> str | None:
    """The heading text of this line, or None if it is not a heading."""
    atx = _ATX.match(line)
    if atx:
        title = atx.group(2).strip()
        return title or None

    # Setext: this line is the title and the next one underlines it.
    if (
        following is not None
        and line.strip()
        and _SETEXT_UNDERLINE.match(following)
        and not _ATX.match(line)
    ):
        return line.strip()

    return None


def _is_setext_underline(line: str, current: list[str]) -> bool:
    return bool(current) and bool(_SETEXT_UNDERLINE.match(line))


__all__ = ["MarkdownParser"]
