"""Citation validation: the specification's three deterministic rules
(§9), exact and provider-free.

This module makes **no provider call and performs no filesystem or network
I/O** — the specification's layering rule (§4) names
`app/generation/citations.py` explicitly as one of the three packages that
must contain no provider calls and be callable as pure functions over
fixture data, the same rule that already governs `app/retrieval/` and
`app/chunking/`.

The specification's three rules, verbatim:

1. Every cited chunk_uid exists in the retrieved evidence set for that
   query. If not, invalid citation, answer rejected.
2. Every sentence containing a factual claim carries at least one citation
   — a coverage metric.
3. Citations pointing to non-selected chunks are invalid.

This project adds one more, load-bearing check that rule 2 cannot be
verified without: the top-level `citations` list and the inline
`[<chunk_uid>]` markers actually present in `answer` must agree exactly.
The specification's output format gives `citations` as one flat list for
the whole answer, with no per-sentence structure — inline markers are what
make "every sentence carries a citation" checkable at all, and requiring
the two to match is what keeps the flat list honest rather than a
disconnected second copy of the same claim.

**"Substantive factual sentence" is not defined by the specification** — §9
introduces the term without a test for it. This project's own, documented
choice, in the specification's own spirit of exactness over nuance: every
non-empty sentence in a non-abstained answer is treated as requiring a
citation. There is no "exempt because it's just a transition" carve-out,
because that carve-out would itself be an undocumented judgment call — and
it is exactly what the versioned prompt instructs the model to produce, so
the two sides of the contract (what the model is told, what this module
checks) agree with each other.
"""

import re

# The exact chunk_uid shape `app.chunking.uid.chunk_uid` produces and
# `evals/retrieval/questions.py` already validates against: sha256[:32],
# lower-case hex.
_CHUNK_UID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_CITATION_MARKER = re.compile(r"\[([0-9a-f]{32})\]")

# A dependency-free sentence boundary: punctuation followed by whitespace
# and the start of a new sentence (capital letter, digit, quote, or a
# citation marker). Not linguistically perfect — no NLP library is
# declared, the same discipline `app/chunking/` applies by counting
# whitespace words rather than importing a tokenizer.
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\[])')


class CitationError(ValueError):
    """The parsed answer's citations fail deterministic validation.

    The message names which rule failed and the offending chunk_uid(s) or
    sentence — never the full evidence text, never the raw model output
    beyond the one offending sentence already being validated.
    """


def split_sentences(text: str) -> list[str]:
    """A deterministic, dependency-free sentence split."""
    stripped = text.strip()
    if not stripped:
        return []
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(stripped) if s.strip()]


def extract_inline_citations(sentence: str) -> frozenset[str]:
    """Every `[<chunk_uid>]` marker in one sentence."""
    return frozenset(_CITATION_MARKER.findall(sentence))


def extract_all_inline_citations(answer_text: str) -> frozenset[str]:
    """Every `[<chunk_uid>]` marker anywhere in the answer."""
    return frozenset(_CITATION_MARKER.findall(answer_text))


def citation_coverage(answer_text: str) -> float:
    """Fraction of sentences carrying at least one citation marker.

    `1.0` for an answer with no sentences at all — there is nothing left
    uncovered, the same reasoning an empty-set precision/recall metric
    would use if the specification defined one here, which it does not.
    """
    sentences = split_sentences(answer_text)
    if not sentences:
        return 1.0
    covered = sum(1 for sentence in sentences if extract_inline_citations(sentence))
    return covered / len(sentences)


def is_fully_covered(answer_text: str) -> bool:
    """Rule 2 as a boolean: every sentence carries at least one citation."""
    return citation_coverage(answer_text) == 1.0


def citation_validity(
    *,
    answer_text: str,
    declared_citations: list[str],
    retrieved_chunk_uids: frozenset[str],
    selected_chunk_uids: frozenset[str],
) -> bool:
    """Rules 1 and 3, plus the declared/inline agreement this project's
    architecture requires — as one boolean, for grounding's independent
    recomputation (`app/generation/grounding.py`). Says nothing about
    abstention; `validate_citations` below is the full, raising,
    abstention-aware gate the generation orchestration actually calls.
    """
    if not all(_CHUNK_UID_PATTERN.match(uid) for uid in declared_citations):
        return False
    declared = frozenset(declared_citations)
    if not declared <= retrieved_chunk_uids:  # rule 1
        return False
    if not declared <= selected_chunk_uids:  # rule 3
        return False
    if declared != extract_all_inline_citations(answer_text):  # declared/inline agreement
        return False
    return True


def abstention_flag_consistent(
    *, abstained: bool, declared_citations: list[str], answer_text: str
) -> bool:
    """The deterministic grounding layer's third check: an abstained
    answer must carry no citations at all, declared or inline. A
    non-abstained answer places no constraint here — its citations are
    checked by `citation_validity` and its coverage by `is_fully_covered`
    instead.
    """
    if not abstained:
        return True
    inline = extract_all_inline_citations(answer_text)
    return not declared_citations and not inline


def validate_citations(
    *,
    answer_text: str,
    declared_citations: list[str],
    retrieved_chunk_uids: frozenset[str],
    selected_chunk_uids: frozenset[str],
    abstained: bool,
) -> None:
    """The hard gate `app/generation/generator.py` calls before an answer
    is ever persisted or returned. Raises `CitationError` on the first
    violation found — the specification's own words are "answer rejected",
    a binary outcome, not a partial-credit report.

    Built from the same boolean checks the module exposes individually, so
    the gate and `app/generation/grounding.py`'s independent recomputation
    can never silently disagree about what each rule means.
    """
    if abstained:
        if not abstention_flag_consistent(
            abstained=True, declared_citations=declared_citations, answer_text=answer_text
        ):
            raise CitationError(
                "an abstained answer must carry no citations, declared or inline"
            )
        return

    for uid in declared_citations:
        if not _CHUNK_UID_PATTERN.match(uid):
            raise CitationError(f"declared citation is not a chunk_uid shape: {uid!r}")

    declared = frozenset(declared_citations)

    unknown = declared - retrieved_chunk_uids
    if unknown:  # rule 1
        raise CitationError(
            f"cited chunk_uid(s) not in the retrieved evidence set: {sorted(unknown)}"
        )

    unselected = declared - selected_chunk_uids
    if unselected:  # rule 3
        raise CitationError(f"cited chunk_uid(s) not selected: {sorted(unselected)}")

    inline = extract_all_inline_citations(answer_text)
    if declared != inline:
        raise CitationError(
            "declared citations do not match the inline [<chunk_uid>] markers "
            f"in the answer text: declared={sorted(declared)} inline={sorted(inline)}"
        )

    if not is_fully_covered(answer_text):  # rule 2
        uncovered = [s for s in split_sentences(answer_text) if not extract_inline_citations(s)]
        raise CitationError(f"sentence(s) with no citation: {uncovered!r}")


__all__ = [
    "CitationError",
    "abstention_flag_consistent",
    "citation_coverage",
    "citation_validity",
    "extract_all_inline_citations",
    "extract_inline_citations",
    "is_fully_covered",
    "split_sentences",
    "validate_citations",
]
