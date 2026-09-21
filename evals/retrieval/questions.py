"""Loading and validating the evaluation question set.

`evals/fixtures/questions/*.yaml` — the specification's own words: each
question has exactly `id`, `text`, `category`, `expected_chunk_uids`,
`expected_abstain`, `filters`. This module is where "exactly" is enforced —
an unknown field, a wrong type, or a category outside the specification's
eight is refused here rather than discovered mid-run.

Pure data validation: no database, no provider. A question set describes
what to ask, not how to answer it — the "no provider calls in retrieval or
eval fixture code" rule applies just as much to this loader as to the data
files it reads.

Note the one intentional naming overlap: a question's own top-level
`category` (one of the specification's eight evaluation categories, e.g.
"metadata-filtered") is a different thing from `filters.category` (a
metadata value like "Policy" that narrows *retrieval*, matching
`documents.category`). Both are named `category` because that is what the
specification calls each of them; they are validated against different
vocabularies below.
"""

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.retrieval.filters import RetrievalFilters

# The specification's own eight evaluation categories, verbatim.
CATEGORIES = frozenset(
    {
        "directly answerable",
        "multi-document",
        "ambiguous",
        "insufficient evidence",
        "conflicting versions",
        "metadata-filtered",
        "citation-sensitive",
        "adversarial",
    }
)

# The specification's own six fields, verbatim — and exactly these six; a
# question with more or fewer is refused, not tolerated.
REQUIRED_FIELDS = frozenset(
    {"id", "text", "category", "expected_chunk_uids", "expected_abstain", "filters"}
)

# What a question's own `filters` mapping may contain: the same five fields
# `RetrievalFilters` accepts, so a question can never ask for a filter the
# retrieval pipeline does not implement.
ALLOWED_FILTER_KEYS = frozenset(
    {"document_id", "department", "category", "tags", "include_superseded"}
)

# sha256(...)[:32] hex, lower-case — the exact shape `app.chunking.uid`
# produces.
_CHUNK_UID_PATTERN = re.compile(r"^[0-9a-f]{32}$")

MINIMUM_QUESTION_COUNT = 40

DEFAULT_QUESTIONS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "questions"


class QuestionSetError(ValueError):
    """The question set is malformed.

    The message names what is wrong and identifies the question (or file)
    it came from; it never dumps an entire file's contents.
    """


@dataclass(frozen=True)
class Question:
    """One evaluation question, fully validated."""

    id: str
    text: str
    category: str
    expected_chunk_uids: frozenset[str]
    expected_abstain: bool
    filters: RetrievalFilters
    source_file: str


def _parse_filters(raw: object, *, question_id: str) -> RetrievalFilters:
    if not isinstance(raw, dict):
        raise QuestionSetError(f"{question_id}: 'filters' must be a mapping")

    unknown = set(raw) - ALLOWED_FILTER_KEYS
    if unknown:
        raise QuestionSetError(
            f"{question_id}: filters has unsupported key(s) {sorted(unknown)}"
        )

    document_id = raw.get("document_id")
    if document_id is not None:
        try:
            document_id = uuid.UUID(str(document_id))
        except ValueError as exc:
            raise QuestionSetError(
                f"{question_id}: filters.document_id is not a valid UUID"
            ) from exc

    for name in ("department", "category"):
        value = raw.get(name)
        if value is not None and not isinstance(value, str):
            raise QuestionSetError(f"{question_id}: filters.{name} must be a string")

    tags = raw.get("tags") or []
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise QuestionSetError(f"{question_id}: filters.tags must be a list of strings")

    include_superseded = raw.get("include_superseded", False)
    if not isinstance(include_superseded, bool):
        raise QuestionSetError(
            f"{question_id}: filters.include_superseded must be a boolean"
        )

    return RetrievalFilters(
        document_id=document_id,
        department=raw.get("department"),
        category=raw.get("category"),
        tags=tuple(tags),
        include_superseded=include_superseded,
    )


def _parse_question(raw: object, *, source_file: str) -> Question:
    if not isinstance(raw, dict):
        raise QuestionSetError(f"{source_file}: each question must be a mapping")

    fields = set(raw)
    missing = REQUIRED_FIELDS - fields
    if missing:
        raise QuestionSetError(
            f"{source_file}: question is missing field(s) {sorted(missing)}"
        )
    unknown = fields - REQUIRED_FIELDS
    if unknown:
        raise QuestionSetError(
            f"{source_file}: question has unsupported field(s) {sorted(unknown)}"
        )

    question_id = raw["id"]
    if not isinstance(question_id, str) or not question_id.strip():
        raise QuestionSetError(f"{source_file}: 'id' must be a non-empty string")

    text = raw["text"]
    if not isinstance(text, str) or not text.strip():
        raise QuestionSetError(f"{question_id}: 'text' must be a non-empty string")

    category = raw["category"]
    if category not in CATEGORIES:
        raise QuestionSetError(
            f"{question_id}: category {category!r} is not one of the "
            f"specification's eight categories"
        )

    expected_chunk_uids = raw["expected_chunk_uids"]
    if not isinstance(expected_chunk_uids, list):
        raise QuestionSetError(f"{question_id}: 'expected_chunk_uids' must be a list")
    for chunk_uid in expected_chunk_uids:
        if not isinstance(chunk_uid, str) or not _CHUNK_UID_PATTERN.match(chunk_uid):
            raise QuestionSetError(
                f"{question_id}: expected_chunk_uids contains {chunk_uid!r}, "
                f"which is not 32 lower-case hex characters"
            )

    expected_abstain = raw["expected_abstain"]
    if not isinstance(expected_abstain, bool):
        raise QuestionSetError(f"{question_id}: 'expected_abstain' must be a boolean")

    filters = _parse_filters(raw["filters"], question_id=question_id)

    return Question(
        id=question_id,
        text=text,
        category=category,
        # A YAML author repeating an id is harmless once collapsed to a set
        # — see tests/test_evals_metrics.py for why counting is set-based.
        expected_chunk_uids=frozenset(expected_chunk_uids),
        expected_abstain=expected_abstain,
        filters=filters,
        source_file=source_file,
    )


def load_question_files(directory: Path = DEFAULT_QUESTIONS_DIR) -> list[Question]:
    """Every `*.yaml` question file in `directory`, parsed and validated
    individually.

    Does not enforce the minimum count or category coverage — that is
    `validate_question_set`'s job, run once over the whole loaded set so a
    shortfall is reported against the total across every file, not
    file-by-file.
    """
    questions: list[Question] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            raise QuestionSetError(
                f"{path.name}: invalid YAML ({type(exc).__name__})"
            ) from exc

        if not isinstance(raw, list):
            raise QuestionSetError(f"{path.name}: must contain a YAML list of questions")

        questions.extend(
            _parse_question(entry, source_file=path.name) for entry in raw
        )

    return questions


def validate_question_set(questions: list[Question]) -> None:
    """Whole-set invariants: the minimum count, unique ids, and every
    category represented at least once."""
    if len(questions) < MINIMUM_QUESTION_COUNT:
        raise QuestionSetError(
            f"the question set has {len(questions)} question(s); the "
            f"specification requires at least {MINIMUM_QUESTION_COUNT}"
        )

    ids = [question.id for question in questions]
    duplicates = sorted({qid for qid in ids if ids.count(qid) > 1})
    if duplicates:
        raise QuestionSetError(f"duplicate question id(s): {duplicates}")

    missing_categories = sorted(CATEGORIES - {question.category for question in questions})
    if missing_categories:
        raise QuestionSetError(
            f"no question covers categor"
            f"{'y' if len(missing_categories) == 1 else 'ies'}: {missing_categories}"
        )


def load_question_set(directory: Path = DEFAULT_QUESTIONS_DIR) -> list[Question]:
    """Load and fully validate the question set: individual questions,
    then whole-set invariants."""
    questions = load_question_files(directory)
    validate_question_set(questions)
    return questions


__all__ = [
    "ALLOWED_FILTER_KEYS",
    "CATEGORIES",
    "DEFAULT_QUESTIONS_DIR",
    "MINIMUM_QUESTION_COUNT",
    "REQUIRED_FIELDS",
    "Question",
    "QuestionSetError",
    "load_question_files",
    "load_question_set",
    "validate_question_set",
]
