"""The evaluation question loader: strict validation, no database, no
provider — every fixture here is a small YAML file written to a temp
directory.
"""

import textwrap
import uuid

import pytest

from app.retrieval.filters import RetrievalFilters
from evals.retrieval.questions import (
    CATEGORIES,
    MINIMUM_QUESTION_COUNT,
    QuestionSetError,
    load_question_files,
    load_question_set,
    validate_question_set,
)

_VALID_UID = "eb40bb7741165afa76f77c3bd4c3083d"[:32]


def _write(tmp_path, filename: str, content: str) -> None:
    (tmp_path / filename).write_text(textwrap.dedent(content))


def _question(
    qid="q001",
    text="How do I request access?",
    category="directly answerable",
    expected_chunk_uids=None,
    expected_abstain=False,
    filters=None,
) -> str:
    uids = expected_chunk_uids if expected_chunk_uids is not None else [_VALID_UID]
    uids_yaml = "[]" if not uids else "[" + ", ".join(f'"{u}"' for u in uids) + "]"
    filters_yaml = filters if filters is not None else "{}"
    return (
        f'- id: "{qid}"\n'
        f'  text: "{text}"\n'
        f'  category: "{category}"\n'
        f"  expected_chunk_uids: {uids_yaml}\n"
        f"  expected_abstain: {str(expected_abstain).lower()}\n"
        f"  filters: {filters_yaml}\n"
    )


# --- individual question parsing -------------------------------------


def test_a_well_formed_question_loads(tmp_path) -> None:
    _write(tmp_path, "a.yaml", _question())

    questions = load_question_files(tmp_path)

    assert len(questions) == 1
    q = questions[0]
    assert q.id == "q001"
    assert q.category == "directly answerable"
    assert q.expected_chunk_uids == {_VALID_UID}
    assert q.expected_abstain is False
    assert q.filters == RetrievalFilters()
    assert q.source_file == "a.yaml"


def test_multiple_files_are_all_loaded(tmp_path) -> None:
    _write(tmp_path, "a.yaml", _question(qid="q001"))
    _write(tmp_path, "b.yaml", _question(qid="q002"))

    questions = load_question_files(tmp_path)

    assert {q.id for q in questions} == {"q001", "q002"}


def test_missing_field_is_refused(tmp_path) -> None:
    _write(
        tmp_path,
        "a.yaml",
        """\
        - id: "q001"
          text: "How do I request access?"
          category: "directly answerable"
          expected_chunk_uids: []
          expected_abstain: false
        """,
    )
    with pytest.raises(QuestionSetError, match="missing field"):
        load_question_files(tmp_path)


def test_unknown_field_is_refused(tmp_path) -> None:
    content = _question().rstrip() + '\n  extra_field: "not allowed"\n'
    _write(tmp_path, "a.yaml", content)

    with pytest.raises(QuestionSetError, match="unsupported field"):
        load_question_files(tmp_path)


@pytest.mark.parametrize("category", CATEGORIES)
def test_every_specified_category_is_accepted(tmp_path, category) -> None:
    _write(tmp_path, "a.yaml", _question(category=category))
    questions = load_question_files(tmp_path)
    assert questions[0].category == category


def test_an_unknown_category_is_refused(tmp_path) -> None:
    _write(tmp_path, "a.yaml", _question(category="not a real category"))
    with pytest.raises(QuestionSetError, match="eight categories"):
        load_question_files(tmp_path)


def test_expected_chunk_uids_must_be_32_lowercase_hex(tmp_path) -> None:
    _write(tmp_path, "a.yaml", _question(expected_chunk_uids=["NOT-VALID"]))
    with pytest.raises(QuestionSetError, match="32 lower-case hex"):
        load_question_files(tmp_path)


def test_expected_chunk_uids_rejects_uppercase(tmp_path) -> None:
    _write(tmp_path, "a.yaml", _question(expected_chunk_uids=[_VALID_UID.upper()]))
    with pytest.raises(QuestionSetError, match="32 lower-case hex"):
        load_question_files(tmp_path)


def test_expected_chunk_uids_may_be_empty(tmp_path) -> None:
    """Insufficient-evidence questions legitimately expect nothing."""
    _write(
        tmp_path,
        "a.yaml",
        _question(category="insufficient evidence", expected_chunk_uids=[], expected_abstain=True),
    )
    questions = load_question_files(tmp_path)
    assert questions[0].expected_chunk_uids == frozenset()


def test_expected_abstain_must_be_boolean(tmp_path) -> None:
    content = _question().replace("expected_abstain: false", 'expected_abstain: "no"')
    _write(tmp_path, "a.yaml", content)
    with pytest.raises(QuestionSetError, match="boolean"):
        load_question_files(tmp_path)


def test_text_must_be_non_empty(tmp_path) -> None:
    _write(tmp_path, "a.yaml", _question(text=""))
    with pytest.raises(QuestionSetError, match="non-empty"):
        load_question_files(tmp_path)


def test_malformed_yaml_fails_clearly(tmp_path) -> None:
    _write(tmp_path, "a.yaml", "this: is: not: valid: yaml: [[[")
    with pytest.raises(QuestionSetError, match="invalid YAML"):
        load_question_files(tmp_path)


def test_a_file_that_is_not_a_list_is_refused(tmp_path) -> None:
    _write(tmp_path, "a.yaml", "id: q001\ntext: not a list\n")
    with pytest.raises(QuestionSetError, match="YAML list"):
        load_question_files(tmp_path)


# --- filters ---------------------------------------------------------------


def test_department_filter_is_parsed(tmp_path) -> None:
    _write(tmp_path, "a.yaml", _question(filters='{department: "Engineering"}'))
    questions = load_question_files(tmp_path)
    assert questions[0].filters == RetrievalFilters(department="Engineering")


def test_tags_filter_is_parsed(tmp_path) -> None:
    _write(tmp_path, "a.yaml", _question(filters='{tags: ["access", "urgent"]}'))
    questions = load_question_files(tmp_path)
    assert questions[0].filters.tags == ("access", "urgent")


def test_include_superseded_filter_is_parsed(tmp_path) -> None:
    _write(tmp_path, "a.yaml", _question(filters="{include_superseded: true}"))
    questions = load_question_files(tmp_path)
    assert questions[0].filters.include_superseded is True


def test_document_id_filter_is_parsed(tmp_path) -> None:
    doc_id = uuid.uuid4()
    _write(tmp_path, "a.yaml", _question(filters=f'{{document_id: "{doc_id}"}}'))
    questions = load_question_files(tmp_path)
    assert questions[0].filters.document_id == doc_id


def test_an_unsupported_filter_key_is_refused(tmp_path) -> None:
    _write(tmp_path, "a.yaml", _question(filters='{page: 3}'))
    with pytest.raises(QuestionSetError, match="unsupported key"):
        load_question_files(tmp_path)


def test_an_invalid_document_id_filter_is_refused(tmp_path) -> None:
    _write(tmp_path, "a.yaml", _question(filters='{document_id: "not-a-uuid"}'))
    with pytest.raises(QuestionSetError, match="not a valid UUID"):
        load_question_files(tmp_path)


# --- whole-set validation ----------------------------------------------


def test_fewer_than_forty_questions_is_refused() -> None:
    from evals.retrieval.questions import Question

    def make(i):
        return Question(
            id=f"q{i:03d}",
            text="text",
            category="directly answerable",
            expected_chunk_uids=frozenset(),
            expected_abstain=True,
            filters=RetrievalFilters(),
            source_file="x.yaml",
        )

    questions = [make(i) for i in range(MINIMUM_QUESTION_COUNT - 1)]
    with pytest.raises(QuestionSetError, match="at least"):
        validate_question_set(questions)


def test_duplicate_ids_across_files_are_refused(tmp_path) -> None:
    from evals.retrieval.questions import Question

    def make(i):
        return Question(
            id=f"q{i:03d}",
            text="text",
            category="directly answerable",
            expected_chunk_uids=frozenset(),
            expected_abstain=True,
            filters=RetrievalFilters(),
            source_file="x.yaml",
        )

    # Enough distinct questions to clear the minimum-count check, plus one
    # deliberate id collision -- so this test isolates the duplicate-id
    # rule from the count rule rather than tripping the count check first.
    questions = [make(i) for i in range(MINIMUM_QUESTION_COUNT)]
    questions[1] = Question(
        id=questions[0].id,  # collides with questions[0]
        text="text",
        category="directly answerable",
        expected_chunk_uids=frozenset(),
        expected_abstain=True,
        filters=RetrievalFilters(),
        source_file="x.yaml",
    )

    with pytest.raises(QuestionSetError, match="duplicate question id"):
        validate_question_set(questions)


def test_a_missing_category_across_the_whole_set_is_refused() -> None:
    from evals.retrieval.questions import Question

    def make(i, category):
        return Question(
            id=f"q{i:03d}",
            text="text",
            category=category,
            expected_chunk_uids=frozenset(),
            expected_abstain=True,
            filters=RetrievalFilters(),
            source_file="x.yaml",
        )

    # Every category except "adversarial", repeated to reach the minimum.
    remaining = sorted(CATEGORIES - {"adversarial"})
    questions = [
        make(i, remaining[i % len(remaining)]) for i in range(MINIMUM_QUESTION_COUNT)
    ]
    with pytest.raises(QuestionSetError, match="adversarial"):
        validate_question_set(questions)


def test_load_question_set_combines_loading_and_validation(tmp_path) -> None:
    """A directory with too few questions fails at `load_question_set`,
    not only when `validate_question_set` is called separately."""
    _write(tmp_path, "a.yaml", _question())
    with pytest.raises(QuestionSetError, match="at least"):
        load_question_set(tmp_path)
