"""`evals/fixtures/splits.yaml` and `evals/answers/splits.py`: the frozen
dev/test split (D4) is internally consistent and matches the live,
frozen milestone 7 question set exactly -- no duplicate id, no id in both
halves, no question missing, no stale id left over from a question that
no longer exists.
"""

import pytest

from evals.answers.splits import DEFAULT_SPLITS_PATH, SplitError, apply_split, load_split_ids
from evals.retrieval.questions import load_question_set


def test_split_file_has_no_duplicate_or_overlapping_ids() -> None:
    dev, test = load_split_ids()
    assert len(dev & test) == 0


def test_split_covers_every_live_question_exactly_once() -> None:
    dev, test = load_split_ids()
    questions = load_question_set()
    live_ids = {q.id for q in questions}

    assert dev | test == live_ids
    assert len(dev) + len(test) == len(live_ids)


def test_apply_split_partitions_every_question_into_exactly_one_half() -> None:
    questions = load_question_set()
    split = apply_split(questions)

    assert len(split.dev) + len(split.test) == len(questions)
    dev_ids = {q.id for q in split.dev}
    test_ids = {q.id for q in split.test}
    assert dev_ids.isdisjoint(test_ids)
    assert dev_ids | test_ids == {q.id for q in questions}


def test_apply_split_raises_when_a_live_question_is_missing_from_the_split(tmp_path) -> None:
    import yaml

    questions = load_question_set()
    # Drop the last question from both halves of a copy of the real split,
    # so the live set now has one id the split file does not know about.
    dev, test = load_split_ids()
    stale_path = tmp_path / "splits.yaml"
    dropped = sorted(dev)[0]
    stale_path.write_text(yaml.safe_dump({"dev": sorted(dev - {dropped}), "test": sorted(test)}))

    with pytest.raises(SplitError):
        apply_split(questions, stale_path)


def test_apply_split_raises_when_the_split_has_a_stale_id(tmp_path) -> None:
    import yaml

    questions = load_question_set()
    dev, test = load_split_ids()
    stale_path = tmp_path / "splits.yaml"
    stale_path.write_text(
        yaml.safe_dump({"dev": sorted(dev) + ["nonexistent-question-id"], "test": sorted(test)})
    )

    with pytest.raises(SplitError):
        apply_split(questions, stale_path)


def test_load_split_ids_rejects_a_duplicate_within_one_half(tmp_path) -> None:
    import yaml

    path = tmp_path / "splits.yaml"
    path.write_text(yaml.safe_dump({"dev": ["a", "a"], "test": ["b"]}))
    with pytest.raises(SplitError):
        load_split_ids(path)


def test_load_split_ids_rejects_an_id_in_both_halves(tmp_path) -> None:
    import yaml

    path = tmp_path / "splits.yaml"
    path.write_text(yaml.safe_dump({"dev": ["a"], "test": ["a"]}))
    with pytest.raises(SplitError):
        load_split_ids(path)


def test_default_splits_path_is_the_committed_fixture() -> None:
    assert DEFAULT_SPLITS_PATH.exists()
    assert DEFAULT_SPLITS_PATH.name == "splits.yaml"
