"""Loading the frozen dev/test split (`evals/fixtures/splits.yaml`,
locked decision D4) and applying it to the live, loaded question set.

Pure data loading and validation: no database, no provider — the same
discipline `evals/retrieval/questions.py` applies to the question files
themselves. This module's whole job is integrity: every question in the
live set must appear in exactly one half of the split, and the split must
never silently drift out of sync with the question files it was frozen
against.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from evals.retrieval.questions import Question

DEFAULT_SPLITS_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "splits.yaml"
)


class SplitError(ValueError):
    """The split file is malformed, or does not exactly match the live
    question set. Never silently ignored or auto-repaired — a stale split
    is a calibration bug waiting to happen."""


@dataclass(frozen=True)
class QuestionSplit:
    """The live question set, partitioned by the frozen split file."""

    dev: list[Question]
    test: list[Question]


def load_split_ids(path: Path = DEFAULT_SPLITS_PATH) -> tuple[frozenset[str], frozenset[str]]:
    """The frozen dev/test id sets, exactly as checked in — no
    recomputation, no re-stratification. Raises `SplitError` if the file
    is malformed or assigns any id to both halves."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "dev" not in raw or "test" not in raw:
        raise SplitError(f"{path.name}: must be a mapping with 'dev' and 'test' lists")

    dev_ids = raw["dev"]
    test_ids = raw["test"]
    if not isinstance(dev_ids, list) or not all(isinstance(i, str) for i in dev_ids):
        raise SplitError(f"{path.name}: 'dev' must be a list of question ids")
    if not isinstance(test_ids, list) or not all(isinstance(i, str) for i in test_ids):
        raise SplitError(f"{path.name}: 'test' must be a list of question ids")

    dev = frozenset(dev_ids)
    test = frozenset(test_ids)

    if len(dev) != len(dev_ids):
        raise SplitError(f"{path.name}: 'dev' contains a duplicate question id")
    if len(test) != len(test_ids):
        raise SplitError(f"{path.name}: 'test' contains a duplicate question id")

    overlap = dev & test
    if overlap:
        raise SplitError(
            f"{path.name}: question id(s) {sorted(overlap)} appear in both dev and test"
        )

    return dev, test


def apply_split(
    questions: list[Question], path: Path = DEFAULT_SPLITS_PATH
) -> QuestionSplit:
    """Partition the live, loaded question set by the frozen split file.

    Raises `SplitError` if the split and the live question set disagree
    about which ids exist — a question added, removed, or renamed since
    the split was frozen must be caught here, not silently dropped from
    calibration or silently left unsplit.
    """
    dev_ids, test_ids = load_split_ids(path)
    live_ids = frozenset(q.id for q in questions)

    missing_from_split = live_ids - (dev_ids | test_ids)
    if missing_from_split:
        raise SplitError(
            f"{path.name}: question id(s) {sorted(missing_from_split)} exist in "
            "the live question set but are not assigned in the frozen split"
        )
    stale_in_split = (dev_ids | test_ids) - live_ids
    if stale_in_split:
        raise SplitError(
            f"{path.name}: question id(s) {sorted(stale_in_split)} are assigned "
            "in the frozen split but no longer exist in the live question set"
        )

    by_id = {q.id: q for q in questions}
    return QuestionSplit(
        dev=[by_id[qid] for qid in sorted(dev_ids)],
        test=[by_id[qid] for qid in sorted(test_ids)],
    )


__all__ = [
    "DEFAULT_SPLITS_PATH",
    "QuestionSplit",
    "SplitError",
    "apply_split",
    "load_split_ids",
]
