"""Saving an answers run as it goes, so a stopped run can be resumed instead of paid for twice.

A run of 52 questions is about 53 model calls. If the provider fails part-way (a daily quota, an
outage), the questions already answered are real, paid-for results. After every question the run
writes its attempts so far to a checkpoint file; `--resume <file>` skips what is done and carries on.

A checkpoint is not a result. Only a run that finishes every question writes an `eval_runs` row and
a report; a stopped run writes this file and nothing official. A checkpoint made with a different
model, threshold, corpus or question set is refused: mixing answers from two configurations into one
set of numbers would be a fabricated result.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID

from evals.answers.metrics import AnswerAttempt, AttemptOutcome
from evals.answers.semantic import SemanticGroundingResult, SentenceEntailment

SCHEMA_VERSION = 1


class CheckpointError(ValueError):
    """A checkpoint that cannot be used for this run. The message says why and never carries answers."""


@dataclass(frozen=True)
class Checkpoint:
    fingerprint: dict
    attempts: list[AnswerAttempt]
    persisted_query_ids: list[UUID]
    stopped_early: str | None


def fingerprint(
    *,
    llm_model: str | None,
    embedding_model: str,
    rerank_model: str,
    abstention_threshold: float | None,
    question_ids: list[str],
) -> dict:
    """What has to be the same for two partial runs to be one run."""
    return {
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "rerank_model": rerank_model,
        "abstention_threshold": abstention_threshold,
        "question_ids": list(question_ids),
    }


def _attempt_to_dict(attempt: AnswerAttempt) -> dict:
    data = asdict(attempt)
    data["outcome"] = attempt.outcome.value
    if attempt.semantic is not None:
        data["semantic"]["sentence_scores"] = [
            {**score, "cited_chunk_uids": sorted(score["cited_chunk_uids"])}
            for score in data["semantic"]["sentence_scores"]
        ]
    return data


def _attempt_from_dict(data: dict) -> AnswerAttempt:
    fields = dict(data)
    fields["outcome"] = AttemptOutcome(fields["outcome"])
    semantic = fields.get("semantic")
    if semantic is not None:
        fields["semantic"] = SemanticGroundingResult(
            status=semantic["status"],
            reason=semantic["reason"],
            sentence_scores=[
                SentenceEntailment(
                    sentence=score["sentence"],
                    cited_chunk_uids=frozenset(score["cited_chunk_uids"]),
                    max_score=score["max_score"],
                    supported=score["supported"],
                )
                for score in semantic["sentence_scores"]
            ],
            unsupported_claim_rate=semantic["unsupported_claim_rate"],
        )
    return AnswerAttempt(**fields)


def save_checkpoint(
    path: Path,
    *,
    fingerprint: dict,
    attempts: list[AnswerAttempt],
    persisted_query_ids: list[UUID],
    stopped_early: str | None = None,
) -> None:
    """Write the checkpoint whole, then move it into place, so a crash mid-write never leaves a
    half-written file where a good one was."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "stopped_early": stopped_early,
        "attempts": [_attempt_to_dict(a) for a in attempts],
        "persisted_query_ids": [str(q) for q in persisted_query_ids],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def load_checkpoint(path: Path, *, expected_fingerprint: dict) -> Checkpoint:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CheckpointError(f"no checkpoint at {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"the checkpoint at {path} cannot be read ({type(exc).__name__})") from exc

    if raw.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointError(f"the checkpoint at {path} has an unsupported format version")

    saved = raw.get("fingerprint", {})
    different = sorted(key for key in expected_fingerprint if saved.get(key) != expected_fingerprint[key])
    if different:
        raise CheckpointError(
            f"the checkpoint was made with a different configuration ({', '.join(different)}); "
            "answers from two configurations must not be combined. Start a new run instead."
        )

    try:
        attempts = [_attempt_from_dict(item) for item in raw["attempts"]]
        query_ids = [UUID(item) for item in raw["persisted_query_ids"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(f"the checkpoint at {path} is damaged ({type(exc).__name__})") from exc

    return Checkpoint(
        fingerprint=saved,
        attempts=attempts,
        persisted_query_ids=query_ids,
        stopped_early=raw.get("stopped_early"),
    )


__all__ = [
    "Checkpoint",
    "CheckpointError",
    "fingerprint",
    "load_checkpoint",
    "save_checkpoint",
]
