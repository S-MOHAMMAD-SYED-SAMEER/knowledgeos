"""The thing that actually picks jobs up.

Polling, not an in-process background task. The difference is durability: a
job table exists so that work survives, and a task attached to the request
that created it cannot retry after a restart and cannot touch the jobs that
were already queued before the process started. Milestone 2 left real `queued`
rows behind, and this is what finally drains them.

**Claiming.** A job is taken with `SELECT … FOR UPDATE SKIP LOCKED`, which is
PostgreSQL's answer to two workers reaching for the same row: the second one
sees the row is taken and moves to the next instead of blocking on it. The
claim and the status change commit together, so a claimed job is visibly
`parsing` to everybody else. That makes the design safe if it is ever run in
more than one process, without a broker, a lock table, or anything else to go
wrong.

**Attempts.** Every try increments `attempts`. A job that fails below the
bound goes back to `queued` and will be picked up again; at the bound it stops
at `failed`. There is no backoff — the specification asks for retries bounded
by `attempts` and nothing more, and a delay nobody specified is a delay nobody
can reason about.

**Transactions.** A job's work is one transaction: stage statuses, the
checksum, the page count and the chunks all commit together, or none of them
do. The bookkeeping for a failure is its own transaction afterwards, because
the first thing a failure does is roll back the work — and the record that it
failed must survive that rollback.

No global state: the scheduler is created, held by the application's lifespan,
and stopped with it.
"""

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.db.session import get_sessionmaker
from app.ingestion.pipeline import StageFailed, run_job
from app.models import IngestionJob, JobStatus
from app.storage import Storage, get_storage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Outcome:
    """What one pass of the runner did."""

    job_id: object
    status: JobStatus
    attempts: int
    detail: str | None = None


def claim_next_job(session: Session, *, commit: bool = True) -> IngestionJob | None:
    """Take the oldest queued job, or return None if there is none.

    `FOR UPDATE SKIP LOCKED` is the whole mechanism: a row another worker has
    already locked is skipped rather than waited for, so two runners never
    take the same job and neither one blocks.

    **The claim is committed before any work begins**, and that ordering is
    not incidental. `attempts` is what bounds retrying, so it has to survive
    the rollback that a failure performs — counted inside the work's own
    transaction it would be undone by every failure, and a job that failed
    for ever would be retried for ever. Committing here also means a job lost
    to a crash has its attempt counted rather than being retried invisibly.

    Releasing the row lock at that commit is safe: the job is no longer
    `queued`, and the query above only ever selects jobs that are.

    `commit=False` exists for the test that has to hold the lock open to
    prove another connection skips the row.
    """
    job = session.execute(
        select(IngestionJob)
        .where(IngestionJob.status == JobStatus.QUEUED)
        .order_by(IngestionJob.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    ).scalar_one_or_none()

    if job is None:
        return None

    job.attempts += 1
    job.status = JobStatus.PARSING
    job.started_at = _now()
    job.stage_error = None
    if commit:
        session.commit()
    else:
        session.flush()
    return job


def process_one(
    session: Session, storage: Storage, settings: Settings
) -> Outcome | None:
    """Claim a job and run it. Returns None when there is nothing queued."""
    job = claim_next_job(session)
    if job is None:
        return None

    job_id = job.id
    attempts = job.attempts

    try:
        run_job(session, job, storage, settings)
    except StageFailed as exc:
        session.rollback()
        return _record_failure(session, job_id, exc.stage, exc.detail, settings)
    except Exception as exc:  # noqa: BLE001 - nothing may escape a job
        session.rollback()
        logger.exception("Job %s failed unexpectedly", job_id)
        return _record_failure(
            session,
            job_id,
            JobStatus.PARSING,
            f"an unexpected error occurred ({type(exc).__name__})",
            settings,
        )

    job.completed_at = _now()
    session.commit()
    logger.info("Job %s reached %s on attempt %d.", job_id, job.status, attempts)
    return Outcome(job_id=job_id, status=JobStatus.INDEXING, attempts=attempts)


def run_pending(
    sessions: sessionmaker[Session] | None = None,
    storage: Storage | None = None,
    settings: Settings | None = None,
    limit: int = 10,
) -> list[Outcome]:
    """Drain up to `limit` queued jobs. One session per job.

    A session each, so one job's failure cannot leave another job's work in a
    rolled-back transaction.
    """
    make_session = sessions or get_sessionmaker()
    resolved = settings or get_settings()
    store = storage or get_storage()

    outcomes: list[Outcome] = []
    for _ in range(limit):
        with make_session() as session:
            outcome = process_one(session, store, resolved)
        if outcome is None:
            break
        outcomes.append(outcome)
    return outcomes


class IngestionRunner:
    """Owns the scheduler, and nothing else owns it.

    Created by the application's lifespan and stopped by it. Held on the
    instance rather than in a module-level variable, so two applications in
    one process (which the test suite builds routinely) do not share one.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._scheduler = None

    @property
    def running(self) -> bool:
        return self._scheduler is not None and self._scheduler.running

    def start(self) -> None:
        from apscheduler.schedulers.background import BackgroundScheduler

        if self.running:
            return
        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(
            self._tick,
            "interval",
            seconds=self._settings.ingestion_poll_seconds,
            id="ingestion",
            # A slow batch must not have a second one start beside it.
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.start()
        logger.info(
            "Ingestion runner polling every %.1fs.",
            self._settings.ingestion_poll_seconds,
        )

    def stop(self) -> None:
        if self._scheduler is None:
            return
        # Let a job in flight finish: it holds a row lock and an open
        # transaction, and killing it mid-write is how a half-written version
        # happens.
        self._scheduler.shutdown(wait=True)
        self._scheduler = None
        logger.info("Ingestion runner stopped.")

    def _tick(self) -> None:
        try:
            run_pending(settings=self._settings)
        except Exception:  # noqa: BLE001 - a poll must never kill the scheduler
            logger.exception("An ingestion poll failed.")


# --- internals --------------------------------------------------------------


def _record_failure(
    session: Session,
    job_id,
    stage: JobStatus,
    detail: str,
    settings: Settings,
) -> Outcome:
    """Write down that a job failed, in a transaction of its own.

    The work has already been rolled back, which is what the failure required
    — so this reads the job again and records the outcome separately.
    """
    job = session.get(IngestionJob, job_id)
    if job is None:  # pragma: no cover - the row was deleted underneath us
        return Outcome(job_id=job_id, status=JobStatus.FAILED, attempts=0, detail=detail)

    exhausted = job.attempts >= settings.max_attempts
    job.status = JobStatus.FAILED if exhausted else JobStatus.QUEUED
    # Stage context, no document content: the specification requires errors
    # stored "without raw document content", and this string is readable
    # through the API.
    job.stage_error = f"{stage}: {detail}"
    job.completed_at = _now() if exhausted else None
    session.commit()

    logger.warning(
        "Job %s failed at %s on attempt %d of %d.",
        job_id,
        stage,
        job.attempts,
        settings.max_attempts,
    )
    return Outcome(
        job_id=job_id, status=job.status, attempts=job.attempts, detail=job.stage_error
    )


def _now():
    from datetime import UTC, datetime

    return datetime.now(UTC)


__all__ = [
    "IngestionRunner",
    "Outcome",
    "claim_next_job",
    "process_one",
    "run_pending",
]
