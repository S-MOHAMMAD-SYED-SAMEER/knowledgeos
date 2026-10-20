"""The abstention threshold the application ships with, and where it came from.

A module with no imports, so `app/config.py` and `app/generation/abstention.py` can both read the same
numbers without importing each other.

PROVENANCE (read this before trusting the number)
-------------------------------------------------
`1.4451` is the cross-encoder rerank score below which the application abstains before the model is
ever called. It came from the calibration in `evals/answers/calibration.py` (`calibrate_threshold`) run
on the DEV split of the 2026-10-16 answers evaluation, against the gold as it stood then (gold-v1).

It is **calibrated on the same 52 questions it is then evaluated on**, so it is **in-sample**: the
test split was held out from the calibration, but it is a split of the same small, author-written
question set, and only 4 of the 29 dev questions expect an abstention. Do not read any abstention figure
scored with it as a measure of how it would do on questions it has not seen.

The decision to ship it as the default, unchanged by the gold-v2 edits, was Sameer's on 2026-10-20.
`docs/EVAL.md` has the numbers.

It is on the scale of `cross-encoder/ms-marco-MiniLM-L-6-v2`'s scores. A different reranker (the
passthrough used to measure what reranking adds, or a test double) scores on another scale, so the
threshold is applied only when that cross-encoder did the reranking.
"""

DEFAULT_ABSTENTION_RERANK_THRESHOLD: float | None = 1.4451

#: The reranker whose score scale `DEFAULT_ABSTENTION_RERANK_THRESHOLD` is on.
CALIBRATED_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def effective_threshold(configured: float | None, rerank_model: str) -> float | None:
    """The threshold to apply for a query reranked by `rerank_model`.

    `None` (no threshold configured) stays `None`. A threshold is applied only to the scale it was
    calibrated on.
    """
    if configured is None:
        return None
    return configured if rerank_model == CALIBRATED_RERANK_MODEL else None
