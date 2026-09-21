"""`POST /answers/{answer_id}/feedback` — the specification's own feedback
endpoint (§11), writing to the specification's own `feedback` table (§5).

`create_feedback` is the one write path. Both this JSON endpoint and the
UI's feedback form (`app/ui/routes.py`) call it — never two implementations
of "does this answer exist, then insert a row" — the same reuse discipline
milestone 10's own locked decisions apply to the query pipeline.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.schemas import FeedbackIn, FeedbackOut
from app.db.session import get_session
from app.models import Answer, Feedback

router = APIRouter(tags=["feedback"])


def create_feedback(
    session: Session, *, answer_id: uuid.UUID, rating: str, reason: str | None
) -> Feedback:
    """Insert one feedback row. Callers check the answer exists first —
    this function only writes."""
    row = Feedback(answer_id=answer_id, rating=rating, reason=reason)
    session.add(row)
    session.commit()
    return row


@router.post(
    "/answers/{answer_id}/feedback",
    response_model=FeedbackOut,
    status_code=status.HTTP_201_CREATED,
)
def submit_feedback(
    answer_id: uuid.UUID,
    body: FeedbackIn,
    session: Annotated[Session, Depends(get_session)],
) -> FeedbackOut:
    if session.get(Answer, answer_id) is None:
        raise HTTPException(status_code=404, detail="No such answer")

    row = create_feedback(
        session, answer_id=answer_id, rating=body.rating, reason=body.reason
    )
    return FeedbackOut.model_validate(row)


__all__ = ["create_feedback", "router", "submit_feedback"]
