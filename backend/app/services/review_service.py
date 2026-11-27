"""Review state, derived from the append-only review tables. Predictions are never modified."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AnalysisReview, Beat, Prediction, Review


def latest_review_status(session: Session, analysis_id: str | None = None) -> dict[str, str]:
    """beat_id -> status of its most recent review."""
    stmt = select(Review.beat_id, Review.status).join(Beat).order_by(Review.created_at)
    if analysis_id:
        stmt = stmt.where(Beat.analysis_id == analysis_id)
    return {beat_id: status for beat_id, status in session.execute(stmt)}


def pending_counts(session: Session) -> dict[str, int]:
    """analysis_id -> flagged windows without a current reviewer label."""
    status = latest_review_status(session)
    counts: dict[str, int] = {}
    rows = session.execute(select(Beat.analysis_id, Beat.id).join(Prediction).where(Prediction.review_flag.is_(True)))
    for analysis_id, beat_id in rows:
        if status.get(beat_id) != "reviewed":
            counts[analysis_id] = counts.get(analysis_id, 0) + 1
    return counts


def signed_off(session: Session) -> set[str]:
    """Analyses whose most recent sign-off is 'reviewed'."""
    latest = {a: s for a, s in session.execute(
        select(AnalysisReview.analysis_id, AnalysisReview.status).order_by(AnalysisReview.created_at))}
    return {a for a, s in latest.items() if s == "reviewed"}
