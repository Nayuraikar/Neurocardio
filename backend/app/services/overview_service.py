from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Analysis, AnalysisReview, ECGRecord, Report, Review, Beat
from app.services.review_service import pending_counts, signed_off


def fail_interrupted_analyses(session: Session) -> None:
    for row in session.scalars(select(Analysis).where(Analysis.status.in_(("running", "queued")))):
        row.status = "failed"
        row.error_message = "interrupted: the server stopped before this analysis finished"
        row.completed_at = datetime.now(timezone.utc)
    session.commit()


def overview(session: Session, model_version: str) -> dict:
    count = lambda stmt: int(session.execute(stmt).scalar_one())
    completed = session.scalars(select(Analysis.id).where(Analysis.status == "completed")).all()
    pending, done = pending_counts(session), signed_off(session)
    needing = [a for a in completed if pending.get(a) and a not in done]
    return {"recordings": count(select(func.count()).select_from(ECGRecord)),
            "completed_analyses": len(completed),
            "review_required": len(needing),
            "pending_review_windows": sum(pending[a] for a in needing),
            "first_review_analysis_id": needing[0] if needing else None,
            "model_version": model_version, "activity": activity(session),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def activity(session: Session, limit: int = 8) -> list[dict]:
    """Events built from stored timestamps only."""
    events = []
    for r in session.scalars(select(ECGRecord).order_by(ECGRecord.uploaded_at.desc()).limit(limit)):
        events.append({"type": "upload", "at": r.uploaded_at, "label": r.filename, "analysis_id": None})
    stmt = select(Analysis, ECGRecord.filename).join(ECGRecord).order_by(Analysis.started_at.desc()).limit(limit)
    for a, name in session.execute(stmt):
        events.append({"type": f"analysis {a.status}", "at": a.completed_at or a.started_at, "label": name, "analysis_id": a.id})
    stmt = (select(Review, Beat.analysis_id, ECGRecord.filename).join(Beat, Review.beat_id == Beat.id)
            .join(Analysis, Beat.analysis_id == Analysis.id).join(ECGRecord).order_by(Review.created_at.desc()).limit(limit))
    for r, analysis_id, name in session.execute(stmt):
        events.append({"type": f"window {r.status}", "at": r.created_at, "label": name, "analysis_id": analysis_id})
    stmt = select(AnalysisReview, ECGRecord.filename).join(Analysis, AnalysisReview.analysis_id == Analysis.id).join(ECGRecord)
    for r, name in session.execute(stmt.order_by(AnalysisReview.created_at.desc()).limit(limit)):
        events.append({"type": f"analysis marked {r.status}", "at": r.created_at, "label": name, "analysis_id": r.analysis_id})
    stmt = select(Report, ECGRecord.filename).join(Analysis, Report.analysis_id == Analysis.id).join(ECGRecord)
    for r, name in session.execute(stmt.order_by(Report.created_at.desc()).limit(limit)):
        events.append({"type": f"{r.format.upper()} report", "at": r.created_at, "label": name, "analysis_id": r.analysis_id})
    events.sort(key=lambda e: e["at"].replace(tzinfo=None), reverse=True)
    return events[:limit]
