from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models import Analysis, Beat, ECGRecord, Review


def get_analysis(session: Session, analysis_id: str) -> Analysis | None:
    return session.scalar(select(Analysis).where(Analysis.id == analysis_id).options(
        selectinload(Analysis.record), selectinload(Analysis.beats).selectinload(Beat.prediction),
        selectinload(Analysis.beats).selectinload(Beat.reviews)))


def get_record(session: Session, record_id: str) -> ECGRecord | None:
    return session.get(ECGRecord, record_id)


def get_beat(session: Session, beat_id: str) -> Beat | None:
    return session.scalar(select(Beat).where(Beat.id == beat_id).options(
        selectinload(Beat.prediction), selectinload(Beat.analysis).selectinload(Analysis.record),
        selectinload(Beat.reviews)))
