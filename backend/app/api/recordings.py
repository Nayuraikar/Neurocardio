"""Recording-centred endpoints used by the UI."""
from __future__ import annotations

import threading

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_db
from app.core.errors import ServiceError
from app.ml.preprocessing import LEFT_SAMPLES, WINDOW_SAMPLES, extract_windows, filter_continuous
from app.models import Analysis, AnalysisReview, Beat, ECGRecord, Explanation
from app.schemas.api import SignOffIn
from app.services.analysis_service import create_analysis, execute_analysis
from app.services.overview_service import overview
from app.services.quality_service import assess
from app.services.review_service import pending_counts, signed_off
from app.utils.serializers import analysis_dict, beat_dict, record_dict

router = APIRouter()
ANALYSIS_LOCK = threading.Lock()  # one inference job at a time


def _record(db: Session, record_id: str) -> ECGRecord:
    row = db.get(ECGRecord, record_id)
    if row is None:
        raise HTTPException(404, "recording not found")
    return row


def load_signal(record: ECGRecord) -> np.ndarray:
    try:
        return np.asarray(np.load(record.signal_path), dtype=np.float32)
    except (OSError, ValueError) as exc:
        raise HTTPException(410, "the stored signal file for this recording is missing or unreadable") from exc


def _filtered(record: ECGRecord) -> tuple[np.ndarray, np.ndarray]:
    raw = load_signal(record)
    try:
        return raw, filter_continuous(raw, record.sampling_rate)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/overview")
def get_overview(request: Request, db: Session = Depends(get_db)):
    return overview(db, request.app.state.config["selected_model"])


@router.get("/recordings")
def recordings(db: Session = Depends(get_db), limit: int = Query(500, ge=1, le=2000)):
    rows = db.scalars(select(ECGRecord).order_by(ECGRecord.uploaded_at.desc()).limit(limit)
                      .options(selectinload(ECGRecord.analyses))).all()
    pending, done = pending_counts(db), signed_off(db)
    out = []
    for r in rows:
        runs = sorted(r.analyses, key=lambda a: a.started_at, reverse=True)
        latest = runs[0] if runs else None
        out.append({**record_dict(r), "latest_analysis": analysis_dict(latest) if latest else None,
                    "pending_review_count": pending.get(latest.id, 0) if latest else 0,
                    "analysis_reviewed": bool(latest and latest.id in done)})
    return out


@router.get("/recordings/{record_id}")
def recording(record_id: str, db: Session = Depends(get_db)):
    return record_dict(_record(db, record_id))


@router.get("/recordings/{record_id}/signal")
def recording_signal(record_id: str, start_sample: int = Query(0, ge=0), count: int = Query(3600, ge=1, le=10000),
                     filtered: bool = False, db: Session = Depends(get_db)):
    record = _record(db, record_id)
    signal = _filtered(record)[1] if filtered else load_signal(record)
    start = min(start_sample, len(signal))
    part = signal[start:start + count]
    return {"record_id": record_id, "lead": record.lead, "units": record_dict(record)["units"],
            "sampling_rate": record.sampling_rate, "filtered": filtered, "start_sample": start,
            "time_seconds": (np.arange(start, start + len(part)) / record.sampling_rate).round(5).tolist(),
            "amplitude": part.astype(float).round(5).tolist()}


@router.get("/recordings/{record_id}/envelope")
def envelope(record_id: str, bins: int = Query(600, ge=50, le=2000), db: Session = Depends(get_db)):
    """Min/max of the whole raw record per bin, for the navigator."""
    signal = load_signal(_record(db, record_id))
    parts = np.array_split(signal, min(bins, len(signal)))
    return {"min": [float(p.min()) for p in parts], "max": [float(p.max()) for p in parts]}


@router.get("/recordings/{record_id}/window")
def window(record_id: str, sample: int = Query(..., ge=0), db: Session = Depends(get_db)):
    """The three stages of one model window: raw, bandpass filtered, standardized model input."""
    record = _record(db, record_id)
    raw, filtered = _filtered(record)
    batch = extract_windows(filtered, np.asarray([sample], dtype=np.int64))
    if not len(batch.windows):
        raise HTTPException(422, "beat window crosses a recording edge")
    lo, hi = sample - LEFT_SAMPLES, sample - LEFT_SAMPLES + WINDOW_SAMPLES
    stats = lambda x: {"mean": float(np.mean(x)), "sd": float(np.std(x))}
    z = batch.windows[0, :, 0]
    return {"sample": sample, "start_sample": lo, "sampling_rate": record.sampling_rate, "units": record_dict(record)["units"],
            "raw": raw[lo:hi].astype(float).tolist(), "filtered": filtered[lo:hi].astype(float).tolist(),
            "model_input": z.astype(float).tolist(),
            "stats": {"raw": stats(raw[lo:hi]), "filtered": stats(filtered[lo:hi]), "model_input": stats(z)}}


@router.get("/recordings/{record_id}/quality")
def quality(record_id: str, analysis_id: str | None = None, db: Session = Depends(get_db)):
    record = _record(db, record_id)
    analysis = _latest(db, record_id, analysis_id)
    raw, filtered = _filtered(record)
    positions = db.scalars(select(Beat.sample_index).where(Beat.analysis_id == analysis.id)).all()
    # edge beats were not stored as windows; they count as invalid beat positions
    edges = [0] * analysis.skipped_edge_count
    return {"analysis_id": analysis.id, "lead": record.lead,
            **assess(raw, filtered, record.sampling_rate, np.asarray([*positions, *edges], dtype=np.int64))}


@router.post("/analyze", status_code=202)
def analyze(record_id: str, request: Request, db: Session = Depends(get_db)):
    """Start an analysis in the background. Poll GET /analyses/{id} for status and summary.stage."""
    record = _record(db, record_id)
    classifier = request.app.state.classifier
    if classifier is None:
        raise HTTPException(503, "the model is not loaded")
    analysis = create_analysis(db, record, classifier)
    analysis_id, factory = analysis.id, request.app.state.SessionLocal

    def work():
        with ANALYSIS_LOCK, factory() as session:
            row = session.get(Analysis, analysis_id)
            if row is None:  # deleted while queued
                return
            try:
                execute_analysis(session, row, session.get(ECGRecord, row.record_id), classifier)
            except ServiceError:
                pass  # stored on the analysis row as status=failed

    response = analysis_dict(analysis)
    threading.Thread(target=work, daemon=True).start()
    return response


def _latest(db: Session, record_id: str, analysis_id: str | None) -> Analysis:
    _record(db, record_id)
    stmt = select(Analysis).where(Analysis.record_id == record_id, Analysis.status == "completed")
    if analysis_id:
        stmt = stmt.where(Analysis.id == analysis_id)
    row = db.scalar(stmt.order_by(Analysis.started_at.desc()).limit(1))
    if row is None:
        raise HTTPException(404, "no completed analysis for this recording")
    return row


@router.get("/recordings/{record_id}/predictions")
def predictions(record_id: str, analysis_id: str | None = None, db: Session = Depends(get_db)):
    analysis = _latest(db, record_id, analysis_id)
    rows = db.scalars(select(Beat).where(Beat.analysis_id == analysis.id).order_by(Beat.beat_index)
                      .options(selectinload(Beat.prediction), selectinload(Beat.reviews))).all()
    return {"analysis_id": analysis.id, "model_version": analysis.model_version, "completed_at": analysis.completed_at,
            "predictions": [beat_dict(b) for b in rows]}


@router.get("/recordings/{record_id}/explanations")
def stored_explanations(record_id: str, analysis_id: str | None = None, db: Session = Depends(get_db)):
    analysis = _latest(db, record_id, analysis_id)
    rows = db.execute(select(Explanation, Beat.beat_index).join(Beat).where(Beat.analysis_id == analysis.id)
                      .order_by(Explanation.created_at)).all()
    return {"analysis_id": analysis.id, "model_version": analysis.model_version,
            "explanations": [{"id": e.id, "beat_id": e.beat_id, "beat_index": i, "method": e.method,
                              "class": e.target_class, "created_at": e.created_at} for e, i in rows]}


@router.get("/history")
def history(db: Session = Depends(get_db), limit: int = Query(500, ge=1, le=2000)):
    pending, done = pending_counts(db), signed_off(db)
    rows = db.execute(select(Analysis, ECGRecord.filename).join(ECGRecord).order_by(Analysis.started_at.desc()).limit(limit))
    return [{**analysis_dict(a), "filename": name, "pending_review_count": pending.get(a.id, 0),
             "analysis_reviewed": a.id in done,
             "needs_review": a.status == "completed" and bool(pending.get(a.id)) and a.id not in done} for a, name in rows]


def _sign_off(row: AnalysisReview, model_version: str) -> dict:
    return {"id": row.id, "analysis_id": row.analysis_id, "status": row.status, "note": row.note, "reviewer": row.reviewer,
            "created_at": row.created_at, "model_version": model_version}


@router.get("/analyses/{analysis_id}/sign-offs")
def sign_offs(analysis_id: str, db: Session = Depends(get_db)):
    analysis = db.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    return [_sign_off(r, analysis.model_version) for r in sorted(analysis.sign_offs, key=lambda r: r.created_at, reverse=True)]


@router.post("/analyses/{analysis_id}/sign-offs")
def create_sign_off(analysis_id: str, payload: SignOffIn, db: Session = Depends(get_db)):
    analysis = db.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    if analysis.status != "completed":
        raise HTTPException(409, "only completed analyses can be marked reviewed")
    row = AnalysisReview(analysis_id=analysis_id, **payload.model_dump())
    db.add(row)
    db.commit()
    return _sign_off(row, analysis.model_version)
