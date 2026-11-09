from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import CONFIG_PATH, PREPROCESSING_VERSION, RESULTS_DIR
from app.core.errors import ServiceError
from app.ml.explainability import ModelExplainer, compute_explanation_cache_key
from app.ml.preprocessing import extract_windows, filter_continuous
from app.models import Analysis, Beat, Explanation, ECGRecord, Review
from app.repositories.queries import get_analysis, get_beat
from app.schemas.api import AnalysisOut, BeatOut, RecordOut, ReviewIn, ReviewOut
from app.services.analysis_service import run_analysis
from app.services.model_info import read_model_info
from app.services.recording_service import upload_recording
from app.services.report_service import generate_report
from app.utils.serializers import analysis_dict, beat_dict, review_dict

router = APIRouter(prefix="/api/v1")


def get_db(request: Request):
    db = request.app.state.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/health")
def health(request: Request):
    classifier = request.app.state.classifier
    return {"status": "ok", "model_loaded": classifier is not None,
            "model_sha256": classifier.sha256 if classifier else None}


@router.get("/model-info")
def model_info(request: Request):
    info = read_model_info(request.app.state.results_dir)
    classifier = request.app.state.classifier
    config = request.app.state.config
    return {**info, "input_shape": [None, 720, 1], "output_shape": [None, 4],
            "classes": config["classes"], "sampling_rate": config["sampling_frequency"],
            "lead": config["lead"], "rr_features": config["rr_features"],
            "model_sha256": classifier.sha256, "config_sha256": request.app.state.config_sha256,
            "preprocessing_version": PREPROCESSING_VERSION}


@router.post("/ecg/upload", response_model=RecordOut)
async def upload_ecg(request: Request, files: list[UploadFile] = File(...), sampling_rate: float | None = Form(None),
                     lead: str | None = Form(None), db: Session = Depends(get_db)):
    try:
        record, preview = await upload_recording(files, sampling_rate, lead, db,
                                                  request.app.state.data_dir, request.app.state.max_upload_bytes)
    except ServiceError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc
    result = RecordOut.model_validate(record).model_dump(mode="json")
    result["preview"] = preview
    return result


@router.get("/ecg/records")
def list_records(db: Session = Depends(get_db), limit: int = Query(100, ge=1, le=500)):
    rows = db.scalars(select(ECGRecord).order_by(ECGRecord.uploaded_at.desc()).limit(limit)).all()
    return [RecordOut.model_validate(row).model_dump(mode="json") for row in rows]


@router.post("/analyses", response_model=AnalysisOut)
def create_analysis(record_id: str, request: Request, db: Session = Depends(get_db)):
    record = db.get(ECGRecord, record_id)
    if record is None:
        raise HTTPException(404, "recording not found")
    try:
        analysis = run_analysis(db, record, request.app.state.classifier)
    except ServiceError as exc:
        failed = db.scalar(select(Analysis).where(Analysis.record_id == record_id).order_by(Analysis.started_at.desc()))
        detail = {"message": exc.detail, "analysis_id": failed.id if failed else None, "status": "failed"}
        raise HTTPException(exc.status_code, detail) from exc
    return analysis_dict(analysis)


@router.get("/analyses", response_model=list[AnalysisOut])
def list_analyses(db: Session = Depends(get_db), limit: int = Query(100, ge=1, le=500)):
    rows = db.scalars(select(Analysis).order_by(Analysis.started_at.desc()).limit(limit)).all()
    return [analysis_dict(x) for x in rows]


@router.get("/analyses/{analysis_id}", response_model=AnalysisOut)
def analysis_detail(analysis_id: str, db: Session = Depends(get_db)):
    analysis = db.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    return analysis_dict(analysis)


@router.delete("/analyses/{analysis_id}")
def delete_analysis(analysis_id: str, request: Request, db: Session = Depends(get_db)):
    analysis = db.scalar(select(Analysis).where(Analysis.id == analysis_id).options(selectinload(Analysis.reports)))
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    report_paths = [Path(row.path) for row in analysis.reports]
    db.delete(analysis)
    db.commit()
    reports_root = (request.app.state.data_dir / "reports").resolve()
    removed_files = 0
    for path in report_paths:
        resolved = path.resolve()
        if resolved.is_relative_to(reports_root) and resolved.is_file():
            resolved.unlink()
            removed_files += 1
    return {"deleted": True, "analysis_id": analysis_id, "report_files_removed": removed_files}


@router.get("/analyses/{analysis_id}/beats", response_model=list[BeatOut])
def analysis_beats(analysis_id: str, db: Session = Depends(get_db), offset: int = Query(0, ge=0),
                   limit: int = Query(1000, ge=1, le=10000)):
    analysis = db.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    rows = db.scalars(select(Beat).where(Beat.analysis_id == analysis_id).order_by(Beat.beat_index).offset(offset).limit(limit)
                      .options(selectinload(Beat.prediction), selectinload(Beat.reviews))).all()
    return [beat_dict(b) for b in rows]


@router.get("/analyses/{analysis_id}/signal")
def analysis_signal(analysis_id: str, start_sample: int = Query(0, ge=0), count: int = Query(1200, ge=1, le=10000),
                    filtered: bool = False, db: Session = Depends(get_db)):
    analysis = get_analysis(db, analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    signal = np.asarray(np.load(analysis.record.signal_path), dtype=np.float32)
    if filtered:
        signal = filter_continuous(signal, analysis.record.sampling_rate)
    stop = min(len(signal), start_sample + count)
    idx = np.arange(start_sample, stop, dtype=np.int64)
    return {"analysis_id": analysis_id, "lead": analysis.record.lead, "sampling_rate": analysis.record.sampling_rate,
            "filtered": filtered, "start_sample": start_sample,
            "sample_indices": idx.tolist(), "time_seconds": (idx / analysis.record.sampling_rate).tolist(),
            "amplitude": signal[start_sample:stop].astype(float).tolist()}


@router.get("/analyses/{analysis_id}/explanations")
def explanation(analysis_id: str, request: Request, beat: str, target_class: str = Query(..., alias="class", pattern="^[NSVF]$"),
                method: str = Query("gradcam", pattern="^(gradcam|ig)$"), db: Session = Depends(get_db)):
    analysis = get_analysis(db, analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    if analysis.status != "completed":
        raise HTTPException(409, "explanations require a completed analysis")
    b = get_beat(db, beat)
    if b is None or b.analysis_id != analysis_id:
        raise HTTPException(404, "beat not found in this analysis")

    ig_steps = 512 if method == "ig" else None
    baseline_name = "all-zero normalized window" if method == "ig" else None
    cache_key = compute_explanation_cache_key(
        analysis_id=analysis.id,
        beat_id=b.id,
        target_class=target_class,
        method=method,
        ig_steps=ig_steps,
        baseline_name=baseline_name,
        model_sha256=analysis.model_sha256,
        preprocessing_version=analysis.preprocessing_version,
    )
    cached = db.scalar(select(Explanation).where(Explanation.cache_key == cache_key,
                                                Explanation.method == method,
                                                Explanation.target_class == target_class))
    if cached:
        return {"id": cached.id, "beat_id": cached.beat_id, "method": cached.method, "class": cached.target_class,
                "attributions": cached.attributions, "metadata": cached.metadata_json, "cached": True}
    signal = np.asarray(np.load(analysis.record.signal_path), dtype=np.float32)
    filtered = filter_continuous(signal, analysis.record.sampling_rate)
    batch = extract_windows(filtered, np.asarray([b.sample_index], dtype=np.int64))
    if not len(batch.windows):
        raise HTTPException(422, "beat window crosses a recording edge")
    explainer = ModelExplainer(request.app.state.classifier.model, ig_steps=512, ig_batch_size=32)
    class_index = ["N", "S", "V", "F"].index(target_class)
    if method == "gradcam":
        attrs, grad_norm = explainer.gradcam_with_diagnostics(batch.windows, class_index)
        values = attrs[0]
        gradient_l2_norm = float(grad_norm[0])
        if not np.isfinite(gradient_l2_norm):
            raise HTTPException(500, "Grad-CAM returned a non-finite conv3 gradient norm")
        metadata = {"target_layer": "conv3", "layer": "conv3", "gradient_l2_norm": gradient_l2_norm,
                    "normalization": "ReLU, per-window max after bilinear upsampling to 720 samples",
                    "normalisation": "ReLU, per-window max after bilinear upsampling to 720 samples"}
    else:
        result = explainer.integrated_gradients(batch.windows, class_index)
        values = result["attributions"][0]
        metadata = {"layer": "logits", "steps": explainer.ig_steps, "baseline": "all-zero normalized window",
                    "score_difference": float(result["score_difference"][0]),
                    "attribution_sum": float(result["attribution_sum"][0]),
                    "completeness_error": float(result["completeness_error"][0])}
    metadata["cache_key"] = cache_key
    row = Explanation(cache_key=cache_key, beat_id=beat, method=method, target_class=target_class,
                      attributions=np.asarray(values, dtype=np.float32).tolist(), metadata_json=metadata)
    db.add(row); db.commit(); db.refresh(row)
    return {"id": row.id, "beat_id": beat, "method": method, "class": target_class,
            "attributions": row.attributions, "metadata": row.metadata_json, "cached": False}


@router.get("/reviews", response_model=list[ReviewOut])
def list_reviews(db: Session = Depends(get_db), status: str | None = None,
                 analysis_id: str | None = None, limit: int = Query(500, ge=1, le=5000)):
    stmt = select(Review).join(Beat).order_by(Review.created_at.desc()).limit(limit)
    if status:
        stmt = stmt.where(Review.status == status)
    if analysis_id:
        stmt = stmt.where(Beat.analysis_id == analysis_id)
    rows = db.scalars(stmt).all()
    return [review_dict(r) for r in rows]


@router.post("/reviews", response_model=ReviewOut)
def create_review(payload: ReviewIn, db: Session = Depends(get_db)):
    beat = get_beat(db, payload.beat_id)
    if beat is None or beat.prediction is None:
        raise HTTPException(404, "predicted beat not found")
    row = Review(beat_id=beat.id, original_prediction=beat.prediction.predicted_class,
                 reviewed_label=payload.reviewed_label, note=payload.note, reviewer=payload.reviewer,
                 status="reviewed")
    db.add(row); db.commit(); db.refresh(row)
    return review_dict(row)


@router.get("/reports/{analysis_id}")
def report(analysis_id: str, request: Request, format: str = Query("pdf", pattern="^(pdf|csv)$"),
           db: Session = Depends(get_db)):
    try:
        path, _ = generate_report(db, analysis_id, format, request.app.state.data_dir / "reports")
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    media = "application/pdf" if format == "pdf" else "text/csv"
    return FileResponse(path, media_type=media, filename=path.name)


@router.get("/reports/{analysis_id}/csv")
def report_csv(analysis_id: str, request: Request, db: Session = Depends(get_db)):
    try:
        path, _ = generate_report(db, analysis_id, "csv", request.app.state.data_dir / "reports")
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return FileResponse(path, media_type="text/csv", filename=path.name)
