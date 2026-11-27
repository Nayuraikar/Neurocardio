from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.services import study_service
from app.services.model_service import model_config

router = APIRouter()


@router.get("/status")
def status(request: Request, db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT count(*) FROM ecg_records")).scalar_one()
        database_ok = True
    except Exception:
        database_ok = False
    classifier = request.app.state.classifier
    return {"api": "ok", "model_loaded": classifier is not None, "database_ok": database_ok,
            "model_sha256": classifier.sha256 if classifier else None,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}


@router.get("/model/metrics")
def metrics(request: Request):
    return study_service.model_metrics(request.app.state)


@router.get("/study")
def study(request: Request):
    return study_service.study(request.app.state.artifacts_dir)


@router.get("/study/operating-points")
def operating_points(request: Request):
    return study_service.operating_points(request.app.state.artifacts_dir)


@router.get("/model/config")
def config(request: Request):
    if request.app.state.classifier is None:
        raise HTTPException(503, "the model is not loaded")
    return model_config(request.app.state.classifier, request.app.state.config, request.app.state.config_sha256)
