from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from app.services import dataset_service as ds
from app.services.study_service import read_csv, read_json

router = APIRouter(prefix="/dataset")


def _mitdb(request: Request):
    path = request.app.state.results_dir / "mitdb"
    if not path.is_dir() or not any(path.glob("*.hea")):
        raise HTTPException(503, "the MIT-BIH records are not available on this machine")
    return path


def _check(request: Request, record: str):
    mitdb = _mitdb(request)
    if not record.isdigit() or not (mitdb / f"{record}.hea").exists():
        raise HTTPException(404, "record not found")
    return mitdb


@router.get("/summary")
def summary(request: Request):
    state = request.app.state
    metrics, problem = read_json(state.artifacts_dir / "metrics.json")
    study, study_problem = read_json(state.artifacts_dir / "multidb_data_report.json")
    return {"app_model": metrics and {"dataset": metrics["dataset"], "preprocessing": metrics["preprocessing"]},
            "app_model_problem": None if metrics else problem or "Model evaluation has not been generated.",
            "study": study, "study_problem": None if study else study_problem or "Study data report not generated yet.",
            "minority_concentration": read_csv(state.results_dir / "minority_concentration_by_split.csv")}


@router.get("/records")
def records(request: Request):
    return ds.list_records(_mitdb(request))


@router.get("/representatives")
def representatives(request: Request):
    return {"rule": "For each class: the training window closest to the mean of that class’s standardized training windows. "
                    "A stated selection rule, not a claim that the beat is typical.",
            "examples": ds.representatives(_mitdb(request))}


@router.get("/records/{record}/segment")
def segment(record: str, request: Request, start: float = Query(0, ge=0), seconds: float = Query(10, ge=2, le=30)):
    return ds.segment(_check(request, record), record, start, seconds)


@router.get("/records/{record}/window")
def window(record: str, request: Request, label: str = Query(..., alias="class", pattern="^[NSVF]$"), index: int = Query(0, ge=0)):
    r = ds.load_record(_check(request, record), record)
    positions = ds.class_positions(r, label)
    if not positions:
        raise HTTPException(404, f"record {record} has no complete {label} window")
    index = min(index, len(positions) - 1)
    return {"record": record, "split": ds.record_split(record), "label": label, "index": index, "count": len(positions),
            **ds.window_payload(r, positions[index])}
