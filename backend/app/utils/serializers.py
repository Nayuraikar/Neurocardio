import json
from functools import lru_cache
from pathlib import Path

from app.core.config import CONFIG_PATH
from app.models import Analysis, Beat, ECGRecord, Prediction, Review


@lru_cache(maxsize=1)
def _model_splits() -> dict[str, str]:
    split = json.loads(CONFIG_PATH.read_text())["split"]
    return {name: part for part in ("train", "validation", "test") for name in split[f"{part}_records"]}


def known_split(filename: str) -> str | None:
    """Split of the loaded model's MIT-BIH record with this file name. A name match only: contents are not compared."""
    return _model_splits().get(Path(filename).stem)


def record_dict(r: ECGRecord) -> dict:
    meta = r.metadata_json or {}
    units, signals = meta.get("units"), meta.get("signals")
    unit = units[signals.index(r.lead)] if units and signals and r.lead in signals else None
    return {"id": r.id, "filename": r.filename, "format": r.format, "sampling_rate": r.sampling_rate, "lead": r.lead,
            "duration_seconds": r.duration_seconds, "sample_count": r.sample_count, "uploaded_at": r.uploaded_at,
            "units": unit, "signals": signals or meta.get("columns"), "has_annotation": bool(r.annotation_path),
            "input_warnings": meta.get("input_warnings", []),
            "model_split_by_name": known_split(r.filename) if r.format == "wfdb" else None}


def analysis_dict(a: Analysis) -> dict:
    return {"id": a.id, "record_id": a.record_id, "status": a.status, "progress": a.progress,
            "model_version": a.model_version, "model_sha256": a.model_sha256, "config_sha256": a.config_sha256,
            "preprocessing_version": a.preprocessing_version, "peak_source": a.peak_source,
            "beat_count": a.beat_count, "skipped_edge_count": a.skipped_edge_count,
            "summary": a.summary_json or {}, "error_message": a.error_message,
            "started_at": a.started_at, "completed_at": a.completed_at}


def prediction_dict(p: Prediction) -> dict:
    return {"predicted_class": p.predicted_class, "probabilities": p.probabilities,
            "confidence": p.confidence, "margin": p.margin, "review_flag": p.review_flag,
            "review_reason": p.review_reason}


def review_dict(r: Review) -> dict:
    return {"id": r.id, "beat_id": r.beat_id, "original_prediction": r.original_prediction,
            "reviewed_label": r.reviewed_label, "note": r.note, "reviewer": r.reviewer,
            "status": r.status, "created_at": r.created_at}


def beat_dict(b: Beat) -> dict:
    reviews = sorted(b.reviews, key=lambda x: x.created_at, reverse=True)
    return {"id": b.id, "beat_index": b.beat_index, "sample_index": b.sample_index,
            "timestamp_seconds": b.timestamp_seconds,
            "prediction": prediction_dict(b.prediction) if b.prediction else None,
            "review": review_dict(reviews[0]) if reviews else None}
