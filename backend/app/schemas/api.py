from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class RecordOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    filename: str
    format: str
    sampling_rate: float
    lead: str
    duration_seconds: float
    sample_count: int
    uploaded_at: datetime
    preview: list[dict[str, Any]] = Field(default_factory=list)


class PredictionOut(BaseModel):
    predicted_class: str
    probabilities: list[float]
    confidence: float
    margin: float
    review_flag: bool
    review_reason: str | None


class BeatOut(BaseModel):
    id: str
    beat_index: int
    sample_index: int
    timestamp_seconds: float
    prediction: PredictionOut
    review: dict[str, Any] | None = None


class AnalysisOut(BaseModel):
    id: str
    record_id: str
    status: str
    progress: int
    model_version: str
    model_sha256: str
    config_sha256: str
    preprocessing_version: str
    peak_source: str | None
    beat_count: int
    skipped_edge_count: int
    summary: dict[str, Any]
    error_message: str | None
    started_at: datetime
    completed_at: datetime | None


class ReviewIn(BaseModel):
    beat_id: str
    reviewed_label: str | None = Field(default=None, pattern="^[NSVF]$")
    note: str | None = None
    reviewer: str = "local-reviewer"


class ReviewOut(BaseModel):
    id: str
    beat_id: str
    original_prediction: str
    reviewed_label: str | None
    note: str | None
    reviewer: str
    status: str
    created_at: datetime
