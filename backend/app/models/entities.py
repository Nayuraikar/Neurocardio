from datetime import datetime, timezone
import uuid

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ECGRecord(Base):
    __tablename__ = "ecg_records"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    filename: Mapped[str] = mapped_column(String(255))
    format: Mapped[str] = mapped_column(String(16))
    sampling_rate: Mapped[float] = mapped_column(Float)
    lead: Mapped[str] = mapped_column(String(64))
    duration_seconds: Mapped[float] = mapped_column(Float)
    sample_count: Mapped[int] = mapped_column(Integer)
    signal_path: Mapped[str] = mapped_column(Text)
    wfdb_base_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    annotation_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    analyses: Mapped[list["Analysis"]] = relationship(back_populates="record", cascade="all, delete-orphan")


class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    record_id: Mapped[str] = mapped_column(ForeignKey("ecg_records.id", ondelete="CASCADE"), index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    model_sha256: Mapped[str] = mapped_column(String(64))
    config_sha256: Mapped[str] = mapped_column(String(64))
    preprocessing_version: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    peak_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    beat_count: Mapped[int] = mapped_column(Integer, default=0)
    skipped_edge_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    summary_json: Mapped[dict] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    record: Mapped[ECGRecord] = relationship(back_populates="analyses")
    beats: Mapped[list["Beat"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")
    reports: Mapped[list["Report"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")


class Beat(Base):
    __tablename__ = "beats"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"), index=True)
    beat_index: Mapped[int] = mapped_column(Integer)
    sample_index: Mapped[int] = mapped_column(Integer)
    timestamp_seconds: Mapped[float] = mapped_column(Float)
    analysis: Mapped[Analysis] = relationship(back_populates="beats")
    prediction: Mapped["Prediction"] = relationship(back_populates="beat", uselist=False, cascade="all, delete-orphan")
    explanations: Mapped[list["Explanation"]] = relationship(back_populates="beat", cascade="all, delete-orphan")
    reviews: Mapped[list["Review"]] = relationship(back_populates="beat", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    beat_id: Mapped[str] = mapped_column(ForeignKey("beats.id", ondelete="CASCADE"), unique=True, index=True)
    predicted_class: Mapped[str] = mapped_column(String(1))
    probabilities: Mapped[list[float]] = mapped_column(JSON)
    confidence: Mapped[float] = mapped_column(Float)
    margin: Mapped[float] = mapped_column(Float)
    review_flag: Mapped[bool] = mapped_column(default=False)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    beat: Mapped[Beat] = relationship(back_populates="prediction")


class Explanation(Base):
    __tablename__ = "explanations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    cache_key: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    beat_id: Mapped[str] = mapped_column(ForeignKey("beats.id", ondelete="CASCADE"), index=True)
    method: Mapped[str] = mapped_column(String(24))
    target_class: Mapped[str] = mapped_column(String(1))
    attributions: Mapped[list[float]] = mapped_column(JSON)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    beat: Mapped[Beat] = relationship(back_populates="explanations")


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    beat_id: Mapped[str] = mapped_column(ForeignKey("beats.id", ondelete="CASCADE"), index=True)
    original_prediction: Mapped[str] = mapped_column(String(1))
    reviewed_label: Mapped[str | None] = mapped_column(String(1), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer: Mapped[str] = mapped_column(String(128), default="local-reviewer")
    status: Mapped[str] = mapped_column(String(24), default="reviewed")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    beat: Mapped[Beat] = relationship(back_populates="reviews")


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"), index=True)
    format: Mapped[str] = mapped_column(String(8))
    path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    analysis: Mapped[Analysis] = relationship(back_populates="reports")
