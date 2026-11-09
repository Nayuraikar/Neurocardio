from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models import Analysis, Beat, Report


def _analysis_with_beats(session: Session, analysis_id: str) -> Analysis | None:
    return session.scalar(select(Analysis).where(Analysis.id == analysis_id).options(
        selectinload(Analysis.record), selectinload(Analysis.beats).selectinload(Beat.prediction),
        selectinload(Analysis.beats).selectinload(Beat.reviews)))


def generate_report(session: Session, analysis_id: str, fmt: str, report_dir: Path) -> tuple[Path, Analysis]:
    analysis = _analysis_with_beats(session, analysis_id)
    if analysis is None:
        raise LookupError("analysis not found")
    if analysis.status != "completed":
        raise ValueError("reports are available only for completed analyses")
    if fmt not in ("pdf", "csv"):
        raise ValueError("format must be pdf or csv")
    report_dir.mkdir(parents=True, exist_ok=True)
    target = report_dir / f"{analysis.id}.{fmt}"
    beats = sorted(analysis.beats, key=lambda b: b.beat_index)
    if fmt == "csv":
        with target.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["beat_index", "sample_index", "timestamp_seconds", "predicted_class", "prob_N", "prob_S", "prob_V", "prob_F",
                        "confidence", "margin", "review_flag", "reviewed_label", "review_note"])
            for beat in beats:
                p = beat.prediction
                latest = max(beat.reviews, key=lambda r: r.created_at) if beat.reviews else None
                w.writerow([beat.beat_index, beat.sample_index, beat.timestamp_seconds, p.predicted_class,
                            *p.probabilities, p.confidence, p.margin, p.review_flag,
                            latest.reviewed_label if latest else "", latest.note if latest else ""])
    else:
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(str(target), pagesize=landscape(letter), title="NeuroCardio Research Analysis")
        story = [Paragraph("NeuroCardio ECG research analysis", styles["Title"]), Spacer(1, 10)]
        story.append(Paragraph("Research prototype • retrospective beat classification • not a diagnostic device", styles["BodyText"]))
        story.append(Spacer(1, 10))
        meta = [["Recording", analysis.record.filename], ["Lead / rate", f"{analysis.record.lead} / {analysis.record.sampling_rate:g} Hz"],
                ["Duration / samples", f"{analysis.record.duration_seconds:.3f} s / {analysis.record.sample_count}"],
                ["Analysis", analysis.id], ["Model", f"{analysis.model_version} ({analysis.model_sha256[:12]}…)"],
                ["Config SHA-256", analysis.config_sha256], ["Preprocessing", analysis.preprocessing_version],
                ["Peak source", analysis.peak_source], ["Valid / edge-skipped beats", f"{analysis.beat_count} / {analysis.skipped_edge_count}"]]
        story.append(Table(meta, colWidths=[130, 560], style=TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#DCE5D8")),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#B9C4B6")), ("VALIGN", (0, 0), (-1, -1), "TOP")])) )
        story.append(Spacer(1, 12))
        story.append(Paragraph("Prediction summary (beat classes only; no recording-level rhythm diagnosis)", styles["Heading2"]))
        summary = analysis.summary_json or {}
        story.append(Paragraph(json.dumps({"class_counts": summary.get("class_counts", {}),
                                          "abnormal_beat_count": summary.get("abnormal_beat_count"),
                                          "flagged_for_review_count": summary.get("flagged_for_review_count"),
                                          "detection": summary.get("detection")}, sort_keys=True), styles["BodyText"]))
        story.append(Spacer(1, 10))
        rows = [["Beat", "Time (s)", "Prediction", "N", "S", "V", "F", "Conf.", "Margin", "Review"]]
        for beat in beats:
            p = beat.prediction
            rows.append([beat.beat_index, f"{beat.timestamp_seconds:.3f}", p.predicted_class,
                         *[f"{q:.4f}" for q in p.probabilities], f"{p.confidence:.4f}", f"{p.margin:.4f}",
                         "Flagged" if p.review_flag else "—"])
        table = Table(rows, repeatRows=1, colWidths=[35, 55, 55, 45, 45, 45, 45, 50, 50, 60])
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263B32")),
                                   ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                   ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD1C8")),
                                   ("FONTSIZE", (0, 0), (-1, -1), 7), ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                                    [colors.white, colors.HexColor("#F7F4EC")])]))
        story.extend([table, Spacer(1, 10), Paragraph("Limitations", styles["Heading2"]),
                      Paragraph("The model was evaluated retrospectively on MIT-BIH. Supraventricular (S) and fusion (F) recall are weak. "
                                "Attribution and selective-review outputs are exploratory and not clinically calibrated. "
                                "This report is for research use and is not a diagnosis.", styles["BodyText"])])
        doc.build(story)
    row = session.scalar(select(Report).where(Report.analysis_id == analysis_id, Report.format == fmt))
    if row is None:
        row = Report(analysis_id=analysis_id, format=fmt, path=str(target))
        session.add(row)
    else:
        row.path = str(target)
    session.commit()
    return target, analysis
