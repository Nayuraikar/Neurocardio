"""PDF and CSV reports. The reader chooses the sections; every report carries the model version and a timestamp."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import ARTIFACTS_DIR, CLASS_COLORS_PATH, CONFIG_PATH, RESULTS_DIR
from app.ml.preprocessing import extract_windows, filter_continuous
from app.models import Analysis, Beat, Report
from app.services.model_service import NORMALIZATION_LABELS
from app.services.study_service import read_csv, read_json

SECTIONS = {
    "recording": "Recording information", "preprocessing": "Processing and model", "waveform": "Waveform excerpts",
    "distribution": "Class distribution", "predictions": "Window predictions", "xai": "Stored explanations",
    "notes": "Reviewer assessments and notes", "performance": "Model performance on held-out data", "limitations": "Limitations",
}
CSV_SECTIONS = ("recording", "preprocessing", "distribution", "predictions", "notes")
WIDTH = 690


def parse_sections(value: str | None) -> list[str]:
    if value is None:
        return list(SECTIONS)
    chosen = [s for s in value.split(",") if s]
    unknown = sorted(set(chosen) - set(SECTIONS))
    if unknown or not chosen:
        raise KeyError(f"unknown or empty report sections: {', '.join(unknown) or '(none)'}")
    return [s for s in SECTIONS if s in chosen]


def class_colors() -> dict[str, str]:
    return {k: v["fill"] for k, v in json.loads(CLASS_COLORS_PATH.read_text())["classes"].items()}


def _analysis(session: Session, analysis_id: str) -> Analysis | None:
    return session.scalar(select(Analysis).where(Analysis.id == analysis_id).options(
        selectinload(Analysis.record), selectinload(Analysis.sign_offs),
        selectinload(Analysis.beats).selectinload(Beat.prediction), selectinload(Analysis.beats).selectinload(Beat.reviews),
        selectinload(Analysis.beats).selectinload(Beat.explanations)))


def _latest_review(beat: Beat):
    return max(beat.reviews, key=lambda r: r.created_at) if beat.reviews else None


def _metadata(a: Analysis, generated: str) -> list[tuple[str, str]]:
    return [("generated_at_utc", generated), ("analysis_id", a.id), ("model_version", a.model_version), ("model_sha256", a.model_sha256)]


def _write_csv(path: Path, a: Analysis, beats: list[Beat], sections: list[str], generated: str) -> None:
    meta = _metadata(a, generated)
    if "recording" in sections:
        r = a.record
        meta += [("recording", r.filename), ("lead", r.lead), ("sampling_rate_hz", f"{r.sampling_rate:g}"),
                 ("duration_seconds", f"{r.duration_seconds:.3f}"), ("uploaded_at_utc", r.uploaded_at.isoformat())]
    if "preprocessing" in sections:
        meta += [("preprocessing_version", a.preprocessing_version), ("config_sha256", a.config_sha256), ("peak_source", a.peak_source)]
    if "distribution" in sections:
        meta += [(f"windows_{k}", str(v)) for k, v in (a.summary_json or {}).get("class_counts", {}).items()]
    meta.append(("notice", "window-level model output for research use; not a diagnosis"))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for key, value in meta:
            handle.write(f"# {key}: {' '.join(str(value).split())}\n")
        if "predictions" in sections or "notes" in sections:
            header = ["beat_index", "sample_index", "timestamp_seconds"]
            if "predictions" in sections:
                header += ["predicted_class", "prob_N", "prob_S", "prob_V", "prob_F", "confidence", "margin", "review_flag"]
            if "notes" in sections:
                header += ["reviewed_label", "review_status", "review_note", "reviewer"]
            writer.writerow(header)
            for beat in beats:
                review = _latest_review(beat)
                if "predictions" not in sections and review is None:
                    continue
                row = [beat.beat_index, beat.sample_index, beat.timestamp_seconds]
                if "predictions" in sections:
                    p = beat.prediction
                    row += [p.predicted_class, *p.probabilities, p.confidence, p.margin, p.review_flag]
                if "notes" in sections:
                    row += [_safe(v) for v in ((review.reviewed_label, review.status, review.note, review.reviewer) if review else ("", "", "", ""))]
                writer.writerow(row)


def _safe(value) -> str:
    """Stops spreadsheet programs from running user text as a formula."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def _trace(values: np.ndarray, label: str, height: int = 90, band: tuple[int, int] | None = None, fill: str = "#777777") -> Drawing:
    d = Drawing(WIDTH, height)
    n = len(values)
    if n < 2:
        return d
    idx = np.unique(np.linspace(0, n - 1, min(n, 1400)).astype(int))
    lo, hi = float(values.min()), float(values.max())
    span = (hi - lo) or 1.0
    if band:
        d.add(Rect(band[0] / (n - 1) * WIDTH, 0, (band[1] - band[0]) / (n - 1) * WIDTH, height - 12,
                   fillColor=colors.HexColor(fill), fillOpacity=0.18, strokeColor=None))
    points = []
    for i in idx:
        points += [float(i / (n - 1) * WIDTH), float(4 + (values[i] - lo) / span * (height - 20))]
    d.add(PolyLine(points, strokeColor=colors.black, strokeWidth=0.6))
    d.add(String(0, height - 9, f"{label}  (min {lo:.2f}, max {hi:.2f})", fontSize=7))
    return d


def _attribution(values: np.ndarray, height: int = 50) -> Drawing:
    d = Drawing(WIDTH, height)
    scale = float(np.abs(values).max()) or 1.0
    zero = height / 2
    d.add(Line(0, zero, WIDTH, zero, strokeColor=colors.grey, strokeWidth=0.3))
    for i, v in enumerate(values):
        if v:
            x = i / (len(values) - 1) * WIDTH
            d.add(Line(x, zero, x, zero + float(v) / scale * (height / 2 - 2),
                       strokeColor=colors.HexColor("#1f5f8b" if v > 0 else "#8a4a12"), strokeWidth=0.8))
    return d


def _distribution(counts: dict) -> Drawing:
    palette = class_colors()
    d = Drawing(WIDTH, 72)
    top = max([*counts.values(), 1])
    for row, label in enumerate("NSVF"):
        y = 56 - row * 17
        d.add(String(0, y + 2, label, fontSize=9))
        d.add(Rect(18, y, counts.get(label, 0) / top * (WIDTH - 90), 11, fillColor=colors.HexColor(palette[label]), strokeColor=None))
        d.add(String(WIDTH - 60, y + 2, f"{counts.get(label, 0):,}", fontSize=9))
    return d


def _table(rows, widths=None, header=True, size=8) -> Table:
    style = [("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9c7b8")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("FONTSIZE", (0, 0), (-1, -1), size)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b3a1e")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    return Table(rows, colWidths=widths, repeatRows=1 if header else 0, style=TableStyle(style))


def _performance(styles) -> list:
    metrics, _ = read_json(ARTIFACTS_DIR / "metrics.json")
    if metrics is None:
        return [Paragraph("Model evaluation figures are not available.", styles["BodyText"])]
    t = metrics["test"]
    story = [Paragraph("Held-out MIT-BIH test windows. These describe the model in general, not this recording.", styles["BodyText"]),
             _table([["Macro-F1", "S recall", "F recall", "Accuracy", "Majority-class accuracy (always N)"],
                     [f"{t['macro']['f1']:.3f}", f"{t['per_class']['S']['recall']:.3f}", f"{t['per_class']['F']['recall']:.3f}",
                      f"{t['accuracy']:.3f}", f"{t['majority_class_baseline_accuracy']:.3f}"]], size=9),
             _table([["Class", "Precision", "Recall", "F1", "Support"]] + [
                 [c, *[f"{t['per_class'][c][k]:.3f}" for k in ("precision", "recall", "f1")], f"{t['per_class'][c]['support']:,}"] for c in "NSVF"], size=8),
             Paragraph(f"S and F are weak: the model misses most of them. Evaluated {escape(metrics['evaluated_at_utc'])}.", styles["BodyText"])]
    story.append(Paragraph(f"Evaluated model version: {escape(metrics['model']['version'])}.", styles["BodyText"]))
    return story


def _limitations(styles) -> list:
    items = ["Window-level model output on retrospective, single-lead data. It is not a diagnosis and says nothing about a patient.",
             "S (supraventricular) and F (fusion) recall are weak. Absence of these classes in the output does not show absence in the signal."]
    for row in read_csv(RESULTS_DIR / "minority_concentration_by_split.csv"):
        if row["split"] == "train" and row["class"] == "F":
            items.append(f"{float(row['top_record_share']):.1%} of training F beats come from one record ({row['top_record']}).")
    items += ["Softmax scores are model scores, not clinical certainty. The review flag does not catch every error.",
              "Attribution maps describe model behaviour, not clinical cause.",
              "Reviewer labels are stored separately and never change the model's predictions."]
    return [Paragraph(f"• {escape(item)}", styles["BodyText"]) for item in items]


def _write_pdf(path: Path, a: Analysis, beats: list[Beat], sections: list[str], generated: str) -> None:
    styles = getSampleStyleSheet()
    styles["Heading2"].keepWithNext = 1
    doc = SimpleDocTemplate(str(path), pagesize=landscape(letter), title="NeuroCardio research report")
    story = [Paragraph("NeuroCardio ECG research report", styles["Title"]),
             Paragraph("Research prototype • window-level model classification • not a diagnosis", styles["BodyText"]),
             Paragraph(f"Generated {generated} • model {escape(a.model_version)} ({a.model_sha256[:12]}…) • analysis {a.id}", styles["BodyText"]),
             Spacer(1, 8)]
    record, summary, config = a.record, a.summary_json or {}, json.loads(CONFIG_PATH.read_text())
    signal = filtered = None
    if {"waveform", "xai"} & set(sections):
        signal = np.asarray(np.load(record.signal_path), dtype=np.float32)
        filtered = filter_continuous(signal, record.sampling_rate)
    for key in sections:
        story += [Spacer(1, 8), Paragraph(SECTIONS[key], styles["Heading2"])]
        if key == "recording":
            story.append(_table([["File", record.filename], ["Lead / rate", f"{record.lead} / {record.sampling_rate:g} Hz"],
                                 ["Duration / samples", f"{record.duration_seconds:.3f} s / {record.sample_count:,}"],
                                 ["Uploaded (UTC)", record.uploaded_at.isoformat(timespec="seconds")],
                                 ["Beat positions from", a.peak_source or "unknown"],
                                 ["Windows analysed / edge beats skipped", f"{a.beat_count:,} / {a.skipped_edge_count}"]], [190, 500], header=False))
            story += [Paragraph(escape(w), styles["BodyText"]) for w in (record.metadata_json or {}).get("input_warnings", [])]
        elif key == "preprocessing":
            band = config["bandpass_filter"]
            story.append(_table([["Bandpass filter", f"{band['type']} order {band['order']}, {band['low_hz']}–{band['high_hz']} Hz, zero-phase"],
                                 ["Window", f"{config['window_size']} samples, {config['window_alignment']}"],
                                 ["Standardization", NORMALIZATION_LABELS.get(config["normalization"], config["normalization"])], ["Preprocessing version", a.preprocessing_version],
                                 ["Config SHA-256", a.config_sha256]], [190, 500], header=False))
        elif key == "waveform":
            n = int(10 * record.sampling_rate)
            story += [_trace(signal[:n], "Raw, first 10 s"), _trace(filtered[:n], "Bandpass filtered, first 10 s")]
        elif key == "distribution":
            story += [_distribution(summary.get("class_counts", {})),
                      Paragraph(f"Flagged for review: {summary.get('flagged_for_review_count', 0)} windows.", styles["BodyText"])]
        elif key == "predictions":
            rows = [["Window", "Time (s)", "Class", "N", "S", "V", "F", "Margin", "Flag"]]
            for beat in beats:
                p = beat.prediction
                rows.append([beat.beat_index + 1, f"{beat.timestamp_seconds:.3f}", p.predicted_class,
                             *[f"{q:.4f}" for q in p.probabilities], f"{p.margin:.4f}", "Review" if p.review_flag else ""])
            story.append(_table(rows, [50, 60, 40, 55, 55, 55, 55, 55, 45], size=7))
        elif key == "xai":
            stored = sorted(((e, b) for b in beats for e in b.explanations), key=lambda x: x[0].created_at, reverse=True)
            if not stored:
                story.append(Paragraph("No explanation was generated for this analysis.", styles["BodyText"]))
            else:
                story.append(Paragraph(f"The {min(6, len(stored))} most recent of {len(stored)} stored maps. They explain model behaviour, "
                                       "not medical causation.", styles["BodyText"]))
            for e, beat in stored[:6]:
                batch = extract_windows(filtered, np.asarray([beat.sample_index], dtype=np.int64))
                meta = e.metadata_json or {}
                error, change = meta.get("completeness_error"), meta.get("score_difference")
                relative = f" ({abs(error / change):.1%} of the score change)" if error is not None and change else ""
                story += [Paragraph(f"Window {beat.beat_index + 1}, {'Integrated Gradients' if e.method == 'ig' else 'Grad-CAM'} for class "
                                    f"{e.target_class} (model class {beat.prediction.predicted_class})"
                                    + (f"; completeness error {error:.4f}{relative}" if error is not None else ""), styles["BodyText"]),
                          _trace(batch.windows[0, :, 0], "Model input (z-score)", 70), _attribution(np.asarray(e.attributions))]
        elif key == "notes":
            for sign in sorted(a.sign_offs, key=lambda s: s.created_at):
                story.append(Paragraph(f"{sign.created_at.isoformat(timespec='seconds')} • {escape(sign.reviewer)} marked the analysis "
                                       f"{sign.status}. {escape(sign.note or '')}", styles["BodyText"]))
            rows = [["Window", "Model", "Reviewer label", "Status", "Reviewer", "Note"]]
            for beat in beats:
                review = _latest_review(beat)
                if review:
                    rows.append([beat.beat_index + 1, beat.prediction.predicted_class, review.reviewed_label or "—", review.status,
                                 review.reviewer, Paragraph(escape(review.note or ""), styles["BodyText"])])
            story.append(_table(rows, [50, 40, 75, 65, 90, 370]) if len(rows) > 1
                         else Paragraph("No window has been assessed by a reviewer.", styles["BodyText"]))
        elif key == "performance":
            story += _performance(styles)
        elif key == "limitations":
            story += _limitations(styles)
    doc.build(story)


def generate_report(session: Session, analysis_id: str, fmt: str, report_dir: Path,
                    sections: list[str] | None = None) -> tuple[Path, Analysis]:
    analysis = _analysis(session, analysis_id)
    if analysis is None:
        raise LookupError("analysis not found")
    if analysis.status != "completed":
        raise ValueError("reports are available only for completed analyses")
    if fmt not in ("pdf", "csv"):
        raise ValueError("format must be pdf or csv")
    sections = sections or list(SECTIONS)
    report_dir.mkdir(parents=True, exist_ok=True)
    target = report_dir / f"{analysis.id}.{fmt}"
    beats = sorted(analysis.beats, key=lambda b: b.beat_index)
    now = datetime.now(timezone.utc)
    (_write_csv if fmt == "csv" else _write_pdf)(target, analysis, beats, sections, now.isoformat(timespec="seconds"))
    row = session.scalar(select(Report).where(Report.analysis_id == analysis_id, Report.format == fmt))
    if row is None:
        row = Report(analysis_id=analysis_id, format=fmt, path=str(target))
        session.add(row)
    row.path, row.created_at = str(target), now
    session.commit()
    return target, analysis
