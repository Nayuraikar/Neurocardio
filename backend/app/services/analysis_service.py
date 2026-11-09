from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import wfdb
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import (ANNOTATION_ALIGNMENT_MIN, ANNOTATION_ALIGNMENT_TOLERANCE_MS,
                             CONFIG_PATH, PREPROCESSING_VERSION, RESULTS_DIR)
from app.core.errors import ServiceError
from app.ml.inference import sha256_file
from app.ml.preprocessing import filter_continuous, extract_windows
from app.ml.rpeak_detection import annotation_alignment, detect_r_peaks
from app.models import Analysis, Beat, ECGRecord, Prediction


def _softmax(logits: np.ndarray) -> np.ndarray:
    v = logits - np.max(logits)
    exp = np.exp(v)
    return exp / exp.sum()


def run_analysis(session: Session, record: ECGRecord, classifier) -> Analysis:
    cfg = json.loads(CONFIG_PATH.read_text())
    policy_path = RESULTS_DIR / "selective_thresholds.json"
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else None
    analysis = Analysis(record_id=record.id, model_version=cfg["selected_model"], model_sha256=classifier.sha256,
                        config_sha256=sha256_file(CONFIG_PATH), preprocessing_version=PREPROCESSING_VERSION,
                        status="running", progress=1, summary_json={})
    session.add(analysis)
    session.commit()
    try:
        signal = np.asarray(np.load(record.signal_path), dtype=np.float32)
        filtered = filter_continuous(signal, record.sampling_rate)
        analysis.progress = 20
        session.commit()
        peak_source = "wfdb_xqrs"
        peaks: np.ndarray
        detector_peaks = detect_r_peaks(filtered, record.sampling_rate)
        alignment = None
        annotation_symbols: dict[int, str] = {}
        if record.annotation_path and record.wfdb_base_path:
            ann_file = Path(record.annotation_path)
            ann = wfdb.rdann(str(Path(record.wfdb_base_path)), ann_file.suffix.lstrip("."))
            mapping = cfg["aami_mapping"]
            pairs = [(int(p), sym) for p, sym in zip(ann.sample, ann.symbol) if sym in mapping]
            if pairs:
                peaks = np.asarray([p for p, _ in pairs], dtype=np.int64)
                annotation_symbols = {p: sym for p, sym in pairs}
                peak_source = "wfdb_annotation"
                alignment = annotation_alignment(peaks, detector_peaks, record.sampling_rate,
                                                 ANNOTATION_ALIGNMENT_TOLERANCE_MS)
                if alignment["matched_fraction"] < ANNOTATION_ALIGNMENT_MIN:
                    raise ServiceError(422, f"annotation alignment check failed during analysis: "
                                       f"{alignment['matched_count']}/{alignment['annotation_count']} AAMI peaks "
                                       f"({alignment['matched_fraction']:.1%}) matched XQRS within "
                                       f"{ANNOTATION_ALIGNMENT_TOLERANCE_MS:g} ms; required "
                                       f"{ANNOTATION_ALIGNMENT_MIN:.0%}.")
            else:
                peaks = detector_peaks
                peak_source = "wfdb_xqrs_no_mapped_annotations"
        else:
            peaks = detector_peaks
        if len(peaks) == 0:
            raise ServiceError(422, "no R peaks detected; analysis produced no predictions")
        batch = extract_windows(filtered, peaks)
        analysis.progress = 55
        analysis.peak_source = peak_source
        analysis.skipped_edge_count = int(len(batch.rejected_sample_indices))
        session.commit()
        if not len(batch.windows):
            raise ServiceError(422, "no complete 720-sample beat windows remain after edge rejection")
        rows = classifier.predict(batch.windows, batch.sample_indices, batch_size=256)
        if len(rows) != len(batch.windows):
            raise RuntimeError("model returned a different number of predictions than valid windows")
        thresholds = policy.get("operating_points", {}).get("balanced", {}) if policy else {}
        tau = float(thresholds.get("tau", 0.0))
        min_margin = float(thresholds.get("margin", 0.0))
        temperature = float(policy.get("temperature", 1.0)) if policy else 1.0
        class_counts = {c: 0 for c in cfg["classes"]}
        flagged = 0
        beat_rows = []
        for i, prediction in enumerate(rows):
            probs = np.asarray(prediction["probabilities"], dtype=np.float64)
            predicted = str(prediction["predicted_class"])
            class_counts[predicted] += 1
            sorted_raw = np.sort(probs)
            confidence = float(sorted_raw[-1])
            margin = float(sorted_raw[-1] - sorted_raw[-2])
            calibrated = _softmax(np.log(np.clip(probs, 1e-12, 1.0)) / temperature)
            cal_sorted = np.sort(calibrated)
            flagged_now = bool(cal_sorted[-1] < tau or cal_sorted[-1] - cal_sorted[-2] < min_margin)
            reason = "exploratory calibration threshold" if flagged_now else None
            flagged += int(flagged_now)
            beat = Beat(analysis_id=analysis.id, beat_index=i, sample_index=int(batch.sample_indices[i]),
                        timestamp_seconds=float(batch.sample_indices[i] / record.sampling_rate))
            beat.prediction = Prediction(predicted_class=predicted, probabilities=probs.tolist(),
                                         confidence=confidence, margin=margin, review_flag=flagged_now,
                                         review_reason=reason)
            beat_rows.append(beat)
        session.add_all(beat_rows)
        analysis.beat_count = len(beat_rows)
        analysis.progress = 90
        abnormal = class_counts.get("S", 0) + class_counts.get("V", 0) + class_counts.get("F", 0)
        analysis.summary_json = {
            "class_counts": class_counts,
            "abnormal_beat_count": abnormal,
            "flagged_for_review_count": flagged,
            "detected_peak_count": int(len(peaks)),
            "valid_window_count": int(len(batch.windows)),
            "skipped_edge_count": int(len(batch.rejected_sample_indices)),
            "detection": {"source": peak_source, "peak_count": int(len(peaks)),
                          "annotation_symbols_available": bool(annotation_symbols),
                          "annotation_alignment": alignment,
                          "annotation_alignment_min_fraction": ANNOTATION_ALIGNMENT_MIN,
                          "annotation_alignment_tolerance_ms": ANNOTATION_ALIGNMENT_TOLERANCE_MS},
            "uncertainty_policy": {"name": "balanced" if policy else "configurable-default",
                                   "provenance": str(policy_path) if policy else "no selective threshold artifact",
                                   "temperature": temperature, "tau": tau, "margin": min_margin,
                                   "label": "exploratory, not clinically calibrated"},
        }
        analysis.status = "completed"
        analysis.progress = 100
        analysis.completed_at = datetime.now(timezone.utc)
        session.commit()
        session.refresh(analysis)
        return analysis
    except Exception as exc:
        session.rollback()
        failed = session.get(Analysis, analysis.id)
        if failed is not None:
            failed.status = "failed"
            failed.progress = 100
            failed.error_message = str(exc)
            failed.completed_at = datetime.now(timezone.utc)
            session.commit()
        if isinstance(exc, ServiceError):
            raise
        raise ServiceError(500, f"analysis failed: {exc}") from exc
