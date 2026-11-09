#!/usr/bin/env python3
"""Verify the selected model against saved predictions and MIT-BIH records.

Run from project root with:
    backend/.venv/bin/python backend/scripts/verify_model_reference.py
The script is read-only with respect to results/ and model artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import keras
import numpy as np
import scipy
import sklearn
import tensorflow as tf
import wfdb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
MODEL = ROOT / "backend/models/neurocardio_final.keras"
CONFIG = ROOT / "backend/models/preprocessing_config.json"
RESULTS = ROOT / "results"
REFERENCE = RESULTS / "test_examples_for_xai.npz"
REPORT_PATH = ROOT / "docs/model-reference-comparison.json"
CLASSES = ["N", "S", "V", "F"]
TOLERANCES = {"max_abs_probability_difference": 1e-4,
              "mean_abs_probability_difference": 1e-5,
              "argmax_agreement_fraction_min": 0.999}
COMMAND = "backend/.venv/bin/python backend/scripts/verify_model_reference.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metric_pack(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict:
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=np.arange(4), zero_division=0)
    cm = confusion_matrix(y, pred, labels=np.arange(4))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(4), average="macro", zero_division=0)),
        "macro_roc_auc_ovr": float(roc_auc_score(y, prob, labels=np.arange(4), multi_class="ovr", average="macro")),
        "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
                      for i, name in enumerate(CLASSES)},
        "confusion_matrix_true_rows_pred_columns": cm.tolist(),
        "true_positive_counts": {name: int(cm[i, i]) for i, name in enumerate(CLASSES)},
    }


def write_report(result: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def verify_identity() -> tuple[dict, object, dict]:
    cfg = json.loads(CONFIG.read_text())
    cv = json.loads((RESULTS / "cv_results.json").read_text())
    test = json.loads((RESULTS / "test_metrics.json").read_text())
    raw = json.loads((RESULTS / "test_raw_primary.json").read_text())
    assert sha256(MODEL) == sha256(RESULTS / "neurocardio_final.keras"), "backend/result model checksums differ"
    assert sha256(CONFIG) == sha256(RESULTS / "preprocessing_config.json"), "backend/result config checksums differ"
    for source, name in ((cfg, "config"), (cv, "cv_results"), (test, "test_metrics"), (raw, "test_raw_primary")):
        selected = source.get("selected_model", source.get("config", source.get("model")))
        assert selected == "bilstm_aug_sqrtw", f"{name} identity mismatch: {selected!r}"
    assert cfg.get("rr_features") is None, "selected config unexpectedly requires RR features"
    assert cfg.get("classes") == CLASSES
    model = keras.saving.load_model(MODEL, compile=False)
    assert len(model.inputs) == 1 and tuple(model.inputs[0].shape[1:]) == (720, 1)
    assert len(model.outputs) == 1 and tuple(model.outputs[0].shape[1:]) == (4,)
    assert all(any(layer.name == lname for layer in model.layers) for lname in ("conv3", "logits"))
    assert model.get_layer("probs").__class__.__name__ == "Softmax"
    return cfg, model, {"test_metrics": test, "test_raw_primary": raw, "cv_results": cv}


def reference_stage(cfg: dict, model, source_metrics: dict) -> tuple[dict, np.ndarray, dict]:
    with np.load(REFERENCE, allow_pickle=False) as z:
        required = {"X", "R", "y", "pred", "prob", "record", "position"}
        assert required <= set(z.files), f"reference NPZ missing keys: {required - set(z.files)}"
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y"], dtype=np.int64)
        saved_pred = np.asarray(z["pred"], dtype=np.int64)
        saved_prob = np.asarray(z["prob"], dtype=np.float32)
        records = np.asarray(z["record"])
        positions = np.asarray(z["position"], dtype=np.int64)
        original_shapes = {k: list(z[k].shape) for k in z.files}
        original_dtypes = {k: str(z[k].dtype) for k in z.files}
    if X.ndim == 2:
        X = X[..., None]
    assert X.shape == (47518, 720, 1), f"unexpected X shape {X.shape}"
    assert saved_prob.shape == (len(X), 4) and saved_pred.shape == y.shape == (len(X),)
    assert np.isfinite(X).all() and np.isfinite(saved_prob).all()
    assert np.allclose(saved_prob.sum(axis=1), 1, atol=1e-4)
    assert np.array_equal(saved_pred, saved_prob.argmax(axis=1)), "saved pred differs from saved probability argmax"
    fresh = np.asarray(model.predict(X, batch_size=256, verbose=0), dtype=np.float32)
    assert fresh.shape == saved_prob.shape and np.isfinite(fresh).all()
    assert np.allclose(fresh.sum(axis=1), 1, atol=1e-4)
    delta = np.abs(fresh.astype(np.float64) - saved_prob.astype(np.float64))
    fresh_pred = fresh.argmax(axis=1)
    agrees = fresh_pred == saved_pred
    agreement = float(agrees.mean())
    per_class_agreement = {}
    for i, name in enumerate(CLASSES):
        mask = saved_pred == i
        per_class_agreement[name] = {"count": int(mask.sum()), "agreement_fraction": float(agrees[mask].mean()) if mask.any() else None}
    worst = []
    disagree_idx = np.flatnonzero(~agrees)
    if disagree_idx.size:
        worst_idx = disagree_idx[np.argsort(delta.max(axis=1)[disagree_idx])[-5:][::-1]]
        worst = [{"record": str(records[i]), "position": int(positions[i]),
                  "saved_pred": CLASSES[int(saved_pred[i])], "fresh_pred": CLASSES[int(fresh_pred[i])],
                  "max_abs_probability_difference": float(delta[i].max())} for i in worst_idx]
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(X), size=500, replace=False)
    batch_1 = np.asarray(model.predict(X[sample_idx], batch_size=1, verbose=0), dtype=np.float32)
    batch_256 = np.asarray(model.predict(X[sample_idx], batch_size=256, verbose=0), dtype=np.float32)
    batch_diff = float(np.max(np.abs(batch_1 - batch_256)))
    metrics = metric_pack(y, fresh_pred, fresh)
    raw = source_metrics["test_raw_primary"]
    final_rows = list(csv.DictReader((RESULTS / "final_summary.csv").open()))
    final_seed = next(row for row in final_rows if row["protocol"] == "Fixed test, seed 42")
    tol_pass = {
        "max_abs_probability_difference": float(delta.max()) <= TOLERANCES["max_abs_probability_difference"],
        "mean_abs_probability_difference": float(delta.mean()) <= TOLERANCES["mean_abs_probability_difference"],
        "argmax_agreement": agreement >= TOLERANCES["argmax_agreement_fraction_min"],
    }
    metrics_pass = (
        np.isclose(metrics["accuracy"], float(raw["accuracy"]), atol=1e-10)
        and np.isclose(metrics["macro_f1"], float(raw["macro_f1"]), atol=1e-10)
        and np.array_equal(metrics["confusion_matrix_true_rows_pred_columns"], raw["confusion_matrix"])
        and np.isclose(metrics["accuracy"], float(final_seed["accuracy"]), atol=1e-10)
        and np.isclose(metrics["macro_f1"], float(final_seed["macro_f1"]), atol=1e-10)
    )
    result = {
        "command": COMMAND,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"tensorflow": tf.__version__, "keras": keras.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "identity": {"selected_model": cfg["selected_model"], "model_sha256": sha256(MODEL),
                     "config_sha256": sha256(CONFIG), "reference_npz_sha256": sha256(REFERENCE),
                     "model_input_shapes": [list(x.shape) for x in model.inputs],
                     "model_output_shapes": [list(x.shape) for x in model.outputs],
                     "classes": cfg["classes"], "rr_features": cfg["rr_features"], "parameter_count": int(model.count_params())},
        "tolerances_preregistered": TOLERANCES,
        "reference_arrays": {"shapes": original_shapes, "dtypes": original_dtypes,
                             "input_used": "X only; reshaped only if rank 2, cast to float32", "R_used": False,
                             "saved_probability_rows_sum_min": float(saved_prob.sum(axis=1).min()),
                             "saved_probability_rows_sum_max": float(saved_prob.sum(axis=1).max()),
                             "saved_pred_equals_prob_argmax": True},
        "reference_equivalence": {"max_abs_probability_difference": float(delta.max()),
                                  "mean_abs_probability_difference": float(delta.mean()),
                                  "argmax_agreement_fraction": agreement,
                                  "per_class_agreement": per_class_agreement,
                                  "worst_disagreement_windows": worst,
                                  "batch_independence_500_seed0_max_abs_difference": batch_diff,
                                  "tolerance_checks": tol_pass},
        "fresh_metrics": metrics,
        "ground_truth_crosscheck": {"test_raw_primary": raw, "final_summary_seed42": final_seed,
                                    "metrics_reproduced": bool(metrics_pass)},
        "raw_record_path": {"status": "not_run_until_reference_probability_and_metric_gates_pass"},
        "overall_pass": bool(all(tol_pass.values()) and metrics_pass),
    }
    write_report(result)
    return result, fresh_pred, {"X": X, "y": y, "records": records, "positions": positions,
                               "saved_prob": saved_prob, "saved_pred": saved_pred, "fresh": fresh,
                               "fresh_pred": fresh_pred}


def raw_record_stage(cfg: dict, model, reference: dict) -> dict:
    import wfdb
    from app.ml.preprocessing import filter_continuous, extract_windows, validate_signal
    from app.ml.rpeak_detection import detect_r_peaks, match_peaks, score_detector

    mapping = cfg["aami_mapping"]
    mitdb = RESULTS / "mitdb"
    records = list(cfg["split"]["test_records"])
    npz_lookup = {(str(r), int(p)): i for i, (r, p) in enumerate(zip(reference["records"], reference["positions"]))}
    raw_windows, raw_labels, raw_records, raw_positions = [], [], [], []
    record_data = {}
    for record in records:
        rec = wfdb.rdrecord(str(mitdb / record), physical=True)
        if float(rec.fs) != float(cfg["sampling_frequency"]):
            raise RuntimeError(f"MIT-BIH record {record} sampling rate {rec.fs} != configured rate")
        lead_idx = rec.sig_name.index(cfg["lead"]) if cfg["lead"] in rec.sig_name else 0
        signal = rec.p_signal[:, lead_idx]
        filtered = filter_continuous(signal, float(rec.fs))
        ann = wfdb.rdann(str(mitdb / record), "atr")
        ann_pos, ann_y = [], []
        for p, sym in zip(ann.sample, ann.symbol):
            if sym in mapping:
                ann_pos.append(int(p)); ann_y.append(CLASSES.index(mapping[sym]))
        batch = extract_windows(filtered, np.asarray(ann_pos, dtype=np.int64))
        label_by_pos = dict(zip(ann_pos, ann_y))
        valid_y = np.asarray([label_by_pos[int(p)] for p in batch.valid_sample_indices], dtype=np.int64)
        for win, pos, label in zip(batch.windows, batch.valid_sample_indices, valid_y):
            raw_windows.append(win); raw_labels.append(int(label)); raw_records.append(record); raw_positions.append(int(pos))
        peaks = detect_r_peaks(filtered, float(rec.fs))
        detected_batch = extract_windows(filtered, peaks)
        record_data[record] = {"signal": filtered, "ann_pos": np.asarray(ann_pos, dtype=np.int64), "ann_y": valid_y,
                               "ann_valid_pos": batch.valid_sample_indices, "ann_windows": batch.windows,
                               "detected": peaks, "detected_valid_pos": detected_batch.valid_sample_indices,
                               "detected_windows": detected_batch.windows, "lead": rec.sig_name[lead_idx]}
    Xraw = np.asarray(raw_windows, dtype=np.float32)
    yraw = np.asarray(raw_labels, dtype=np.int64)
    rraw = np.asarray(raw_records)
    praw = np.asarray(raw_positions, dtype=np.int64)
    indices = np.asarray([npz_lookup[(str(r), int(p))] for r, p in zip(rraw, praw)], dtype=np.int64)
    input_delta = np.abs(Xraw.astype(np.float64) - reference["X"][indices].astype(np.float64))
    raw_prob = np.asarray(model.predict(Xraw, batch_size=256, verbose=0), dtype=np.float32)
    raw_pred = raw_prob.argmax(axis=1)
    saved_prob = reference["saved_prob"][indices]
    saved_pred = reference["saved_pred"][indices]
    pred_agree = raw_pred == saved_pred
    prob_delta = np.abs(raw_prob.astype(np.float64) - saved_prob.astype(np.float64))
    per_record = {}
    for record in records:
        mask = rraw == record
        per_record[record] = {"beats": int(mask.sum()), "input_max_abs_difference": float(input_delta[mask].max()),
                              "input_mean_abs_difference": float(input_delta[mask].mean()),
                              "probability_max_abs_difference": float(prob_delta[mask].max()),
                              "probability_mean_abs_difference": float(prob_delta[mask].mean()),
                              "prediction_agreement_fraction": float(pred_agree[mask].mean())}
    annotation_metrics = metric_pack(yraw, raw_pred, raw_prob)
    annotation_gate = (float(input_delta.max()) <= 1e-6 and float(prob_delta.max()) <= TOLERANCES["max_abs_probability_difference"]
                       and float(prob_delta.mean()) <= TOLERANCES["mean_abs_probability_difference"]
                       and float(pred_agree.mean()) >= TOLERANCES["argmax_agreement_fraction_min"])

    # Evaluate XQRS independently from classifier performance.
    det_tp = det_fp = det_fn = 0
    all_match_class_agreement = []
    match_y, match_pred, match_prob = [], [], []
    false_peak_classes = Counter()
    per_record_detector = {}
    for record in records:
        d = record_data[record]
        ann_pos = d["ann_pos"]
        matches, fp_peaks, missed = match_peaks(ann_pos, d["detected"], tolerance_samples=54)
        # match_peaks returns (detected, reference)
        counts = score_detector(ann_pos, d["detected"], tolerance_samples=54)
        det_tp += counts["tp"]; det_fp += counts["fp"]; det_fn += counts["fn"]
        if len(d["detected_windows"]):
            p_detect = np.asarray(model.predict(d["detected_windows"], batch_size=256, verbose=0), dtype=np.float32)
            pred_detect = p_detect.argmax(axis=1)
        else:
            p_detect = np.empty((0, 4), np.float32); pred_detect = np.empty(0, np.int64)
        det_index = {int(p): i for i, p in enumerate(d["detected_valid_pos"])}
        ann_index = {int(p): i for i, p in enumerate(d["ann_valid_pos"])}
        agree = []
        for dp, ap in matches:
            if dp in det_index and ap in ann_index:
                di, ai = det_index[dp], ann_index[ap]
                agree.append(bool(pred_detect[di] == reference["fresh_pred"][npz_lookup[(record, ap)]]))
                match_y.append(int(d["ann_y"][ai])); match_pred.append(int(pred_detect[di])); match_prob.append(p_detect[di])
        all_match_class_agreement.extend(agree)
        fp_valid = [int(p) for p in fp_peaks if int(p) in det_index]
        for p in fp_valid:
            false_peak_classes[CLASSES[int(pred_detect[det_index[p]])]] += 1
        per_record_detector[record] = {"tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"],
                                       "sensitivity": counts["sensitivity"], "positive_predictivity": counts["positive_predictivity"],
                                       "f1": counts["f1"], "mean_timing_error_ms": counts["mean_timing_error_ms"],
                                       "matched_prediction_agreement_fraction": float(np.mean(agree)) if agree else None,
                                       "false_peak_model_class_counts": {c: int(sum(1 for p in fp_valid if CLASSES[int(pred_detect[det_index[p]])] == c)) for c in CLASSES}}
    det_sens = det_tp / (det_tp + det_fn) if det_tp + det_fn else 0.0
    det_ppv = det_tp / (det_tp + det_fp) if det_tp + det_fp else 0.0
    det_f1 = 2 * det_tp / (2 * det_tp + det_fp + det_fn) if 2 * det_tp + det_fp + det_fn else 0.0
    matched_metrics = metric_pack(np.asarray(match_y), np.asarray(match_pred), np.asarray(match_prob)) if match_y else None
    for fs in (250, 500):
        try:
            validate_signal(np.zeros(900, dtype=np.float32), fs)
            raise AssertionError(f"sampling rate {fs} was not rejected")
        except ValueError as e:
            if "requires 360 Hz" not in str(e):
                raise
    return {
        "status": "run",
        "annotation_peak_path": {"records": len(records), "windows": int(len(Xraw)),
                                  "input_max_abs_difference": float(input_delta.max()),
                                  "input_mean_abs_difference": float(input_delta.mean()),
                                  "probability_max_abs_difference": float(prob_delta.max()),
                                  "probability_mean_abs_difference": float(prob_delta.mean()),
                                  "prediction_agreement_fraction": float(pred_agree.mean()),
                                  "annotation_classifier_metrics": annotation_metrics,
                                  "per_record": per_record, "gate_pass": bool(annotation_gate)},
        "xqrs_detector_only": {"tolerance_samples": 54, "tolerance_ms": 150,
                               "tp": det_tp, "false_detections": det_fp, "missed_aami_beats": det_fn,
                               "sensitivity": det_sens, "positive_predictivity": det_ppv, "f1": det_f1,
                               "per_record": per_record_detector},
        "xqrs_classifier_on_matched_beats": {"matched_predicted_class_agreement_with_annotation_peak":
                                              float(np.mean(all_match_class_agreement)) if all_match_class_agreement else None,
                                              "matched_beat_classifier_metrics": matched_metrics,
                                              "false_peak_model_class_counts": dict(false_peak_classes)},
        "sampling_rate_policy": {"accepted_hz": 360, "tested_rejected_hz": [250, 500], "policy": "reject; no resampling configured"},
    }


def main() -> int:
    base = {"command": COMMAND, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {"tensorflow": tf.__version__, "keras": keras.__version__}}
    try:
        cfg, model, source = verify_identity()
        result, _, refs = reference_stage(cfg, model, source)
        if not result["overall_pass"]:
            print(json.dumps({"overall_pass": False, "reference_equivalence": result["reference_equivalence"],
                              "fresh_metrics": result["fresh_metrics"], "ground_truth_crosscheck": result["ground_truth_crosscheck"]}, indent=2))
            return 1
        raw = raw_record_stage(cfg, model, refs)
        result["raw_record_path"] = raw
        result["overall_pass"] = bool(result["overall_pass"] and raw["annotation_peak_path"]["gate_pass"])
        write_report(result)
        print(json.dumps({"comparison_report": str(REPORT_PATH), "overall_pass": result["overall_pass"],
                          "reference_equivalence": result["reference_equivalence"],
                          "fresh_metrics": result["fresh_metrics"], "raw_record_summary": {
                              "annotation_peak_path": {k: v for k, v in raw["annotation_peak_path"].items() if k != "per_record"},
                              "xqrs_detector_only": {k: v for k, v in raw["xqrs_detector_only"].items() if k != "per_record"},
                              "xqrs_classifier_on_matched_beats": raw["xqrs_classifier_on_matched_beats"],
                              "sampling_rate_policy": raw["sampling_rate_policy"]}}, indent=2))
        return 0 if result["overall_pass"] else 1
    except Exception as exc:
        if REPORT_PATH.exists():
            partial = json.loads(REPORT_PATH.read_text())
        else:
            partial = base
        partial["overall_pass"] = False
        partial["error"] = f"{type(exc).__name__}: {exc}"
        write_report(partial)
        raise


if __name__ == "__main__":
    sys.exit(main())
