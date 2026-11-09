"""Automatic retrospective R-peak detection; annotations are never consumed here."""
from __future__ import annotations

import numpy as np
from wfdb import processing

from .preprocessing import FS_HZ, validate_signal


def detect_r_peaks(signal: np.ndarray, sampling_frequency: float = FS_HZ) -> np.ndarray:
    """Detect peaks with WFDB XQRS. No ground-truth annotations are accepted."""
    x = validate_signal(signal, sampling_frequency)
    peaks = processing.xqrs_detect(sig=x, fs=float(sampling_frequency), learn=True, verbose=False)
    if peaks is None:
        return np.empty(0, dtype=np.int64)
    result = np.asarray(peaks, dtype=np.int64)
    if result.ndim != 1 or (result.size and ((result < 0).any() or (result >= x.size).any())):
        raise RuntimeError("WFDB XQRS returned invalid sample positions")
    return result


def match_peaks(reference: np.ndarray, detected: np.ndarray, tolerance_samples: int = 54) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Greedy one-to-one nearest-time matching, with 150 ms default tolerance."""
    ref = np.sort(np.asarray(reference, dtype=np.int64))
    det = np.sort(np.asarray(detected, dtype=np.int64))
    candidates: list[tuple[int, int, int]] = []
    lo = hi = 0
    for di, d in enumerate(det):
        while lo < len(ref) and ref[lo] < d - tolerance_samples:
            lo += 1
        hi = max(hi, lo)
        while hi < len(ref) and ref[hi] <= d + tolerance_samples:
            candidates.append((abs(int(ref[hi] - d)), di, hi))
            hi += 1
    used_d: set[int] = set()
    used_r: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, di, ri in sorted(candidates):
        if di not in used_d and ri not in used_r:
            used_d.add(di); used_r.add(ri)
            matches.append((int(det[di]), int(ref[ri])))
    unmatched_det = np.asarray([int(v) for i, v in enumerate(det) if i not in used_d], dtype=np.int64)
    unmatched_ref = np.asarray([int(v) for i, v in enumerate(ref) if i not in used_r], dtype=np.int64)
    return matches, unmatched_det, unmatched_ref


def score_detector(reference: np.ndarray, detected: np.ndarray, tolerance_samples: int = 54) -> dict[str, float | int]:
    matches, false_positive, false_negative = match_peaks(reference, detected, tolerance_samples)
    tp, fp, fn = len(matches), len(false_positive), len(false_negative)
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    errors = np.asarray([d - r for d, r in matches], dtype=np.float64)
    return {"tp": tp, "fp": fp, "fn": fn, "sensitivity": sensitivity, "positive_predictivity": ppv,
            "f1": f1, "mean_timing_error_ms": float(errors.mean() * 1000 / FS_HZ) if tp else float("nan"),
            "median_absolute_timing_error_ms": float(np.median(np.abs(errors)) * 1000 / FS_HZ) if tp else float("nan")}


def annotation_alignment(reference: np.ndarray, detected: np.ndarray, sampling_frequency: float = FS_HZ,
                         tolerance_ms: float = 150.0) -> dict[str, float | int]:
    """Return one-to-one annotation coverage against detector peaks."""
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be finite and positive")
    if not np.isfinite(tolerance_ms) or tolerance_ms <= 0:
        raise ValueError("tolerance_ms must be finite and positive")
    annotated = np.asarray(reference, dtype=np.int64)
    detected = np.asarray(detected, dtype=np.int64)
    matches, _, _ = match_peaks(annotated, detected, int(round(sampling_frequency * tolerance_ms / 1000)))
    return {"annotation_count": int(len(annotated)), "detected_count": int(len(detected)),
            "matched_count": int(len(matches)),
            "matched_fraction": float(len(matches) / len(annotated)) if len(annotated) else 1.0,
            "tolerance_ms": float(tolerance_ms)}
