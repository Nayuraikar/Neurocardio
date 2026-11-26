"""Signal quality from measurements on the stored raw signal.

Only what is computed here may be shown. The Good/Moderate/Poor cut-offs are descriptive rules of this app: they have
not been validated against model performance or against expert quality grades.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from app.ml.preprocessing import HIGH_HZ, LOW_HZ, extract_windows

STATES = ("Good", "Moderate", "Poor")
CLIP_MIN_RUN = 3
# metric -> (Good below, Moderate below); anything higher is Poor
THRESHOLDS = {
    "missing_fraction": (1e-9, 0.01),
    "clipping_fraction": (0.001, 0.01),
    "baseline_wander_power_fraction": (0.20, 0.50),
    "high_frequency_power_fraction": (0.05, 0.15),
    "invalid_window_fraction": (0.02, 0.10),
}
LABELS = {
    "missing_fraction": ("Missing samples", "Share of samples that are NaN or infinite."),
    "clipping_fraction": ("Clipping", f"Share of samples in runs of {CLIP_MIN_RUN}+ identical values at the record's maximum or minimum."),
    "baseline_wander_power_fraction": ("Baseline-wander power", f"Share of raw signal power below {LOW_HZ} Hz (mean removed, Welch)."),
    "high_frequency_power_fraction": ("High-frequency power", f"Share of raw signal power above {HIGH_HZ:g} Hz (mean removed, Welch)."),
    "invalid_window_fraction": ("Invalid windows", "Share of beat positions with no usable window: it crosses a record edge or is flat."),
}


def state_of(metric: str, value: float) -> str:
    good, moderate = THRESHOLDS[metric]
    return STATES[0] if value < good else STATES[1] if value < moderate else STATES[2]


def clipping_fraction(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    at_rail = (x == x.max()) | (x == x.min())
    edges = np.flatnonzero(np.diff(np.concatenate([[0], at_rail.astype(np.int8), [0]])))
    clipped = 0
    for start, stop in zip(edges[::2], edges[1::2]):
        run = x[start:stop]
        if len(run) >= CLIP_MIN_RUN and np.all(run == run[0]):
            clipped += len(run)
    return clipped / x.size


def band_fractions(x: np.ndarray, fs: float) -> tuple[float, float]:
    frequencies, power = welch(x - x.mean(), fs=fs, nperseg=min(len(x), int(8 * fs)), detrend=False)
    total = float(power.sum())
    if total <= 0:
        return 0.0, 0.0
    return float(power[frequencies < LOW_HZ].sum() / total), float(power[frequencies > HIGH_HZ].sum() / total)


def assess(raw: np.ndarray, filtered: np.ndarray, fs: float, peaks: np.ndarray) -> dict:
    finite = np.isfinite(raw)
    clean = raw[finite].astype(np.float64)
    baseline, high = band_fractions(clean, fs)
    batch = extract_windows(filtered, peaks.astype(np.int64))
    flat = 0
    for p in batch.sample_indices:
        flat += int(np.ptp(filtered[p - 360:p + 360]) == 0)
    invalid = len(batch.rejected_sample_indices) + flat
    values = {"missing_fraction": float(1 - finite.mean()), "clipping_fraction": float(clipping_fraction(clean)),
              "baseline_wander_power_fraction": baseline, "high_frequency_power_fraction": high,
              "invalid_window_fraction": float(invalid / len(peaks)) if len(peaks) else 0.0}
    metrics = [{"key": k, "label": LABELS[k][0], "definition": LABELS[k][1], "value": v, "state": state_of(k, v),
                "good_below": THRESHOLDS[k][0], "moderate_below": THRESHOLDS[k][1]} for k, v in values.items()]
    overall = max((m["state"] for m in metrics), key=STATES.index)
    return {"overall": overall, "metrics": metrics,
            "counts": {"samples": int(raw.size), "missing_samples": int((~finite).sum()), "beat_positions": int(len(peaks)),
                       "edge_windows": int(len(batch.rejected_sample_indices)), "flat_windows": flat},
            "rule": "Overall state is the worst metric state.",
            "caveat": "Cut-offs are descriptive rules of this app. They have not been validated against model accuracy "
                      "or expert grading, so Good does not mean the predictions are right."}
