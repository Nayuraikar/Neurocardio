"""Training-matched ECG filtering and beat-window preparation.

The selected export was trained at exactly 360 Hz. The notebook specifies no
resampling operation, so signals at other rates are rejected rather than
silently resampled with an undocumented method.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt

FS_HZ = 360.0
WINDOW_SAMPLES = 720
LEFT_SAMPLES = WINDOW_SAMPLES // 2
FILTER_ORDER = 4
LOW_HZ = 0.5
HIGH_HZ = 40.0
NORMALIZATION_EPSILON = 1e-8
CLASS_NAMES = ("N", "S", "V", "F")


@dataclass(frozen=True)
class WindowBatch:
    windows: np.ndarray  # (beats, 720, 1), float32
    sample_indices: np.ndarray  # original R peak positions
    valid_sample_indices: np.ndarray
    rejected_sample_indices: np.ndarray
    warnings: tuple[str, ...]


def validate_signal(signal: Sequence[float] | np.ndarray, sampling_frequency: float) -> np.ndarray:
    """Validate a single channel without changing its values or sample rate."""
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be finite and positive")
    if not np.isclose(sampling_frequency, FS_HZ, rtol=0.0, atol=1e-9):
        raise ValueError(f"selected model requires {FS_HZ:g} Hz; received {sampling_frequency:g} Hz; resampling is not configured")
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional after channel selection")
    if x.size < 30:
        raise ValueError("signal is too short for zero-phase filtering")
    if not np.isfinite(x).all():
        raise ValueError("signal contains NaN or infinite values")
    return x


def filter_continuous(signal: Sequence[float] | np.ndarray, sampling_frequency: float = FS_HZ) -> np.ndarray:
    """Apply the notebook's 4th-order 0.5–40 Hz Butterworth SOS filter."""
    x = validate_signal(signal, sampling_frequency)
    sos = butter(FILTER_ORDER, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=FS_HZ, output="sos")
    # Matches notebook: filter the full continuous record before windowing.
    return sosfiltfilt(sos, x).astype(np.float32)


def extract_windows(
    filtered_signal: Sequence[float] | np.ndarray,
    r_peak_indices: Sequence[int] | np.ndarray,
) -> WindowBatch:
    """Extract [peak-360, peak+360) windows and z-score each independently."""
    x = np.asarray(filtered_signal, dtype=np.float32)
    peaks_raw = np.asarray(r_peak_indices)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("filtered_signal must be a finite 1-D array")
    if peaks_raw.ndim != 1 or not np.issubdtype(peaks_raw.dtype, np.integer):
        raise ValueError("R-peak indices must be a 1-D integer array")
    peaks = peaks_raw.astype(np.int64, copy=False)
    valid_mask = (peaks - LEFT_SAMPLES >= 0) & (peaks + LEFT_SAMPLES <= x.size)
    valid = peaks[valid_mask]
    rejected = peaks[~valid_mask]
    if valid.size:
        raw = np.stack([x[p - LEFT_SAMPLES:p + LEFT_SAMPLES] for p in valid]).astype(np.float32, copy=False)
        means = raw.mean(axis=1, keepdims=True)
        stds = raw.std(axis=1, keepdims=True)
        norm = ((raw - means) / (stds + NORMALIZATION_EPSILON)).astype(np.float32)
        windows = norm[..., None]
    else:
        windows = np.empty((0, WINDOW_SAMPLES, 1), dtype=np.float32)
    warns = (f"Rejected {len(rejected)} beat(s) whose centered window crossed a recording edge.",) if len(rejected) else ()
    return WindowBatch(windows, valid.copy(), valid.copy(), rejected.copy(), warns)


def prepare_recording(signal: Sequence[float] | np.ndarray, r_peak_indices: Sequence[int] | np.ndarray,
                      sampling_frequency: float = FS_HZ) -> tuple[np.ndarray, WindowBatch]:
    """Filter full signal first, then extract/normalize centered beat windows."""
    filtered = filter_continuous(signal, sampling_frequency)
    return filtered, extract_windows(filtered, r_peak_indices)
