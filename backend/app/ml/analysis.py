"""Retrospective end-to-end raw ECG analysis for the selected ECG-only model."""
from __future__ import annotations

from typing import Sequence
import numpy as np
from .inference import ECGClassifier
from .preprocessing import FS_HZ, filter_continuous, extract_windows, validate_signal
from .rpeak_detection import detect_r_peaks


def analyze_recording(signal: Sequence[float] | np.ndarray, sampling_frequency: float, lead: str,
                      classifier: ECGClassifier | None = None) -> dict[str, object]:
    """Analyze an entire recording retrospectively; requires exactly 360 Hz."""
    x = validate_signal(signal, sampling_frequency)
    if not lead or not lead.strip():
        raise ValueError("lead/channel name is required")
    if x.size < 720:
        raise ValueError("recording is shorter than 720 samples; no complete model window can fit")
    filtered = filter_continuous(x, sampling_frequency)
    peaks = detect_r_peaks(filtered, sampling_frequency)
    batch = extract_windows(filtered, peaks)
    if len(batch.windows):
        model = classifier or ECGClassifier()
        predictions = model.predict(batch.windows, batch.sample_indices)
    else:
        predictions = []
    warnings = list(batch.warnings)
    status = "complete"
    if len(peaks) == 0:
        status = "no_r_peaks"
        warnings.append("No R peaks detected; no beat predictions were produced.")
    elif len(batch.windows) == 0:
        status = "no_valid_windows"
    return {"mode": "offline_retrospective", "status": status, "sampling_frequency": FS_HZ, "lead": lead,
            "sample_count": int(x.size), "duration_seconds": float(x.size / FS_HZ),
            "r_peak_count": int(len(peaks)), "valid_beat_count": int(len(batch.windows)),
            "rejected_beat_count": int(len(batch.rejected_sample_indices)),
            "rejected_sample_indices": batch.rejected_sample_indices.tolist(),
            "warnings": warnings, "predictions": predictions,
            "filtered_signal": filtered}
