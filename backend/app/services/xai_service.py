"""Read-time facts about stored attribution maps."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb
from scipy.stats import spearmanr

from app.models import ECGRecord

CENTRE = 360
SMOOTH = 15          # samples; used only to find regions, never to change the stored map
REGION_FRACTION = 0.30


def reference_labels(record: ECGRecord, aami_mapping: dict) -> dict[int, str] | None:
    """sample -> N/S/V/F from the uploaded annotation file, or None if there is none."""
    if not record.annotation_path or not record.wfdb_base_path:
        return None
    suffix = Path(record.annotation_path).suffix.lstrip(".")
    try:
        ann = wfdb.rdann(str(Path(record.wfdb_base_path)), suffix)
    except (OSError, ValueError):
        return None
    return {int(s): aami_mapping[sym] for s, sym in zip(ann.sample, ann.symbol) if sym in aami_mapping}


def top_regions(values: list[float], fs: float, count: int = 3) -> list[dict]:
    """Largest contiguous stretches where the smoothed |attribution| passes a fraction of its maximum."""
    v = np.asarray(values, dtype=np.float64)
    total = float(np.abs(v).sum())
    if total == 0:
        return []
    smooth = np.convolve(np.abs(v), np.ones(SMOOTH) / SMOOTH, mode="same")
    mask = np.concatenate([[False], smooth >= REGION_FRACTION * smooth.max(), [False]])
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    regions = []
    for start, stop in zip(edges[::2], edges[1::2]):
        part = v[start:stop]
        regions.append({"start_sample": int(start), "end_sample": int(stop),
                        "start_ms": float((start - CENTRE) / fs * 1000), "end_ms": float((stop - CENTRE) / fs * 1000),
                        "share_of_total_absolute": float(np.abs(part).sum() / total),
                        "signed_share": float(part.sum() / total), "contains_centre_beat": bool(start <= CENTRE < stop)})
    return sorted(regions, key=lambda r: r["share_of_total_absolute"], reverse=True)[:count]


def rank_correlation(a: list[float], b: list[float]) -> float | None:
    x, y = np.asarray(a), np.abs(np.asarray(b))
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)
