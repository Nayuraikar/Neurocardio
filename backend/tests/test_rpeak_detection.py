import numpy as np
from app.ml.rpeak_detection import match_peaks, score_detector


def test_one_to_one_matching_and_metrics():
    ref = np.array([100, 300, 500])
    det = np.array([103, 105, 302, 700])
    matches, fp, fn = match_peaks(ref, det, tolerance_samples=10)
    assert len(matches) == 2 and len(fp) == 2 and fn.tolist() == [500]
    metrics = score_detector(ref, det, tolerance_samples=10)
    assert metrics["sensitivity"] == 2 / 3
    assert metrics["positive_predictivity"] == 2 / 4
