import numpy as np
import pytest
from app.ml.preprocessing import filter_continuous, extract_windows, prepare_recording


def test_window_contract_center_and_edges():
    x = np.arange(1440, dtype=np.float32)
    batch = extract_windows(x, np.array([0, 360, 1080, 1439], dtype=np.int64))
    assert batch.windows.shape == (2, 720, 1)
    assert batch.valid_sample_indices.tolist() == [360, 1080]
    assert batch.rejected_sample_indices.tolist() == [0, 1439]
    assert batch.windows.dtype == np.float32
    expected_center = (x[360] - x[:720].mean()) / (x[:720].std() + 1e-8)
    assert batch.windows[0, 360, 0] == pytest.approx(float(expected_center), abs=2e-6)
    assert float(batch.windows[0, :, 0].mean()) == pytest.approx(0, abs=1e-6)
    assert float(batch.windows[0, :, 0].std()) == pytest.approx(1, abs=1e-5)


def test_constant_window_zero_variance_is_finite_zeros():
    batch = extract_windows(np.ones(1440, np.float32), np.array([720], np.int64))
    assert np.isfinite(batch.windows).all()
    assert np.count_nonzero(batch.windows) == 0


def test_sampling_rate_must_match_training_rate():
    with pytest.raises(ValueError, match="resampling is not configured"):
        filter_continuous(np.zeros(1000), 250)


def test_nan_rejected():
    x = np.zeros(1000); x[500] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        filter_continuous(x)


def test_prepare_filters_full_signal_before_windowing():
    fs = 360
    t = np.arange(3600) / fs
    raw = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 80 * t)
    filtered, batch = prepare_recording(raw, np.array([720, 1800, 2880], np.int64))
    assert filtered.shape == raw.shape
    assert batch.windows.shape == (3, 720, 1)
