import json
from pathlib import Path
import numpy as np
import pytest
import wfdb

from app.ml.analysis import analyze_recording
from app.ml.inference import CLASS_NAMES
from app.ml.preprocessing import extract_windows, filter_continuous, prepare_recording

ROOT = Path(__file__).resolve().parents[2]


def reference_arrays():
    with np.load(ROOT / "results/test_examples_for_xai.npz", allow_pickle=False) as z:
        return z["X"], z["prob"], z["pred"], z["record"], z["position"]


def test_batch_sizes_1_7_256_match_reference(model):
    X, ref_prob, _, _, pos = reference_arrays()
    X = np.asarray(X[:263], dtype=np.float32)
    outputs = []
    for batch_size in (1, 7, 256):
        outputs.append(np.asarray(model.model.predict(X, batch_size=batch_size, verbose=0), dtype=np.float32))
    for out in outputs:
        assert np.allclose(out, ref_prob[:263], atol=1e-4, rtol=0)
        assert np.isfinite(out).all()
        assert np.allclose(out.sum(1), 1, atol=1e-4)
    assert np.allclose(outputs[0], outputs[1], atol=1e-4, rtol=0)
    assert np.allclose(outputs[1], outputs[2], atol=1e-4, rtol=0)


def test_integer_and_float64_windows_are_cast_to_float32(model):
    X, _, _, _, positions = reference_arrays()
    X = X[:3]
    indices = positions[:3]
    for typed in (X.astype(np.int16), X.astype(np.float64)):
        result = model.predict(typed, indices)
        assert len(result) == 3
        assert all(np.isfinite(r["probabilities"]).all() for r in result)


def test_wrong_shape_empty_and_bad_batch_size(model):
    with pytest.raises(ValueError, match="shape"):
        model.predict(np.zeros((1, 720), dtype=np.float32))
    assert model.predict(np.empty((0, 720, 1), dtype=np.float32)) == []
    with pytest.raises(ValueError, match="batch_size"):
        model.predict(np.empty((0, 720, 1), dtype=np.float32), batch_size=0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_window_values_rejected(model, bad):
    X, _, _, _, _ = reference_arrays()
    bad_window = np.array(X[:1], dtype=np.float32, copy=True)
    bad_window[0, 4, 0] = bad
    with pytest.raises(ValueError, match="NaN or infinite"):
        model.predict(bad_window)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_signal_values_rejected(bad):
    signal = np.ones(1200, dtype=np.float64)
    signal[500] = bad
    with pytest.raises(ValueError, match="NaN or infinite"):
        filter_continuous(signal)


def test_flat_signal_produces_finite_normalized_window_and_probabilities(model):
    filtered, batch = prepare_recording(np.zeros(1440, dtype=np.float32), np.array([720], dtype=np.int64))
    assert np.isfinite(filtered).all()
    assert np.isfinite(batch.windows).all()
    result = model.predict(batch.windows, batch.sample_indices)
    assert len(result) == 1
    p = np.asarray(result[0]["probabilities"])
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    assert p.sum() == pytest.approx(1, abs=1e-4)


def test_short_signal_is_explicitly_rejected(model):
    with pytest.raises(ValueError, match="shorter than 720 samples"):
        analyze_recording(np.zeros(719, dtype=np.float32), 360, "MLII", model)


def test_no_detected_peaks_is_explicit_empty_result(model):
    result = analyze_recording(np.zeros(1440, dtype=np.float32), 360, "MLII", model)
    assert result["status"] == "no_r_peaks"
    assert result["r_peak_count"] == 0
    assert result["predictions"] == []
    assert any("No R peaks" in w for w in result["warnings"])


def test_edge_beats_are_skipped_and_counted():
    x = np.zeros(2000, dtype=np.float32)
    peaks = np.array([0, 359, 360, 1640, 1641, 1999], dtype=np.int64)
    batch = extract_windows(x, peaks)
    assert batch.valid_sample_indices.tolist() == [360, 1640]
    assert batch.rejected_sample_indices.tolist() == [0, 359, 1641, 1999]
    assert len(batch.warnings) == 1 and "Rejected 4" in batch.warnings[0]


@pytest.mark.parametrize("fs", [250, 500])
def test_unsupported_sampling_rates_rejected(fs):
    with pytest.raises(ValueError, match="requires 360 Hz"):
        filter_continuous(np.zeros(1440, dtype=np.float32), fs)


def test_deterministic_repeat_predictions(model):
    X, _, _, _, pos = reference_arrays()
    first = model.predict(X[:16], pos[:16])
    second = model.predict(X[:16], pos[:16])
    assert [r["predicted_class"] for r in first] == [r["predicted_class"] for r in second]
    assert np.array_equal(np.asarray([r["probabilities"] for r in first]),
                          np.asarray([r["probabilities"] for r in second]))


@pytest.mark.parametrize("record", ["213", "232"])
def test_raw_annotation_record_reproduces_reference(record, model):
    cfg = json.loads((ROOT / "backend/models/preprocessing_config.json").read_text())
    with np.load(ROOT / "results/test_examples_for_xai.npz", allow_pickle=False) as z:
        mask = z["record"] == record
        ref_X, ref_prob, ref_pred, ref_pos = z["X"][mask], z["prob"][mask], z["pred"][mask], z["position"][mask]
    wf = wfdb.rdrecord(str(ROOT / "results/mitdb" / record), physical=True)
    lead_index = wf.sig_name.index("MLII") if "MLII" in wf.sig_name else 0
    filtered = filter_continuous(wf.p_signal[:, lead_index], wf.fs)
    ann = wfdb.rdann(str(ROOT / "results/mitdb" / record), "atr")
    mapping = cfg["aami_mapping"]
    positions = np.asarray([int(p) for p, sym in zip(ann.sample, ann.symbol) if sym in mapping], dtype=np.int64)
    batch = extract_windows(filtered, positions)
    wanted = set(int(p) for p in ref_pos)
    selected = np.asarray([int(p) in wanted for p in batch.valid_sample_indices])
    rebuilt_positions = batch.valid_sample_indices[selected]
    rebuilt = batch.windows[selected]
    lookup = {int(p): i for i, p in enumerate(rebuilt_positions)}
    order = np.asarray([lookup[int(p)] for p in ref_pos], dtype=np.int64)
    rebuilt = rebuilt[order]
    assert rebuilt.shape == ref_X.shape
    assert np.max(np.abs(rebuilt - ref_X)) <= 1e-6
    fresh = np.asarray(model.model.predict(rebuilt, batch_size=256, verbose=0))
    assert np.max(np.abs(fresh - ref_prob)) <= 1e-4
    assert np.array_equal(fresh.argmax(1), ref_pred)
