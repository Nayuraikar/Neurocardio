import numpy as np
from app.ml.inference import ECGClassifier, CLASS_NAMES


def test_selected_model_loads_with_expected_signature(model):
    clf = model
    assert len(clf.model.inputs) == 1
    assert tuple(clf.model.inputs[0].shape[1:]) == (720, 1)
    assert len(clf.model.outputs) == 1
    assert tuple(clf.model.outputs[0].shape[1:]) == (4,)
    assert list(CLASS_NAMES) == ["N", "S", "V", "F"]


def test_reference_npz_golden_window_prediction(model):
    from pathlib import Path
    ref_path = Path(__file__).resolve().parents[2] / "results" / "test_examples_for_xai.npz"
    assert ref_path.exists(), f"Reference NPZ missing: {ref_path}"
    with np.load(ref_path, allow_pickle=False) as z:
        X, prob, positions = z["X"], z["prob"], z["position"]
        idx = 0
        actual = model.predict(X[idx:idx + 1], positions[idx:idx + 1])[0]
        assert actual["predicted_class"] == ("N", "S", "V", "F")[int(np.argmax(prob[idx]))]
        assert np.allclose(actual["probabilities"], prob[idx], atol=1e-5, rtol=1e-5)
