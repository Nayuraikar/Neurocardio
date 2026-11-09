from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
TOLERANCES = {"max": 1e-4, "mean": 1e-5, "agreement": 0.999}


def test_full_saved_reference_equivalence(model):
    ref_path = ROOT / "results/test_examples_for_xai.npz"
    assert ref_path.exists()
    with np.load(ref_path, allow_pickle=False) as z:
        X, y, saved_pred, saved_prob = z["X"], z["y"], z["pred"], z["prob"]
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[..., None]
    fresh = np.asarray(model.model.predict(X, batch_size=256, verbose=0), dtype=np.float32)
    delta = np.abs(fresh.astype(np.float64) - saved_prob.astype(np.float64))
    pred = fresh.argmax(axis=1)
    assert float(delta.max()) <= TOLERANCES["max"]
    assert float(delta.mean()) <= TOLERANCES["mean"]
    assert float(np.mean(pred == saved_pred)) >= TOLERANCES["agreement"]
    assert np.array_equal(pred, saved_pred)
    assert accuracy_score(y, pred) == accuracy_score(y, saved_pred)
    assert f1_score(y, pred, average="macro") == f1_score(y, saved_pred, average="macro")
