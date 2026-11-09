from pathlib import Path
import numpy as np
import pytest

from app.ml.explainability import ModelExplainer

ROOT = Path(__file__).resolve().parents[2]


def reference_batch():
    with np.load(ROOT / "results/test_examples_for_xai.npz", allow_pickle=False) as z:
        return np.asarray(z["X"][[0, 1500, 30000, 45000]], dtype=np.float32)


def test_gradcam_finite_shape_reproducible_and_class_specific(model):
    explainer = ModelExplainer(model.model)
    x = reference_batch()[:1]
    cam1, norm1 = explainer.gradcam_with_diagnostics(x, [0])
    cam2, norm2 = explainer.gradcam_with_diagnostics(x, [0])
    assert cam1.shape == (1, 720)
    assert np.isfinite(cam1).all() and np.isfinite(norm1).all() and norm1[0] > 0
    assert cam1.min() >= 0 and cam1.max() <= 1
    assert np.array_equal(cam1, cam2) and np.array_equal(norm1, norm2)
    cams = explainer.gradcam(np.repeat(x, 4, axis=0), [0, 1, 2, 3])
    assert cams.shape == (4, 720) and np.isfinite(cams).all()
    assert any(not np.allclose(cams[0], cams[i]) for i in (1, 2, 3))


def test_integrated_gradients_finite_completeness_and_baseline_sensitivity(model):
    explainer = ModelExplainer(model.model, ig_steps=64, ig_batch_size=16)
    x = reference_batch()[:1]
    zero = explainer.integrated_gradients(x, 0)
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 0.1, size=x.shape).astype(np.float32)
    noisy = explainer.integrated_gradients(x, 0, baseline=baseline)
    for result in (zero, noisy):
        assert result["attributions"].shape == (1, 720)
        assert all(np.isfinite(value).all() for value in result.values())
    assert not np.allclose(zero["attributions"], noisy["attributions"])
    repeated = explainer.integrated_gradients(x, 0)
    assert np.array_equal(zero["attributions"], repeated["attributions"])


def test_explainability_rejects_invalid_input_and_target(model):
    explainer = ModelExplainer(model.model)
    with pytest.raises(ValueError, match="shape"):
        explainer.gradcam(np.zeros((1, 10), dtype=np.float32), 0)
    bad = np.zeros((1, 720, 1), dtype=np.float32); bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        explainer.integrated_gradients(bad, 0)
    with pytest.raises(ValueError, match="target_class"):
        explainer.gradcam(np.zeros((1, 720, 1), dtype=np.float32), 4)
