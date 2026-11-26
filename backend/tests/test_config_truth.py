"""The text the UI shows about the model must match how the code really behaves."""
import json

import numpy as np

from app.core.config import CONFIG_PATH
from app.ml import preprocessing as P
from app.ml.inference import ECGClassifier
from app.services.model_service import model_config

CFG = json.loads(CONFIG_PATH.read_text())


def test_code_constants_equal_config():
    band = CFG["bandpass_filter"]
    assert (P.FS_HZ, P.WINDOW_SAMPLES) == (CFG["sampling_frequency"], CFG["window_size"])
    assert (P.FILTER_ORDER, P.LOW_HZ, P.HIGH_HZ) == (band["order"], band["low_hz"], band["high_hz"])
    assert P.NORMALIZATION_EPSILON == CFG["normalization_epsilon"]
    assert list(P.CLASS_NAMES) == CFG["classes"]


def test_windows_are_zscored_individually():
    rng = np.random.default_rng(0)
    signal = (rng.normal(5.0, 3.0, 5000) + np.linspace(0, 40, 5000)).astype(np.float32)
    windows = P.extract_windows(signal, np.array([600, 2500, 4300])).windows[..., 0]
    assert windows.shape == (3, 720)
    assert np.allclose(windows.mean(axis=1), 0, atol=1e-5) and np.allclose(windows.std(axis=1), 1, atol=1e-4)


def test_model_text_comes_from_loaded_layers():
    info = model_config(ECGClassifier(), CFG, "x")
    assert (info["architecture"], info["input_shape"], info["normalization"]) == ("CNN-BiLSTM", "720×1", "per-window z-score")
    assert [s["label"] for s in info["pipeline"]][1] == "Bandpass filter"
