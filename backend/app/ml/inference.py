"""Verified selected-model loading and batched probability inference."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Sequence

import numpy as np

CLASS_NAMES = ("N", "S", "V", "F")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "neurocardio_final.keras"
EXPECTED_SHA256 = "0e7083d56c31a4e43f6cd332b375b8480446485b60b80f52c3c7c5380508becc"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class ECGClassifier:
    """Loads the exported Keras artifact once, without custom-object fallback."""
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, verify_checksum: bool = True):
        self.model_path = Path(model_path)
        digest = sha256_file(self.model_path)
        if verify_checksum and digest != EXPECTED_SHA256:
            raise ValueError(f"Model checksum mismatch: {digest}")
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required to load the NeuroCardio model") from exc
        self.model = tf.keras.models.load_model(self.model_path, compile=False, safe_mode=True)
        self.sha256 = digest
        self._validate_signature()

    def _validate_signature(self) -> None:
        if len(self.model.inputs) != 1:
            raise ValueError(f"Expected one model input, found {len(self.model.inputs)}")
        if tuple(self.model.inputs[0].shape[1:]) != (720, 1):
            raise ValueError(f"Unexpected model input shape {self.model.inputs[0].shape}")
        if len(self.model.outputs) != 1 or tuple(self.model.outputs[0].shape[1:]) != (4,):
            raise ValueError(f"Unexpected model output signature: {self.model.outputs}")

    def predict(self, windows: np.ndarray, sample_indices: Sequence[int] | np.ndarray | None = None,
                batch_size: int = 256) -> list[dict[str, object]]:
        x = np.asarray(windows, dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != (720, 1):
            raise ValueError("windows must have shape (beats, 720, 1)")
        if not np.isfinite(x).all():
            raise ValueError("windows contain NaN or infinite values")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        positions = np.arange(len(x), dtype=np.int64) if sample_indices is None else np.asarray(sample_indices, dtype=np.int64)
        if positions.shape != (len(x),):
            raise ValueError("sample_indices must have one value per window")
        output: list[dict[str, object]] = []
        for start in range(0, len(x), batch_size):
            probs = np.asarray(self.model.predict(x[start:start + batch_size], verbose=0), dtype=np.float64)
            if probs.shape != (min(batch_size, len(x) - start), 4):
                raise RuntimeError(f"Model returned unexpected probability shape {probs.shape}")
            if not np.isfinite(probs).all() or (probs < -1e-6).any() or (probs > 1 + 1e-6).any():
                raise RuntimeError("Model returned invalid probability values")
            if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
                raise RuntimeError("Model outputs do not sum to one")
            for j, row in enumerate(probs):
                pred = int(np.argmax(row))
                output.append({"sample_index": int(positions[start + j]), "probabilities": row.tolist(),
                               "predicted_class": CLASS_NAMES[pred], "confidence": float(row[pred])})
        return output
