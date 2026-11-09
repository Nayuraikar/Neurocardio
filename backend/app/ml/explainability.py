"""Gradient-based explanations for the verified NeuroCardio Keras model.

These are technical attributions of model output, not clinically validated
explanations. Grad-CAM targets the actual `conv3` layer; Integrated Gradients
targets actual `logits` and uses zero z-scored signal as its default baseline.
"""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

import numpy as np
import tensorflow as tf


def compute_explanation_cache_key(
    *,
    analysis_id: str,
    beat_id: str,
    target_class: str,
    method: str,
    ig_steps: int | None,
    baseline_name: str | None,
    model_sha256: str,
    preprocessing_version: str,
    cache_version: str = "v3-conv3-gradcam",
) -> str:
    key_dict = {
        "cache_version": cache_version,
        "analysis_id": analysis_id,
        "beat_id": beat_id,
        "target_class": target_class,
        "method": method,
        "ig_steps": ig_steps if method == "ig" else None,
        "baseline_name": baseline_name if method == "ig" else None,
        "model_sha256": model_sha256,
        "preprocessing_version": preprocessing_version,
    }
    raw = json.dumps(key_dict, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ModelExplainer:
    def __init__(self, model: tf.keras.Model, ig_steps: int = 64, ig_batch_size: int = 32):
        if ig_steps < 2:
            raise ValueError("ig_steps must be at least 2")
        if ig_batch_size < 2:
            raise ValueError("ig_batch_size must be at least 2")
        if len(model.inputs) != 1 or tuple(model.inputs[0].shape[1:]) != (720, 1):
            raise ValueError("explainability requires the selected one-input (720,1) model")
        self.model = model
        self.ig_steps = int(ig_steps)
        self.ig_batch_size = int(ig_batch_size)
        self.grad_model = tf.keras.Model(
            model.inputs, [model.get_layer("conv3").output, model.get_layer("logits").output]
        )
        self.logit_model = tf.keras.Model(model.inputs, model.get_layer("logits").output)

    @staticmethod
    def _batch(windows: np.ndarray) -> np.ndarray:
        x = np.asarray(windows, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :, None]
        elif x.ndim == 2 and x.shape == (720, 1):
            x = x[None, ...]
        if x.ndim != 3 or x.shape[1:] != (720, 1):
            raise ValueError("windows must have shape (720,), (720,1), or (batch,720,1)")
        if not np.isfinite(x).all():
            raise ValueError("windows contain NaN or infinite values")
        return x

    @staticmethod
    def _targets(target_class: int | Sequence[int], batch_size: int) -> np.ndarray:
        targets = np.asarray(target_class, dtype=np.int64)
        if targets.ndim == 0:
            targets = np.full(batch_size, int(targets), dtype=np.int64)
        if targets.shape != (batch_size,) or ((targets < 0) | (targets > 3)).any():
            raise ValueError("target_class must be 0..3 or one valid class index per window")
        return targets

    def gradcam_with_diagnostics(self, windows: np.ndarray, target_class: int | Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized Grad-CAM maps and per-window gradient L2 norms."""
        x = self._batch(windows)
        targets = self._targets(target_class, len(x))
        xt = tf.convert_to_tensor(x, dtype=tf.float32)
        target_t = tf.convert_to_tensor(targets, dtype=tf.int32)
        with tf.GradientTape() as tape:
            conv, logits = self.grad_model(xt, training=False)
            selected = tf.gather(logits, target_t, axis=1, batch_dims=1)
            objective = tf.reduce_sum(selected)
        grads = tape.gradient(objective, conv)
        if grads is None:
            raise RuntimeError("Grad-CAM gradients are disconnected from conv3")
        grad_norms = tf.norm(grads, axis=[1, 2]).numpy().astype(np.float32)
        weights = tf.reduce_mean(grads, axis=1, keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(conv * weights, axis=-1, keepdims=True))
        # Explicit rank-4 [batch, height, width, channels] avoids interpreting
        # a batch of CAMs as one unbatched image.
        up = tf.image.resize(cam[..., None], (720, 1), method="bilinear")
        up = tf.squeeze(up, axis=[-1, -2])
        maxima = tf.reduce_max(up, axis=1, keepdims=True)
        normalized = tf.where(maxima > 0, up / tf.maximum(maxima, tf.keras.backend.epsilon()), tf.zeros_like(up))
        result = normalized.numpy().astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("Grad-CAM produced non-finite attributions")
        return result, grad_norms

    def gradcam(self, windows: np.ndarray, target_class: int | Sequence[int]) -> np.ndarray:
        """Return one ReLU, per-window-max-normalized Grad-CAM vector per beat."""
        return self.gradcam_with_diagnostics(windows, target_class)[0]

    def integrated_gradients(self, windows: np.ndarray, target_class: int | Sequence[int],
                             baseline: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Compute trapezoidal IG per window; process windows sequentially to bound memory."""
        x = self._batch(windows)
        targets = self._targets(target_class, len(x))
        b = np.zeros_like(x) if baseline is None else self._batch(baseline)
        if b.shape[0] == 1 and x.shape[0] > 1:
            b = np.repeat(b, x.shape[0], axis=0)
        if b.shape != x.shape:
            raise ValueError("baseline must match windows or contain one broadcastable window")
        attrs_all, delta_all, sum_all, error_all = [], [], [], []
        alphas = tf.linspace(0.0, 1.0, self.ig_steps + 1)
        for i, target in enumerate(targets):
            xi, bi = x[i:i + 1], b[i:i + 1]
            delta = xi - bi
            grad_integral = tf.zeros((720, 1), dtype=tf.float32)
            for start in range(0, self.ig_steps + 1, self.ig_batch_size):
                end = min(start + self.ig_batch_size, self.ig_steps + 1)
                alpha_chunk = alphas[start:end]
                interp = bi[:, None, ...] + alpha_chunk[None, :, None, None] * delta[:, None, ...]
                flat = tf.reshape(interp, (-1, 720, 1))
                target_t = tf.fill((end - start,), tf.cast(target, tf.int32))
                with tf.GradientTape() as tape:
                    tape.watch(flat)
                    logits = self.logit_model(flat, training=False)
                    selected = tf.gather(logits, target_t, axis=1, batch_dims=1)
                    objective = tf.reduce_sum(selected)
                grads = tape.gradient(objective, flat)
                if grads is None:
                    raise RuntimeError("Integrated Gradients are disconnected from the input")
                grads = tf.reshape(grads, (end - start, 720, 1))
                weights = tf.ones((end - start,), dtype=tf.float32)
                weights = tf.tensor_scatter_nd_update(weights, [[0]], [0.5]) if start == 0 else weights
                if end == self.ig_steps + 1:
                    weights = tf.tensor_scatter_nd_update(weights, [[end - start - 1]], [0.5])
                grad_integral += tf.reduce_sum(grads * weights[:, None, None], axis=0)
            avg_grads = (grad_integral / self.ig_steps)[None, ...]
            attrs = tf.convert_to_tensor(delta) * avg_grads
            end_score = self.logit_model(tf.convert_to_tensor(xi), training=False)[0, target]
            base_score = self.logit_model(tf.convert_to_tensor(bi), training=False)[0, target]
            score_delta = end_score - base_score
            attr_sum = tf.reduce_sum(attrs)
            attrs_all.append(tf.squeeze(attrs, axis=-1)[0].numpy())
            delta_all.append(float(score_delta.numpy()))
            sum_all.append(float(attr_sum.numpy()))
            error_all.append(float((attr_sum - score_delta).numpy()))
        output = {"attributions": np.asarray(attrs_all, dtype=np.float32),
                  "score_difference": np.asarray(delta_all, dtype=np.float32),
                  "attribution_sum": np.asarray(sum_all, dtype=np.float32),
                  "completeness_error": np.asarray(error_all, dtype=np.float32)}
        if any(not np.isfinite(v).all() for v in output.values()):
            raise RuntimeError("Integrated Gradients produced non-finite values")
        return output
