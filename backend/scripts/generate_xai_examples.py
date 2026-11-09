#!/usr/bin/env python3
"""Generate real MIT-BIH attribution examples from saved test windows."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.ml.inference import ECGClassifier
from app.ml.explainability import ModelExplainer

CLASSES = ("N", "S", "V", "F")
OUT = ROOT / "docs/xai-examples"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / "results/test_examples_for_xai.npz", allow_pickle=False) as z:
        X, y, pred, prob, record, position = (z[k] for k in ("X", "y", "pred", "prob", "record", "position"))
    model = ECGClassifier()
    explainer = ModelExplainer(model.model, ig_steps=512)
    chosen = []
    for c in range(4):
        candidates = np.flatnonzero((y == c) & (pred == c))
        if not candidates.size:
            raise RuntimeError(f"no correctly classified reference beat for class {CLASSES[c]}")
        candidates = candidates[np.argsort(prob[candidates, c])[::-1]][:100]
        candidate_maps, _ = explainer.gradcam_with_diagnostics(np.asarray(X[candidates], dtype=np.float32), c)
        nonzero = np.flatnonzero(candidate_maps.max(axis=1) > 0)
        if not nonzero.size:
            raise RuntimeError(f"no nonzero ReLU Grad-CAM map found among top reference beats for {CLASSES[c]}")
        chosen.append(int(candidates[int(nonzero[0])]))
    batch = np.asarray(X[chosen], dtype=np.float32)
    targets = np.arange(4, dtype=np.int64)
    fresh_prob = np.asarray(model.model.predict(batch, batch_size=4, verbose=0), dtype=np.float32)
    print("selected real beat batch and probabilities", flush=True)
    cam, grad_norm = explainer.gradcam_with_diagnostics(batch, targets)
    print("Grad-CAM complete", flush=True)
    if not np.isfinite(grad_norm).all() or np.any(grad_norm <= 0):
        raise RuntimeError(f"non-finite or zero Grad-CAM gradients: {grad_norm}")
    ig_zero = explainer.integrated_gradients(batch, targets)
    print("zero-baseline Integrated Gradients complete", flush=True)
    noise_rng = np.random.default_rng(0)
    noise_baseline = noise_rng.normal(0.0, 0.1, size=batch.shape).astype(np.float32)
    ig_noise = explainer.integrated_gradients(batch, targets, baseline=noise_baseline)
    print("noise-baseline Integrated Gradients complete", flush=True)
    sensitivity = np.sum(np.abs(ig_zero["attributions"] - ig_noise["attributions"]), axis=1)
    relative_sensitivity = sensitivity / np.maximum(np.sum(np.abs(ig_zero["attributions"]), axis=1), 1e-12)
    print("attribution diagnostics computed", flush=True)
    rows = []
    time_s = np.arange(720, dtype=np.float32) / 360.0 - 1.0
    for j, c in enumerate(targets):
        print(f"rendering {CLASSES[c]}", flush=True)
        rows.append({"class": CLASSES[int(c)], "record": str(record[chosen[j]]), "sample_position": int(position[chosen[j]]),
                     "probabilities": fresh_prob[j].tolist(), "gradcam_gradient_l2_norm": float(grad_norm[j]),
                     "gradcam_max": float(cam[j].max()), "ig_zero_score_difference": float(ig_zero["score_difference"][j]),
                     "ig_zero_attribution_sum": float(ig_zero["attribution_sum"][j]),
                     "ig_zero_completeness_error": float(ig_zero["completeness_error"][j]),
                     "ig_noise_completeness_error": float(ig_noise["completeness_error"][j]),
                     "ig_zero_vs_noise_l1": float(sensitivity[j]),
                     "ig_zero_vs_noise_relative_l1": float(relative_sensitivity[j])})
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
        axes[0].plot(time_s, batch[j, :, 0], color="#292D2A", linewidth=0.85)
        axes[0].axvline(0, color="#C66B52", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("z-score")
        axes[0].set_title(f"Class {CLASSES[c]} • record {record[chosen[j]]} • sample {position[chosen[j]]}")
        axes[1].fill_between(time_s, 0, cam[j], color="#9881B6", alpha=0.72)
        axes[1].set_ylabel("Grad-CAM")
        axes[2].plot(time_s, ig_zero["attributions"][j], color="#263B32", linewidth=0.85)
        axes[2].axhline(0, color="#777", linewidth=0.5)
        axes[2].set_ylabel("IG attribution")
        axes[2].set_xlabel("Time from annotated R peak (s)")
        for ax in axes:
            ax.grid(alpha=0.18)
        fig.savefig(OUT / f"{CLASSES[c]}.png", dpi=160)
        plt.close(fig)
        print(f"saved {CLASSES[c]}.png", flush=True)
    np.savez_compressed(OUT / "attributions.npz", windows=batch, targets=targets, classes=np.asarray(CLASSES),
                        records=np.asarray([str(record[i]) for i in chosen]),
                        positions=position[chosen], probabilities=fresh_prob, gradcam=cam,
                        gradcam_gradient_l2_norm=grad_norm,
                        integrated_gradients_zero=ig_zero["attributions"],
                        ig_zero_score_difference=ig_zero["score_difference"],
                        ig_zero_attribution_sum=ig_zero["attribution_sum"],
                        ig_zero_completeness_error=ig_zero["completeness_error"],
                        integrated_gradients_noise=ig_noise["attributions"],
                        ig_noise_completeness_error=ig_noise["completeness_error"],
                        noise_baseline=noise_baseline)
    (OUT / "measurements.json").write_text(json.dumps({"source": "results/test_examples_for_xai.npz",
        "attribution_method": "Grad-CAM conv3; Integrated Gradients logits, 512 trapezoidal intervals",
        "zero_baseline": "all zeros in per-window z-scored signal space",
        "sensitivity_baseline": "fixed-seed-0 Gaussian N(0,0.1) per z-scored sample",
        "examples": rows}, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "examples": rows}, indent=2))


if __name__ == "__main__":
    main()
