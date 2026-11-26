"""Scoring shared by experiments.py and final_test.py."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix

CLASSES = ["N", "S", "V", "F"]
GRID = np.linspace(-4, 4, 33)                          # log2 probability scales; N is fixed at 1


def cm(y, pred):
    return confusion_matrix(y, pred, labels=range(4)).astype(np.float64)


def from_cm(c):
    tp = np.diag(c)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.nan_to_num(tp / c.sum(0))
        r = np.nan_to_num(tp / c.sum(1))
        f = np.nan_to_num(2 * tp / (c.sum(0) + c.sum(1)))
    return {"precision": p, "recall": r, "f1": f, "macro_f1": float(f.mean()), "accuracy": float(tp.sum() / c.sum()),
            "majority_baseline_accuracy": float(c.sum(1).max() / c.sum())}


def report(y, pred) -> dict:
    c = cm(y, pred)
    m = from_cm(c)
    return {"n": int(c.sum()), "accuracy": m["accuracy"], "majority_baseline_accuracy": m["majority_baseline_accuracy"],
            "macro_f1": m["macro_f1"], "confusion_matrix": c.astype(int).tolist(),
            "per_class": {k: {"precision": float(m["precision"][i]), "recall": float(m["recall"][i]), "f1": float(m["f1"][i]),
                              "support": int(c[i].sum())} for i, k in enumerate(CLASSES)}}


def fit_thresholds(probs, y) -> np.ndarray:
    """Per-class probability scales maximising macro-F1 on the given VALIDATION data (coordinate search)."""
    logs = np.zeros(4)
    score = lambda lg: from_cm(cm(y, (probs * 2.0 ** lg).argmax(1)))["macro_f1"]
    best = score(logs)
    for _ in range(3):
        for k in (1, 2, 3):
            for g in sorted(GRID, key=abs):             # ties keep the scale closest to 1
                trial = logs.copy()
                trial[k] = g
                s = score(trial)
                if s > best + 1e-12:
                    best, logs = s, trial
    return 2.0 ** logs


def record_cms(y, pred, record, order):
    return np.stack([cm(y[record == r], pred[record == r]) if (record == r).any() else np.zeros((4, 4)) for r in order])


METRICS = {"macro_f1": lambda m: m["macro_f1"], "S_f1": lambda m: m["f1"][1], "F_f1": lambda m: m["f1"][3],
           "S_recall": lambda m: m["recall"][1], "F_recall": lambda m: m["recall"][3]}


def compare(new_cms: list, base_cms: list, resamples: int = 2000, seed: int = 20241127) -> dict:
    """new_cms/base_cms: one (records,4,4) array per seed, same record order.
    Applies the pre-registered rule: gain > 2 x larger seed SD, record bootstrap 95% CI above 0,
    and gain still positive with any single record left out."""
    rng = np.random.default_rng(seed)
    n = new_cms[0].shape[0]
    picks = rng.integers(0, n, (resamples, n))
    out = {}
    for name, fn in METRICS.items():
        val = lambda cms, idx: float(np.mean([fn(from_cm(c[idx].sum(0))) for c in cms]))
        per_seed = lambda cms: np.array([fn(from_cm(c.sum(0))) for c in cms])
        a, b = per_seed(new_cms), per_seed(base_cms)
        gain = float(a.mean() - b.mean())
        boot = np.array([val(new_cms, i) - val(base_cms, i) for i in picks])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        loro = [val(new_cms, np.delete(np.arange(n), i)) - val(base_cms, np.delete(np.arange(n), i)) for i in range(n)]
        sd = max([float(x.std(ddof=1)) for x in (a, b) if len(x) > 1] or [float("nan")])
        passes = {"gain_exceeds_2sd": bool(gain > 2 * sd) if np.isfinite(sd) else None,
                  "bootstrap_ci_above_zero": bool(lo > 0), "survives_leave_one_record_out": bool(min(loro) > 0)}
        out[name] = {"gain": gain, "seed_sd_larger": None if not np.isfinite(sd) else sd, "bootstrap_ci95": [float(lo), float(hi)],
                     "leave_one_record_out_min_gain": float(min(loro)), **passes,
                     "meaningful": bool(all(v for v in passes.values() if v is not None))}
    return out
