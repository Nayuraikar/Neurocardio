"""Scoring shared by experiments.py and final_test.py."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

CLASSES = ["N", "S", "V", "F"]
GRID = np.linspace(-4, 4, 33)                          # log2 probability scales; N is fixed at 1


def cm(y, pred):
    return np.bincount(4 * np.asarray(y, np.int64) + np.asarray(pred, np.int64), minlength=16).reshape(4, 4).astype(np.float64)


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


# ------------------------------------------------------------------ calibration
def temperature(probs, y) -> float:
    lp = np.log(np.clip(probs, 1e-12, 1))

    def nll(t):
        z = lp / np.exp(t)
        z -= z.max(1, keepdims=True)
        return -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def scaled(probs, t):
    z = np.log(np.clip(probs, 1e-12, 1)) / t
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def calibration(probs, y, bins=15) -> dict:
    conf, ok = probs.max(1), (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = sum(m.mean() * abs(ok[m].mean() - conf[m].mean())
              for lo, hi in zip(edges[:-1], edges[1:]) if (m := (conf > lo) & (conf <= hi)).any())
    return {"ece_15": float(ece), "brier": float(((probs - np.eye(4)[y]) ** 2).sum(1).mean()),
            "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())}


# ------------------------------------------------------------------ selective review
def reviewed_mask(confidence, budget: float) -> np.ndarray:
    """The `budget` share of beats with the lowest confidence (ties broken by position)."""
    mask = np.zeros(len(confidence), bool)
    mask[np.argsort(confidence, kind="stable")[:int(round(budget * len(confidence)))]] = True
    return mask


def after_review(y, pred, mask) -> np.ndarray:
    """Predictions if a PERFECT reviewer corrects every reviewed beat. An upper bound, not a measured workflow."""
    return np.where(mask, y, pred)
