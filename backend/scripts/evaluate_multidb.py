"""Validation-only evaluation and model selection for the multi-database study.

  python -u evaluate_multidb.py [--out DIR]

Reads each run's validation_predictions.npz. Never opens a test file.
Writes <out>/analysis/multidb/: validation_report.{json,md}, selection.json (everything final_test.py needs, frozen),
per_patient_validation.csv, risk_coverage.{csv,png}.

Thresholds and temperature are tuned on validation, so scoring them on the same beats is optimistic. Every
"cross-fitted" number fits them on four patient folds and applies them to the fifth. The frozen values used on the
test sets are fitted on all validation beats. Validation also drove early stopping, so even cross-fitted validation
numbers are optimistic; only the test results are unbiased.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_common as A  # noqa: E402
import models_multidb as M  # noqa: E402
import train as T  # noqa: E402
import train_multidb as TM  # noqa: E402

CLASSES = A.CLASSES
BUDGETS = (0.05, 0.10)
FOLDS, RESAMPLES, BOOT_SEED = 5, 2000, 20241127
STACK = "stacked_ensemble"
COVERAGE = np.round(np.arange(1.0, 0.495, -0.01), 2)
# pre-declared paired comparisons: each step of the ablation against the step before it
ABLATION = [("rr_fusion", "cnn_bilstm"), ("rr_fusion_attn", "rr_fusion"), ("rr_fusion_attn_shead", "rr_fusion_attn")]


# ------------------------------------------------------------------ fitting on validation
def patient_folds(patient: np.ndarray, y: np.ndarray, k: int = FOLDS) -> np.ndarray:
    """Fold id per beat; whole patients, S-heavy patients dealt first so every fold has some S."""
    order = sorted(set(patient), key=lambda p: (-int(((patient == p) & (y == 1)).sum()), -int((patient == p).sum()), p))
    load, assign = np.zeros((k, 2)), {}
    for p in order:
        m = patient == p
        f = min(range(k), key=lambda j: (load[j, 0], load[j, 1], j))
        assign[p] = f
        load[f] += (int((y[m] == 1).sum()), int(m.sum()))
    return np.array([assign[p] for p in patient])


def crossfit(probs, y, folds):
    """-> (thresholded predictions, temperature-scaled probabilities), each fitted without the scored patient's fold."""
    pred, cal = np.empty(len(y), np.int64), np.empty_like(probs)
    for f in np.unique(folds):
        fit, use = folds != f, folds == f
        pred[use] = (probs[use] * A.fit_thresholds(probs[fit], y[fit])).argmax(1)
        cal[use] = A.scaled(probs[use], A.temperature(probs[fit], y[fit]))
    return pred, cal


def stack_features(member_probs: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.log(np.clip(p, 1e-7, 1)) for p in member_probs], axis=1)


def fit_stacker(X, y) -> dict:
    weights = T.class_weights("sqrt_weighted_ce", np.bincount(y, minlength=4).astype(np.float64))
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(X, y, sample_weight=weights[y])
    assert list(clf.classes_) == [0, 1, 2, 3], "validation lacks a class"
    return {"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()}


def apply_stacker(model: dict, X) -> np.ndarray:
    z = X @ np.array(model["coef"]).T + np.array(model["intercept"])
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


# ------------------------------------------------------------------ scoring (shared with final_test.py)
def boot_metrics() -> dict:
    out = {"macro_f1": lambda m: m["macro_f1"], "accuracy": lambda m: m["accuracy"]}
    for i, c in enumerate(CLASSES):
        for k in ("precision", "recall", "f1"):
            out[f"{c}_{k}"] = lambda m, i=i, k=k: m[k][i]
    return out


def bootstrap_ci(cms_by_seed: list[np.ndarray], keys=None) -> dict:
    """Patient bootstrap. cms_by_seed: one (patients,4,4) array per seed; the statistic is the mean over seeds."""
    metrics = boot_metrics()
    keys = keys or list(metrics)
    n = cms_by_seed[0].shape[0]
    picks = np.random.default_rng(BOOT_SEED).integers(0, n, (RESAMPLES, n))
    vals = {k: np.empty(RESAMPLES) for k in keys}
    for j, idx in enumerate(picks):
        ms = [A.from_cm(c[idx].sum(0)) for c in cms_by_seed]
        for k in keys:
            vals[k][j] = np.mean([metrics[k](m) for m in ms])
    return {k: [float(v) for v in np.percentile(x, [2.5, 97.5])] for k, x in vals.items()}


def mean_sd(values) -> dict:
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def review_stats(y, pred, conf, budget, mask=None) -> dict:
    mask = A.reviewed_mask(conf, budget) if mask is None else mask
    after = A.after_review(y, pred, mask)
    out = {"reviewed_share": float(mask.mean()),
           "errors_sent_to_review": float((mask & (pred != y)).sum() / max((pred != y).sum(), 1))}
    for i, c in ((1, "S"), (3, "F"), (2, "V")):
        total = (y == i).sum()
        if not total:
            out[c] = None
            continue
        model_recall = float(((pred == i) & (y == i)).sum() / total)
        out[c] = {"model_recall_all_beats": model_recall,
                  "recall_on_unreviewed": float(((pred == i) & (y == i) & ~mask).sum() / max(((y == i) & ~mask).sum(), 1)),
                  "recall_after_perfect_review": float(((after == i) & (y == i)).sum() / total),
                  "recall_after_random_review": model_recall + float(mask.mean()) * (1 - model_recall),
                  "missed_beats_sent_to_review": float((mask & (y == i) & (pred != i)).sum() / max(((y == i) & (pred != i)).sum(), 1))}
    return out


def average(dicts: list):
    """Element-wise mean of equally shaped nested dicts of numbers (None stays None)."""
    first = dicts[0]
    if first is None:
        return None
    if isinstance(first, dict):
        return {k: average([d[k] for d in dicts]) for k in first}
    return float(np.mean(dicts))


def per_patient(y, preds: list, patient) -> list[dict]:
    rows = []
    for p in sorted(set(patient)):
        m = patient == p
        cms = np.mean([A.cm(y[m], pred[m]) for pred in preds], axis=0)       # mean over seeds
        row = {"patient": p, "beats": int(m.sum())}
        for i, c in ((1, "S"), (3, "F")):
            support = int(cms[i].sum())
            row[c] = {"support": support, "true_positive": float(cms[i, i]), "false_positive": float(cms[:, i].sum() - cms[i, i]),
                      "recall": float(cms[i, i] / support) if support else None,
                      "precision": float(cms[i, i] / cms[:, i].sum()) if cms[:, i].sum() else None}
        rows.append(row)
    return rows


def summarize(y, patient, db, preds: list, raw: list, calibrated: list, with_ci: bool = True) -> dict:
    """preds / raw / calibrated: one array per seed. Metrics are per seed, reported as mean and SD over seeds."""
    reports = [A.report(y, p) for p in preds]
    order = sorted(set(patient))
    out = {"n_beats": int(len(y)), "n_patients": len(order), "seeds": len(preds),
           "macro_f1": mean_sd([r["macro_f1"] for r in reports]), "accuracy": mean_sd([r["accuracy"] for r in reports]),
           "per_class": {c: {**{m: mean_sd([r["per_class"][c][m] for r in reports]) for m in ("precision", "recall", "f1")},
                             "support": reports[0]["per_class"][c]["support"]} for c in CLASSES},
           "confusion_matrix_mean": np.mean([r["confusion_matrix"] for r in reports], axis=0).round(1).tolist(),
           "per_database": {}, "per_patient": per_patient(y, preds, patient),
           "calibration_raw": average([A.calibration(p, y) for p in raw]),
           "calibration_temperature_scaled": average([A.calibration(p, y) for p in calibrated]),
           "review": {str(b): average([review_stats(y, p, c.max(1), b) for p, c in zip(preds, calibrated)]) for b in BUDGETS}}
    for name in sorted(set(db)):
        m = db == name
        sub = [A.report(y[m], p[m]) for p in preds]
        out["per_database"][name] = {"beats": int(m.sum()), "macro_f1": mean_sd([r["macro_f1"] for r in sub]),
                                     "accuracy": mean_sd([r["accuracy"] for r in sub]),
                                     "per_class": {c: {**{k: mean_sd([r["per_class"][c][k] for r in sub]) for k in ("precision", "recall", "f1")},
                                                       "support": sub[0]["per_class"][c]["support"]} for c in CLASSES}}
    if with_ci:
        cms = [A.record_cms(y, p, patient, order) for p in preds]
        out["bootstrap_ci95_patients"] = bootstrap_ci(cms)
        for b in BUDGETS:
            after = [A.record_cms(y, A.after_review(y, p, A.reviewed_mask(c.max(1), b)), patient, order) for p, c in zip(preds, calibrated)]
            ci = bootstrap_ci(after, ["S_recall", "F_recall"])
            out["bootstrap_ci95_patients"].update({f"{k}_after_perfect_review_{int(b * 100)}pct": v for k, v in ci.items()})
    return out


def risk_coverage(y, preds, calibrated) -> list[dict]:
    rows = []
    for cov in COVERAGE:
        vals = []
        for p, c in zip(preds, calibrated):
            mask = A.reviewed_mask(c.max(1), 1 - cov)
            st = review_stats(y, p, c.max(1), 0, mask)
            vals.append({"risk_on_unreviewed": float((p != y)[~mask].mean()),
                         **{f"{k}_{m}": st[k][m] if st[k] else float("nan") for k in ("S", "F")
                            for m in ("recall_on_unreviewed", "recall_after_perfect_review")}})
        rows.append({"coverage": float(cov), **average(vals)})
    return rows


def plot_curves(path: Path, curves: dict, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: curve plot skipped (the CSV holds the data)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [("risk_on_unreviewed", "error rate on unreviewed beats", lambda c: c),
              ("S_recall_after_perfect_review", "S recall after perfect review", lambda c: 1 - c),
              ("F_recall_after_perfect_review", "F recall after perfect review", lambda c: 1 - c)]
    for ax, (key, label, x) in zip(axes, panels):
        for i, (name, rows) in enumerate(curves.items()):
            ax.plot([x(r["coverage"]) for r in rows], [r[key] for r in rows], label=name,
                    color=plt.cm.tab10(i % 10), linestyle="-" if i < 5 else "--")
        ax.set_xlabel("coverage" if key.startswith("risk") else "share of beats reviewed")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fmt(s: dict) -> str:
    return f"{s['mean']:.3f}" + ("" if s["sd"] is None else f"±{s['sd']:.3f}")


def table(rows: dict) -> list[str]:
    L = ["| model | macro-F1 | " + " | ".join(f"{c} P | {c} R | {c} F1" for c in CLASSES) + " | accuracy |", "|" + "---|" * 15]
    for name, e in rows.items():
        cells = " | ".join(f"{fmt(e['per_class'][c]['precision'])} | {fmt(e['per_class'][c]['recall'])} | {fmt(e['per_class'][c]['f1'])}" for c in CLASSES)
        L.append(f"| {name} | {fmt(e['macro_f1'])} | {cells} | {fmt(e['accuracy'])} |")
    return L


def comparison_line(title: str, v: dict) -> str:
    parts = []
    for k in ("S_f1", "F_f1", "macro_f1"):
        x = v[k]
        parts.append(f"{k} {x['gain']:+.3f} [{x['bootstrap_ci95'][0]:+.3f}, {x['bootstrap_ci95'][1]:+.3f}] "
                     + ("meaningful" if x["meaningful"] else "not meaningful"))
    return f"- {title}: " + "; ".join(parts)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="evaluate smoke runs (<out> must be the smoke folder)")
    T.add_path_args(ap)
    a = ap.parse_args()
    env = T.Env(a.data, a.splits, a.out, a.model_config, a.allow_cpu)
    split, split_sha = TM.load_split(env)
    seeds = TM.SEEDS[:2] if a.smoke else TM.SEEDS
    tag = "__smoke" if a.smoke else ""

    registry, probs, members = {}, {}, {}
    y = patient = db = record = None
    for name in M.ALL_MODELS:
        for s in seeds:
            rid = TM.run_id(name, s, tag)
            entry = TM.completed(env, rid)
            if not entry:
                sys.exit(f"missing run {rid}: finish run_multidb.py first")
            fp = entry["fingerprint"]
            assert fp["split_sha256"] == split_sha and fp["data_sha256"] == split["data_sha256"][TM.DEV_FILE], f"{rid} used another split or data"
            with np.load(env.runs / rid / "validation_predictions.npz", allow_pickle=False) as z:
                p = z["probs"].astype(np.float64)
                if y is None:
                    y, patient, db, record = z["y"], z["patient"].astype(str), z["db"].astype(str), z["record"].astype(str)
                assert np.array_equal(y, z["y"]) and np.array_equal(record, z["record"].astype(str)), f"{rid}: validation order differs"
            probs[name, s] = p / p.sum(1, keepdims=True)
            registry[name, s] = entry
            members.setdefault(name, {})[str(s)] = rid
    assert set(record) <= set(split["validation"]), "predictions hold a non-validation record"
    folds = patient_folds(patient, y)

    # stacked ensemble: one stacker per seed over all base models; validation predictions are cross-fitted over patients
    stackers = {}
    for s in seeds:
        X = stack_features([probs[n, s] for n in M.ALL_MODELS])
        oof = np.empty((len(y), 4))
        for f in np.unique(folds):
            oof[folds == f] = apply_stacker(fit_stacker(X[folds != f], y[folds != f]), X[folds == f])
        probs[STACK, s] = oof
        stackers[str(s)] = fit_stacker(X, y)
    print("stacker fitted", flush=True)

    results, selection, curves = {}, {}, {}
    for name in [*M.ALL_MODELS, STACK]:
        argmax = [probs[name, s].argmax(1) for s in seeds]
        cross = [crossfit(probs[name, s], y, folds) for s in seeds]
        thr, cal = [c[0] for c in cross], [c[1] for c in cross]
        macro = lambda preds: float(np.mean([A.from_cm(A.cm(y, p))["macro_f1"] for p in preds]))
        rule = "thresholds" if macro(thr) > macro(argmax) else "argmax"
        chosen = thr if rule == "thresholds" else argmax
        raw = [probs[name, s] for s in seeds]
        frozen = {}
        for s in seeds:
            t = A.temperature(probs[name, s], y)
            conf = A.scaled(probs[name, s], t).max(1)
            frozen[str(s)] = {"temperature": t,
                              "thresholds": A.fit_thresholds(probs[name, s], y).tolist() if rule == "thresholds" else [1.0] * 4,
                              "review_confidence_cutoffs": {str(b): float(np.quantile(conf, b)) for b in BUDGETS}}
        selection[name] = {"rule": rule, "score": macro(chosen), "macro_f1_argmax": macro(argmax),
                           "macro_f1_thresholds_crossfit": macro(thr),
                           "macro_f1_thresholds_in_sample_optimistic": macro(
                               [(probs[name, s] * A.fit_thresholds(probs[name, s], y)).argmax(1) for s in seeds]),
                           "frozen": frozen}
        results[name] = {"rule": rule, "chosen": summarize(y, patient, db, chosen, raw, cal),
                         "argmax": summarize(y, patient, db, argmax, raw, cal, with_ci=False), "_preds": chosen}
        curves[name] = risk_coverage(y, chosen, cal)
        print(f"{name:22s} argmax {macro(argmax):.4f} | cross-fitted thresholds {macro(thr):.4f} | rule {rule}", flush=True)

    primary = max(selection, key=lambda n: selection[n]["score"])
    order = sorted(set(patient))
    cms = {n: [A.record_cms(y, p, patient, order) for p in results[n]["_preds"]] for n in results}
    pairs = [*ABLATION, *([(primary, "cnn_bilstm")] if primary != "cnn_bilstm" else [])]
    comparisons = {f"{new} vs {base}": A.compare(cms[new], cms[base], RESAMPLES) for new, base in pairs}
    for r in results.values():
        r.pop("_preds")

    parameters = {n: {k: registry[n, seeds[0]].get(k) for k in
                      ("parameter_count", "trainable_parameter_count", "tree_count", "leaf_count", "feature_count", "best_iterations")
                      if registry[n, seeds[0]].get(k) is not None} for n in M.ALL_MODELS}
    out = env.out / "analysis/multidb"
    out.mkdir(parents=True, exist_ok=True)
    counts = np.bincount(y, minlength=4)
    report = {"smoke": a.smoke, "split": TM.SPLIT, "seeds": seeds, "validation_counts": dict(zip(CLASSES, counts.tolist())),
              "majority_class": A.report(y, np.zeros(len(y), int)), "parameters": parameters, "results": results,
              "ablation_comparisons": comparisons, "primary": primary, "created_at": T.now()}
    T.write_json(out / "validation_report.json", report)
    T.write_json(out / "selection.json", {
        "smoke": a.smoke, "primary": primary, "selection_rule": "highest mean validation macro-F1 over seeds; each candidate uses "
        "cross-fitted thresholds only if they beat argmax", "seeds": seeds, "candidates": selection, "members": members,
        "stack_member_order": M.ALL_MODELS, "stackers": stackers, "split_sha256": split_sha,
        "data_sha256": split["data_sha256"], "protocol": TM.PROTOCOL, "created_at": T.now()})

    with (out / "per_patient_validation.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "patient", "beats"] + [f"{c}_{k}" for c in "SF" for k in ("support", "tp", "fp", "recall", "precision")])
        for name, r in results.items():
            for row in r["chosen"]["per_patient"]:
                w.writerow([name, row["patient"], row["beats"]] + [row[c][k] for c in "SF" for k in
                           ("support", "true_positive", "false_positive", "recall", "precision")])
    with (out / "risk_coverage.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", *curves[primary][0]])
        w.writeheader()
        for name, rows in curves.items():
            w.writerows({"model": name, **row} for row in rows)
    plot_curves(out / "risk_coverage.png", curves, "Validation (cross-fitted temperature and thresholds)" + (" - SMOKE" if a.smoke else ""))

    P = results[primary]["chosen"]
    L = ["# Validation report (multidb_v1)" + (" - SMOKE, NOT RESULTS" if a.smoke else ""), "",
         "Validation drove early stopping and selection, so these numbers are optimistic. Test sets were not opened.",
         f"Beats: " + ", ".join(f"{c} {n:,}" for c, n in zip(CLASSES, counts)) + f"; patients {len(order)}; seeds {seeds}.", "",
         "## Chosen decision rule per model (thresholds are cross-fitted over patients)", *table({n: r["chosen"] for n, r in results.items()}),
         "", "## Raw argmax", *table({n: r["argmax"] for n, r in results.items()}),
         "", "Majority class (always N): accuracy " f"{report['majority_class']['accuracy']:.3f}, macro-F1 {report['majority_class']['macro_f1']:.3f}.",
         "", "## Selection", "| model | rule | argmax | thresholds cross-fitted | thresholds in-sample (optimistic) |", "|---|---|---|---|---|"]
    L += [f"| {n} | {s['rule']} | {s['macro_f1_argmax']:.4f} | {s['macro_f1_thresholds_crossfit']:.4f} | "
          f"{s['macro_f1_thresholds_in_sample_optimistic']:.4f} |" for n, s in selection.items()]
    L += ["", f"**Primary model: {primary}** (rule: {selection[primary]['rule']}). Choosing the best of {len(selection)} candidates is itself optimistic.",
          "", "## Pre-declared paired comparisons (patient bootstrap; rule in docs/multidb-protocol.md)"]
    L += [comparison_line(k, v) for k, v in comparisons.items()]
    L += ["", "## Parameters", "| model | parameters | trainable | trees | leaves | features |", "|---|---|---|---|---|---|"]
    L += [f"| {n} | " + " | ".join(str(p.get(k, "-")) for k in
          ("parameter_count", "trainable_parameter_count", "tree_count", "leaf_count", "feature_count")) + " |" for n, p in parameters.items()]
    L += ["", "## Calibration and selective review (cross-fitted temperature)",
          "Review recall assumes a reviewer who corrects every reviewed beat. It is an upper bound; compare with random review.",
          "| model | ECE raw | ECE scaled | " + " | ".join(f"S recall {int(b * 100)}% (random) | F recall {int(b * 100)}% (random)" for b in BUDGETS) + " |",
          "|" + "---|" * 7]
    for n, r in results.items():
        e = r["chosen"]
        cell = lambda b, c: (f"{e['review'][str(b)][c]['recall_after_perfect_review']:.3f} ({e['review'][str(b)][c]['recall_after_random_review']:.3f})"
                             if e["review"][str(b)][c] else "-")
        L.append(f"| {n} | {e['calibration_raw']['ece_15']:.4f} | {e['calibration_temperature_scaled']['ece_15']:.4f} | "
                 + " | ".join(cell(b, c) for b in BUDGETS for c in ("S", "F")) + " |")
    L += ["", f"## Primary model: 95% CI over {P['n_patients']} patients"]
    L += [f"- {k}: [{v[0]:.3f}, {v[1]:.3f}]" for k, v in P["bootstrap_ci95_patients"].items() if k.startswith(("macro", "S_", "F_"))]
    L += ["", "## Primary model by database", *table(P["per_database"])]
    L += ["", "## Primary model: per-patient S and F (mean over seeds)", "| patient | S support | S recall | S false pos | F support | F recall | F false pos |",
          "|---|---|---|---|---|---|---|"]
    num = lambda v: "-" if v is None else f"{v:.3f}"
    L += [f"| {r['patient']} | {r['S']['support']} | {num(r['S']['recall'])} | {r['S']['false_positive']:.0f} | {r['F']['support']} | "
          f"{num(r['F']['recall'])} | {r['F']['false_positive']:.0f} |" for r in P["per_patient"]
          if r["S"]["support"] or r["F"]["support"] or r["S"]["false_positive"] >= 1 or r["F"]["false_positive"] >= 1]
    (out / "validation_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:30]), flush=True)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
