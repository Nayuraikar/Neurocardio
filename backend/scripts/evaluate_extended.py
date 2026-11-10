"""Extended held-out evaluation of the saved NeuroCardio model.

Nothing is tuned on test records. The only fitted quantity is the temperature
(validation records). Outputs go to results/extended_eval/.

Run: backend/.venv/bin/python backend/scripts/evaluate_extended.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.explainability import ModelExplainer  # noqa: E402
from app.ml.inference import ECGClassifier  # noqa: E402
from app.ml.preprocessing import CLASS_NAMES, FS_HZ, extract_windows, filter_continuous  # noqa: E402
from app.ml.rpeak_detection import detect_r_peaks, match_peaks  # noqa: E402

CFG = json.loads((ROOT / "backend/models/preprocessing_config.json").read_text())
MITDB = ROOT / "results/mitdb"
OUT = ROOT / "results/extended_eval"
OUT.mkdir(parents=True, exist_ok=True)
AAMI = CFG["aami_mapping"]
CIDX = {c: i for i, c in enumerate(CLASS_NAMES)}
TEST_RECS = CFG["split"]["test_records"]
VAL_RECS = CFG["split"]["validation_records"]
SEED = 20240611
K = 4


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- data
def load_record(rec: str):
    r = wfdb.rdrecord(str(MITDB / rec))
    ch = r.sig_name.index("MLII") if "MLII" in r.sig_name else 0
    sig = r.p_signal[:, ch]
    ann = wfdb.rdann(str(MITDB / rec), "atr")
    if int(round(r.fs)) != 360:
        raise ValueError(f"{rec}: fs={r.fs}")
    pk, lab = [], []
    for s, sym in zip(ann.sample, ann.symbol):
        if sym in AAMI:
            pk.append(int(s)); lab.append(CIDX[AAMI[sym]])
    return sig, np.asarray(pk, dtype=np.int64), np.asarray(lab, dtype=np.int64)


def record_windows(rec: str):
    sig, pk, lab = load_record(rec)
    filt = filter_continuous(sig)
    wb = extract_windows(filt, pk)
    keep = np.isin(pk, wb.sample_indices)
    return filt, wb.windows, pk[keep], lab[keep], pk, lab


def build_split(recs):
    W, Y, R = [], [], []
    for rec in recs:
        _, w, _, y, _, _ = record_windows(rec)
        W.append(w); Y.append(y); R.append(np.full(len(y), rec))
    return np.concatenate(W), np.concatenate(Y), np.concatenate(R)


def get_logits(clf, W, bs=512):
    lm = ModelExplainer(clf.model).logit_model
    return np.asarray(lm.predict(W, batch_size=bs, verbose=0), dtype=np.float64)


# ---------------------------------------------------------------- metrics
def cm_of(y, p):
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    return cm


def metrics_from_cm(cm):
    cm = cm.astype(np.float64)
    tot = cm.sum()
    tp = np.diag(cm); fn = cm.sum(1) - tp; fp = cm.sum(0) - tp; tn = tot - tp - fn - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = tp / (tp + fn); ppv = tp / (tp + fp); spec = tn / (tn + fp)
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {"sens": sens, "ppv": ppv, "spec": spec, "f1": f1, "acc": tp.sum() / tot}


def full_metrics(y, p, prob):
    cm = cm_of(y, p)
    m = metrics_from_cm(cm)
    present = [i for i in range(K) if cm[i].sum() > 0]
    out = {
        "confusion_matrix": cm.tolist(),
        "per_class": {c: {"sensitivity": float(m["sens"][i]), "ppv": float(m["ppv"][i]),
                          "specificity": float(m["spec"][i]), "f1": float(m["f1"][i]),
                          "support": int(cm[i].sum())} for i, c in enumerate(CLASS_NAMES)},
        "accuracy": float(m["acc"]),
        "macro_f1": float(np.nanmean(m["f1"][present])),
        "balanced_accuracy": float(np.nanmean(m["sens"][present])),
        "mcc": float(matthews_corrcoef(y, p)),
        "cohen_kappa": float(cohen_kappa_score(y, p)),
    }
    if prob is not None:
        out["pr_auc"] = {c: float(average_precision_score((y == i).astype(int), prob[:, i]))
                         if (y == i).any() else None for i, c in enumerate(CLASS_NAMES)}
    return out


def ece_score(prob, y, n_bins=15):
    conf = prob.max(1); pred = prob.argmax(1); correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            bins.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return float(ece), bins


def calib_metrics(prob, y):
    ece, bins = ece_score(prob, y)
    onehot = np.eye(K)[y]
    return {"ece_15": ece, "brier": float(((prob - onehot) ** 2).sum(1).mean()),
            "nll": float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)).mean())}, bins


# ---------------------------------------------------------------- main
def main():
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    clf = ECGClassifier()
    expl = ModelExplainer(clf.model)
    log("loading test windows")
    Wt, yt, rt = build_split(TEST_RECS)
    log(f"test beats: {len(yt)}")
    Lt = get_logits(clf, Wt)
    Pt = softmax(Lt, axis=1)
    pt = Pt.argmax(1)
    res: dict = {"n_test_beats": int(len(yt)), "seed": SEED}

    # 1. AAMI metrics
    res["aami"] = full_metrics(yt, pt, Pt)

    # 2. calibration + temperature scaling on validation
    log("validation logits / temperature")
    Wv, yv, _ = build_split(VAL_RECS)
    Lv = get_logits(clf, Wv)
    nll = lambda logT: -log_softmax(Lv / np.exp(logT), axis=1)[np.arange(len(yv)), yv].mean()
    T = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
    Pt_ts = softmax(Lt / T, axis=1)
    cal_b, bins_b = calib_metrics(Pt, yt)
    cal_a, bins_a = calib_metrics(Pt_ts, yt)
    res["calibration"] = {"temperature_fit_on_validation": T, "n_validation_beats": int(len(yv)),
                          "test_before": cal_b, "test_after_temperature_scaling": cal_a,
                          "val_nll_before": float(nll(0.0)), "val_nll_after": float(nll(np.log(T)))}
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for a, bins, title, c in ((ax[0], bins_b, f"Before (ECE={cal_b['ece_15']:.3f})", "#c0392b"),
                              (ax[1], bins_a, f"After T={T:.2f} (ECE={cal_a['ece_15']:.3f})", "#2471a3")):
        a.plot([0, 1], [0, 1], "k--", lw=1)
        a.plot([b[0] for b in bins], [b[1] for b in bins], "o-", color=c)
        a.set_title(title); a.set_xlabel("Mean confidence")
    ax[0].set_ylabel("Accuracy"); fig.suptitle("Reliability diagram (test, 15 bins)")
    fig.tight_layout(); fig.savefig(OUT / "reliability.png", dpi=140); plt.close(fig)

    # 3. record-level bootstrap
    log("bootstrap")
    recs = np.array(TEST_RECS)
    rec_cm = {r: cm_of(yt[rt == r], pt[rt == r]) for r in recs}
    stats = {"macro_f1": [], "V_recall": [], "S_recall": [], "F_recall": [], "accuracy": []}
    for _ in range(1000):
        cm = sum(rec_cm[r] for r in rng.choice(recs, len(recs), replace=True))
        m = metrics_from_cm(cm)
        present = cm.sum(1) > 0
        stats["macro_f1"].append(np.nanmean(m["f1"][present]))
        stats["V_recall"].append(m["sens"][2]); stats["S_recall"].append(m["sens"][1])
        stats["F_recall"].append(m["sens"][3]); stats["accuracy"].append(m["acc"])
    res["bootstrap_record_level"] = {
        k: {"point": float(res["aami"][k] if k in ("macro_f1", "accuracy")
                          else res["aami"]["per_class"][k[0]]["sensitivity"]),
            "ci95_low": float(np.nanpercentile(v, 2.5)), "ci95_high": float(np.nanpercentile(v, 97.5)),
            "n_valid_resamples": int(np.isfinite(v).sum())} for k, v in stats.items()}
    res["bootstrap_record_level"]["n_resamples"] = 1000

    # 4. noise robustness (test windows, fixed seeds; model unchanged)
    log("noise robustness")
    def renorm(x):
        x = x - x.mean(1, keepdims=True)
        return (x / (x.std(1, keepdims=True) + 1e-8)).astype(np.float32)
    def run_cond(W):
        P = softmax(get_logits(clf, W), axis=1); p = P.argmax(1)
        m = full_metrics(yt, p, None)
        return {"macro_f1": m["macro_f1"], "V_recall": m["per_class"]["V"]["sensitivity"],
                "S_recall": m["per_class"]["S"]["sensitivity"], "accuracy": m["accuracy"]}
    nrng = np.random.default_rng(SEED + 1)
    noise = {"clean": run_cond(Wt)}
    for snr in (24, 18, 12, 6):
        sd = 10 ** (-snr / 20)  # windows are unit-variance, so noise sd sets SNR directly
        Wn = renorm(Wt[..., 0] + nrng.normal(0, sd, Wt.shape[:2]))[..., None]
        noise[f"snr_{snr}"] = run_cond(Wn)
    t = np.arange(720) / FS_HZ
    n = len(Wt)
    amp = nrng.uniform(0.0, 0.3, (n, 1)); f = nrng.uniform(0.15, 0.5, (n, 1)); ph = nrng.uniform(0, 2 * np.pi, (n, 1))
    Wb = renorm(Wt[..., 0] + amp * np.sin(2 * np.pi * f * t[None] + ph))[..., None]
    noise["baseline_wander"] = run_cond(Wb)
    res["noise_robustness"] = {"conditions": noise,
                               "note": "SNR relative to unit-variance z-scored window; baseline wander amp~U(0,0.3), f~U(0.15,0.5) Hz (training aug max amp 0.1); windows re-z-scored"}
    snrs = [24, 18, 12, 6]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, c in (("macro_f1", "#2471a3"), ("V_recall", "#c0392b")):
        ax.plot(snrs, [noise[f"snr_{s}"][key] for s in snrs], "o-", color=c, label=key)
        ax.axhline(noise["clean"][key], color=c, ls=":", lw=1)
    ax.invert_xaxis(); ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Score"); ax.legend()
    ax.set_title("Noise robustness (dotted = clean)"); fig.tight_layout()
    fig.savefig(OUT / "noise_robustness.png", dpi=140); plt.close(fig)

    # 5. end-to-end with XQRS peaks
    log("end-to-end XQRS")
    e2e_y, e2e_p, e2e_prob = [], [], []
    missed = np.zeros(K, dtype=int); n_ref = np.zeros(K, dtype=int); extra = 0; n_det = 0; per_rec_pk = {}
    for rec in TEST_RECS:
        sig, pk, lab = load_record(rec)
        filt = filter_continuous(sig)
        det = detect_r_peaks(sig)
        matches, un_det, un_ref = match_peaks(pk, det, int(round(0.150 * FS_HZ)))
        n_det += len(det); extra += len(un_det)
        ref_lab = dict(zip(pk.tolist(), lab.tolist()))
        for r_ in lab: n_ref[r_] += 1
        for r_ in un_ref: missed[ref_lab[int(r_)]] += 1
        m_det = np.array([d for d, _ in matches], dtype=np.int64)
        m_lab = np.array([ref_lab[r_] for _, r_ in matches], dtype=np.int64)
        if len(m_det):
            wb = extract_windows(filt, m_det)
            keep = np.isin(m_det, wb.sample_indices)
            P = softmax(get_logits(clf, wb.windows), axis=1)
            e2e_y.append(m_lab[keep]); e2e_p.append(P.argmax(1)); e2e_prob.append(P)
            missed_edge = (~keep).sum()
            if missed_edge:
                for l_ in m_lab[~keep]: missed[l_] += 1
    ey, ep, eP = np.concatenate(e2e_y), np.concatenate(e2e_p), np.concatenate(e2e_prob)
    m_e = full_metrics(ey, ep, eP)
    cm_miss = cm_of(ey, ep).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec_with_miss = np.diag(cm_miss) / (cm_miss.sum(1) + missed)
    res["end_to_end_xqrs"] = {"matched_beat_metrics": m_e, "tolerance_ms": 150,
                              "n_annotated_beats": int(n_ref.sum()), "n_detected_peaks": int(n_det),
                              "missed_peaks_total": int(missed.sum()),
                              "missed_by_class": {c: int(missed[i]) for i, c in enumerate(CLASS_NAMES)},
                              "extra_peaks_total": int(extra),
                              "recall_counting_missed_as_errors": {c: float(rec_with_miss[i]) for i, c in enumerate(CLASS_NAMES)}}

    # 6. PVC burden (annotated-peak predictions)
    rows = []
    for r in recs:
        m = rt == r
        rows.append((str(r), 100 * (yt[m] == 2).mean(), 100 * (pt[m] == 2).mean(), int(m.sum())))
    tv = np.array([x[1] for x in rows]); pv = np.array([x[2] for x in rows])
    res["pvc_burden"] = {"mae_percentage_points": float(np.abs(tv - pv).mean()),
                         "pearson_r": float(np.corrcoef(tv, pv)[0, 1]),
                         "per_record": [{"record": a, "true_V_pct": b, "pred_V_pct": c, "n_beats": d} for a, b, c, d in rows]}
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(tv, pv, s=28, color="#2471a3"); mx = max(tv.max(), pv.max()) * 1.05 + 1
    ax.plot([0, mx], [0, mx], "k--", lw=1)
    for a, b, c, _ in rows:
        if abs(b - c) > 5: ax.annotate(a, (b, c), fontsize=7)
    ax.set_xlabel("True V (%)"); ax.set_ylabel("Predicted V (%)")
    ax.set_title(f"PVC burden per record (MAE {res['pvc_burden']['mae_percentage_points']:.2f} pp)")
    fig.tight_layout(); fig.savefig(OUT / "pvc_burden.png", dpi=140); plt.close(fig)

    # 7. XAI faithfulness
    log("XAI faithfulness")
    idx = np.sort(rng.choice(len(yt), 200, replace=False))
    Wx = Wt[idx]; tgt = pt[idx]; p0 = Pt[idx, tgt]
    cam = expl.gradcam(Wx, tgt)
    cam_tie = cam + rng.uniform(0, 1e-6, cam.shape)  # random tie-break among zero-attribution samples
    fracs = [0, 0.05, 0.10, 0.20, 0.30, 0.50]
    def prob_after(mask):
        Wm = Wx.copy(); Wm[..., 0][mask] = 0.0  # zero == window mean in z-scored space
        return softmax(get_logits(clf, Wm), axis=1)[np.arange(len(idx)), tgt]
    order = np.argsort(-cam_tie, axis=1)
    curves = {"gradcam": [], "random": []}
    for fr in fracs:
        k = int(round(fr * 720))
        m_g = np.zeros((len(idx), 720), bool); m_r = np.zeros_like(m_g)
        if k:
            np.put_along_axis(m_g, order[:, :k], True, axis=1)
            for i in range(len(idx)): m_r[i, rng.choice(720, k, replace=False)] = True
        curves["gradcam"].append(prob_after(m_g)); curves["random"].append(prob_after(m_r))
    drop = {m: [float((p0 - c).mean()) for c in cs] for m, cs in curves.items()}
    meanp = {m: [float(c.mean()) for c in cs] for m, cs in curves.items()}
    # weight randomization (full re-initialisation) on the first 20 sampled beats
    rand_model = tf_clone_random(clf.model)
    rexp = ModelExplainer(rand_model)
    cam_r = rexp.gradcam(Wx[:20], tgt[:20])
    rho = []
    for a, b in zip(cam[:20], cam_r):
        rho.append(float(spearmanr(a, b)[0]) if a.std() > 0 and b.std() > 0 else float("nan"))
    res["xai_faithfulness"] = {
        "n_beats": 200, "target": "predicted class", "fractions_deleted": fracs,
        "mean_pred_class_prob_gradcam_deletion": meanp["gradcam"],
        "mean_pred_class_prob_random_deletion": meanp["random"],
        "mean_prob_drop_gradcam": drop["gradcam"], "mean_prob_drop_random": drop["random"],
        "mean_drop_over_fractions_gradcam": float(np.mean(drop["gradcam"][1:])),
        "mean_drop_over_fractions_random": float(np.mean(drop["random"][1:])),
        "clean_mean_pred_prob": float(p0.mean()),
        "weight_randomization": {"n_beats": 20, "spearman_orig_vs_randomized_mean": float(np.nanmean(rho)),
                                 "spearman_per_beat": rho,
                                 "n_degenerate_maps": int(np.isnan(rho).sum()),
                                 "note": "all weights re-initialised; low |rho| => explanation depends on learned weights"}}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([f * 100 for f in fracs], meanp["gradcam"], "o-", color="#c0392b", label="Grad-CAM top-k")
    ax.plot([f * 100 for f in fracs], meanp["random"], "s-", color="#7f8c8d", label="Random k")
    ax.set_xlabel("Samples zeroed (%)"); ax.set_ylabel("Mean predicted-class probability")
    ax.set_title("Deletion curve (200 beats)"); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "xai_deletion.png", dpi=140); plt.close(fig)

    (OUT / "extended_eval.json").write_text(json.dumps(res, indent=2))
    write_summary(res)
    log("done")


def tf_clone_random(model):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    return tf.keras.models.clone_model(model)  # fresh random initialisation, same architecture/names


def write_summary(res):
    base = json.loads((ROOT / "results/test_metrics.json").read_text())["selected_test_metrics"]
    a = res["aami"]; rows = []
    def add(name, old, new):
        d = "" if old is None else f"{new - old:+.4f}"
        rows.append(f"| {name} | {'' if old is None else f'{old:.4f}'} | {new:.4f} | {d} |")
    add("Accuracy", base["accuracy"], a["accuracy"])
    add("Macro-F1", base["macro_f1"], a["macro_f1"])
    for c in CLASS_NAMES:
        add(f"Sensitivity {c}", base["sensitivity"][c], a["per_class"][c]["sensitivity"])
    for c in CLASS_NAMES:
        add(f"Specificity {c}", base["specificity"][c], a["per_class"][c]["specificity"])
    for c in CLASS_NAMES:
        add(f"PR-AUC {c}", base["auprc_per_class"][c], a["pr_auc"][c])
    add("Macro PR-AUC", base["macro_auprc"], float(np.mean(list(a["pr_auc"].values()))))
    add("Balanced accuracy", None, a["balanced_accuracy"]); add("MCC", None, a["mcc"]); add("Cohen's kappa", None, a["cohen_kappa"])
    for c in CLASS_NAMES:
        add(f"PPV {c}", None, a["per_class"][c]["ppv"]); add(f"F1 {c}", None, a["per_class"][c]["f1"])
    cal = res["calibration"]
    L = ["# Extended evaluation (held-out test records)", "",
         f"Test beats: {res['n_test_beats']}. Seed {res['seed']}. Only the temperature was fit (validation).", "",
         "## Comparison with test_metrics.json", "", "| Metric | Existing | Recomputed | Diff |", "|---|---|---|---|", *rows, "",
         "## Calibration", "", "| | ECE(15) | Brier | NLL |", "|---|---|---|---|",
         f"| Test, raw | {cal['test_before']['ece_15']:.4f} | {cal['test_before']['brier']:.4f} | {cal['test_before']['nll']:.4f} |",
         f"| Test, T={cal['temperature_fit_on_validation']:.3f} | {cal['test_after_temperature_scaling']['ece_15']:.4f} | {cal['test_after_temperature_scaling']['brier']:.4f} | {cal['test_after_temperature_scaling']['nll']:.4f} |", "",
         "## Record-level bootstrap (1000 resamples, 95% CI)", "", "| Metric | Point | 95% CI |", "|---|---|---|"]
    for k, v in res["bootstrap_record_level"].items():
        if isinstance(v, dict):
            L.append(f"| {k} | {v['point']:.4f} | [{v['ci95_low']:.4f}, {v['ci95_high']:.4f}] |")
    L += ["", "## Noise robustness", "", "| Condition | Macro-F1 | V recall | S recall | Accuracy |", "|---|---|---|---|---|"]
    for k, v in res["noise_robustness"]["conditions"].items():
        L.append(f"| {k} | {v['macro_f1']:.4f} | {v['V_recall']:.4f} | {v['S_recall']:.4f} | {v['accuracy']:.4f} |")
    e = res["end_to_end_xqrs"]; em = e["matched_beat_metrics"]
    L += ["", "## End-to-end (XQRS peaks, 150 ms match)", "",
          f"Annotated {e['n_annotated_beats']}, detected {e['n_detected_peaks']}, missed {e['missed_peaks_total']} "
          f"({e['missed_by_class']}), extra {e['extra_peaks_total']}.", "",
          f"Matched-beat accuracy {em['accuracy']:.4f}, macro-F1 {em['macro_f1']:.4f}. Recall with misses counted as errors: "
          + ", ".join(f"{c} {v:.4f}" for c, v in e["recall_counting_missed_as_errors"].items()) + ".", "",
          "## PVC burden", "", f"MAE {res['pvc_burden']['mae_percentage_points']:.2f} percentage points; Pearson r {res['pvc_burden']['pearson_r']:.3f}.", "",
          "## XAI faithfulness", "", "| % zeroed | Grad-CAM mean P | Random mean P | Grad-CAM drop | Random drop |", "|---|---|---|---|---|"]
    x = res["xai_faithfulness"]
    for i, f in enumerate(x["fractions_deleted"]):
        L.append(f"| {f*100:.0f} | {x['mean_pred_class_prob_gradcam_deletion'][i]:.4f} | {x['mean_pred_class_prob_random_deletion'][i]:.4f} | "
                 f"{x['mean_prob_drop_gradcam'][i]:.4f} | {x['mean_prob_drop_random'][i]:.4f} |")
    w = x["weight_randomization"]
    L += ["", f"Mean drop (5-50%): Grad-CAM {x['mean_drop_over_fractions_gradcam']:.4f} vs random {x['mean_drop_over_fractions_random']:.4f}. "
          f"Weight randomization: mean Spearman {w['spearman_orig_vs_randomized_mean']:.3f} (n=20, {w['n_degenerate_maps']} degenerate maps).", "",
          "Figures: reliability.png, noise_robustness.png, pvc_burden.png, xai_deletion.png."]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
