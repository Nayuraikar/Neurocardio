"""Diagnostics for noise + XAI sections of evaluate_extended.py. Writes results/extended_eval/diagnostics.md"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_extended as E  # noqa: E402
from app.ml.preprocessing import extract_windows, filter_continuous  # noqa: E402

clf = E.ECGClassifier(); expl = E.ModelExplainer(clf.model)
Wt, yt, rt = E.build_split(E.TEST_RECS)
SEED = E.SEED
L = ["# Noise / XAI diagnostics", ""]

def renorm(x):
    x = x - x.mean(1, keepdims=True)
    return (x / (x.std(1, keepdims=True) + 1e-8)).astype(np.float32)

def pred(W):
    return softmax(E.get_logits(clf, W), axis=1).argmax(1)

def met(p):
    m = E.full_metrics(yt, p, None)
    return m["macro_f1"], m["per_class"]["V"]["sensitivity"], m["per_class"]["S"]["sensitivity"], m["accuracy"]

def post(snr, seed):
    sd = 10 ** (-snr / 20)
    r = np.random.default_rng(seed)
    Wn = renorm(Wt[..., 0] + r.normal(0, sd, Wt.shape[:2]))[..., None]
    return Wn, sd

clean_p = pred(Wt); cm = met(clean_p)
L += [f"Clean: macroF1 {cm[0]:.4f} Vrec {cm[1]:.4f} Srec {cm[2]:.4f} acc {cm[3]:.4f}", ""]

# 1
L += ["## 1. Post-z-score noise, high SNR", "| SNR | macroF1 | V recall |", "|---|---|---|"]
for snr in (60, 40, 30):
    Wn, _ = post(snr, SEED + 1); mm = met(pred(Wn))
    L.append(f"| {snr} | {mm[0]:.4f} | {mm[1]:.4f} |")

# 2
Wn, sd = post(24, SEED + 1); pn = pred(Wn)
raw_noise = Wn[..., 0]  # after renorm
r = np.random.default_rng(SEED + 1); nz = r.normal(0, sd, Wt.shape[:2])
noisy_pre = Wt[..., 0] + nz
L += ["", "## 2. snr_24 detail",
      f"- noise sd used {sd:.5f}; expected 10**(-24/20) = {10**(-24/20):.5f}; empirical std of noise {nz.std():.5f}; "
      f"mean per-window std of clean {Wt[...,0].std(1).mean():.4f} (std-of-stds {Wt[...,0].std(1).std():.2e})",
      f"- clean windows: mean {Wt.mean():.2e}, std {Wt.std():.4f}; noisy pre-renorm: mean {noisy_pre.mean():.2e}, std {noisy_pre.std():.4f}; "
      f"after renorm: mean {Wn.mean():.2e}, std {Wn.std():.4f}",
      f"- empirical SNR {10*np.log10((Wt[...,0]**2).mean()/(nz**2).mean()):.2f} dB",
      f"- pred histogram clean {np.bincount(clean_p, minlength=4).tolist()}, snr_24 {np.bincount(pn, minlength=4).tolist()}, true {np.bincount(yt, minlength=4).tolist()}"]

# 3 pipeline-faithful
def raw_noise_eval(snr, seed):
    r = np.random.default_rng(seed); Ws = []
    for rec in E.TEST_RECS:
        sig, pk, lab = E.load_record(rec)
        sd = sig.std() * 10 ** (-snr / 20)
        filt = filter_continuous(sig + r.normal(0, sd, sig.shape))
        wb = extract_windows(filt, pk); Ws.append(wb.windows)
    return np.concatenate(Ws)
L += ["", "## 3. Post-z-score vs pipeline-faithful (noise on raw, then 0.5-40 Hz bandpass, window, z-score)",
      "SNR vs raw record power (std of raw signal incl. baseline offset/drift).", "",
      "| SNR | post: macroF1 | post: V rec | raw-faithful: macroF1 | raw-faithful: V rec | raw-faithful: S rec |", "|---|---|---|---|---|---|"]
for snr in (24, 18, 12, 6):
    a = met(pred(post(snr, SEED + 1)[0]))
    Wr = raw_noise_eval(snr, SEED + 2); assert len(Wr) == len(yt)
    b = met(pred(Wr))
    L.append(f"| {snr} | {a[0]:.4f} | {a[1]:.4f} | {b[0]:.4f} | {b[1]:.4f} | {b[2]:.4f} |")

# 4
sd_aug = E.CFG["augmentation_training_only"]["gaussian_noise_sd_on_zscored_window"]
L += ["", "## 4. Training augmentation noise",
      f"- gaussian_noise_sd_on_zscored_window = {sd_aug} on unit-variance windows -> SNR {-20*np.log10(sd_aug):.1f} dB "
      f"({sd_aug*100:.0f}% of signal std). Eval SNR 24 dB = sd {10**(-24/20):.3f} ({10**(-24/20)/sd_aug:.1f}x train); 6 dB = sd {10**(-6/20):.3f} ({10**(-6/20)/sd_aug:.0f}x train)."]

# 5
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(len(yt), 200, replace=False))
Wx = Wt[idx]; tgt = clf_t = pred(Wt)[idx]
p0 = softmax(E.get_logits(clf, Wx), axis=1)[np.arange(200), tgt]
cam = expl.gradcam(Wx, tgt); order = np.argsort(-(cam + rng.uniform(0, 1e-6, cam.shape)), axis=1)
def apply(mask, interp):
    Wm = Wx.copy(); x = Wm[..., 0]
    for i in range(200):
        m = mask[i]
        if m.any():
            x[i, m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), x[i, ~m]) if (interp and (~m).any()) else 0.0
    return softmax(E.get_logits(clf, Wm), axis=1)[np.arange(200), tgt]
def masks(fr):
    k = int(round(fr * 720)); g = np.zeros((200, 720), bool); c = g.copy(); s = g.copy()
    np.put_along_axis(g, order[:, :k], True, axis=1)
    for i in range(200):
        st = rng.integers(0, 720 - k + 1); c[i, st:st + k] = True
        s[i, rng.choice(720, k, replace=False)] = True
    return g, c, s
L += ["", "## 5. XAI deletion: mean prob drop (n=200, clean mean P {:.4f})".format(p0.mean()),
      "Zero = replace with 0 (window mean); Interp = linear interpolation across deleted samples. Contig = random contiguous block of equal length; Scatter = original random-k.", "",
      "| % | CAM zero | Contig zero | CAM interp | Contig interp | Scatter interp |", "|---|---|---|---|---|---|"]
for fr in (0.05, 0.10, 0.20, 0.30, 0.50):
    g, c, s = masks(fr)
    d = [(p0 - apply(g, False)).mean(), (p0 - apply(c, False)).mean(), (p0 - apply(g, True)).mean(),
         (p0 - apply(c, True)).mean(), (p0 - apply(s, True)).mean()]
    L.append(f"| {fr*100:.0f} | " + " | ".join(f"{v:.4f}" for v in d) + " |")
# 6 MoRF vs LeRF, interpolation replacement, attributions smoothed (15-sample moving average) so top-k is contiguous-ish
ig = expl.integrated_gradients(Wx, tgt)["attributions"].reshape(200, 720)
def smooth(a):
    return np.stack([np.convolve(r_, np.ones(15) / 15, mode="same") for r_ in a])
r6 = np.random.default_rng(SEED + 7)
attr = {"Grad-CAM": smooth(cam.reshape(200, 720)), "IG (signed)": smooth(ig)}
L += ["", "## 6. MoRF vs LeRF (interp replacement, attribution smoothed 15 samples, n=200)",
      "Mean prob drop. MoRF = delete highest-attribution first; LeRF = lowest first. Faithful => MoRF >> LeRF.", "",
      "| % | CAM MoRF | CAM LeRF | IG MoRF | IG LeRF |", "|---|---|---|---|---|"]
ords = {}
for name, a in attr.items():
    a = a + r6.uniform(0, 1e-9, a.shape)
    ords[name] = (np.argsort(-a, axis=1), np.argsort(a, axis=1))
for fr in (0.05, 0.10, 0.20, 0.30, 0.50):
    k = int(round(fr * 720)); row = []
    for name in attr:
        for o in ords[name]:
            m = np.zeros((200, 720), bool); np.put_along_axis(m, o[:, :k], True, axis=1)
            row.append((p0 - apply(m, True)).mean())
    L.append(f"| {fr*100:.0f} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
L += ["", "## 7. Augmentation (from preprocessing_config.json)",
      "- Classes augmented: S (2 copies incl. original), F (4 copies incl. original); N, V none. Gaussian sd 0.03 on z-scored window; also amp scale U(0.9,1.1), baseline wander amp U(0,0.1) 0.15-0.5 Hz, circular shift +-15 samples. Whether the noise is applied to every copy or the original: NOT FOUND."]
(E.OUT / "diagnostics.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
