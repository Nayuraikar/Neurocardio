# NeuroCardio: Phase 1 source-of-truth sheet (2026-11-27)

Statuses: IMPLEMENTED / PARTIAL / MISSING / MISMATCH / UNVERIFIED.
Numbers come from `metrics.json` (M) and `xai_evaluation.json` (X) in this folder unless another source is named.
M and X were produced by audit scripts held in the session scratchpad, not yet in the repo.
No project source code was changed.

## 1. Model: IMPLEMENTED
Loaded `backend/models/neurocardio_final.keras`. SHA-256 `0e7083d5…508becc` matches the pinned value and `results/neurocardio_final.keras`.
`results/bilstm_aug_sqrtw.keras` has a different file hash but identical weights.

| Layer | Output | Params |
|---|---|---|
| ecg_input | 720×1 | 0 |
| conv1 (32, k7, ReLU) | 720×32 | 256 |
| BatchNorm / MaxPool 2 | 360×32 | 128 |
| conv2 (64, k5, ReLU) | 360×64 | 10,304 |
| BatchNorm / MaxPool 2 | 180×64 | 256 |
| conv3 (128, k3, ReLU), no BN or pool after it | 180×128 | 24,704 |
| Bidirectional LSTM 64, concat | 128 | 98,816 |
| Dropout 0.4 | 128 | 0 |
| Dense 64 ReLU | 64 | 8,256 |
| logits (Dense 4) / probs (Softmax) | 4 | 260 |

Total 142,980. **CNN-BiLSTM, input 720×1.** 187 is not used at runtime (it appears in `results/window_length_audit.csv`).
README.md describes "Conv1D 32/64/128 + BatchNorm + MaxPool"; only the first two convolutions have BN and pooling.

## 2. Preprocessing
| Item | Value | Evidence |
|---|---|---|
| Sampling rate | 360 Hz, other rates rejected, no resampling | preprocessing.py |
| Filter | Butterworth order 4, 0.5–40 Hz, zero-phase, whole record | preprocessing.py:55 |
| Window | 720 samples (2 s), R-peak centred; edge beats dropped | preprocessing.py:16 |
| Overlap | no fixed stride; median RR 268 samples, median adjacent overlap 62.8% | M.preprocessing.overlap |
| Normalization | **per-window z-score**, eps 1e-8. Neither per-patient z-score nor min-max | preprocessing.py:75 |

A 2 s window usually contains neighbouring beats. A highlight can fall on a beat other than the one being classified.

**Training vs inference**
- Test windows: IMPLEMENTED. Rebuilt windows match the training export for all 47,518 windows (max difference 1.2e-7; probabilities 9.3e-6; argmax agreement 100%).
- App float32 storage path: max window difference 2.9e-6, argmax agreement 100%.
- Train and validation window code: UNVERIFIED. The training notebook is not in the repo (README.md:70, docs/artifact-inventory.md:7).

**Skew and input risks**
- Without `.atr`, the app uses XQRS peaks, while the notebook evaluation used annotated peaks (docs/artifact-inventory.md:13). Measured effect is in section 5.
- With `.atr`, the app takes beat positions from the annotations (analysis_service.py:54). The UI never shows `peak_source`. MISMATCH.
- No paced-recording check. Without MLII, the first channel is used (recording_service.py:139).
- `evaluate_extended.py:241` ran XQRS on the raw signal; the app uses the filtered one (analysis_service.py:43).

## 3. Dataset: IMPLEMENTED (counts recomputed from `results/mitdb`)
48 records on disk, all 360 Hz. 44 used, forming 43 subject groups (202 merged into 201; both in validation).
Excluded paced records: 102, 104, 107, 217 (733 AAMI-mapped beats). 102 and 104 have no MLII. In 114, MLII is the second channel.
Classes: N, S, V, F (AAMI). Symbols outside the mapping are dropped.

| Split | Records / subjects | N | S | V | F | Usable total | Annotated total |
|---|---|---|---|---|---|---|---|
| Train (before augmentation) | 16 / 16 | 34,478 | 692 | 2,937 | 376 | 38,483 | 38,528 |
| Validation | 7 / 6 | 13,394 | 306 | 868 | 39 | 14,607 | 14,621 |
| Test | 21 / 21 | 42,149 | 1,782 | 3,200 | 387 | 47,518 | 47,569 |

`split_selection.json` and `patient_class_distribution.csv` report **annotated** counts. The UI must label which count it shows.
Concentration (`minority_concentration_by_split.csv`): 98.9% of training F beats come from record 208; 77.6% of test S beats from record 232.

## 4. Split
Programmatic disjointness assertion on patient groups and records: **PASS**. No byte-identical windows across splits.
- Train: 101 106 109 112 115 116 118 119 122 203 207 208 209 215 220 230
- Validation: 108 114 124 201(+202) 205 223
- Test: 100 103 105 111 113 117 121 123 200 210 212 213 214 219 221 222 228 231 232 233 234

| Claim | Status |
|---|---|
| Split matches config, JSON and .npy files | IMPLEMENTED |
| Per-window normalization uses no dataset statistics | IMPLEMENTED |
| Augmentation and class weights use training data only | UNVERIFIED (config says so; no code; weight values not stored) |
| BatchNorm statistics from training only | UNVERIFIED |
| Selected by validation macro-F1 | IMPLEMENTED for stored scores. Five non-RR variants re-scored exactly; cnn_ce and bilstm_ce models absent; RR variants not re-scored. Margin over bilstm_rr_sqrtw is 0.4858 vs 0.4835 |
| Test never used for tuning or early stopping | UNVERIFIED. History files have only validation columns. Temperature and thresholds are marked validation-only; temperature refit on validation gives 1.2975 |

Caveats for the "test patients not used" panel:
- **CORRECTION (2026-11-27, model track):** the test set is exactly de Chazal DS2 without record 202, and train + validation is DS1 plus 202 (`backend/scripts/make_splits.py`). `split_selection.json` says the split was drawn from 50,000 class-count-constrained assignments; chance draws cannot produce this partition. How the test set was fixed is **UNVERIFIED**, so the UI must not state the selection method as fact.
- All nine variants have stored test scores. Some beat the selected model on test (RR+Aug macro-F1 0.557 vs 0.473).
- The grouped-CV pool includes the test patients.

## 5. Metrics (fresh run)
Confusion-matrix row sums equal class support: **PASS** on both splits. ROC and PR curves are saved in M.

| | Validation (selection; not final) | **Test (final)** |
|---|---|---|
| Accuracy | 0.9259 | **0.9061** |
| Always-predict-N accuracy | 0.9170 | **0.8870** |
| Balanced accuracy | 0.5106 | 0.4987 |
| Macro P / R / F1 | 0.476 / 0.511 / 0.486 | 0.493 / 0.499 / 0.473 |
| Weighted P / R / F1 | 0.931 / 0.926 / 0.927 | 0.900 / 0.906 / 0.899 |
| ROC-AUC macro / weighted | 0.847 / 0.918 | 0.837 / 0.856 |
| PR-AUC macro | 0.472 | 0.506 |
| MCC / kappa | 0.561 / 0.559 | 0.552 / 0.550 |

| Test class | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC | Support |
|---|---|---|---|---|---|---|---|
| N | 0.952 | 0.949 | 0.625 | 0.951 | 0.850 | 0.973 | 42,149 |
| S | 0.342 | 0.097 | 0.993 | 0.151 | 0.798 | 0.153 | 1,782 |
| V | 0.627 | 0.892 | 0.962 | 0.736 | 0.981 | 0.879 | 3,200 |
| F | 0.050 | 0.057 | 0.991 | 0.053 | 0.719 | 0.017 | 387 |

Test confusion matrix (rows true, columns predicted, N S V F):
N `40009 264 1511 365` / S `1412 172 181 17` / V `242 67 2855 36` / F `357 0 8 22`

- Record bootstrap 95% CI (2,000): accuracy 0.839–0.955; macro-F1 0.395–0.530; S recall 0.062–0.409; V 0.787–0.937; F 0.000–0.385.
- ECE(15): 0.0415 raw, 0.0309 temperature-scaled.
- XQRS end to end, app pipeline: 47,477 detections, 205 annotated beats missed or unwindowed, 65 extra; matched accuracy 0.910, macro-F1 0.476; recall with misses as errors N 0.952, S 0.095, V 0.860, F 0.054.
- Review flag (app uses "balanced"): flags 16.1% of test windows; error among unflagged 5.2% against a 3% target (validation 2.9%); catches 53% of test errors; flags only 11.6% of true F.
- Training history: 19 epochs, best validation macro-F1 at epoch 11 (`hist_bilstm_aug_sqrtw.csv`).

**Leakage flag (>99%): not triggered** (0.906). This does not prove there is no leakage; see the unverified rows in section 4.

## 6. XAI (X: 30 windows per **predicted** class, n=120, seed 20240611, CPU)
Share of sampled predictions that were correct: N 100%, S 37%, V 40%, F 3%. Per-class XAI results therefore often explain wrong predictions.
The fast kernels were compared with the app's `ModelExplainer` on 2 windows: Grad-CAM difference 0; IG at equal steps ≤1.2e-7.

**Grad-CAM: IMPLEMENTED.** Target conv3 (180×128) (explainability.py:56), target-class logit, bilinear upsampling to 720 (explainability.py:100), per-window max normalization. Independent NumPy version differs by at most 2.5e-6. 20/120 maps were all zero (N 7, S 3, V 4, F 6).

**IG: PARTIAL.** Zero baseline (explainability.py:118), logit output, trapezoidal rule. App uses 512 steps (routes.py:160); the class default is 64.
- Requested 50-step setting, 120 windows: relative completeness error median 11.7%, p95 70%, max 329%; 67.5% exceed 5%. **50 steps does not pass.**
- 5 check windows, relative error (%):

| Predicted | 50 | 200 | 512 |
|---|---|---|---|
| N | 3.3 | 1.9 | 0.95 |
| S | 15.9 | 1.1 | 0.97 |
| V | 18.1 | 8.0 | **12.5** |
| F | 18.3 | 0.16 | 0.95 |
| N | 21.5 | 19.8 | 1.4 |

- At 512, 4/5 check windows and 2/2 app-path windows are within 1.5%; one is 12.5%. This is too few windows to give a pass rate. Cause of the outlier not investigated.
- App 512-step IG took roughly 20 s per window in the audit's app-path check (timing includes a 50-step run and Grad-CAM).

**Weight randomization (10 windows; small sample)**, mean Spearman against trained-model maps, 3 full re-initialisations:
- Signed IG: -0.04 to 0.00. Passes.
- Absolute IG: 0.23 to 0.25. Some structure survives.
- Grad-CAM: -0.10 to 0.19 (mean absolute 0.20–0.30); 3–5 of 10 maps degenerate. No strong similarity, weak evidence.
- Cascading: after randomizing only logits, absolute IG stays at 0.67.

**Deletion / insertion** (20 steps, 50-step IG, signed ranking, 15-sample smoothing, 3 smoothed-random controls). Values are AUC advantage over random with 95% CI; positive is better than random.

| Test | Grad-CAM | IG |
|---|---|---|
| Deletion, zero | 0.087 [0.041, 0.138] | 0.140 [0.103, 0.178] |
| Deletion, blur | 0.071 [0.012, 0.131] | 0.189 [0.153, 0.227] |
| Deletion, interpolate | 0.114 [0.061, 0.166] | 0.285 [0.252, 0.321] |
| Insertion, zero | 0.078 [0.020, 0.136] | 0.135 [0.100, 0.168] |
| Insertion, blur | 0.013 [-0.038, 0.059] (not different) | 0.136 [0.098, 0.174] |

IG beats random in 5/5; Grad-CAM in 4/5, by a smaller margin. Grad-CAM is worse than random for some classes in some variants.
Limits: perturbed windows are off-distribution; the IG ranking used 50 steps despite its completeness error (512-step ranking not tested).
These tests say nothing about physiological or causal relevance.

## 7. Discrepancies

### 7a. Old metric files vs fresh run: 233 comparisons, 4 differences
| Item | Old | New | Reason |
|---|---|---|---|
| Bootstrap macro-F1 lower | 0.3908 | 0.3951 | likely resampling noise (1,000 vs 2,000 resamples) |
| Bootstrap S recall upper | 0.4130 | 0.4091 | same |
| Bootstrap V recall lower | 0.7786 | 0.7872 | same |
| End-to-end S recall | 0.0948 | 0.0954 | raw vs filtered detector input |

PVC burden (MAE 4.46, r 0.595) was recomputed and matches.
UNVERIFIED: grouped CV (fold models absent; pooled accuracy and recall agree with its confusion matrix), seeds 11/22/33, noise robustness (not re-run).

### 7b. Hardcoded UI numbers (values correct today, but not sourced)
| Location | UI | Artifact |
|---|---|---|
| App.tsx:126 accuracy / macro-F1 | 90.6% / 47.3% | 90.61 / 47.27 |
| App.tsx:126 S·V·F recall | 9.7 / 89.2 / 5.7% | 9.65 / 89.22 / 5.68 |
| App.tsx:979 silent fallbacks | .906 / .473 / .837 | .9061 / .4727 / .8371 |
| ModelFlow.tsx:59 parameters | 142,980 | 142,980 |

### 7c. XAI documentation and UI claims: MISMATCH
- README.md:76 says Grad-CAM is no better than a random contiguous block under interpolation. The new test, with a different control, sample and metric, finds a modest advantage in 4/5 variants. The two are not directly comparable. Replace the text with X results and state the protocol.
- ExplainDrawer.tsx:161 "Beats driving this decision": it ranks other same-class beats by attribution size. Those beats do not drive this prediction.
- ExplainDrawer.tsx:117 displays absolute IG; the faithfulness test used signed ranking. The tooltip at line 8 therefore overstates support.

### 7d. UI behaviour: MISMATCH / MISSING
- EcgPlot.tsx:217 (and the hover text) pairs the **reviewer's** class with the **model's** confidence.
- App.tsx:444 and :474 apply frontend-only cut-offs (V ≥1%, S/F ≥5%) to give recording-level verdicts. Below them it says "Predominantly normal".
- App.tsx:398 "Reset to model" saves a review with a null label and status "reviewed", so the beat leaves the pending queue.
- App.tsx:118 shows reports as unavailable although a `reports` table exists.
- Ground-truth labels are not stored per beat, so "ground truth if available" is MISSING.
- Reviews are in a separate table and leave predictions unchanged: IMPLEMENTED.

## 8. Placeholder sweep
No hits for `98.2`, `Math.random`, mock/fake/dummy data or `187` in `frontend/src` or `backend/app`. `128` and `12` hits are layer widths, column sizes or layout constants.

| Location | Hit |
|---|---|
| App.tsx:126 | hardcoded metrics, window size, architecture name |
| App.tsx:979 | metric fallbacks |
| App.tsx:980 | typed-in runtime facts |
| App.tsx:827, :837 | fallbacks for IG baseline and `512` steps |
| ModelFlow.tsx:5–17, :58, :59 | typed-in architecture, shapes, preprocessing, parameter count |
| HomePage.tsx:49, :81 | typed-in input and pipeline text |
| HomePage.tsx:19–44, :127 | "Illustrative" trace and flow (labelled) |
| App.tsx:111 | "Illustrative" trace (labelled) |
| shared.ts:8 | typed-in claim about S/F recall |

Class colours are defined in shared.ts:4 and repeated as literals in App.css and App.tsx.

## 9. Database snapshot (read-only)
12 recordings, 7 completed analyses, 14,357 predictions (1,526 flagged), 30 cached explanations, 0 reviews, 1 report.
All analyses use the current model hash. This was taken before any new app use.
