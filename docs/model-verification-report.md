# NeuroCardio model verification report

This report records measurements from the local verification command and
read-only Kaggle outputs. The prototype uses retrospective MIT-BIH data and is
not a clinically validated diagnostic device.

## Model identity

The backend model and copied backend config match their `/results` sources by
SHA-256. `cv_results.json`, `test_metrics.json`, and `test_raw_primary.json`
identify the selected model as `bilstm_aug_sqrtw`; the config has
`rr_features: null` and one model input. The model loaded as one `ecg_input`
with shape `(None,720,1)` and one output of shape `(None,4)`. The output class
order from config is N, S, V, F. `conv3` and `logits` are present. The model
has 142,980 parameters. The archive contains standard Keras layers only; no
custom object is needed for inference.

The committed-run notebook path was absent. As instructed, identity and
metrics were determined from files under `results/`; no external notebook was
used as evidence.

| Artifact | SHA-256 | Evidence command |
|---|---|---|
| Backend `.keras` | `0e7083d56c31a4e43f6cd332b375b8480446485b60b80f52c3c7c5380508becc` | `shasum -a 256 backend/models/neurocardio_final.keras results/neurocardio_final.keras` |
| `/results` `.keras` | same value as backend | same command |
| Backend preprocessing config | `0177a64052b5d5fb260b271b8069bab65c1265da791d57c9f9edb71884aedae4` | `shasum -a 256 backend/models/preprocessing_config.json results/preprocessing_config.json` |
| `/results` preprocessing config | same value as backend | same command |

Model loading and configuration inspection used:
`MPLCONFIGDIR=/tmp/neurocardio-mpl backend/.venv/bin/python backend/scripts/verify_model_reference.py`.
Environment: arm64, Python 3.11.15, TensorFlow 2.20.0, Keras 3.15.1;
`backend/.venv/bin/python -m pip check` reported `No broken requirements found.`
The archive metadata records Keras 3.13.2 and save date
`2026-11-08@18:55:56`; it deserialized without custom objects.

## Preregistered equivalence tolerances and observations

These tolerances were written before fresh inference and were not changed:

| Measure | Tolerance | Observed | Result |
|---|---:|---:|---|
| Maximum absolute probability difference | ≤ 1e-4 | 9.268522262573242e-06 | PASS |
| Mean absolute probability difference | ≤ 1e-5 | 2.9065758149077487e-08 | PASS |
| Argmax agreement with saved `pred` | ≥ 99.9% | 100% | PASS |
| Batch size 1 vs 256, 500 windows, seed 0 | measured | 4.172325134277344e-07 max absolute difference | PASS |

Evidence for all values in this table:
`MPLCONFIGDIR=/tmp/neurocardio-mpl backend/.venv/bin/python backend/scripts/verify_model_reference.py`.
The reference NPZ SHA-256 is
`7f28287060e089829186c7d7dd77c73154a5e6c708a511619fd102251b037f54`.
Its `X` array was `(47518,720,1)` float32 and was passed directly as the only
model input. `R` had shape `(47518,5)` and was not used. Saved probability
rows summed from 0.9999998211860657 to 1.000000238418579, and saved `pred`
equaled saved probability argmax for every row. There were no disagreement
windows; therefore no worst-disagreement records exist. Per saved-predicted
class, agreement was 100%: N 42,020 rows, S 503, V 4,555, F 440.

## Fresh classifier metrics

Fresh prediction metrics from the same verification command:

| Class | Precision | Recall | F1 | Support | True positives |
|---|---:|---:|---:|---:|---:|
| N | 0.9521418372 | 0.9492277397 | 0.9506825553 | 42,149 | 40,009 |
| S | 0.3419483101 | 0.0965207632 | 0.1505470460 | 1,782 | 172 |
| V | 0.6267837541 | 0.8921875000 | 0.7362991618 | 3,200 | 2,855 |
| F | 0.0500000000 | 0.0568475452 | 0.0532043531 | 387 | 22 |

Accuracy: **0.906140830843049**. Macro-F1: **0.47268327905036756**.
Macro ROC-AUC, one-vs-rest: **0.8370565003894894**. Confusion matrix
(true rows, predicted columns):

```text
[[40009, 264, 1511, 365],
 [ 1412, 172,  181,  17],
 [  242,  67, 2855,  36],
 [  357,   0,    8,  22]]
```

The accuracy, macro-F1 and confusion matrix reproduce `test_raw_primary.json`
and the seed-42 fixed-test row in `final_summary.csv`. The fresh metrics are
stored in [model-reference-comparison.json](model-reference-comparison.json).
Evidence command: the verification command above.

S and F recall are weak: fixed test S recall was 0.0965207632 and F recall
was 0.0568475452. The `/results/cv_results.json` pooled grouped-CV values are
accuracy 0.8796517176, macro-F1 0.4210558688, S recall 0.0535971223, V recall
0.72976445396 and F recall 0.01246882793. These are values read from that JSON,
not new model predictions.

## Raw MIT-BIH annotation-peak reproduction

The verification command processed all 21 `split.test_records` locally. It
selected MLII when present, filtered the complete recording, extracted mapped
annotation-centered windows, and compared them against matching `(record,
position)` entries in the reference NPZ.

| Measure | Observed |
|---|---:|
| Rebuilt windows | 47,518 |
| Maximum absolute input difference | 1.1920928955078125e-07 |
| Mean absolute input difference | 3.5850503165133483e-13 |
| Maximum absolute probability difference | 9.268522262573242e-06 |
| Mean absolute probability difference | 2.9062807848178147e-08 |
| Prediction agreement | 100% |

Per-record classifier agreement and errors (same command):

| Record | Windows | Maximum input difference | Maximum probability difference | Prediction agreement |
|---|---:|---:|---:|---:|
| 100 | 2,270 | 5.960464477539063e-08 | 2.652406692504883e-06 | 100% |
| 105 | 2,565 | 1.490116119384766e-08 | 2.086162567138672e-06 | 100% |
| 213 | 3,247 | 5.960464477539063e-08 | 1.9073486328125e-06 | 100% |
| 232 | 1,780 | 2.9802322387695312e-08 | 4.082918167114258e-06 | 100% |
| 234 | 2,751 | 1.490116119384766e-08 | 1.1920928955078125e-06 | 100% |

The complete per-record values are in `model-reference-comparison.json`.

## Automatic detector evaluation (kept separate from classifier metrics)

WFDB XQRS was run on the filtered signal without annotation input. Annotation
positions were used only after detection for one-to-one nearest-error matching
within 54 samples (150 ms at 360 Hz). This detector result is not classifier
accuracy.

| Detector measure | Result |
|---|---:|
| Records | 21 |
| Matched detections (TP) | 47,412 |
| False detections | 65 |
| Missed AAMI-mapped beats | 157 |
| Sensitivity | 0.9966995312 |
| Positive predictivity | 0.9986309160 |
| F1 | 0.9976642889 |

Of the 65 false detections, 63 had complete windows and were classified: 36
were predicted N and 27 V. Two false detections were excluded because their
720-sample windows crossed signal edges. On matched detections, class
predictions agreed with the annotation-centered prediction for 0.9926315345
of beats. Classifier metrics on matched detector beats were accuracy
0.9101004983, macro-F1 0.4755221943, macro ROC-AUC 0.8361595277. Their
supports and full confusion matrix are in the JSON. Record-level detector
results and matched prediction agreement are in `docs/rpeak-evaluation.csv`.
Evidence command: the verification command above; the CSV was regenerated
from its saved JSON output.

## Preprocessing and limits

The copied config specifies a single MLII channel at 360 Hz; fourth-order
Butterworth bandpass from 0.5 to 40 Hz in SOS form; zero-phase `sosfiltfilt`
over the entire continuous signal before segmentation; windows `[peak-360,
peak+360)` with the peak at index 360; population-standard-deviation
per-window z-score `(x-mean)/(std+1e-8)`; float32 model input. A window crossing
either recording boundary is skipped, not padded. The configured policy has
no resampling: 250 Hz and 500 Hz test inputs were both rejected with a clear
360 Hz requirement error. Evidence: the same verification command and
`backend/tests/test_inference_robustness.py`.

The evaluation is retrospective on MIT-BIH records and a fixed patient-wise
DS2 test partition. It does not establish performance on new devices, leads,
or populations. Grouped-CV notes in the config warn that the pooled folds
include original test patients. Single-lead MLII and 360 Hz are required.
The preprocessing config records the selection rule as highest validation
macro-F1 among CNN-BiLSTM variants and states it was not changed after viewing
test results; the fixed test set was not used for model selection. The grouped
CV pool nevertheless includes original test patients, so that CV summary is
not independent of the fixed test partition across the complete workflow.
Minority S and F recall is weak. This is a research prototype, not a diagnostic
device or clinical validation.
