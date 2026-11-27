# Results: multi-database study (multidb_v1)

**Generated file.** `backend/tools/build_results_doc.py` writes it from the committed reports; do not edit it by hand.

## Sources

| Content | File (under `backend/artifacts/`) |
|---|---|
| Frozen choice: primary model, thresholds, temperatures, review cut-offs | `analysis/multidb/selection.json` |
| Validation scores | `analysis/multidb/validation_report.json` |
| Test scores, intervals, comparisons, review (one evaluation) | `analysis/multidb_final/final_report.json` (readable form: `analysis/multidb_final/final_report.md`; lock: `FINAL_TEST_LOCK.json`) |
| Beat and patient counts | `multidb_data_report.json` |
| The app's own model evaluation | `metrics.json` |

Protocol and the disclosures fixed before training: [multidb-protocol.md](multidb-protocol.md).

**Gradient-boosted trees (GBM) is the PRIMARY model.** It was chosen on validation (rule: highest mean validation macro-F1 over seeds; each candidate uses cross-fitted thresholds only if they beat argmax). The other study models are SECONDARY and are listed in a fixed order, never ranked by test score.

Checked by the generator: the three reports name the same primary; the SHA-256 of `selection.json` equals the one recorded in the final report; selection is dated 2026-11-27T04:46:59+00:00 and the test was opened 2026-11-27T05:05:55+00:00. These dates were written by the study scripts; they are not independent proof.

**The app does not run the primary model.** Its live model is the deployed CNN-BiLSTM. No study model, threshold, temperature or review cut-off is applied to uploaded recordings, and the API cannot serve the primary.

## What the final test showed

A result counts as improved only if, as declared before the test, the gain exceeds twice the larger seed SD, the paired patient-bootstrap 95% CI excludes zero, and the gain stays positive with any one patient left out. No multiple-comparison correction is applied.

- **Macro-F1 against the deployed model:** improved on MIT-BIH DS2 without record 202 (0.560 against 0.473); improved on INCART held-out patients (unseen patients) (0.620 against 0.457). The comparison is not like for like: the study models trained on more data from more databases. The `cnn_bilstm` row (deployed architecture, new data) is the control for that.
- **S and F:** S and F recall and F1 did NOT meaningfully improve on either comparable test set. Lower than the deployed model: S recall on INCART held-out patients (unseen patients) (-0.103).
- **F recall of the primary:** MIT-BIH DS2, all records 0.062, 95% CI 0.000 to 0.067 (includes zero); MIT-BIH DS2 without record 202 0.061, 95% CI 0.000 to 0.064 (includes zero); INCART held-out patients (unseen patients) 0.113, 95% CI 0.000 to 0.459 (includes zero).
- **Validation S recall (0.656) against test:** MIT-BIH DS2, all records 0.188 (validation overstated it); MIT-BIH DS2 without record 202 0.174 (validation overstated it); INCART held-out patients (unseen patients) 0.483 (validation overstated it).
- **S and F depend on a few patients.** DS2: patient mitdb:232 holds 75.2% of the 1,837 S beats; patient mitdb:213 holds 93.3% of the 388 F beats. INCART test: patient incartdb:p10 holds 72.5% of the 561 S beats; patient incartdb:p02 holds 38.2% of the 68 F beats.
- **Recall after review is an upper bound:** it assumes a reviewer who corrects every reviewed beat. It is shown beside random review of the same share.
- **DS2 is not untouched.** It had been scored before this study. This is one evaluation within the study, not a first look.

## Test results

Study models: mean over 3 training seeds (±SD where shown). The seeds measure training randomness only. Read macro-F1 and S/F recall first; accuracy is dominated by N and must be read beside the majority-class row.

### MIT-BIH DS2, all records (22 records)

Not a fair comparison with the deployed model: record 202 drove its early stopping. 49,652 beats: N 44,208, S 1,837, V 3,219, F 388.

| Model | Role | Macro-F1 | S recall | F recall | S F1 | F F1 | S precision | F precision | V recall | Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| Gradient-boosted trees (GBM) | **PRIMARY** | 0.555 ±0.008 | 0.188 ±0.010 | 0.062 ±0.023 | 0.220 | 0.074 | 0.265 | 0.092 | 0.978 | 0.937 |
| Deployed CNN-BiLSTM (the app's model) | Reference | 0.472 | 0.100 | 0.057 | 0.155 | 0.052 | 0.349 | 0.048 | 0.890 | 0.907 |
| Majority class (always N) | Baseline | 0.235 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.890 |
| CNN-BiLSTM (study retrain) | SECONDARY | 0.538 ±0.008 | 0.296 ±0.021 | 0.016 ±0.015 | 0.243 | 0.031 | 0.208 | 0.301 | 0.940 | 0.920 |
| CNN-BiLSTM + RR fusion | SECONDARY | 0.569 ±0.034 | 0.375 ±0.088 | 0.077 ±0.052 | 0.288 | 0.116 | 0.233 | 0.299 | 0.936 | 0.919 |
| RR fusion + attention | SECONDARY | 0.567 ±0.107 | 0.487 ±0.328 | 0.028 ±0.025 | 0.411 | 0.032 | 0.356 | 0.044 | 0.915 | 0.927 |
| RR fusion + attention + S head | SECONDARY | 0.567 ±0.033 | 0.537 ±0.221 | 0.005 ±0.005 | 0.409 | 0.009 | 0.335 | 0.057 | 0.876 | 0.929 |
| CNN only | SECONDARY | 0.538 ±0.016 | 0.249 ±0.063 | 0.020 ±0.008 | 0.219 | 0.036 | 0.196 | 0.199 | 0.959 | 0.922 |
| LSTM only | SECONDARY | 0.556 ±0.016 | 0.259 ±0.023 | 0.087 ±0.067 | 0.216 | 0.146 | 0.188 | 0.669 | 0.905 | 0.917 |
| ResNet1D | SECONDARY | 0.541 ±0.006 | 0.278 ±0.006 | 0.007 ±0.010 | 0.256 | 0.013 | 0.240 | 0.139 | 0.926 | 0.928 |
| Stacked ensemble | SECONDARY | 0.571 ±0.009 | 0.312 ±0.026 | 0.029 ±0.023 | 0.308 | 0.055 | 0.305 | 0.738 | 0.947 | 0.939 |

95% CI, bootstrap over patients:

| Model | Macro-F1 | S recall | F recall |
|---|---|---|---|
| Gradient-boosted trees (GBM) (PRIMARY) | 0.507 to 0.628 | 0.074 to 0.692 | 0.000 to 0.067 |
| Deployed CNN-BiLSTM | 0.396 to 0.532 | 0.067 to 0.388 | 0.000 to 0.385 |
| Majority class | no interval computed | no interval computed | no interval computed |

Selective review. **After review is an upper bound (perfect reviewer).**

| Budget | Cut-off | Reviewed | Model | Class | Model recall | After review (upper bound) | After random review |
|---|---|---|---|---|---|---|---|
| 5% | ranked within set | 5.0% | Gradient-boosted trees (GBM) (PRIMARY) | S | 0.188 | 0.371 | 0.228 |
| 5% | ranked within set | 5.0% | Gradient-boosted trees (GBM) (PRIMARY) | F | 0.062 | 0.250 | 0.109 |
| 10% | ranked within set | 10.0% | Gradient-boosted trees (GBM) (PRIMARY) | S | 0.188 | 0.532 | 0.269 |
| 10% | ranked within set | 10.0% | Gradient-boosted trees (GBM) (PRIMARY) | F | 0.062 | 0.422 | 0.156 |
| 5% | fixed on validation | 7.8% | Gradient-boosted trees (GBM) (PRIMARY) | S | 0.188 | 0.459 | 0.251 |
| 5% | fixed on validation | 7.8% | Gradient-boosted trees (GBM) (PRIMARY) | F | 0.062 | 0.358 | 0.135 |
| 10% | fixed on validation | 14.5% | Gradient-boosted trees (GBM) (PRIMARY) | S | 0.188 | 0.706 | 0.306 |
| 10% | fixed on validation | 14.5% | Gradient-boosted trees (GBM) (PRIMARY) | F | 0.062 | 0.526 | 0.198 |
| 5% | ranked within set | 5.0% | Deployed CNN-BiLSTM | S | 0.100 | 0.189 | 0.145 |
| 5% | ranked within set | 5.0% | Deployed CNN-BiLSTM | F | 0.057 | 0.070 | 0.104 |
| 10% | ranked within set | 10.0% | Deployed CNN-BiLSTM | S | 0.100 | 0.280 | 0.190 |
| 10% | ranked within set | 10.0% | Deployed CNN-BiLSTM | F | 0.057 | 0.082 | 0.151 |
| — | — | — | Majority class | S | 0.000 | not defined (no confidence score) | not defined |
| — | — | — | Majority class | F | 0.000 | not defined (no confidence score) | not defined |

Calibration, expected calibration error (15 bins): primary 0.0406 raw, 0.0295 temperature-scaled; deployed 0.0390 raw.

### MIT-BIH DS2 without record 202 (21 records)

The fair comparison with the deployed model. 47,518 beats: N 42,149, S 1,782, V 3,200, F 387.

| Model | Role | Macro-F1 | S recall | F recall | S F1 | F F1 | S precision | F precision | V recall | Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| Gradient-boosted trees (GBM) | **PRIMARY** | 0.560 ±0.007 | 0.174 ±0.010 | 0.061 ±0.022 | 0.235 | 0.074 | 0.365 | 0.095 | 0.978 | 0.944 |
| Deployed CNN-BiLSTM (the app's model) | Reference | 0.473 | 0.097 | 0.057 | 0.151 | 0.053 | 0.342 | 0.050 | 0.892 | 0.906 |
| Majority class (always N) | Baseline | 0.235 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.887 |
| CNN-BiLSTM (study retrain) | SECONDARY | 0.541 ±0.009 | 0.284 ±0.021 | 0.016 ±0.015 | 0.251 | 0.031 | 0.227 | 0.301 | 0.941 | 0.924 |
| CNN-BiLSTM + RR fusion | SECONDARY | 0.578 ±0.035 | 0.364 ±0.087 | 0.078 ±0.052 | 0.313 | 0.116 | 0.276 | 0.299 | 0.937 | 0.927 |
| RR fusion + attention | SECONDARY | 0.577 ±0.115 | 0.487 ±0.340 | 0.028 ±0.025 | 0.446 | 0.034 | 0.412 | 0.049 | 0.915 | 0.934 |
| RR fusion + attention + S head | SECONDARY | 0.581 ±0.036 | 0.536 ±0.224 | 0.005 ±0.005 | 0.454 | 0.009 | 0.400 | 0.058 | 0.878 | 0.938 |
| CNN only | SECONDARY | 0.544 ±0.018 | 0.235 ±0.063 | 0.020 ±0.008 | 0.236 | 0.036 | 0.236 | 0.208 | 0.959 | 0.930 |
| LSTM only | SECONDARY | 0.557 ±0.015 | 0.241 ±0.023 | 0.087 ±0.067 | 0.217 | 0.147 | 0.200 | 0.669 | 0.906 | 0.921 |
| ResNet1D | SECONDARY | 0.546 ±0.007 | 0.265 ±0.004 | 0.007 ±0.010 | 0.270 | 0.013 | 0.280 | 0.140 | 0.927 | 0.934 |
| Stacked ensemble | SECONDARY | 0.578 ±0.009 | 0.294 ±0.029 | 0.029 ±0.023 | 0.329 | 0.055 | 0.376 | 0.738 | 0.948 | 0.946 |

95% CI, bootstrap over patients:

| Model | Macro-F1 | S recall | F recall |
|---|---|---|---|
| Gradient-boosted trees (GBM) (PRIMARY) | 0.507 to 0.648 | 0.068 to 0.694 | 0.000 to 0.064 |
| Deployed CNN-BiLSTM | 0.390 to 0.536 | 0.063 to 0.430 | 0.000 to 0.385 |
| Majority class | no interval computed | no interval computed | no interval computed |

Primary against the deployed model:

| Metric | Primary | Deployed | Majority class | Gain | 95% CI of gain | Smallest gain, one patient out | Verdict |
|---|---|---|---|---|---|---|---|
| Macro-F1 | 0.560 | 0.473 | 0.235 | +0.088 | +0.030 to +0.172 | +0.064 | improved |
| S F1 | 0.235 | 0.151 | 0.000 | +0.085 | -0.031 to +0.361 | +0.005 | did NOT meaningfully improve |
| F F1 | 0.074 | 0.053 | 0.000 | +0.021 | -0.064 to +0.067 | -0.026 | did NOT meaningfully improve |
| S recall | 0.174 | 0.097 | 0.000 | +0.077 | -0.014 to +0.530 | +0.007 | did NOT meaningfully improve |
| F recall | 0.061 | 0.057 | 0.000 | +0.004 | -0.359 to +0.020 | -0.227 | did NOT meaningfully improve |

Selective review. **After review is an upper bound (perfect reviewer).**

| Budget | Cut-off | Reviewed | Model | Class | Model recall | After review (upper bound) | After random review |
|---|---|---|---|---|---|---|---|
| 5% | ranked within set | 5.0% | Gradient-boosted trees (GBM) (PRIMARY) | S | 0.174 | 0.385 | 0.215 |
| 5% | ranked within set | 5.0% | Gradient-boosted trees (GBM) (PRIMARY) | F | 0.061 | 0.283 | 0.108 |
| 10% | ranked within set | 10.0% | Gradient-boosted trees (GBM) (PRIMARY) | S | 0.174 | 0.565 | 0.256 |
| 10% | ranked within set | 10.0% | Gradient-boosted trees (GBM) (PRIMARY) | F | 0.061 | 0.444 | 0.155 |
| 5% | fixed on validation | 6.9% | Gradient-boosted trees (GBM) (PRIMARY) | S | 0.174 | 0.448 | 0.230 |
| 5% | fixed on validation | 6.9% | Gradient-boosted trees (GBM) (PRIMARY) | F | 0.061 | 0.357 | 0.126 |
| 10% | fixed on validation | 13.5% | Gradient-boosted trees (GBM) (PRIMARY) | S | 0.174 | 0.700 | 0.285 |
| 10% | fixed on validation | 13.5% | Gradient-boosted trees (GBM) (PRIMARY) | F | 0.061 | 0.525 | 0.188 |
| 5% | ranked within set | 5.0% | Deployed CNN-BiLSTM | S | 0.097 | 0.182 | 0.142 |
| 5% | ranked within set | 5.0% | Deployed CNN-BiLSTM | F | 0.057 | 0.070 | 0.104 |
| 10% | ranked within set | 10.0% | Deployed CNN-BiLSTM | S | 0.097 | 0.273 | 0.187 |
| 10% | ranked within set | 10.0% | Deployed CNN-BiLSTM | F | 0.057 | 0.085 | 0.151 |
| — | — | — | Majority class | S | 0.000 | not defined (no confidence score) | not defined |
| — | — | — | Majority class | F | 0.000 | not defined (no confidence score) | not defined |

Calibration, expected calibration error (15 bins): primary 0.0366 raw, 0.0263 temperature-scaled; deployed 0.0415 raw.

### INCART held-out patients (unseen patients) (10 patients)

Unseen patients, not an unseen database: other INCART patients were in training. 59,806 beats: N 52,441, S 561, V 6,736, F 68.

| Model | Role | Macro-F1 | S recall | F recall | S F1 | F F1 | S precision | F precision | V recall | Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| Gradient-boosted trees (GBM) | **PRIMARY** | 0.620 ±0.007 | 0.483 ±0.034 | 0.113 ±0.008 | 0.433 | 0.160 | 0.393 | 0.274 | 0.828 | 0.972 |
| Deployed CNN-BiLSTM (the app's model) | Reference | 0.457 | 0.586 | 0.103 | 0.204 | 0.031 | 0.124 | 0.019 | 0.760 | 0.871 |
| Majority class (always N) | Baseline | 0.234 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.877 |
| CNN-BiLSTM (study retrain) | SECONDARY | 0.576 ±0.000 | 0.608 ±0.118 | 0.015 ±0.025 | 0.466 | 0.019 | 0.383 | 0.025 | 0.773 | 0.957 |
| CNN-BiLSTM + RR fusion | SECONDARY | 0.602 ±0.031 | 0.677 ±0.157 | 0.098 ±0.037 | 0.515 | 0.043 | 0.418 | 0.029 | 0.790 | 0.961 |
| RR fusion + attention | SECONDARY | 0.534 ±0.039 | 0.242 ±0.155 | 0.123 ±0.042 | 0.235 | 0.069 | 0.248 | 0.061 | 0.791 | 0.953 |
| RR fusion + attention + S head | SECONDARY | 0.575 ±0.050 | 0.525 ±0.293 | 0.078 ±0.008 | 0.379 | 0.089 | 0.299 | 0.116 | 0.750 | 0.961 |
| CNN only | SECONDARY | 0.561 ±0.049 | 0.283 ±0.110 | 0.123 ±0.042 | 0.271 | 0.113 | 0.265 | 0.115 | 0.807 | 0.964 |
| LSTM only | SECONDARY | 0.577 ±0.014 | 0.683 ±0.042 | 0.044 ±0.039 | 0.452 | 0.047 | 0.346 | 0.053 | 0.781 | 0.953 |
| ResNet1D | SECONDARY | 0.598 ±0.023 | 0.597 ±0.053 | 0.078 ±0.022 | 0.435 | 0.105 | 0.346 | 0.158 | 0.773 | 0.966 |
| Stacked ensemble | SECONDARY | 0.583 ±0.015 | 0.664 ±0.100 | 0.044 ±0.015 | 0.422 | 0.067 | 0.314 | 0.169 | 0.760 | 0.965 |

95% CI, bootstrap over patients:

| Model | Macro-F1 | S recall | F recall |
|---|---|---|---|
| Gradient-boosted trees (GBM) (PRIMARY) | 0.464 to 0.718 | 0.235 to 0.563 | 0.000 to 0.459 |
| Deployed CNN-BiLSTM | 0.330 to 0.540 | 0.100 to 0.724 | 0.013 to 0.435 |
| Majority class | no interval computed | no interval computed | no interval computed |

Primary against the deployed model:

| Metric | Primary | Deployed | Majority class | Gain | 95% CI of gain | Smallest gain, one patient out | Verdict |
|---|---|---|---|---|---|---|---|
| Macro-F1 | 0.620 | 0.457 | 0.234 | +0.163 | +0.094 to +0.220 | +0.118 | improved |
| S F1 | 0.433 | 0.204 | 0.000 | +0.228 | -0.016 to +0.431 | +0.033 | did NOT meaningfully improve |
| F F1 | 0.160 | 0.031 | 0.000 | +0.128 | -0.018 to +0.350 | +0.035 | did NOT meaningfully improve |
| S recall | 0.483 | 0.586 | 0.000 | -0.103 | -0.466 to +0.200 | -0.346 | lower; not a meaningful improvement |
| F recall | 0.113 | 0.103 | 0.000 | +0.010 | -0.179 to +0.274 | -0.076 | did NOT meaningfully improve |

Selective review. **After review is an upper bound (perfect reviewer).**

| Budget | Cut-off | Reviewed | Model | Class | Model recall | After review (upper bound) | After random review |
|---|---|---|---|---|---|---|---|
| 5% | ranked within set | 5.0% | Gradient-boosted trees (GBM) (PRIMARY) | S | 0.483 | 0.924 | 0.509 |
| 5% | ranked within set | 5.0% | Gradient-boosted trees (GBM) (PRIMARY) | F | 0.113 | 0.520 | 0.157 |
| 10% | ranked within set | 10.0% | Gradient-boosted trees (GBM) (PRIMARY) | S | 0.483 | 0.957 | 0.535 |
| 10% | ranked within set | 10.0% | Gradient-boosted trees (GBM) (PRIMARY) | F | 0.113 | 0.637 | 0.201 |
| 5% | fixed on validation | 3.1% | Gradient-boosted trees (GBM) (PRIMARY) | S | 0.483 | 0.844 | 0.499 |
| 5% | fixed on validation | 3.1% | Gradient-boosted trees (GBM) (PRIMARY) | F | 0.113 | 0.377 | 0.140 |
| 10% | fixed on validation | 4.9% | Gradient-boosted trees (GBM) (PRIMARY) | S | 0.483 | 0.922 | 0.508 |
| 10% | fixed on validation | 4.9% | Gradient-boosted trees (GBM) (PRIMARY) | F | 0.113 | 0.515 | 0.156 |
| 5% | ranked within set | 5.0% | Deployed CNN-BiLSTM | S | 0.586 | 0.643 | 0.607 |
| 5% | ranked within set | 5.0% | Deployed CNN-BiLSTM | F | 0.103 | 0.191 | 0.148 |
| 10% | ranked within set | 10.0% | Deployed CNN-BiLSTM | S | 0.586 | 0.693 | 0.628 |
| 10% | ranked within set | 10.0% | Deployed CNN-BiLSTM | F | 0.103 | 0.279 | 0.193 |
| — | — | — | Majority class | S | 0.000 | not defined (no confidence score) | not defined |
| — | — | — | Majority class | F | 0.000 | not defined (no confidence score) | not defined |

Calibration, expected calibration error (15 bins): primary 0.0189 raw, 0.0146 temperature-scaled; deployed 0.0677 raw.

## Validation results (not test results)

Validation drove early stopping and chose the primary from all the candidates, so these figures are optimistic. The deployed model was not scored on this data.

| Model | Role | Macro-F1 | S recall | F recall | S F1 | F F1 | S precision | F precision | V recall | Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| Gradient-boosted trees (GBM) | **PRIMARY** | 0.697 ±0.002 | 0.656 ±0.005 | 0.274 ±0.015 | 0.722 | 0.168 | 0.802 | 0.121 | 0.938 | 0.976 |
| Majority class (always N) | Baseline | 0.235 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.887 |
| CNN-BiLSTM (study retrain) | SECONDARY | 0.651 ±0.012 | 0.711 ±0.034 | 0.215 ±0.133 | 0.663 | 0.085 | 0.621 | 0.061 | 0.877 | 0.962 |
| CNN-BiLSTM + RR fusion | SECONDARY | 0.685 ±0.009 | 0.765 ±0.076 | 0.312 ±0.086 | 0.755 | 0.099 | 0.750 | 0.059 | 0.887 | 0.971 |
| RR fusion + attention | SECONDARY | 0.641 ±0.032 | 0.652 ±0.039 | 0.308 ±0.115 | 0.681 | 0.065 | 0.714 | 0.037 | 0.875 | 0.953 |
| RR fusion + attention + S head | SECONDARY | 0.659 ±0.010 | 0.705 ±0.016 | 0.207 ±0.077 | 0.697 | 0.103 | 0.689 | 0.074 | 0.847 | 0.964 |
| CNN only | SECONDARY | 0.667 ±0.004 | 0.654 ±0.025 | 0.350 ±0.064 | 0.629 | 0.171 | 0.609 | 0.119 | 0.920 | 0.961 |
| LSTM only | SECONDARY | 0.643 ±0.023 | 0.652 ±0.064 | 0.329 ±0.058 | 0.628 | 0.108 | 0.605 | 0.065 | 0.845 | 0.959 |
| ResNet1D | SECONDARY | 0.652 ±0.018 | 0.661 ±0.063 | 0.232 ±0.064 | 0.654 | 0.094 | 0.647 | 0.064 | 0.881 | 0.964 |
| Stacked ensemble | SECONDARY | 0.695 ±0.009 | 0.813 ±0.009 | 0.215 ±0.066 | 0.770 | 0.110 | 0.732 | 0.081 | 0.893 | 0.975 |

## Consistency checks

- Deployed model on DS2 without record 202: the final report and `metrics.json` agree on beat count and macro-F1 (within 1e-9).
- **Known defect, MIT-BIH DS2, all records:** the per-patient F supports in `final_report.json` add to 387, but the set has 388.
- **Known defect, MIT-BIH DS2 without record 202:** the per-patient F supports in `final_report.json` add to 386, but the set has 387.
- **Known defect, INCART held-out patients (unseen patients):** the per-patient F supports in `final_report.json` add to 67, but the set has 68.
- Cause of the per-patient gaps: `backend/scripts/evaluate_multidb.py`, function `per_patient`, applies `int()` to a floating-point mean over seeds, which can round a whole number down. Pooled per-class scores are computed separately and are not affected. Patient shares above come from the data report, which counts beats directly. The study scripts and the final report were left unchanged.

## Limitations

- One fixed split. Seed SD does not measure how much results would move with other patients.
- Few patients carry most S and F beats; the patient-bootstrap intervals for S and F are wide.
- INCART F is a very small sample (68 beats).
- DS2 was scored before the study, and the RR variants' stored scores were known when RR fusion was chosen.
- INCART is not an external database: other INCART patients were used in training and validation.
- Offline method: RR features use the following beat and the whole-record median, and are computed from reference beat positions after beats outside N/S/V/F are removed. No beat-detector error is included.
- SVDB has no record-to-subject map (one record is assumed to be one subject), and overlap with MIT-BIH subjects cannot be checked.
- Windows are centred on each database's own annotations, which are not aligned the same way ([multidb-data-report.md](multidb-data-report.md)).
- The comparison with the deployed model mixes the effect of more training data with the effect of the method.
- Window-level results on retrospective recordings. Not a diagnosis, and no patient-level claim.
- Packaging did not retrain or re-evaluate anything. This page reports the committed files; it does not re-verify the evaluation that produced them.
