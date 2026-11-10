# Extended evaluation (held-out test records)

Test beats: 47518. Seed 20240611. Only the temperature was fit (validation).

## Comparison with test_metrics.json

| Metric | Existing | Recomputed | Diff |
|---|---|---|---|
| Accuracy | 0.9061 | 0.9061 | +0.0000 |
| Macro-F1 | 0.4727 | 0.4727 | +0.0000 |
| Sensitivity N | 0.9492 | 0.9492 | +0.0000 |
| Sensitivity S | 0.0965 | 0.0965 | +0.0000 |
| Sensitivity V | 0.8922 | 0.8922 | +0.0000 |
| Sensitivity F | 0.0568 | 0.0568 | +0.0000 |
| Specificity N | 0.6254 | 0.6254 | +0.0000 |
| Specificity S | 0.9928 | 0.9928 | +0.0000 |
| Specificity V | 0.9616 | 0.9616 | +0.0000 |
| Specificity F | 0.9911 | 0.9911 | +0.0000 |
| PR-AUC N | 0.9732 | 0.9732 | +0.0000 |
| PR-AUC S | 0.1533 | 0.1533 | -0.0000 |
| PR-AUC V | 0.8793 | 0.8793 | +0.0000 |
| PR-AUC F | 0.0172 | 0.0172 | -0.0000 |
| Macro PR-AUC | 0.5057 | 0.5057 | +0.0000 |
| Balanced accuracy |  | 0.4987 |  |
| MCC |  | 0.5523 |  |
| Cohen's kappa |  | 0.5502 |  |
| PPV N |  | 0.9521 |  |
| F1 N |  | 0.9507 |  |
| PPV S |  | 0.3419 |  |
| F1 S |  | 0.1505 |  |
| PPV V |  | 0.6268 |  |
| F1 V |  | 0.7363 |  |
| PPV F |  | 0.0500 |  |
| F1 F |  | 0.0532 |  |

## Calibration

| | ECE(15) | Brier | NLL |
|---|---|---|---|
| Test, raw | 0.0415 | 0.1551 | 0.3904 |
| Test, T=1.298 | 0.0309 | 0.1540 | 0.3419 |

## Record-level bootstrap (1000 resamples, 95% CI)

| Metric | Point | 95% CI |
|---|---|---|
| macro_f1 | 0.4727 | [0.3908, 0.5292] |
| V_recall | 0.8922 | [0.7786, 0.9367] |
| S_recall | 0.0965 | [0.0626, 0.4130] |
| F_recall | 0.0568 | [0.0000, 0.3846] |
| accuracy | 0.9061 | [0.8388, 0.9549] |

## Noise robustness

| Condition | Macro-F1 | V recall | S recall | Accuracy |
|---|---|---|---|---|
| clean | 0.4727 | 0.8922 | 0.0965 | 0.9061 |
| snr_24 | 0.0484 | 0.0000 | 0.6302 | 0.0319 |
| snr_18 | 0.0231 | 0.0000 | 0.2800 | 0.0186 |
| snr_12 | 0.0124 | 0.0000 | 0.0909 | 0.0115 |
| snr_6 | 0.0040 | 0.0000 | 0.0000 | 0.0081 |
| baseline_wander | 0.4675 | 0.8678 | 0.1027 | 0.9016 |

## End-to-end (XQRS peaks, 150 ms match)

Annotated 47569, detected 47477, missed 205 ({'N': 71, 'S': 4, 'V': 126, 'F': 4}), extra 65.

Matched-beat accuracy 0.9100, macro-F1 0.4752. Recall with misses counted as errors: N 0.9517, S 0.0948, V 0.8601, F 0.0543.

## PVC burden

MAE 4.46 percentage points; Pearson r 0.595.

## XAI faithfulness

| % zeroed | Grad-CAM mean P | Random mean P | Grad-CAM drop | Random drop |
|---|---|---|---|---|
| 0 | 0.9520 | 0.9520 | 0.0000 | 0.0000 |
| 5 | 0.7389 | 0.2080 | 0.2130 | 0.7439 |
| 10 | 0.6240 | 0.0244 | 0.3280 | 0.9276 |
| 20 | 0.5024 | 0.0145 | 0.4496 | 0.9374 |
| 30 | 0.3507 | 0.0137 | 0.6013 | 0.9382 |
| 50 | 0.2265 | 0.0138 | 0.7255 | 0.9382 |

Mean drop (5-50%): Grad-CAM 0.4635 vs random 0.8971. Weight randomization: mean Spearman -0.257 (n=20, 2 degenerate maps).

Figures: reliability.png, noise_robustness.png, pvc_burden.png, xai_deletion.png.
