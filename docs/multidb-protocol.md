# Protocol: inter-patient, cross-database beat classification (multidb_v1)

Written 2026-11-27, before any full training run of this study existed. Only CPU smoke runs (2,048 training beats,
one epoch) had been made. This replaces the stage 1/2/3 grid in `experiment-protocol.md`; earlier entries in
`experiments.json` are kept.

## Disclosures to carry into the paper
1. **DS2 is not untouched.** Nine earlier variants and an audit were scored on it. The stored scores show the RR variants
   above the deployed model, which was known when RR fusion was chosen here. DS2 is sealed within this study only.
2. **INCART is not a fully external database.** 22 of its 32 patients are used for training and validation. The INCART test
   measures unseen *patients* from a database the model has seen. It must not be described as unseen-database generalisation.
3. **F and S are concentrated.** DS2: patient 213 holds 93% of F, patient 232 holds 75% of S. INCART test: patient p10
   holds 73% of S, and there are only 68 F beats. Training: record 208 holds 73% of F.
4. **Offline method.** RR features use the following beat and the whole-record median. Beat positions come from
   reference annotations, and beats outside N/S/V/F are removed before RR intervals are computed. No R-peak detector error
   is included.
5. **Unverifiable identity.** SVDB has no record-to-subject map (one record = one subject is assumed), and overlap between
   SVDB and MIT-BIH subjects cannot be checked.
6. **Alignment differs by database.** Windows are centred on each database's annotation (`multidb-data-report.md`).
7. **The comparison with the deployed model is not like for like.** The new models train on about 7 times more beats from
   three databases. A gain over the deployed model mixes the effect of more data with the effect of the method. The
   `cnn_bilstm` control (deployed architecture, same new data) separates the two.

## Data and split
`backend/scripts/build_multidb.py` -> `backend/splits/multidb_v1.json`, counts in `multidb-data-report.md`.
Train: 16 MIT-BIH DS1, 62 SVDB records, 15 INCART patients. Validation: MIT-BIH 108 114 124 205 223, 16 SVDB records,
7 INCART patients. Tests: DS2 (22 records); 10 INCART patients. MIT-BIH 201 is dropped from development because it is the
same subject as DS2 record 202. SVDB and INCART patients were assigned by a seeded random search on label counts only.
Assertions (`split_checks.py`, run before every training run): no record or patient in two parts, test_ds2 equals DS2,
no DS2 record or subject in development. The dev file must match the SHA-256 recorded in the split.

## Models
| ID | Description | RR |
|---|---|---|
| cnn_bilstm | deployed architecture, retrained on the new data (ablation control) | no |
| rr_fusion | + Dense(16) RR branch, late fusion | yes |
| rr_fusion_attn | + additive attention pooling over the BiLSTM steps, replacing the last states | yes |
| rr_fusion_attn_shead | + auxiliary S-versus-rest head, training only (loss weight 0.5) | yes |
| cnn, lstm, resnet1d | waveform-only baselines | no |
| gbm | histogram gradient boosting on 121 handcrafted features, including RR | yes |
| stacked_ensemble | multinomial logistic regression on the log-probabilities of all 8, one per seed | - |

`rr_context` from the earlier track is not trained: it is a near-duplicate of `rr_fusion` (143,300 vs 144,100
parameters; same inputs, differing by one Dense(16) layer and the dropout position). `rr_fusion` matches the original
notebook's `bilstm_rr_sqrtw` layout.
The waveform-only baselines get no RR input, so compare them with `cnn_bilstm` for architecture, not with RR models.

## Training
Fixed split; seeds 42, 11, 22; batch 64; Adam 1e-3; at most 25 epochs; early stopping patience 5 on validation macro-F1
with best weights restored; learning rate x0.5 after 2 flat epochs; sqrt-inverse-frequency class weights; the deployed
model's training-only augmentation (S x2, F x4). GBM: same class weights, no augmentation, early stopping on validation
macro-F1 every 20 iterations (patience 5, at most 400). Seeds measure training randomness only, not split variation.

## Selection (validation only)
- Each candidate uses per-class thresholds only if patient-cross-fitted thresholds (5 patient folds) beat raw argmax.
- Primary model = highest mean validation macro-F1 under that rule. Temperature, thresholds, review cut-offs and
  stacker weights are then frozen in `selection.json`.
- The stacker is fitted on validation; its validation predictions are cross-fitted over patients.
- Validation also drove early stopping and it picks the best of 9 candidates, so validation scores are optimistic.

## Decision rule for "improved" (fixed in advance)
A gain in S F1, F F1, macro-F1, or S/F recall counts only if all of these hold: the mean gain exceeds 2 x the larger
seed SD; the paired patient bootstrap 95% CI excludes 0; and the gain stays positive with any single patient left out.
Against the single deployed model, the SD is that of the new model's seeds. Otherwise the report says it did not
meaningfully improve. No multiple-comparison correction is applied: report all pre-declared comparisons
(each ablation step, primary vs cnn_bilstm, primary vs deployed).

## Reporting
Per-class precision/recall/F1, macro-F1, accuracy (mean±SD over seeds); patient-bootstrap 95% CIs (2,000 resamples);
per-patient S and F; ECE (15 bins), Brier and NLL before and after temperature scaling; risk-coverage curves; S and F
recall at 5% and 10% review budgets. Review recall assumes a perfect reviewer, so it is an upper bound and is shown next
to random review. Test results are reported both with a within-set ranking and with the cut-off fixed on validation.

## Final evaluation
`final_test.py`, once. It scores the primary model, the deployed model and the majority class on DS2 and the INCART
held-out patients. `--secondary-all` also scores the other candidates in the same pass; decide before running, and report
them as secondary. The deployed model used record 202 for early stopping, so the headline comparison on MIT-BIH is DS2
without 202 (21 records). All 22 records are reported as well.
