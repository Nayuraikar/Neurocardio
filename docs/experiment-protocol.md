# Model-improvement protocol

Written 2026-11-27, before any training result from this track existed.

## Disclosure: the test set is not untouched
- Test scores for nine earlier model variants are stored in `results/`; the audit also reported per-class test results.
- Stored scores show the RR-feature variants above the deployed model on test. That knowledge exists when choosing experiment (c).
- "Evaluated once" therefore means once within this track. It is not a first look.

## Split provenance (corrects the Phase 1 sheet)
- The project test set is exactly de Chazal DS2 without record 202. Train + validation is exactly DS1 plus 202.
- `results/split_selection.json` describes 50,000 constrained random draws. That cannot by chance produce this partition
  (C(43,21) is about 1.05e12). How the test set was fixed is UNVERIFIED.
- Standard DS1/DS2 has one subject on both sides: 201 (DS1) and 202 (DS2).

## Exclusions
- Paced records 102, 104, 107, 217 (AAMI practice; 102 and 104 also lack MLII).
- Beat symbols outside the project AAMI mapping, including the Q group. This is a 4-class N/S/V/F task.
- Beats whose 720-sample window crosses a record edge.
- DS1 annotated counts here are N 45,866, S 944, V 3,788, F 415. Published tables differ by a few beats.

## Step 1: reproduction (fixed split, seeds 42, 11, 22)
`train.py` is a reconstruction (see its header). **Pass:** the deployed model's validation macro-F1 (0.4858)
lies within mean ± 2 SD of the three seeds. The test set is not scored. If it fails, experiments stop until understood.

## Step 2: selection protocol
- 5 subject-wise folds over the 22 DS1 records: `backend/splits/cv5_dev.json`. No DS2 record, including 202, is used.
  (Changed before any result existed; the first draft included 202.) Record 202 is used only to early-stop the
  reproduction runs, because the deployed model did.
- Iteration k: evaluate fold k; early-stop and tune thresholds on fold k+1; train on the other three.
- Score: macro-F1 and per-class metrics on pooled held-out-fold predictions, per seed; report mean ± SD over seeds 42, 11, 22.
- **F limitation:** record 208 holds 373 of 415 development F beats. Two iterations train on 35 or 37 F beats;
  several early-stop folds have 3-7. F results from this data cannot support a general claim.

## Experiments
| ID | Variant | Change from baseline |
|---|---|---|
| a1 | focal_g2 | focal loss, gamma 2, no class weights |
| a2 | cb_focal | class-balanced focal loss, beta 0.999, gamma 2 |
| b | balanced_sampler | class-uniform batches, unweighted cross-entropy (no double correction) |
| c | rr_context | five RR timing features from beat positions, joined before the dense layer; uses the next beat (offline only) |
| d | thresholds | per-class probability scales fit on each early-stop fold, applied to its held-out fold |
| e | ensemble | mean probabilities of 5 seeds (adds seeds 33, 44) |

Ablation: each of a-e alone, then the combination of those that help.

## Decision rules (fixed in advance)
- Configuration chosen = highest mean pooled-CV macro-F1.
- S or F counts as **meaningfully improved** only if all hold: the mean pooled F1 gain exceeds 2 × the larger seed SD;
  a paired record-level bootstrap 95% CI of the gain excludes 0; and the gain stays positive when the single most
  influential record is left out. Otherwise the report says it did not meaningfully improve.

## Final evaluation (one pass)
Train the chosen configuration on `backend/splits/ds1_ds2.json` (inner training set = the current training set;
inner validation = current validation minus 202). Run inference on the 22 DS2 records **once**.
From that single set of predictions report (1) DS2, and (2) its 21-record subset, which is compared with the current
baseline and the majority class. Report per-class P/R/F1, macro-F1, per-record S/F and ECE (temperature fit on
inner validation). The script refuses to run if a final result already exists.

## Implementation rules (fixed before any result; see backend/scripts)
- **Combined variant:** take the imbalance method with the highest mean CV macro-F1 among baseline, focal_g2, cb_focal and
  balanced_sampler. Train "that method + RR context" only if RR context beats the baseline **and** the best method is not
  the baseline. These are alternatives, so two imbalance methods are never stacked.
- **Thresholds:** probability scales for S, V, F (N fixed at 1), grid 2^-4 to 2^4, coordinate search maximising macro-F1.
- **Ensembles:** 3 and 5 seeds for the baseline and for the best single or combined variant.
- **Meaningful rule with ensembles:** an ensemble is a single value, so the seed SD used is the baseline's.
- **Choice:** stage 3 reports the highest CV macro-F1 row. Picking among many rows on the same folds is optimistic; a person
  makes the choice, and the one final evaluation is the check.
- **Deterministic ops:** measured once per project (`gpu_check.py`); kept if they work on every code path and cost <= 10%.
  Batch size stays 64. Mixed precision is off.
- **Final script:** a lock is written before the sealed file is opened; predictions are saved immediately. A second
  configuration is refused as long as the earlier output is attached. This is a safeguard, not tamper-proof.
