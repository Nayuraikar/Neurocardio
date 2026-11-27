# Multi-database study on Kaggle

Protocol: `docs/multidb-protocol.md`. Data counts: `docs/multidb-data-report.md`.

## Upload
1. Create a **private** Kaggle Dataset with `neurocardio_kaggle.zip` and `data/multidb_dev.npz`.
2. Do **not** upload the two `*_SEALED.npz` files yet.
   Data are derived from the MIT-BIH Arrhythmia, MIT-BIH Supraventricular Arrhythmia and St Petersburg INCART databases
   on PhysioNet. Keep their citations and licence terms.
3. Create a notebook from `neurocardio_kaggle.ipynb`, attach the dataset, set Accelerator to **GPU T4 x2**.

`neurocardio_dev.npz` and `neurocardio_test_SEALED.npz` belong to the old single-database track. Do not upload them.

## Run
Run cells 1-3 interactively first. Cell 3 (smoke) must end with `SMOKE OK`. Then use
**Save Version -> Save & Run All (Commit)**: a committed run keeps `/kaggle/working`; an interactive session that dies can lose it.

If it prints `[TIME BUDGET]`, attach that notebook's output as an input (Add Input -> Your Work) and commit again.
Finished runs are skipped. A run cut off in the middle starts again from epoch 1.
Registry entries from earlier outputs, including reproduction runs, are merged into `experiments.json` and never removed.
To carry your earlier reproduction results forward, attach that old notebook output once.

Watch the `ms/batch` figure in each epoch line: the estimate assumes 24.1 ms for every architecture.

Output: `analysis/multidb/validation_report.md`, `selection.json`, `per_patient_validation.csv`, `risk_coverage.{csv,png}`,
`experiments.json`, `runs/`, `logs/`.

## Final evaluation: once
Read the validation report. Decide whether to score the other candidates too (`--secondary-all`); the choice is locked.
Add a dataset with both `*_SEALED.npz` files, attach the finished study output, run cells 1 and 2 with
`ALLOW_SEALED = True`, skip cells 3 and 4, then run:

    !python -u /kaggle/temp/neurocardio/backend/scripts/final_test.py --confirm EVALUATE-TEST-ONCE [--secondary-all]

Then run cell 5. Output is in `final_multidb/`. **Keep that output**: the lock only protects you while it is attached.
If the script is interrupted after predictions are saved, rerun it with the same arguments.

## Bring back
`neurocardio_results.zip`.
