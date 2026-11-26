# NeuroCardio model-improvement track on Kaggle

## Upload
1. Create a **private** Kaggle Dataset with `neurocardio_kaggle.zip` and `data/neurocardio_dev.npz`.
2. Do **not** upload `data/neurocardio_test_SEALED.npz` yet.
   (Data derived from the MIT-BIH Arrhythmia Database on PhysioNet; keep its attribution and licence terms.)
3. Create a notebook from `neurocardio_kaggle.ipynb`, attach the dataset, set Accelerator to GPU T4.

## Run one stage per session
Set `STAGE` in cell 4, then use **Save Version -> Save & Run All (Commit)**. The committed run keeps `/kaggle/working`
as the notebook output. An interactive session that dies can lose `/kaggle/working`.

For stage 2 and 3, attach the **previous stage's notebook output** as an input (Add Input -> Your Work), or upload the
downloaded results zip to a dataset. Earlier results are found under `/kaggle/input` and restored automatically.
Order: 1, then 2, then 3. Stage 1 first: it contains the reproduction gate, and nothing else counts until it passes.

If a stage prints `[TIME BUDGET]`, attach its output and run the same stage again. Finished runs are skipped.
If stage 1 prints `[STOP] Reproduction failed`, do not continue.

## After stage 3
Read `analysis/ablation.md`, choose a configuration, then run the final test **once**, by hand. Add a dataset with the
sealed file, attach the stage 3 output, run cells 1-2 but delete the SEALED check line in cell 2, then:

    !python -u /kaggle/temp/neurocardio/backend/scripts/final_test.py --config <name> [--thresholds] [--ensemble 3|5] \
        --confirm EVALUATE-TEST-ONCE

Names map to ablation rows: `rr_context+thresholds+ensemble5` is `--config rr_context --thresholds --ensemble 5`.
Save the output (`final/` holds the lock, predictions and report). The script refuses a second configuration,
but only if the earlier output is attached, so never delete it.

## Bring back
Download `neurocardio_results_stage*.zip`. It holds `experiments.json` (every run, including failed and interrupted
ones), `analysis/`, per-run configs with class weights, histories, predictions and models.
