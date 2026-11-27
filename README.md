# NeuroCardio

NeuroCardio is a CNN-BiLSTM ECG beat classifier (N/S/V/F, AAMI classes) with explainability and human review. It is a research prototype, not a diagnostic device.

## Quick start (macOS/Linux)

Prerequisites: Python 3.11 (developed with 3.11.15), Node.js 22 (developed with v22.12.0) and npm.

```sh
./setup.sh      # creates backend/.venv, installs Python deps, runs npm install
./run.sh        # starts API (127.0.0.1:8000) and frontend; Ctrl+C stops both
```

Requires Node ^20.19, ^22.13 or >=24, and Python 3.11.
The first launch takes up to a minute while TensorFlow loads.

Open http://localhost:5173 and upload the files from `sample_data/` (select a record's `.hea`, `.dat` and `.atr` together, e.g. 100).

### Manual steps (also for Windows)

```sh
# one-time
python3.11 -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt   # Windows: backend\.venv\Scripts\python
cd frontend && npm install && cd ..

# terminal 1 (project root)
backend/.venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000

# terminal 2
cd frontend && npm run dev -- --host localhost

# tests
cd backend && PYTHONPATH=. .venv/bin/python -m pytest -q

# demo windows (training and validation patients only)
backend/.venv/bin/python backend/tools/build_demo_samples.py

# end-to-end and number checks (see frontend/e2e/README.md)
cd frontend && npm run e2e
```

API docs: http://127.0.0.1:8000/docs. The Vite dev server proxies `/api` to port 8000. `backend/data/` (SQLite DB, uploads, reports) is created automatically on first start.

## Features

- Overview, three-stage Analysis Studio, ECG viewer with replay, three-panel view
- Signal-quality measurements, confidence and uncertainty explorer, priority review queue
- XAI Studio (signed Integrated Gradients at 512 steps, Grad-CAM, per-window completeness, offline faithfulness results)
- Misclassification lab, model performance, model comparison, model card, dataset explorer
- Demo Playground and Challenge Mode (training and validation patients only)
- History, review workspace (reviewer labels never change predictions), report builder (PDF and CSV), batch analysis, comparison, presentation mode

Study results appear only when `selection.json`, `validation_report.json` and `final_report.json` from a non-smoke run are
placed in `backend/artifacts/` (or `backend/artifacts/multidb/`). Until then the UI says they have not been run.

## Model

- Input: 720×1 R-peak-centred window, 0.5–40 Hz Butterworth bandpass, per-window z-score
- Architecture: CNN-BiLSTM. Three Conv1D layers (32, 64, 128 filters; batch normalization and max pooling after the first two) → BiLSTM 64 per direction → Dropout 0.4 → Dense 64 → Dense 4 softmax; 142,980 parameters
- Selected by validation macro-F1; the config records sqrt-weighted cross-entropy (unverified: the training code is not in the repository)
- Files: `backend/models/`. Details in `docs/inference-pipeline.md`.

## Evaluation

Patient-wise held-out MIT-BIH test, window level (source: `backend/artifacts/metrics.json`).

| | Macro-F1 | S recall | F recall | Accuracy |
|---|---|---|---|---|
| This model | 0.473 | 0.097 | 0.057 | 0.906 |
| Always answer N (majority class) | 0.235 | 0 | 0 | 0.887 |

**The model misses most S and F beats.** Accuracy is only slightly above the majority-class baseline. 98.9% of the training F beats come from record 208. Test scores exist for other variants in `results/`, so this test set is not untouched. See `docs/limitations.md`.

## Limitations

- Retrospective analysis of recordings only; not real-time or live.
- Not a diagnosis: it classifies individual beat windows, not recordings or patients.
- The training notebook/code is not included in this repository.
- Calibration: ECE 0.042 -> 0.031 after temperature scaling (T=1.30, fit on validation records).
- Small minority classes: record-level bootstrap 95% CIs are S recall [0.06, 0.41] and F recall [0.00, 0.38], too wide to draw conclusions about S/F performance.
- End-to-end with XQRS-detected peaks, V recall (misses counted as errors) drops from 0.892 to 0.860.
- PVC burden per record: MAE 4.46 percentage points, Pearson r = 0.595.
- Noise: stable with white noise added to the raw signal down to 12 dB SNR, but collapses if noise reaches the z-scored window (macro-F1 0.27 at 30 dB, ~0.05 at 24 dB). Only white noise was tested. Hypothesis (untested): only S and F were augmented in training (Gaussian noise sd 0.03 among the transforms), so N and V never saw noisy beats, and noise may act as a class cue; under heavy noise the model predicts almost everything as F/S.
- Explanations: results are mixed and come from small samples. In `backend/artifacts/xai_evaluation.json`, Integrated Gradients beats smoothed random controls in all five deletion/insertion variants and Grad-CAM in four of five, using 50-step IG (the app uses 512). Signed IG passes weight randomization; absolute IG and Grad-CAM give weaker evidence. None of this shows clinical or causal relevance. Completeness is reported per window and can exceed 5% error even at 512 steps.
- Details: `results/extended_eval/summary.md` and `results/extended_eval/diagnostics.md`.
- Some large research artifacts (extra model variants, `results/test_examples_for_xai.npz`, >100 MB) are not in the repo; the tests that need that file are skipped when it is absent.

## Troubleshooting

- Port in use: stop the other process on 8000/5173, or change `--port` (and the proxy in `frontend/vite.config.ts`).
- TensorFlow install fails (Apple Silicon/Windows): use Python 3.11 64-bit and upgrade pip; `tensorflow==2.20.0` is pinned.
- Model not found: confirm `backend/models/neurocardio_final.keras` exists and `shasum -a 256 -c neurocardio_final.keras.sha256` passes (run in `backend/models`).
- CORS / API errors: the backend allows `http://localhost:5173`; open the app at `localhost`, not `127.0.0.1`, and keep the backend on port 8000.
- An evaluation panel says it has not been generated: check the JSON files in `backend/artifacts/`.

## Structure

```
backend/app/     FastAPI app (api, ml, services, models)
backend/models/  trained model + preprocessing config
backend/tests/   pytest suite
frontend/src/    React + Vite UI
results/         metrics/configs and results/mitdb records
sample_data/     demo records 100, 106, 228
docs/            methodology, reports, user guide
```
