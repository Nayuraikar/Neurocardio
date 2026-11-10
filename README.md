# NeuroCardio

NeuroCardio is a CNN-BiLSTM ECG beat classifier (N/S/V/F, AAMI classes) with explainability and human review. It is a research prototype and decision-support tool, not a diagnostic device.

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
```

API docs: http://127.0.0.1:8000/docs. The Vite dev server proxies `/api` to port 8000. `backend/data/` (SQLite DB, uploads, reports) is created automatically on first start.

## Features

- Dashboard and upload (WFDB or CSV)
- ECG viewer: paper grid, zoom/pan, overview strip, R-peaks, RR/bpm, class-shaded abnormal beats
- Single verdict card with composition bar
- Per-beat Explain drawer (Integrated Gradients by default, Grad-CAM as the coarse alternative; heatmap plus top contributing beats)
- Doctor review/edit; model and reviewed values are stored separately
- Model architecture page
- PDF and CSV report export

## Model

- Input: 720x1 R-peak-centred window, 0.5-40 Hz Butterworth bandpass, per-window z-score
- Architecture: Conv1D 32/64/128 + BatchNorm + MaxPool -> BiLSTM 64/direction -> Dropout 0.4 -> Dense 64 -> Dense 4 softmax; 142,980 parameters
- Selected by validation macro-F1; sqrt-weighted cross-entropy
- Files: `backend/models/` (`neurocardio_final.keras`, `.sha256`, `preprocessing_config.json`). Details in `docs/inference-pipeline.md`.

## Evaluation

Fixed record-wise split: 16 train / 7 validation / 21 test records; 4 paced records excluded.

| Accuracy | Macro-F1 | Macro ROC-AUC |
|---|---|---|
| 0.906 | 0.473 | 0.837 |

Recall for S is about 0.10 and for F about 0.06: the model largely misses these minority classes. See `docs/limitations.md` and `docs/model-verification-report.md`.

## Limitations

- Retrospective analysis of recordings only; not real-time or live.
- Decision-support only, not a diagnosis; it classifies beats, not whole recordings.
- The training notebook/code is not included in this repository.
- Calibration: ECE 0.042 -> 0.031 after temperature scaling (T=1.30, fit on validation records).
- Small minority classes: record-level bootstrap 95% CIs are S recall [0.06, 0.41] and F recall [0.00, 0.38], too wide to draw conclusions about S/F performance.
- End-to-end with XQRS-detected peaks, V recall (misses counted as errors) drops from 0.892 to 0.860.
- PVC burden per record: MAE 4.46 percentage points, Pearson r = 0.595.
- Noise: stable with white noise added to the raw signal down to 12 dB SNR, but collapses if noise reaches the z-scored window (macro-F1 0.27 at 30 dB, ~0.05 at 24 dB). Only white noise was tested. Hypothesis (untested): only S and F were augmented in training (Gaussian noise sd 0.03 among the transforms), so N and V never saw noisy beats, and noise may act as a class cue; under heavy noise the model predicts almost everything as F/S.
- Explanations pass a weight-randomization check. Deletion tests are mixed: Integrated Gradients is faithful under MoRF vs LeRF (mean probability drop 0.28 vs -0.02 at 50% deleted), while Grad-CAM is weak (0.07 vs 0.03) and, with interpolated replacement, no better than deleting a random contiguous block.
- Details: `results/extended_eval/summary.md` and `results/extended_eval/diagnostics.md`.
- Some large research artifacts (extra model variants, `results/test_examples_for_xai.npz`, >100 MB) are not in the repo; the tests that need that file are skipped when it is absent.

## Troubleshooting

- Port in use: stop the other process on 8000/5173, or change `--port` (and the proxy in `frontend/vite.config.ts`).
- TensorFlow install fails (Apple Silicon/Windows): use Python 3.11 64-bit and upgrade pip; `tensorflow==2.20.0` is pinned.
- Model not found: confirm `backend/models/neurocardio_final.keras` exists and `shasum -a 256 -c neurocardio_final.keras.sha256` passes (run in `backend/models`).
- CORS / API errors: the backend allows `http://localhost:5173`; open the app at `localhost`, not `127.0.0.1`, and keep the backend on port 8000.
- `results/` files missing: `/api/v1/model-info` reads metrics JSONs from `results/`.

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
