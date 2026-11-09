# NeuroCardio

**Understand the rhythm. Explore the signal.** A local, explainable ECG beat-classification research prototype. It is not a clinically validated diagnostic device.

## Requirements

- macOS/Linux with Python 3.11 and the project venv at `backend/.venv`.
- Node/npm and the dependencies already present in `frontend/node_modules`.
- Local model/config files in `backend/models/` and the Kaggle-derived evaluation/data files in `results/`.

No Docker, database server, or network service is required. Backend pins record the installed environment in `backend/requirements.txt`.

## Run

Terminal 1, from the project root:

```sh
MPLCONFIGDIR=/tmp/neurocardio-mpl backend/.venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Terminal 2:

```sh
cd frontend
npm run dev -- --host localhost
```

The frontend Vite proxy forwards `/api` to `127.0.0.1:8000`. FastAPI docs are at `http://127.0.0.1:8000/docs`.

## Model and data

The backend loads the verified `bilstm_aug_sqrtw` export once on startup. It expects one `(720,1)` window per beat at 360 Hz, with class order N/S/V/F and no RR input. Filtering and windowing are defined in [the inference pipeline](docs/inference-pipeline.md). `/results` is read-only source evidence. Runtime recordings and SQLite data are stored under `backend/data/`.

The model classifies beats; it does not diagnose an entire recording. Fixed-test recall is low for S and F. See [limitations](docs/limitations.md), [verification](docs/model-verification-report.md), and [the user guide](docs/user-guide.md).

## Checks

```sh
MPLCONFIGDIR=/tmp/neurocardio-mpl TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 backend/.venv/bin/python -m pytest backend/tests -v -p no:cacheprovider
cd frontend && npm run build
```

See [testing report](docs/testing-report.md) for exact results and pending browser checks.
