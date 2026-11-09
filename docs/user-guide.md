# User guide

## Start locally

From the project root:

```sh
MPLCONFIGDIR=/tmp/neurocardio-mpl backend/.venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

In another terminal:

```sh
cd frontend && npm run dev -- --host localhost
```

Open the Vite URL printed by npm. The frontend proxies `/api` requests to the local backend. Analysis data and uploads stay in `backend/data/`.

## Run an analysis

1. Open **New analysis**.
2. Select one CSV and provide its sample rate and lead column, or select the companion WFDB `.hea`, `.dat`, and optional `.atr` files together. WFDB base names must match; AAMI annotation peaks must align with detected XQRS peaks (default minimum coverage 90% within 150 ms).
3. Validate to upload the signal and view the actual downsampled signal preview.
4. Start offline analysis. The selected model requires MLII at 360 Hz; unsupported rates are rejected.
5. Review the result status, predicted beat classes, probability table, edge-window count, and peak source. A failed run is shown as failed.
6. Filter the beat table by class or review flag, and load subsequent pages as needed. Table and selected-beat labels are one-based for display; the API retains zero-based indices. Select a beat to inspect its probability vector. Open its attribution view for Grad-CAM or Integrated Gradients.
7. Add a human-reviewed class and note in the selected-beat panel. The reviewed label remains separate from the original model prediction.
8. Export PDF or CSV from Reports. On Recordings, delete a saved analysis with the confirmation dialog; its dependent beats, predictions, explanations, reviews, report rows and generated report files are deleted, while the recording itself remains.

## Read results carefully

Class probabilities describe model outputs for beat windows. They do not establish a recording-level rhythm diagnosis. Review flags are exploratory and not clinically calibrated. S and F recall are weak in retrospective evaluation. Treat every output as research information for review, not medical advice or a care decision.

## Sample measured result

The saved MIT-BIH test reference includes record 213, sample 478: reference label N, predicted N, with saved probabilities N `0.9996776581`, S `0.0000098508`, V `0.0000077557`, F `0.0003046927`. This is a reference beat example, not a newly uploaded recording. Fresh inference reproduced the saved outputs within the preregistered probability tolerance; see [model verification](model-verification-report.md).
