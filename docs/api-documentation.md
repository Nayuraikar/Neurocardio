# Local API

Base path: `/api/v1`. The service listens on `127.0.0.1:8000` by default. FastAPI's generated schema is available at `/docs` while the local API runs. JSON error bodies use `detail`; uploads and rate/lead issues use 4xx responses.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | API and model-load status |
| GET | `/model-info` | Selected model signature and metrics read from `results/` |
| POST | `/ecg/upload` | Multipart CSV or WFDB upload; returns metadata and downsampled signal preview |
| GET | `/ecg/records` | Locally stored recordings |
| POST | `/analyses?record_id=…` | Run a retrospective analysis synchronously |
| GET | `/analyses` | Recent analyses |
| GET | `/analyses/{id}` | Status, provenance and aggregate beat counts |
| DELETE | `/analyses/{id}` | Delete an analysis and its beat predictions, explanations, reviews, report rows, and generated report files; preserves the uploaded recording |
| GET | `/analyses/{id}/beats?offset=0&limit=1000` | Beat-level predictions and latest review |
| GET | `/analyses/{id}/signal?start_sample=0&count=1200&filtered=false` | Actual uploaded ECG samples and time coordinates |
| GET | `/analyses/{id}/explanations?beat=…&class=N&method=gradcam` | Compute/cache a real model attribution |
| GET | `/reviews` | Saved reviews, optionally filter by status/analysis_id |
| POST | `/reviews` | Save a separate human label and note |
| GET | `/reports/{analysis_id}?format=pdf` | Generate/download a PDF |
| GET | `/reports/{analysis_id}?format=csv` | Generate/download beat-level CSV |

## Upload fields

For CSV, submit `files` (one CSV), `sampling_rate` and `lead` (the selected column). WFDB uploads submit a `.hea`, matching `.dat`, and optionally a matching `.atr`; all three base names must be identical and the `.hea` record identifier must match its filename. The WFDB sampling rate and available lead names are read from the header. If AAMI-mapped annotation peaks exist, upload and analysis compare them one-to-one with XQRS peaks. By default at least 90% of annotation peaks must match within 150 ms. Configure with `NEUROCARDIO_ANNOTATION_ALIGNMENT_MIN` (fraction from 0 to 1) and `NEUROCARDIO_ANNOTATION_ALIGNMENT_TOLERANCE_MS` (positive milliseconds). A failed alignment is a 422 response and the annotation is not used. The current selected model accepts exactly 360 Hz. Other rates are rejected, not resampled. The configured byte cap is 64 MiB. Signal values must be finite and non-flat, the file must support at least one full 720-sample beat window, and a peak must be detected during upload validation.

## Analysis behavior

An included annotation file is used as the peak source when it contains AAMI-mapped beat annotations and passes the XQRS alignment check; this is recorded as `wfdb_annotation`. Otherwise the service uses WFDB XQRS. Analysis is retrospective. Edge windows are skipped and counted. The JSON summary reports class counts, S+V+F beat count, review flags, peak source and uncertainty-policy provenance. It intentionally does not assign a recording-level rhythm diagnosis. Beat API indices remain zero-based; the UI displays one-based beat numbers.

Analysis currently runs synchronously; `progress` is persisted, but the endpoint does not provide a background job queue. A failure is persisted with `status: failed` and an error message. The local UI must render this state as failure. Beat indices in API responses remain zero-based. The frontend displays `beat_index + 1`; its explanation request passes the selected beat's API identifier and does not submit the display number.

## API workflow

```mermaid
sequenceDiagram
  participant UI as Local frontend
  participant API as FastAPI
  participant ML as Verified ML service
  participant DB as SQLite
  UI->>API: POST /ecg/upload
  API->>DB: Save metadata and signal path
  API-->>UI: Recording id and preview
  UI->>API: POST /analyses?record_id=id
  API->>ML: Filter, peaks, windows, probabilities
  ML-->>API: Beat predictions
  API->>DB: Store Analysis, Beat, Prediction
  API-->>UI: Completed analysis or explicit failure
  UI->>API: POST /reviews or GET /reports/{id}
```

The frontend origin is restricted to local Vite at `http://localhost:5173`. This prototype has no authentication; do not expose its API to a network.
