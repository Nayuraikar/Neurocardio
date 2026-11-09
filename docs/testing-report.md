# Testing report

## Backend

Command:

```sh
MPLCONFIGDIR=/tmp/neurocardio-mpl TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 backend/.venv/bin/python -m pytest backend/tests -v -p no:cacheprovider
```

Final run after the Grad-CAM API/UI regression: **50 passed, 0 failed, 0 skipped, 0 errors**, 9 dependency deprecation warnings, in 193.48 seconds. The warnings are from WFDB/Pandas annotation handling and Starlette's deprecated TestClient/httpx integration. No tests were skipped or xfailed.

The suite includes real Keras loading and saved-window equivalence, preprocessing and edge behavior, MIT-BIH annotation reproduction for records 213 and 232, XQRS matching, model explainability checks, FastAPI TestClient routes, actual MIT-BIH WFDB and CSV uploads, record 213 analysis compared with Stage 3 saved probabilities, review separation, Grad-CAM caching, PDF/CSV exports, failed-analysis persistence, oversize/invalid uploads, no-detected-peak state, and persistence after constructing a second app against the same SQLite file. WFDB regression tests cover mismatched `.hea`/`.dat`/`.atr` basenames, internal header record name, sampling rate, valid record trios, and record 223 signal paired with record 228 annotation contents renamed to `223.atr`. Delete tests assert dependent beat, prediction, explanation, review and report rows plus generated report files are removed. The new Grad-CAM API test independently computes `conv3` Grad-CAM from a real record 213 beat with `tf.GradientTape`, checks max absolute difference `<=1e-5`, `target_layer=conv3`, finite norm, normalization metadata, distinct IG output/cache key, and Grad-CAM total variation under 25% of IG. The initial strict comparison differed by `1.0222e-5` because the test compared the saved NPZ window with a separately reconstructed API window. The test now obtains the exact deployed-preprocessed window from the API's stored recording; tolerance was not loosened.

The first full Stage 5 run found one API schema bug: the `RecordOut` response model removed the upload preview field. The schema was corrected to include preview data; the next full run passed all 43 tests. No model, training, or reference artifacts were changed.

## Frontend

Command:

```sh
cd frontend && npm run build
```

Result: TypeScript project build and Vite production build passed. The final output bundle is 732.30 kB (220.91 kB gzip); Vite reports its configured 500 kB chunk-size advisory. Node was `v22.12.0`; npm was `10.9.2`.

Frontend automated test count: **0 passed, 0 failed, 0 skipped** because `vitest`, `@testing-library/react`, and `jsdom` are absent from the actual `frontend/node_modules` and there is no test script in `package.json`. `playwright` and `@playwright/test` are also absent. No install was attempted, per instruction. Browser-level checks and screenshots are pending because browser tooling is unavailable.

The prompt reported that uPlot and DM font packages were installed, but inspection found no `uplot` or `@fontsource` package under `frontend/node_modules`. The UI draws the actual API waveform using SVG, with wheel/button zoom and horizontal-scroll pan; it uses the requested font-family names with local system fallbacks. Dedicated uPlot charting, DM font assets, selected-beat marker highlighting, and an attribution overlay aligned directly over the waveform remain pending. No fabricated signal or chart values are used.

## Integration checklist

| Flow / case | Evidence |
|---|---|
| Real WFDB record upload and 213 annotation-based model output | Backend API integration test; all beat probabilities compared against the Stage 3 NPZ within `1e-4` |
| Real signal exported to CSV and upload preview | Backend API integration test |
| Wrong format, missing lead/companion, 250 Hz, NaN/inf, flatline, short signal | Backend API validation tests |
| Oversize request | Backend API test using a test-specific 8-byte cap |
| No detected peaks | Explicit empty result test using a real MIT-BIH signal and an empty detector return; model is not invoked |
| Failure state and persisted error | Backend API test corrupts only its temporary uploaded copy with NaNs; restores it after assertion |
| Signal, Grad-CAM, review, PDF and CSV | Backend API integration test |
| Restart persistence | A second TestClient app reads the same SQLite file |
| Inference/XAI injected runtime failures | Pending; model predictions were not mocked |
| Multi-record upload and browser rendering | Pending |
| Browser screenshots | Pending; browser automation is not installed |
| WFDB cross-record annotation rejection | Backend tests pass: different filename base rejected; matching filename with record 228 annotation bytes on record 223 signal rejected by default alignment gate |
| Analysis deletion | Backend test confirms dependent DB rows and generated report files are removed while the source recording remains |

## Manual browser checklist (pending)

- [ ] Open the dashboard with the backend running; confirm empty states and API-backed totals.
- [ ] Upload a real CSV and WFDB pair; confirm validation errors and actual signal preview.
- [ ] Run record 213; confirm analysis status, class table, actual waveform, raw/filtered toggle and beat markers.
- [ ] Zoom and pan the waveform; select a beat and confirm probabilities and model/preprocessing provenance.
- [ ] Request Grad-CAM and Integrated Gradients; compare the displayed window and attributions with the selected beat.
- [ ] Submit a reviewed label and note; confirm original and reviewed classes remain distinct.
- [ ] Download PDF and CSV; confirm report metadata, beat rows and research-use notice.
- [ ] Confirm failed analyses remain visible as failures and do not show success metrics.

These checks are not marked passed because no browser automation or screenshot-capable tool is available in the environment.

## Grad-CAM/UI regression run

Command: `MPLCONFIGDIR=/tmp/neurocardio-mpl TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 backend/.venv/bin/python -m pytest backend/tests -v -p no:cacheprovider | tail -20` → **50 passed, 0 failed, 0 skipped, 0 errors**, 9 warnings, 193.48 s.

Command: `cd frontend && npm run build | tail -20` → TypeScript and Vite production build passed; 2,593 modules transformed. One advisory remains because the minified JS bundle exceeds 500 kB.

The reported UI showed `logits` and no gradient norm while Grad-CAM was selected. Stored API rows for analysis `b1848afd-d695-482d-97be-4930823e9d2e` showed separate IG (`logits`) and Grad-CAM (`conv3`) outputs. Audit found the frontend accepted whichever asynchronous response completed last and rendered it under the currently selected method. It now ignores stale requests and requires returned method/class to match selection. The API Grad-CAM remains an independent conv3 calculation; its cache version was bumped to avoid reuse of older entries. Browser interaction remains pending because browser automation is unavailable.
