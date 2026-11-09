# NeuroCardio stage progress

## Stage 3 — model verification and reference equivalence

| Item | Implemented | Tested | Verified | Pending | Evidence command and result |
|---|---|---|---|---|---|
| Environment and artifact identity | Yes | Yes | Yes | None | `backend/.venv/bin/python -m pip check` → no broken requirements; SHA-256 model/config copies match `/results`. |
| Model loading and signature | Yes | Yes | Yes | None | `MPLCONFIGDIR=/tmp/neurocardio-mpl backend/.venv/bin/python backend/scripts/verify_model_reference.py` → one `(None,720,1)` input, `(None,4)` output, selected model/config all `bilstm_aug_sqrtw`, RR null. |
| Saved probability equivalence | Yes | Yes | Yes | None | Same command → max diff `9.268522262573242e-06`, mean diff `2.9065758149077487e-08`, argmax agreement `1.0`; all preregistered tolerances passed. |
| Test metrics | Yes | Yes | Yes | None | Same command → accuracy `0.906140830843049`, macro-F1 `0.47268327905036756`, macro ROC-AUC `0.8370565003894894`; confusion matrix and S/V/F TPs reproduce result files. |
| Annotation-peak raw records | Yes | Yes | Yes | None | Same command → 21 records / 47,518 windows; max input difference `1.1920928955078125e-07`, prediction agreement `1.0`. |
| XQRS detector and sampling policy | Yes | Yes | Yes | None | Same command → 47,412 TP, 65 false detections, 157 missed beats at 150 ms; 250/500 Hz rejected. Detector metrics are separate from classifier metrics. |
| Robustness, explainability and raw-record pytest coverage | Yes | Yes | Yes | None | `MPLCONFIGDIR=/tmp/neurocardio-mpl TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 backend/.venv/bin/python -m pytest backend/tests -v -p no:cacheprovider` → **30 passed**, 0 failed, 0 skipped; four WFDB/Pandas deprecation warnings (118.61 s). |

The first pytest run had one test-expectation error (26 passed, 1 failed): the
test assumed the center of a linear ramp equaled the window mean. The test was
corrected to assert the center's z-score formula; production preprocessing was
unchanged. The rerun above passed all tests.

## Stage 4 — explainability

| Item | Implemented | Tested | Verified | Pending | Evidence command and result |
|---|---|---|---|---|---|
| Grad-CAM on actual `conv3` | Yes | Yes | Yes | None | `MPLCONFIGDIR=/tmp/neurocardio-mpl TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 backend/.venv/bin/python backend/scripts/generate_xai_examples.py` → finite nonzero gradient norms and normalized maps for N/S/V/F real test beats. |
| Integrated Gradients | Yes | Yes | Yes | None | Same generator command → 512-step trapezoidal IG, zero baseline, completeness errors recorded for each class, fixed-seed Gaussian baseline sensitivity measured. |
| Persisted real examples | Yes | Yes | Yes | None | Same generator command → four PNGs, compressed NPZ and measurements JSON under `docs/xai-examples/`. |
| Explainability pytest coverage | Yes | Yes | Yes | None | `MPLCONFIGDIR=/tmp/neurocardio-mpl TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 backend/.venv/bin/python -m pytest backend/tests/test_explainability.py -v` → **3 passed**; also included in the 30-test full run above. |

The first Grad-CAM upsampling implementation treated a batch as an unbatched
image and failed; dimensions were corrected and the tests passed. Initial
64-step IG showed a large completeness error on one example, so measured
integration was increased to 512 intervals. Attributions are baseline
sensitive and are explicitly not clinically validated explanations.

## Stage 5 — backend (resumed)

| Item | Implemented | Tested | Verified | Pending | Evidence command and result |
|---|---|---|---|---|---|
| FastAPI TestClient prerequisite (`httpx`) | Yes | Yes | Yes | None | On resume, `backend/.venv/bin/python -m pip check` → no broken requirements; `pip freeze` records `httpx==0.28.1`. |
| Backend API, persistence and endpoint tests | No | No | No | Yes | Implementation begins after this Stage 0 suite rerun. |

The previous network block is resolved by the user's environment update.
Stage 0 rerun command:
`MPLCONFIGDIR=/tmp/neurocardio-mpl TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 backend/.venv/bin/python -m pytest backend/tests -v -p no:cacheprovider`
→ **30 passed, 0 failed, 0 skipped**. Dependency pins in
`backend/requirements.txt` were regenerated with
`backend/.venv/bin/python -m pip freeze`; no package install or upgrade was run
in this turn.

At the end of the initial resume note, Stages 6–8 had not started; their current status is recorded below.

### Stage 5–8 continuation (2026-11-09)

| Stage / item | Implemented | Tested | Verified | Pending | Evidence command and result |
|---|---|---|---|---|---|
| Stage 5: FastAPI + SQLAlchemy + SQLite lifecycle | Yes | Yes | Yes | Background job queue | Full pytest command below: API startup, health, model info, records, analysis, signal, explanations, reviews, PDF/CSV. Model loads once in app lifespan; database persistence is exercised after app restart. |
| Stage 5: real recording inference | Yes | Yes | Yes | None for single-record acceptance | Full pytest command → real record 213 WFDB upload with annotation peaks; every shared test beat's probabilities match the Stage 3 NPZ within `1e-4`. Separate record 232 was also uploaded. |
| Stage 5: validation, review, reports | Yes | Yes | Yes | None for tested paths | Full pytest command → real CSV and WFDB upload; unsupported rate, missing lead/companion, invalid numeric data, flatline, short, oversize, no peaks, failed analysis, separate review, PDF/CSV export pass. |
| Stage 5: backend suite | Yes | Yes | Yes | None | `MPLCONFIGDIR=/tmp/neurocardio-mpl TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 backend/.venv/bin/python -m pytest backend/tests -v -p no:cacheprovider` → **43 passed, 0 failed, 0 skipped, 0 errors**, 6 warnings, 140.18 s. |
| Stage 6: React workspace pages and API-backed data | Yes | Build/type-check | Partially | Browser rendering | `cd frontend && npm run build` → TypeScript and Vite production build passed (717.83 kB JS bundle). Pages for overview, upload, recordings, results, explainability, review, reports, model information and settings read backend API data. Browser verification pending. |
| Stage 6: chart and typography assets | Partial | Build | No | uPlot, DM font assets, selected-marker highlighting, aligned attribution overlay | Filesystem check found `uplot`, `@fontsource`, `vitest`, `@testing-library/react`, and `jsdom` absent from actual `frontend/node_modules`; no installation attempted. Current waveform is an API-backed SVG with basic wheel/button zoom and horizontal scroll pan; font names use local fallbacks. |
| Stage 7: scripted API integration | Yes | Yes | Partial | Multi-record analysis; injected runtime XAI/inference failure; browser | Full pytest command above → 43 passing, including two-record upload, one real record upload-to-inference, explanation and cache, review, report, and restart path. Pending scenarios were not represented as passed. |
| Stage 8: technical documentation | Yes | Read-through | Yes | Screenshots | `README.md`, architecture, pipeline, verification, explainability, API, testing, limitations, user guide, progress, and measured sample beat excerpt are present. Browser screenshots pending. |

Stage 5's initial complete-suite run found a preview serialization failure: the Pydantic response schema omitted the preview field. The schema was corrected; the following complete suite passed all 43 tests. No source `/results`, model, or model config artifact was modified.

## Resume / remaining work

The backend and local frontend build are implemented. To continue, install no packages without an explicit environment/network change: the current environment lacks the reported Vitest/testing-library/jsdom, uPlot, and DM font modules. Basic SVG zoom and pan and two-record upload are implemented/tested; browser rendering, dedicated uPlot/DM font assets, selected-marker highlighting, attribution aligned over the waveform, multi-record analysis, and explicit injected inference/XAI failure integration cases remain pending. Playwright/screenshots are also pending. No Stage 4 work was started in this continuation.

## Manual QA corrections (2026-11-09)

| Item | Implemented | Tested | Verified | Pending | Evidence command and result |
|---|---|---|---|---|---|
| WFDB companion identity | Yes | Yes | Yes | None | Full pytest below: mismatched `.hea`/`.dat`/`.atr` basenames return 422 naming the files; internal `.hea` record name must equal the upload base; header sampling rate is checked against the 360 Hz model requirement. Valid record trios still upload. |
| Annotation-to-signal alignment | Yes | Yes | Yes | None | Full pytest below: record 223 data with record 228 annotation bytes renamed `223.atr` is rejected even though filenames match. One-to-one XQRS matching tolerance defaults to 150 ms, minimum annotation coverage defaults to 90%; configure via `NEUROCARDIO_ANNOTATION_ALIGNMENT_TOLERANCE_MS` and `NEUROCARDIO_ANNOTATION_ALIGNMENT_MIN`. Check runs at upload and again before annotations are used for analysis. |
| Delete analysis and dependents | Yes | Yes | Yes | None | Full pytest below: `DELETE /api/v1/analyses/{id}` removes Analysis, Beat, Prediction, Explanation, Review and Report rows plus generated report files; retains the uploaded ECG record. Recordings UI confirms before delete. |
| Beat numbering and reachability | Yes | Build/type-check | Partially | Browser interaction | `cd frontend && npm run build` passed. Table and selected-beat panel use one-based display; API indices remain zero-based. Beat table has class and flagged-only filters and fetches 500-beat pages until exhausted; explanation links pass the selected API beat ID. |
| Model info and analysis labels | Yes | Build/type-check | Partially | Browser interaction | Same build passed. Run button says “Run analysis”; shapes render `batch` for null dimension; warning precision/recall figures come from `/model-info`. |
| Full backend suite after regressions | Yes | Yes | Yes | None | `MPLCONFIGDIR=/tmp/neurocardio-mpl TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 backend/.venv/bin/python -m pytest backend/tests -v -p no:cacheprovider` → **48 passed, 0 failed, 0 skipped, 0 errors**, 9 dependency warnings, 175.72 s. |
| Frontend production build after regressions | Yes | Yes | Yes | Browser checks | `cd frontend && npm run build` → TypeScript/Vite passed; bundle 720.12 kB (218.28 kB gzip), Vite 500 kB advisory. |

The API regression subset ran separately first: `MPLCONFIGDIR=/tmp/neurocardio-mpl TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=backend backend/.venv/bin/python -m pytest backend/tests/test_backend_api.py -v -p no:cacheprovider` → **18 passed, 0 failed**, 7 warnings, 47.94 s. The full-suite result above is the final backend count. No package installation occurred. `/results`, the Keras model, and its preprocessing config were not modified.

Resume point: backend safety, deletion and API tests are complete. Continue with browser-level verification when browser automation is available. The installed frontend tree still lacks Vitest/testing-library/jsdom, Playwright, uPlot and DM font packages; no installs were attempted. The frontend build passes, but browser rendering, selected-marker highlighting and attribution aligned over the waveform remain unverified/pending.

## Grad-CAM manual QA regression (2026-11-09)

| Item | Implemented | Tested | Verified | Pending | Evidence command and result |
|---|---|---|---|---|---|
| API Grad-CAM target and diagnostics | Yes | Yes | Yes | None | `MPLCONFIGDIR=/tmp/neurocardio-mpl TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 backend/.venv/bin/python -m pytest backend/tests -v -p no:cacheprovider | tail -20` → independent test-side `tf.GradientTape` reference on `conv3`, max difference `<=1e-5`, finite norm, normalization and method-cache assertions passed. |
| Explainability method response race | Yes | Build | Partially | Browser interaction | `cd frontend && npm run build | tail -20` → TypeScript/Vite build passed. Stale async responses are ignored; data renders only when response method and class match the current selection. Browser reproduction remains pending. |
| Attribution clipping and IG diagnostics | Yes | Build | Partially | Browser interaction | Same build → attribution paths clip to plot bounds; UI shows absolute and percentage completeness error, warns above 5%, and displays Grad-CAM layer/norm. |
| Backend full suite | Yes | Yes | Yes | None | Full command above → **50 passed, 0 failed, 0 skipped, 0 errors**, 9 warnings, 193.48 s. |
| Frontend production build | Yes | Yes | Yes | Browser tests/screenshots | `cd frontend && npm run build | tail -20` → 2,593 modules transformed; JS 732.30 kB (220.91 kB gzip), build passed; Vite emitted its >500 kB chunk advisory. |

Manual QA described `Target layer: logits`, missing gradient norm and an IG-like trace while Grad-CAM was selected. The API route calls `ModelExplainer.gradcam_with_diagnostics()` on `conv3`; stored rows for analysis `b1848afd-d695-482d-97be-4930823e9d2e` separately identified Grad-CAM as `conv3` and IG as `logits`. The fault was frontend asynchronous response handling: a late response for a previous method/class could replace the currently selected result. The effect now ignores stale requests and validates returned method/class before display. The Grad-CAM cache version was bumped so old entries cannot be reused. Regression coverage compares the API result with independent conv3 gradients; no tolerance was loosened. An initial independent comparison exceeded the bound by `1.0222e-5` because it compared the saved NPZ window with an API-reconstructed window; the test now uses the exact deployed-preprocessed beat. No package installation occurred. `/results`, the Keras model and its preprocessing config were not changed.

Resume point: code changes and requested commands are complete. Browser-level interaction remains pending because browser automation is unavailable.
