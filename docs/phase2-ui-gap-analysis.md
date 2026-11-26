# Phase 2: UI gap analysis and change plan

Method: source review at commit f633ce6. The UI was **not** rendered, so visual and responsive behaviour is UNVERIFIED.
Baseline: `tsc` passes; `eslint src` reports 42 errors and 3 warnings; `pytest` 50 passed.
No test or spec files exist under `frontend/`.

## 1. Gap table

### Layout, state, design
| Page / component | Status | What's wrong | Planned change |
|---|---|---|---|
| Overview layout (no sidebar) | MISSING | `/` is a marketing page (HomePage.tsx:60); the dashboard is a separate route with the same top nav (App.tsx:17, :981) | `OverviewLayout`; `/` becomes the Overview |
| Workspace layout (sidebar, topbar, breadcrumbs) | MISSING | Top pill nav only (Nav.tsx) | `WorkspaceLayout` with the 7 sidebar items |
| RecordingContext | MISSING | Each page loads from URL parameters; Explainability opens empty without them (App.tsx:604) | Context holding recording, analysis and window; URL is authoritative, localStorage restores, IDs checked with the API |
| Shared theme | MISMATCH | Class colours in shared.ts:4 and repeated in App.css:835–853 and App.tsx:765–940; PDF has its own palette | One `classColors.json` (fill and text colour per class), read by React and the PDF generator |
| Palette and fonts | MISMATCH | Three token sets across index.css, App.css, pulmify.css; DM Sans and Helvetica | One `tokens.css` with the spec palette; Inter plus JetBrains Mono, self-hosted |
| API access | PARTIAL | Per-component axios calls, `any` types | Typed client and one fetch hook returning loading, error, data, partial |

### Overview
| Item | Status | What's wrong | Planned change |
|---|---|---|---|
| Branding text | MISSING | "From Raw ECG to Clear Decisions" (HomePage.tsx:68) overclaims | "Interpretable AI-powered ECG analysis" with research-use notice |
| System status | MISSING | `/health` has model state only; UI does not show it; no DB check | `GET /api/status`: model loaded, DB query succeeds; API state from the request |
| Four stats | MISMATCH | Beats, class share, flags, analysed recordings (App.tsx:101–118) | `GET /api/overview`: recordings, completed analyses, review required, model version |
| Model scores card | MISMATCH | Typed-in accuracy without baseline (App.tsx:126) | Remove. Link to Model Performance |
| Workspace launcher | MISMATCH | Five identical-style cards (App.tsx:90–122) | Editorial layout: Analysis Studio, Dataset Explorer, Model Performance |
| Recent Activity | PARTIAL | Analyses only; one request per 1,000 beats to count reviews (App.tsx:52–77) | Server-side feed from stored timestamps: upload, analysis, review, report |
| Illustrative traces, marketing sections | MISMATCH | HomePage.tsx:19–58, App.tsx:111 | Delete; no decorative ECG |

### Analysis Studio
| Item | Status | What's wrong | Planned change |
|---|---|---|---|
| 3-stage flow | MISSING | Two buttons on one page (App.tsx:159); a second one-click uploader (UploadCard.tsx) | One page: Upload → Process → Results |
| File name and size | IMPLEMENTED | App.tsx:154 | Keep |
| Validation status, metadata | PARTIAL | Success banner only (App.tsx:158); ID, fs, lead, duration not shown | Metadata sheet from the upload response |
| Remove / replace | PARTIAL | Replace by re-picking; no remove | Per-file remove and "replace all" |
| Specific errors | PARTIAL | Backend messages are specific (recording_service.py) but shown as one plain string | Error codes with title, cause, fix; backend detail shown underneath |
| Paced / missing-MLII input | MISSING | Accepted silently (recording_service.py:139) | Decision D5 |
| 5-step pipeline with live progress | MISSING | Analysis blocks one request; progress is written but not observable (analysis_service.py:44) | Run in a background thread; stage stored in `summary_json`; UI polls. Completed runs on a 30-minute record took 4.8–6.4 s |
| Step labels | MISMATCH (spec vs pipeline) | See C1 | Truthful labels from `/api/model/config` |
| Raw vs processed waveform | MISSING | Raw preview only (App.tsx:160) | Raw, filtered, and one z-scored 720×1 window |
| Results | PARTIAL | Redirects to a long results page with frontend-made verdicts (App.tsx:444, :474) | Windows analysed, per-class counts, per-window probabilities, "V-class windows present: n". Remove verdicts |
| Beat source | MISSING | `peak_source` never shown | Always show: annotation file or XQRS detector |

### ECG Viewer
| Item | Status | What's wrong | Planned change |
|---|---|---|---|
| Waveform as centrepiece | PARTIAL | Shares a long page with 6 other sections (App.tsx:435–557) | Dedicated page |
| Zoom, pan | PARTIAL | Works inside a loaded 10,000-sample chunk (App.tsx:14) | Keep EcgPlot logic; load chunks automatically; full-record navigator |
| Window selection | PARTIAL | Click selects a beat; the 720-sample extent is not what is shaded (±0.25 s) | Draw the true 2 s window; keyboard previous/next |
| Beat markers | MISMATCH | Reviewer class shown with model confidence (EcgPlot.tsx:217, :230) | Model class and confidence together; reviewer label as a separate marker |
| Time / amplitude readout | MISMATCH | Unit "mV" hardcoded (EcgPlot.tsx:230); CSV units are unknown | Unit from recording metadata, else "signal units"; z-score view labelled as unitless |
| Raw / processed toggle | IMPLEMENTED | App.tsx:439 | Keep, labelled "Raw" and "Bandpass filtered" |
| Window table | PARTIAL | Row click jumps and also opens the drawer (App.tsx:468) | Virtualised table; row click jumps |
| Summary | PARTIAL | Distribution exists; 24-bin timeline; no V list or review list | Distribution, V-window locations, windows needing review |
| V window → XAI | MISSING | Drawer, or a link after selecting | Direct link to `/explain?...&window=` |

### XAI Studio
| Item | Status | What's wrong | Planned change |
|---|---|---|---|
| Two implementations | MISMATCH | Page (App.tsx:561) and drawer (ExplainDrawer.tsx) disagree | One page; delete drawer |
| Header | PARTIAL | No ground truth | Derive reference label at read time from the stored `.atr`; otherwise "No reference annotation" |
| Tabs, signed IG | PARTIAL | Select box on page; drawer shows absolute IG (ExplainDrawer.tsx:117) | Two tabs; signed IG default; positive and negative separated |
| Method comparison | MISSING | none | Aligned views plus rank correlation computed from the two stored maps |
| Top influential regions | MISSING | "Beats driving this decision" is something else and false (ExplainDrawer.tsx:161) | Top contiguous regions within this window, with ms offsets and signed share. Delete the beat ranking |
| Completeness | PARTIAL | Shown with a >5% warning (App.tsx:846); fallback `512` (:837) | Always show per window; no fallbacks |
| Evidence panel | MISSING | none | Faithfulness and randomization results from `xai_evaluation.json`, with n, 50-step caveat, and C7 |
| Interpretation text | PARTIAL | "highlighted regions affect the prediction" (ExplainDrawer.tsx:8) | "Explains model behaviour, not medical causation"; note neighbouring beats in the window |
| All-zero Grad-CAM | IMPLEMENTED | App.tsx:745 | Keep |
| IG latency | PARTIAL | Roughly 20 s per window in the audit's eager path | Compiled, batched IG, still 512 steps; test equality with the current output |

### Dataset Explorer
| Item | Status | What's wrong | Planned change |
|---|---|---|---|
| Whole page | MISSING | Only prose on the home page (HomePage.tsx:131) | New page |
| Facts, distribution, imbalance | MISSING | | `GET /api/dataset/summary`; show usable and annotated counts, labelled; show F/S record concentration |
| Representative waveforms | MISSING | | Per class: the window closest to the class mean (stated on screen), plus record and position |
| Record / class / time controls | MISSING | | Reads tracked `results/mitdb` records |
| Preprocessing effect | MISSING | | Raw, filtered, z-scored for the chosen window, computed by the app's own functions |

### Model Performance
| Item | Status | What's wrong | Planned change |
|---|---|---|---|
| Data source | MISMATCH | Reads `results/*.json` (model_info.py), not `artifacts/metrics.json` | `GET /api/model/metrics`; flag stale if model or config hash differs |
| Architecture diagram | MISMATCH | Typed in (ModelFlow.tsx:5–17, :58–59) | Generated from loaded model layers and preprocessing config |
| Headline metrics | MISMATCH | Accuracy shown without baseline (App.tsx:979), with silent fallbacks | `HeadlineMetrics`: accuracy, 0.887 baseline (0.917 for validation) and macro-F1 as one unit |
| S and F weakness | PARTIAL | Small warning under a table | Top-of-page S and F recall with support, bootstrap CI and note |
| Metrics table | PARTIAL | P/R/F1/support only | Add specificity, ROC-AUC, PR-AUC, macro, weighted |
| Confusion matrix | PARTIAL | Raw `<pre>` text | Heatmap with counts, row percentages, support check |
| ROC, PR curves | MISSING | | SVG from artifact points; PR shows prevalence baseline |
| Train / val curves | MISSING | | History: loss and accuracy (both), macro-F1 (validation only; no training series exists); note training data were augmented |
| Model info | PARTIAL | No loss, imbalance method, split, eval date | From artifact; label loss and weighting UNVERIFIED |
| Validation vs test | MISSING | Test only, plus grouped CV unlabelled as unverified | Separate sections; CV removed from headline |
| Split panel | MISSING | | IDs and counts, disjointness result, wording in C6, split-selection statement |
| Calibration, review flag | MISSING | | ECE and the 3% target vs 5.2% observed |
| Metric click → detail | MISSING | | `/models/:metric` |

### History, review, reports
| Item | Status | What's wrong | Planned change |
|---|---|---|---|
| Search and filters | PARTIAL | Search on Recordings only (App.tsx:963); no filters | One History table: All, Completed, Failed, Needs review |
| Open stored results | PARTIAL | Link by analysis exists | Row sets context and opens stored results |
| Review workspace | PARTIAL | Queue table and inline panel (App.tsx:556, :977) | Per-analysis workspace: window labels, notes |
| Mark reviewed | MISSING | No analysis-level state | New `analysis_reviews` table |
| Separate assessment storage | IMPLEMENTED | `reviews` table (entities.py) | Keep; predictions stay read-only |
| Reset behaviour | MISMATCH | Null-label review counts as reviewed (App.tsx:398) | Store status `withdrawn`; beat returns to queue |
| Report: recording info | PARTIAL | No upload time | Complete |
| Report: preprocessing | PARTIAL | Version string and hash only | From model config |
| Report: waveform | MISSING | | Raw and filtered excerpts |
| Report: class distribution | PARTIAL | Raw JSON dump (report_service.py) | Table and bar chart |
| Report: window predictions, confidence | IMPLEMENTED (PDF) | | Keep; CSV gains metadata |
| Report: XAI | MISSING | | Decision D6 |
| Report: reviewer notes | MISSING (PDF) / IMPLEMENTED (CSV) | | Add section |
| Report: version + timestamp | PARTIAL | PDF lacks timestamp; CSV lacks both | Add to both |
| Report preview page | MISSING | Download links only (App.tsx:978) | `/history/:id/report` |
| Settings page | not in spec | Typed-in facts (App.tsx:980) | Delete |

### States
| State | Status | Gap and change |
|---|---|---|
| Empty | PARTIAL | Reports and some tables only. Shared component on every screen |
| Loading skeleton | MISSING | Text placeholders. Skeletons per layout |
| Success | PARTIAL | Upload and review only |
| Error | PARTIAL | Reviews and Reports have no catch (App.tsx:977–978). Error component with retry; backend-down banner |
| Partial failure | PARTIAL | e.g. App.tsx:120. Per-panel failure; also stale metrics, missing artifacts, non-MLII lead |
| Review-ready | MISSING | Banner with pending count and link |

### Backend endpoints
| Spec endpoint | Status | Today | Change |
|---|---|---|---|
| POST /api/upload | MISMATCH (path) | /api/v1/ecg/upload | Add; add error codes |
| POST /api/analyze | MISMATCH | /api/v1/analyses?record_id | Add; asynchronous with polling |
| GET /api/recordings | MISMATCH | /api/v1/ecg/records | Add with latest analysis and review state |
| GET /api/recordings/{id} | MISSING | | Add |
| GET /api/recordings/{id}/predictions | MISSING | by analysis only | Add (latest completed; `?analysis_id=`) |
| GET /api/recordings/{id}/explanations | MISSING | by analysis; GET computes and stores | Add (same compute-and-cache behaviour) |
| GET /api/model/metrics | MISSING | /api/v1/model-info | Add |
| GET /api/dataset/summary | MISSING | | Add |
| POST /api/reviews | MISMATCH (path) | /api/v1/reviews | Add |
| GET /api/reports/{id} | MISMATCH (path) | /api/v1/reports/{id} | Add |
| Also needed | MISSING | | status, overview, model/config, model/xai-evaluation, dataset record signal and examples, analysis review |
| Version and timestamp on every result | PARTIAL | Explanations, reviews, reports lack a model version | Include it in every response; for old rows derive it through the analysis |

## 2. Where the spec conflicts with the evidence
- **C1. Pipeline order.** There is no record-level normalization step. The real order is Raw → Validation → Bandpass filter → Segmentation (R-peak centred) → Input standardization (per-window z-score, 720×1). I propose "Bandpass filter" in place of a separate "Normalization" step.
- **C2. Config as text source.** "CNN-BiLSTM" and 720×1 come from the loaded model's layers; normalization comes from `preprocessing_config.json`. The config string does not prove behaviour, so a test will check `extract_windows` yields mean 0 and SD 1 and that code constants equal the config.
- **C3. Ventricular flag** means only "one or more windows were predicted V". No threshold, no diagnosis.
- **C4. "Review required"** counts completed analyses with pending flagged windows and no "marked reviewed" entry. The pending window count is shown beneath.
- **C5. Representative waveform** is a stated selection rule, not a claim that it is typical.
- **C6. Split panel wording.** The spec's "test patients not used in training or tuning" is only partly verified. Proposed text: "Test patients are disjoint from training and validation patients (checked). Project files record that model selection and calibration used validation data; test non-use during tuning cannot be verified because training code is not in the repository. The patient split was chosen using validation and test class counts (no model predictions)." The panel will also say that test scores exist for all nine variants.
- **C7. XAI evidence** used 50-step IG on 120 windows sampled by predicted class; the app uses 512. This is stated beside the results.

## 3. Rules enforced in code
1. Accuracy appears only through `HeadlineMetrics` (with baseline and macro-F1), including report and README.
2. S and F recall and a weakness note sit above the fold on Model Performance.
3. The split statement is read from the artifact. `evaluate_metrics.py` must first export `split_selection` and the variant test scores.
4. Architecture and preprocessing text come from `/api/model/config`.
5. Every displayed number carries a `data-source` attribute for the Phase 4 checker.

## 4. File-level plan

### Frontend: create (`frontend/src/`)
- `theme/`: `classColors.json`, `tokens.css`, `theme.ts`
- `api/`: `client.ts`, `types.ts`, `useApi.ts`
- `context/RecordingContext.tsx`
- `layouts/`: `OverviewLayout.tsx`, `WorkspaceLayout.tsx`, `Sidebar.tsx`, `Topbar.tsx`, `Breadcrumbs.tsx`
- `components/states/`: `Empty`, `Skeleton`, `ErrorState`, `PartialNotice`, `SuccessBanner`, `ReviewReady`
- `components/waveform/`: `WaveformViewer` (from EcgPlot), `Navigator`, `AttributionPlot`
- `components/`: `ClassBadge`, `ClassDistribution`, `WindowTable`, `HeadlineMetrics`, `ConfusionMatrix`, `CurveChart`, `ArchitectureDiagram`, `UploadDropzone`, `PipelineSteps`, `SplitPanel`
- `pages/`: `Overview`, `AnalysisStudio`, `Viewer`, `ExplainStudio`, `DatasetExplorer`, `ModelPerformance`, `MetricDetail`, `History`, `ReviewWorkspace`, `ReportView`

Charts stay hand-written SVG.

### Frontend: rewrite
`App.tsx` (router only), `main.tsx`, `index.html` (title, favicon link), `package.json` (add Inter and JetBrains Mono; remove DM fonts and the seven packages with no imports: plotly.js, react-plotly.js, recharts, uplot, date-fns, react-dropzone, zod).

### Frontend: delete
`HomePage.tsx`, `Nav.tsx`, `UploadCard.tsx`, `ModelFlow.tsx`, `ExplainDrawer.tsx`, `EcgPlot.tsx` (after porting), `shared.ts`, `App.css`, `pulmify.css`, `index.css`, `assets/hero.png`, `assets/react.svg`, `assets/vite.svg`, `public/icons.svg`.

### Backend: create
- `app/api/`: `system.py`, `recordings.py`, `model.py`, `dataset.py` (new routes)
- `app/services/`: `model_service.py`, `dataset_service.py`, `overview_service.py`, `explanation_service.py`, `review_service.py`
- `tests/`: `test_api_ui_endpoints.py`, `test_config_truth.py`

### Backend: modify
- `api/routes.py`: serve existing and spec paths under `/api` and keep `/api/v1` (D1)
- `main.py`: router mounting, CORS
- `models/entities.py`: add `AnalysisReview`. New table only; no column changes
- `services/analysis_service.py`: stages, background run, store a relative policy path
- `services/recording_service.py`: error codes, input warnings (D5)
- `services/report_service.py`: 9 sections, CSV metadata, shared colours
- `ml/explainability.py`: compiled batched IG
- `schemas/api.py`, `utils/serializers.py`, `core/config.py`
- `scripts/evaluate_metrics.py`: export split selection and variant scores; regenerate the artifact

### Backend: delete
`services/model_info.py`, after `/api/v1/model-info` is removed with its test updated (D1).

### Repository
Track `backend/artifacts/metrics.json`, `xai_evaluation.json`, `source_of_truth.md` (a fresh clone needs the first two). Ignore `xai_eval_partial.json` and `*.log`. Update `README.md` (baseline beside accuracy; replace the Grad-CAM claim), `docs/limitations.md`, `docs/api-documentation.md`.
`data/neurocardio.sqlite3` is backed up before the first run of new code.

## 5. Build order (one commit each)
0. Audit scripts (staged now) and artifacts
1. Theme, layouts, context, API client, `/api` paths, status, model config
2. Overview and its endpoints
3. Studio and Viewer; asynchronous analysis
4. XAI Studio; compiled IG
5. Dataset and Model Performance; artifact additions
6. History, Review, Reports
7. States, accessibility, responsive polish; delete leftover files

From step 1 until step 6, pages not yet rebuilt show a "not rebuilt yet" placeholder.

## 6. Decisions (defaults apply if you just approve)
| # | Question | Default |
|---|---|---|
| D1 | API paths | UI uses `/api/...`; old routes stay reachable at `/api/v1` except model-info, which is removed |
| D2 | Pipeline step 3 | "Bandpass filter" (C1) |
| D3 | Recording-level verdicts | Remove |
| D4 | Split wording | C6 text |
| D5 | Paced or non-MLII input | Reject if the `.atr` contains paced-beat symbols; warn clearly for a non-MLII lead; state that paced rhythm cannot be detected without annotations |
| D6 | XAI in reports | Include explanations already stored for the analysis; otherwise say none was generated |
| D7 | Phase 4 browser test | Use the installed Playwright package; this needs a browser download |
