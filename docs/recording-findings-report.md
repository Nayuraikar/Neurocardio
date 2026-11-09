# Recording-level findings

## Aggregation behavior

For a completed analysis, the results page retrieves every saved beat prediction through the existing paginated beats API. It derives N/S/V/F totals, percentages, abnormal-class totals (S+V+F), review-flag count, and first/last occurrences from those rows. A summary is shown only when the loaded prediction count matches the backend's analysed-beat count, every prediction has a supported class and finite non-negative timestamp, and the four class counts sum to that beat count. Failed, incomplete, empty, unavailable, or inconsistent analyses show an insufficient-analysis state instead.

Class percentages use `class count / analysed beats × 100` and are rounded to one decimal place for display. Abnormal-class percentage uses `(S+V+F) / analysed beats × 100`. Review counts use the prediction's review flag; a human-reviewed label remains separate and does not replace the original prediction or its probabilities.

Class-specific findings include the number and percentage of beats, first and last one-based beat numbers and timestamps, and a link that filters the detailed table to that predicted class. The timeline divides the actual recording duration into 24 equal segments. Each segment's N/S/V/F composition is calculated from saved beat timestamps; selecting it loads the actual signal around a real beat and filters the detailed classification table. The detailed table retains the four original model probabilities. No clinical threshold or rhythm diagnosis is inferred.

## Verification with recording 228

Queried the local API for the completed analysis and all paginated saved beat rows for recording `228.hea`. The recording duration returned by the API was 1,805.556 seconds. The downloaded rows and their class totals matched the analysis's 2,051 analysed beats:

| Class | Description | Count | Percentage | First occurrence | Last occurrence |
|---|---|---:|---:|---|---|
| N | Normal beats | 1,690 | 82.4% | Beat 1 · 1.203 s | Beat 2,050 · 1,803.078 s |
| S | Supraventricular ectopic beats | 0 | 0.0% | None | None |
| V | Ventricular ectopic beats | 353 | 17.2% | Beat 21 · 17.497 s | Beat 2,051 · 1,803.700 s |
| F | Fusion beats | 8 | 0.4% | Beat 644 · 555.097 s | Beat 1,907 · 1,679.511 s |

The class counts sum to 2,051. The abnormal-class total is 361 (17.6%), and 218 beats are flagged for review. Percentages are rounded to one decimal place. The output supports the phrase “Ventricular ectopic activity detected” as a model-based beat-class finding only; it does not establish a PVC diagnosis, ventricular tachycardia, or another rhythm diagnosis.

## Implementation and verification status

- Files changed: `frontend/src/App.tsx`, `frontend/src/App.css`, `docs/rhythm-interpretation-report.md`, and this report.
- Backend, model, preprocessing, database schema, prediction logic, and stored analysis records were not modified.
- `cd frontend && npm run build` — passed after the findings and timeline changes.
- A read-only local API check fetched all 2,051 prediction rows for the completed 228 analysis and verified the class totals, percentages, flagged-beat count, duration, and first/last timestamps shown above.
- A browser check loaded the actual 228 results. It showed the recording summary, all 2,051 detailed rows, and 24 timeline segments. Selecting the ventricular finding filtered the table to 353 V beats. Selecting an abnormal timeline segment filtered to V and selected beat 21 at 17.497 seconds; its selected-beat details and actual waveform were present.
- The UI states that reviewed labels are saved separately from the model prediction/probabilities. No review was submitted, so review persistence was not re-tested for this change.

The model's beat-class output is experimental and has weak S and F recall. These findings are not a clinical diagnosis and absence of a predicted class does not rule out that class or a rhythm condition.
