## Packaging notes (2026-11-27, real study artifacts)

Run on the `ui-rebuild` working tree with changes **not yet committed**, so the commit named above is the base, not the
code that was tested.

### Other checks

| Check | Where | Result |
|---|---|---|
| Backend tests | main checkout (has the 128 MB git-ignored reference file) | 72 passed |
| Backend tests | clean copy, fresh virtualenv | 57 passed, 15 skipped, 0 failed |
| `tsc -b` and `eslint .` | main checkout and clean copy | passed |
| `npm run build` | clean copy | passed |
| `pip install -r backend/requirements.txt`, `npm ci` | clean copy | succeeded |
| API and UI from the clean copy, throwaway data directory; upload and analyse record 100; main pages | headless Chromium | no console warnings or errors, page errors or HTTP failures |
| `npm audit --omit=dev` | main checkout | 0 vulnerabilities (with development tools: 9 moderate) |

The "clean copy" is a copy of the files git tracks or would add, in an empty folder. It is not a `git clone`, because the
changes were not committed.

### What failed during this packaging, and was fixed

- First checker run on the real reports: 58 passed, 7 failed.
  - Real UI bug: `/sensitivity` scrolled sideways by 43 px at phone width.
  - Checker mistakes: case-sensitive text match against headings that CSS upper-cases; the live-model notice read before
    the study data had loaded; "SHA-256" counted as an unsourced number.
- First clean-copy backend run: 9 failed, 57 passed, 6 skipped. Those 9 tests load
  `results/test_examples_for_xai.npz` through a helper, so the skip rule did not see them. They now skip when it is absent.

### Not verified

- The 15 skipped tests (reference-model equivalence, explainability and inference robustness among them) cannot run from a
  clean clone; they need the git-ignored file.
- Linux, Windows, Node 22 and Node 24: not run. Only macOS with Python 3.11.15 and Node 26.13.1.
- `setup.sh` and `run.sh` were not run; the equivalent README commands were.
- The study itself: no training, no final test, no sealed or prediction file was opened. The checks show that pages match
  the committed reports, not that the reports are correct.
- Layout was inspected by eye on the Comparison, Model Card and Sensitivity pages only. At 390 px the wide study tables
  are cramped, although the page does not scroll sideways.

### Known data defect (left unchanged)

`per_patient` F supports in `final_report.json` are one short in each test set (see `docs/RESULTS.md`). The cause is in
`backend/scripts/evaluate_multidb.py`. Pooled scores are not affected, and the app does not display those rows.
