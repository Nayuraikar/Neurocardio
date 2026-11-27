# Phase 4 checks

`npm run e2e` (from `frontend/`) starts a temporary API and UI on ports 8791 and 5191, drives the demo flow in a
headless browser, checks numbers against the artifacts, tests error paths and greps the sources. It writes
`docs/phase4-test-report.md`. It needs `backend/.venv` and a Playwright Chromium (`npx playwright install chromium`).
It does not use your database and does not open any test-set `.npz` file.
