#!/usr/bin/env bash
cd "$(dirname "$0")"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/neurocardio-mpl"
backend/.venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 &
API=$!
trap 'kill $API 2>/dev/null' EXIT INT TERM
cd frontend && npm run dev -- --host localhost
