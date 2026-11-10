#!/usr/bin/env bash
cd "$(dirname "$0")"
source backend/.venv/bin/activate

(cd backend && exec uvicorn app.main:app --port 8000 > ../backend.log 2>&1) &
BACK=$!
trap 'kill $BACK 2>/dev/null' EXIT INT TERM

echo "Starting backend (first run can take 30-60s)..."
for i in $(seq 1 60); do
  curl -sf http://localhost:8000/api/v1/health >/dev/null && break
  kill -0 $BACK 2>/dev/null || { echo "Backend crashed:"; tail -30 backend.log; exit 1; }
  sleep 2
done
echo "Backend ready. Starting UI at http://localhost:5173"

cd frontend && npm run dev
