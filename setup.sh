#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3.11 -m venv backend/.venv 2>/dev/null || python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install --upgrade pip
backend/.venv/bin/python -m pip install -r backend/requirements.txt
(cd frontend && npm install)
echo "Setup done. Run ./run.sh"
