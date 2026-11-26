#!/usr/bin/env bash
# Stage 3. One Kaggle session. Stops cleanly after about 8 h; run it again to resume. Extra arguments are passed on.
HERE="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python}" -u "$HERE/../backend/scripts/experiments.py" --stage 3 --max-hours 8 "$@"
