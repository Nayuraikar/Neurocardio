#!/usr/bin/env bash
# Runs stages 1, 2, 3 in order.   --smoke: tiny end-to-end check of every step (results mean nothing).
# Other arguments (--data, --splits, --out, --max-hours, --allow-cpu, --resume-from) go to every stage.
# A full run does NOT fit in one Kaggle session: use stage1.sh, stage2.sh, stage3.sh.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python}"
SCRIPTS="$HERE/../backend/scripts"
SMOKE=0
for a in "$@"; do [ "$a" = "--smoke" ] && SMOKE=1; done
START=$(date +%s)
for stage in 1 2 3; do
  "$PY" -u "$SCRIPTS/experiments.py" --stage "$stage" "$@" || { code=$?; echo "stage $stage stopped with exit code $code"; exit $code; }
done
if [ "$SMOKE" = 1 ]; then
  # rehearse the final script on development records; the sealed file is never opened
  REST=(); for a in "$@"; do [ "$a" != "--smoke" ] && REST+=("$a"); done
  "$PY" -u "$SCRIPTS/final_test.py" --smoke --config baseline --thresholds --ensemble 3 ${REST[@]+"${REST[@]}"} | tail -12 \
    || { echo "final_test rehearsal failed"; exit 1; }
  echo "SMOKE OK in $(( $(date +%s) - START )) s"
  "$PY" -u "$SCRIPTS/gpu_check.py" ${REST[@]+"${REST[@]}"} 2>/dev/null | grep -E 'Projection|stage|run|NOTE|deterministic' || true
fi
