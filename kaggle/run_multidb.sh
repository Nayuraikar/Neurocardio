#!/usr/bin/env bash
# Multi-database study: all runs on both GPUs, then the validation evaluation. Run again to resume.
#   --smoke      tiny end-to-end check, including a rehearsal of final_test.py (results mean nothing)
#   --estimate   print the T4 time projection
# Other arguments (--data, --out, --max-hours, --workers, --resume-from, --allow-cpu) are passed on.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python}"
SCRIPTS="$HERE/../backend/scripts"
"$PY" -u "$SCRIPTS/run_multidb.py" "$@"
case " $* " in *" --smoke "*)
  REST=(); for a in "$@"; do [ "$a" != "--smoke" ] && REST+=("$a"); done
  # rehearsal on validation beats standing in as "test"; the sealed files are never opened
  RARGS=(); skip=0
  for a in ${REST[@]+"${REST[@]}"}; do
    if [ "$skip" = 1 ]; then RARGS+=("$a"); skip=0; continue; fi
    case "$a" in --data|--out|--splits|--model-config) RARGS+=("$a"); skip=1 ;; --allow-cpu) RARGS+=("$a") ;; esac
  done
  "$PY" -u "$SCRIPTS/final_test.py" --smoke --secondary-all ${RARGS[@]+"${RARGS[@]}"} > /tmp/final_rehearsal.log 2>&1 \
    || { tail -30 /tmp/final_rehearsal.log; echo "final_test rehearsal failed"; exit 1; }
  echo "SMOKE OK in ${SECONDS} s (including the final_test rehearsal)"
  "$PY" -u "$SCRIPTS/run_multidb.py" --estimate
;; esac
