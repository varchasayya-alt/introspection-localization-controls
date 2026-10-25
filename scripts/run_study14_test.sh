#!/usr/bin/env bash
# Study 14 test run (after calibration): sanity + 400 clean test draws, up to 3 restarts (resuming),
# then the pre-committed analysis.
PY="${PYTHON:-python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${INTROSPECTION_DATA:-$REPO/data}"
LOGDIR="$DATA/logs"
mkdir -p "$LOGDIR"
RL="$LOGDIR/study14_runner.log"
for attempt in 1 2 3 4; do
  echo "[queue] study14 test attempt $attempt $(date '+%F %T')" >> "$RL"
  "$PY" "$REPO/scripts/study14.py" --stage test >> "$LOGDIR/study14_stdout.log" 2>&1
  code=$?
  echo "[queue] exit code $code $(date '+%F %T')" >> "$RL"
  [ $code -eq 0 ] && break
done
"$PY" "$REPO/scripts/study14_analyze.py" >> "$LOGDIR/study14_analysis.log" 2>&1
echo "QUEUE_DONE" >> "$RL"
