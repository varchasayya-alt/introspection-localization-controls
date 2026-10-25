#!/usr/bin/env bash
# Study 15 test run (after dose calibration): sanity + 400 clean test draws, up to 3 restarts (resuming),
# then the pre-committed analysis.
PY="${PYTHON:-python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${INTROSPECTION_DATA:-$REPO/data}"
LOGDIR="$DATA/logs"
mkdir -p "$LOGDIR"
RL="$LOGDIR/study15_runner.log"
for attempt in 1 2 3 4; do
  echo "[queue] study15 test attempt $attempt $(date '+%F %T')" >> "$RL"
  "$PY" "$REPO/scripts/study15.py" --stage test >> "$LOGDIR/study15_stdout.log" 2>&1
  code=$?
  echo "[queue] exit code $code $(date '+%F %T')" >> "$RL"
  [ $code -eq 0 ] && break
done
"$PY" "$REPO/scripts/study15_analyze.py" >> "$LOGDIR/study15_analysis.log" 2>&1
echo "QUEUE_DONE" >> "$RL"
