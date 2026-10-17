#!/usr/bin/env bash
# Runs Study 3 end to end (sanity -> 400 paired trials) with up to 3 automatic restarts.
# Every finished trial is already on disk, so a restart resumes where it stopped.
PY="${PYTHON:-python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="${INTROSPECTION_DATA:-$REPO/data}/logs"
mkdir -p "$LOGDIR"
for attempt in 1 2 3 4; do
  echo "[runner] start attempt $attempt $(date '+%F %T')" >> "$LOGDIR/study3_runner.log"
  "$PY" "$REPO/scripts/study3.py" --stage all >> "$LOGDIR/study3_stdout.log" 2>&1
  code=$?
  echo "[runner] exit code $code $(date '+%F %T')" >> "$LOGDIR/study3_runner.log"
  if [ $code -eq 0 ]; then echo "RUNNER_DONE"; exit 0; fi
done
echo "RUNNER_GAVE_UP"
exit 1
