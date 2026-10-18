#!/usr/bin/env bash
# Runs Study 6 end to end: Llama activation-size step (once), then sanity -> arms A and C, with up
# to 3 automatic restarts. Every finished trial is already on disk, so a restart resumes.
PY="${PYTHON:-python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${INTROSPECTION_DATA:-$REPO/data}"
LOGDIR="$DATA/logs"
mkdir -p "$LOGDIR"
if [ ! -f "$DATA/study6/llama_norms.json" ]; then
  echo "[runner] llama_norms $(date '+%F %T')" >> "$LOGDIR/study6_runner.log"
  "$PY" "$REPO/scripts/study6.py" --stage llama_norms >> "$LOGDIR/study6_stdout.log" 2>&1
fi
for attempt in 1 2 3 4; do
  echo "[runner] start attempt $attempt $(date '+%F %T')" >> "$LOGDIR/study6_runner.log"
  "$PY" "$REPO/scripts/study6.py" --stage all >> "$LOGDIR/study6_stdout.log" 2>&1
  code=$?
  echo "[runner] exit code $code $(date '+%F %T')" >> "$LOGDIR/study6_runner.log"
  if [ $code -eq 0 ]; then echo "RUNNER_DONE"; exit 0; fi
done
echo "RUNNER_GAVE_UP"
exit 1
