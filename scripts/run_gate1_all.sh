#!/usr/bin/env bash
# Runs Gate 1 end to end (sanity -> A -> B -> controls) with up to 3 automatic restarts.
# Every finished trial is already on disk, so a restart resumes where it stopped.
PY=$HOME/.venvs/introspection/Scripts/python
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR=$HOME/IntrospectionData/logs
mkdir -p "$LOGDIR"
for attempt in 1 2 3 4; do
  echo "[runner] start attempt $attempt $(date '+%F %T')" >> "$LOGDIR/gate1_runner.log"
  "$PY" "$REPO/scripts/gate1.py" --stage all >> "$LOGDIR/gate1_stdout.log" 2>&1
  code=$?
  echo "[runner] exit code $code $(date '+%F %T')" >> "$LOGDIR/gate1_runner.log"
  if [ $code -eq 0 ]; then echo "RUNNER_DONE"; exit 0; fi
done
echo "RUNNER_GAVE_UP"
exit 1
