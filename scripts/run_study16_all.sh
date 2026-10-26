#!/usr/bin/env bash
# Study 16 runner: direction calibration, dose calibration, then the test (sanity + 400 clean test
# draws), each resuming after a crash with up to 3 restarts; the pre-committed analysis after the test.
# Stages can be given as arguments (e.g. `run_study16_all.sh dircal`) so results can be committed
# between stages, as the pre-registration requires; no argument runs all three.
PY="${PYTHON:-python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${INTROSPECTION_DATA:-$REPO/data}"
LOGDIR="$DATA/logs"
mkdir -p "$LOGDIR"
RL="$LOGDIR/study16_runner.log"
# Each stage resumes after a crash; up to 3 restarts per stage. A stage that still fails stops the queue.
for STAGE in ${@:-dircal dosecal test}; do
  ok=0
  for attempt in 1 2 3 4; do
    echo "[queue] study16 $STAGE attempt $attempt $(date '+%F %T')" >> "$RL"
    "$PY" "$REPO/scripts/study16.py" --stage $STAGE >> "$LOGDIR/study16_stdout.log" 2>&1
    code=$?
    echo "[queue] exit code $code $(date '+%F %T')" >> "$RL"
    [ $code -eq 0 ] && { ok=1; break; }
  done
  [ $ok -eq 1 ] || { echo "QUEUE_STOPPED at $STAGE" >> "$RL"; exit 1; }
  if [ "$STAGE" = "dosecal" ] && grep -q '"stop": true' "$DATA/study16/doses.json"; then
    echo "QUEUE_STOPPED: dose-calibration stop rule" >> "$RL"; exit 0
  fi
done
case " ${*:-test} " in *" test "*)
  "$PY" "$REPO/scripts/study16_analyze.py" >> "$LOGDIR/study16_analysis.log" 2>&1 ;;
esac
echo "QUEUE_DONE ${*:-all}" >> "$RL"
