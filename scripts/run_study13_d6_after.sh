#!/usr/bin/env bash
# Deviation D6: after the main Study 13 queue has finished (QUEUE_DONE), run the Llama-3.2-1B
# localization with S7 on spans 2-10 (layer 1, then layer 2 secondary), then the D6 summary.
PY="${PYTHON:-python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${INTROSPECTION_DATA:-$REPO/data}"
LOGDIR="$DATA/logs"
RL="$LOGDIR/study13_runner.log"
until grep -q QUEUE_DONE "$RL" 2>/dev/null; do sleep 60; done
for S in 1b_l1 1b_l2; do
  for attempt in 1 2 3 4; do
    echo "[d6] study13_d6 $S attempt $attempt $(date '+%F %T')" >> "$RL"
    "$PY" "$REPO/scripts/study13_d6.py" --setting $S >> "$LOGDIR/study13_stdout.log" 2>&1
    code=$?
    echo "[d6] exit code $code $(date '+%F %T')" >> "$RL"
    [ $code -eq 0 ] && break
  done
done
"$PY" "$REPO/scripts/study13_d6_analyze.py" >> "$LOGDIR/study13_analysis.log" 2>&1
echo "D6_DONE" >> "$RL"
