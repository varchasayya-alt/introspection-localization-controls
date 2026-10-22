#!/usr/bin/env bash
# Overnight queue for Study 13: Llama-3.2-1B (layer 1), Llama-3.2-3B (layer 2), then the secondary
# Llama-3.2-1B layer-2 arm. Up to 3 restarts per setting, which resume. Analysis at the end.
PY="${PYTHON:-python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${INTROSPECTION_DATA:-$REPO/data}"
LOGDIR="$DATA/logs"
mkdir -p "$LOGDIR"
RL="$LOGDIR/study13_runner.log"
for S in 1b_l1 3b_l2 1b_l2; do
  for attempt in 1 2 3 4; do
    echo "[queue] study13 $S attempt $attempt $(date '+%F %T')" >> "$RL"
    "$PY" "$REPO/scripts/study13.py" --setting $S --stage all >> "$LOGDIR/study13_stdout.log" 2>&1
    code=$?
    echo "[queue] exit code $code $(date '+%F %T')" >> "$RL"
    [ $code -eq 0 ] && break
  done
done
"$PY" "$REPO/scripts/study13_analyze.py" >> "$LOGDIR/study13_analysis.log" 2>&1
echo "QUEUE_DONE" >> "$RL"
