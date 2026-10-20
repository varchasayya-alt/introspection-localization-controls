#!/usr/bin/env bash
# Overnight queue for Study 11: Tülu SFT, DPO, final, then (optional, last) Llama-3.1-8B-Instruct.
# Up to 3 automatic restarts per model, which resume; analysis at the end.
PY="${PYTHON:-python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${INTROSPECTION_DATA:-$REPO/data}"
LOGDIR="$DATA/logs"
mkdir -p "$LOGDIR"
RL="$LOGDIR/study11_runner.log"
for M in sft dpo final llama; do
  for attempt in 1 2 3 4; do
    echo "[queue] study11 $M attempt $attempt $(date '+%F %T')" >> "$RL"
    "$PY" "$REPO/scripts/study11.py" --model $M --stage all >> "$LOGDIR/study11_stdout.log" 2>&1
    code=$?
    echo "[queue] exit code $code $(date '+%F %T')" >> "$RL"
    [ $code -eq 0 ] && break
  done
done
"$PY" "$REPO/scripts/study11_analyze.py" >> "$LOGDIR/study11_analysis.log" 2>&1
echo "QUEUE_DONE" >> "$RL"
