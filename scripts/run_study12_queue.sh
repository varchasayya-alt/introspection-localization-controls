#!/usr/bin/env bash
# Overnight queue for Study 12: Tülu final, Llama-3.1-8B-Instruct, then (optional, last) Tülu SFT.
# Per model: refusal direction (once) -> sanity -> arms 0, a, b, c. Up to 3 restarts per model,
# which resume. Analysis at the end. The ablation is applied only at runtime (hooks).
PY="${PYTHON:-python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${INTROSPECTION_DATA:-$REPO/data}"
LOGDIR="$DATA/logs"
mkdir -p "$LOGDIR"
RL="$LOGDIR/study12_runner.log"
[ -f "$DATA/study12/refusal_prompts.json" ] || "$PY" "$REPO/scripts/study12.py" --stage prep >> "$LOGDIR/study12_stdout.log" 2>&1
for M in final llama sft; do
  for attempt in 1 2 3 4; do
    echo "[queue] study12 $M attempt $attempt $(date '+%F %T')" >> "$RL"
    "$PY" "$REPO/scripts/study12.py" --model $M --stage all >> "$LOGDIR/study12_stdout.log" 2>&1
    code=$?
    echo "[queue] exit code $code $(date '+%F %T')" >> "$RL"
    [ $code -eq 0 ] && break
  done
done
"$PY" "$REPO/scripts/study12_analyze.py" >> "$LOGDIR/study12_analysis.log" 2>&1
echo "QUEUE_DONE" >> "$RL"
