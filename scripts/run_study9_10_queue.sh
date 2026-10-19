#!/usr/bin/env bash
# Overnight queue: Study 10 (SFT, DPO, final Tülu) then Study 9 (base Llama-3.1-8B: vectors once,
# then sanity/positive control -> arms A and C). Up to 3 automatic restarts per step, which resume.
PY="${PYTHON:-python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${INTROSPECTION_DATA:-$REPO/data}"
LOGDIR="$DATA/logs"
mkdir -p "$LOGDIR"
RL="$LOGDIR/study9_10_runner.log"
run() {  # run with up to 3 restarts
  for attempt in 1 2 3 4; do
    echo "[queue] $* attempt $attempt $(date '+%F %T')" >> "$RL"
    "$PY" "$@" >> "$LOGDIR/study9_10_stdout.log" 2>&1
    code=$?
    echo "[queue] exit code $code $(date '+%F %T')" >> "$RL"
    [ $code -eq 0 ] && return 0
  done
  return 1
}
for C in sft dpo final; do run "$REPO/scripts/study10.py" --ckpt $C --stage all; done
if [ "$(ls "$DATA/study9/base_vectors/"*.pt 2>/dev/null | wc -l)" -lt 20 ]; then
  run "$REPO/scripts/study9.py" --stage vectors
fi
run "$REPO/scripts/study9.py" --stage all
"$PY" "$REPO/scripts/study9_10_analyze.py" >> "$LOGDIR/study9_10_analysis.log" 2>&1
echo "QUEUE_DONE" >> "$RL"
