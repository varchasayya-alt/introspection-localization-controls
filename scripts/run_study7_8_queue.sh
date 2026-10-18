#!/usr/bin/env bash
# Chained queue for Studies 7 (Tülu-3-8B-SFT) and 8 (Tülu-3-8B-DPO), same protocol as Study 6.
# Per study: vectors (once) -> sanity -> arms A and C, with up to 3 automatic restarts that resume
# where they stopped. Study 8 starts when Study 7 ends, whatever Study 7's outcome.
PY="${PYTHON:-python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${INTROSPECTION_DATA:-$REPO/data}"
LOGDIR="$DATA/logs"
mkdir -p "$LOGDIR"
for S in 7 8; do
  RL="$LOGDIR/study${S}_runner.log"; SO="$LOGDIR/study${S}_stdout.log"
  if ! ls "$DATA/study$S/tulu_vectors/"*.pt >/dev/null 2>&1 || \
     [ "$(ls "$DATA/study$S/tulu_vectors/"*.pt | wc -l)" -lt 20 ]; then
    echo "[queue] vectors $(date '+%F %T')" >> "$RL"
    "$PY" "$REPO/scripts/study6.py" --study $S --stage vectors >> "$SO" 2>&1
  fi
  for attempt in 1 2 3 4; do
    echo "[queue] study $S start attempt $attempt $(date '+%F %T')" >> "$RL"
    "$PY" "$REPO/scripts/study6.py" --study $S --stage all >> "$SO" 2>&1
    code=$?
    echo "[queue] study $S exit code $code $(date '+%F %T')" >> "$RL"
    [ $code -eq 0 ] && break
  done
done
"$PY" "$REPO/scripts/study7_8_analyze.py" >> "$LOGDIR/study7_8_analysis.log" 2>&1
echo "QUEUE_DONE"
