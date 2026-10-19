#!/usr/bin/env bash
# RIDI world-frame EqNIO filter runner -- uses EqNIO's stock main_filter.py (NOT the RIEKF body port).
# Usage: run_per_seq_ridi_wf_eqnio.sh <model_dir> <out_dir>
# Ported from Server B (chankyokim-Alienware). All machine-specific paths are now variables
# that can be overridden from the environment; defaults are relative to the repository root
# (GINIO/). PY = python interpreter, MODEL_DIR = directory with model_torchscript.pt +
# parameters.json (e.g. checkpoints/tlio_ginio_w_ln189), ROOT = dataset root, OUT = output root.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
PY=${PY:-python3}
SRC_DIR="$HERE"
MODEL_DIR=${1:?model_dir required}
OUT=${2:?out_dir required}

ROOT=${ROOT:-$REPO_ROOT/local_data_ridi_wf}
LIST=$ROOT/test_list.txt

mkdir -p "$OUT"
LOG=${LOG:-/tmp/eqnio_filter_ridi_wf_$(basename "$OUT").log}
: > "$LOG"
cd "$SRC_DIR"

IDX=-1
while read SEQ; do
  IDX=$((IDX+1))
  [ -z "$SEQ" ] && continue
  NPY="$OUT/$SEQ/not_vio_state.txt.npy"
  if [ -f "$NPY" ]; then
    echo "[$(date +%H:%M:%S)] skip idx=$IDX seq=$SEQ" | tee -a "$LOG"
    continue
  fi
  rm -rf "$OUT/$SEQ"
  echo "[$(date +%H:%M:%S)] RUN idx=$IDX seq=$SEQ" | tee -a "$LOG"
  $PY main_filter.py \
      --root_dir "$ROOT" \
      --model_path "$MODEL_DIR/model_torchscript.pt" \
      --model_param_path "$MODEL_DIR/parameters.json" \
      --out_dir "$OUT" \
      --dataset_number "$IDX" \
      --save_as_npy --update_freq 20.0 >> "$LOG" 2>&1
  rc=$?
  if [ -f "$NPY" ]; then
    echo "[$(date +%H:%M:%S)] DONE idx=$IDX seq=$SEQ rc=$rc" | tee -a "$LOG"
  else
    echo "[$(date +%H:%M:%S)] FAIL idx=$IDX seq=$SEQ rc=$rc" | tee -a "$LOG"
  fi
done < "$LIST"

echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a "$LOG"
