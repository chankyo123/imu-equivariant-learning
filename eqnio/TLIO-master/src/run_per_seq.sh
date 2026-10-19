#!/usr/bin/env bash
# Run main_filter_riekf.py one sequence at a time so a single-sequence crash cannot poison the next.
# Skips sequences whose .npy already exists. (TLIO body-frame, RIEKF + GINIO network.)
# Ported from Server B (chankyokim-Alienware). All machine-specific paths are now variables
# that can be overridden from the environment; defaults are relative to the repository root
# (GINIO/). PY = python interpreter, MODEL_DIR = directory with model_torchscript.pt +
# parameters.json (e.g. checkpoints/tlio_ginio_w_ln189), ROOT = dataset root, OUT = output root.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
PY=${PY:-python3}
SRC_DIR="$HERE"
MODEL_DIR=${MODEL_DIR:-$REPO_ROOT/checkpoints/tlio_ginio_w_ln189}
ROOT=${ROOT:-$REPO_ROOT/local_data_bodyframe/tlio_golden}
OUT=${OUT:-$REPO_ROOT/filter_outputs/ln_body_lfa_eqnio_filter_riekf}
LOG=${LOG:-/tmp/eqnio_riekf_per_seq.log}

mkdir -p "$OUT"
: > "$LOG"

cd "$SRC_DIR"

idx=-1
while read -r SEQ; do
  idx=$((idx+1))
  [ -z "$SEQ" ] && continue
  NPY="$OUT/$SEQ/not_vio_state.txt.npy"
  if [ -f "$NPY" ]; then
    echo "[$(date +%H:%M:%S)] skip idx=$idx seq=$SEQ (npy exists)" | tee -a "$LOG"
    continue
  fi
  # Wipe partial output dir if any (to avoid txt-stub poisoning)
  rm -rf "$OUT/$SEQ"
  echo "[$(date +%H:%M:%S)] RUN idx=$idx seq=$SEQ" | tee -a "$LOG"
  $PY main_filter_riekf.py \
      --root_dir "$ROOT" \
      --model_path "$MODEL_DIR/model_torchscript.pt" \
      --model_param_path "$MODEL_DIR/parameters.json" \
      --out_dir "$OUT" \
      --dataset_number "$idx" \
      --use_riekf True \
      --body_frame True \
      --input_3 False \
      --save_as_npy \
      --update_freq 20.0 >> "$LOG" 2>&1
  rc=$?
  if [ -f "$NPY" ]; then
    echo "[$(date +%H:%M:%S)] DONE idx=$idx seq=$SEQ rc=$rc" | tee -a "$LOG"
  else
    echo "[$(date +%H:%M:%S)] FAIL idx=$idx seq=$SEQ rc=$rc (no npy)" | tee -a "$LOG"
  fi
done < "$ROOT/test_list.txt"

echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a "$LOG"
