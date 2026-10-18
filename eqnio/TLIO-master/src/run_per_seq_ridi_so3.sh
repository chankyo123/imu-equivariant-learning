#!/usr/bin/env bash
# RIDI per-sequence EqNIO+RIEKF body-frame filter runner -- SO(3)-rotated variant (RIDI ID/SO(3)).
# Reads the rotated copy produced by scripts/ridi/generate_ridi_so3_dataset.py and passes the
# per-sequence R_so3_init.npy to the filter (--so3_rot_path).
# Ported from Server B (chankyokim-Alienware). All machine-specific paths are now variables
# that can be overridden from the environment; defaults are relative to the repository root
# (GINIO/). PY = python interpreter, MODEL_DIR = directory with model_torchscript.pt +
# parameters.json (e.g. checkpoints/tlio/ginio_w_ln189), ROOT = dataset root, OUT = output root.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
PY=${PY:-python3}
SRC_DIR="$HERE"
MODEL_DIR=${MODEL_DIR:-$REPO_ROOT/checkpoints/tlio/ginio_w_ln189}
ROOT=${ROOT:-$REPO_ROOT/local_data_bodyframe/ridi_tlio_so3}
OUT=${OUT:-$REPO_ROOT/filter_outputs/ln_body_lfa_eqnio_filter_riekf_ridi_so3}
LOG=${LOG:-/tmp/eqnio_riekf_ridi_so3.log}

mkdir -p "$OUT"
: > "$LOG"
cd "$SRC_DIR"

IDX=-1
while read SEQ; do
  IDX=$((IDX+1))
  [ -z "$SEQ" ] && continue
  NPY="$OUT/$SEQ/not_vio_state.txt.npy"
  if [ -f "$NPY" ]; then
    echo "[$(date +%H:%M:%S)] skip idx=$IDX seq=$SEQ (npy exists)" | tee -a "$LOG"
    continue
  fi
  rm -rf "$OUT/$SEQ"
  echo "[$(date +%H:%M:%S)] RUN idx=$IDX seq=$SEQ" | tee -a "$LOG"
  $PY main_filter_riekf.py \
      --root_dir "$ROOT" \
      --model_path "$MODEL_DIR/model_torchscript.pt" \
      --model_param_path "$MODEL_DIR/parameters.json" \
      --out_dir "$OUT" \
      --dataset_number "$IDX" \
      --use_riekf True --body_frame True --input_3 False \
      --save_as_npy --update_freq 20.0 \
      --so3_rot_path "$ROOT/{seq}/R_so3_init.npy" >> "$LOG" 2>&1
  rc=$?
  if [ -f "$NPY" ]; then
    echo "[$(date +%H:%M:%S)] DONE idx=$IDX seq=$SEQ rc=$rc" | tee -a "$LOG"
  else
    echo "[$(date +%H:%M:%S)] FAIL idx=$IDX seq=$SEQ rc=$rc" | tee -a "$LOG"
  fi
done < "$ROOT/test_list.txt"

echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a "$LOG"
