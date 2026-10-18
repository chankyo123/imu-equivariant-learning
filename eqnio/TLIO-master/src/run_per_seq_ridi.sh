#!/usr/bin/env bash
# RIDI per-sequence EqNIO+RIEKF body-frame filter runner. Reads RIDI test_list.txt and runs one
# sequence per fresh python process (state-leak safe).
#
# Usage: run_per_seq_ridi.sh <out_dir> [extra_args...]
# extra_args are appended to the python invocation (e.g. --so3_rot_path "<root>/{seq}/R_so3_init.npy").
# Ported from Server B (chankyokim-Alienware). All machine-specific paths are now variables
# that can be overridden from the environment; defaults are relative to the repository root
# (GINIO/). PY = python interpreter, MODEL_DIR = directory with model_torchscript.pt +
# parameters.json (e.g. checkpoints/tlio/ginio_w_ln189), ROOT = dataset root, OUT = output root.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
PY=${PY:-python3}
SRC_DIR="$HERE"
OUT=${1:?out_dir required}
shift
EXTRA="$@"

MODEL_DIR=${MODEL_DIR:-$REPO_ROOT/checkpoints/tlio/ginio_w_ln189}
ROOT=${ROOT:-$REPO_ROOT/local_data_bodyframe/ridi_tlio}

# main_filter reads test_list.txt and indexes into it via --dataset_number,
# so idx == position in test_list (zero-based).
LIST=$ROOT/test_list.txt
mkdir -p "$OUT"
LOG=${LOG:-/tmp/eqnio_riekf_ridi_$(basename "$OUT").log}
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
      --save_as_npy --update_freq 20.0 $EXTRA >> "$LOG" 2>&1
  rc=$?
  if [ -f "$NPY" ]; then
    echo "[$(date +%H:%M:%S)] DONE idx=$IDX seq=$SEQ rc=$rc" | tee -a "$LOG"
  else
    echo "[$(date +%H:%M:%S)] FAIL idx=$IDX seq=$SEQ rc=$rc" | tee -a "$LOG"
  fi
done < "$LIST"

echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a "$LOG"
