#!/bin/bash
# Ported from Server B (chankyokim-Alienware). Paths are relative to the repository root:
# the script cd's there, so it can be launched from anywhere. Expects local_data/tlio_golden/
# (TLIO golden, world frame) and writes runs to models_eq/<name>/ and logs to logs/<name>.log.
# Override the interpreter with PY=... (default: python3 on PATH).
set +e
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python3}
mkdir -p logs models_eq

run_one() {
  local out=$1; shift
  echo "===== START $(basename $out) at $(date) ====="
  $PY src/main_net.py --mode train \
    --root_dir local_data/tlio_golden/ \
    --body_frame False \
    --batch_size 4096 --lr 2e-4 -j 16 --seed 42 \
    --out_dir $out "$@" 2>&1 | tee logs/$(basename $out).log
  echo "===== END $(basename $out) at $(date) (rc=$?) ====="
}

# A. Bias sweep (4 runs) -- the core sweep
for bs in 0.5 0.25 0.1 0.0; do
  run_one models_eq/chol_A_bs${bs} \
    --arch ln_resnet_chol --epochs 81 \
    --bias_scale ${bs} --mse_anchor_weight 0.1 --mse_warmup_epochs 10 \
    --init_log_a_scale -1.897
done

# B. Anchor sweep (2 runs)
for aw in 0.0 0.3; do
  run_one models_eq/chol_B_anchor${aw} \
    --arch ln_resnet_chol --epochs 81 \
    --bias_scale 0.25 --mse_anchor_weight ${aw} --mse_warmup_epochs 10 \
    --init_log_a_scale -1.897
done

# C. Init log_a_scale sweep (2 runs)
for ils in -1.0 -2.5; do
  run_one models_eq/chol_C_init${ils} \
    --arch ln_resnet_chol --epochs 81 \
    --bias_scale 0.25 --mse_anchor_weight 0.1 --mse_warmup_epochs 10 \
    --init_log_a_scale ${ils}
done

# E. Architecture comparison (chol vs diag vs spectral, same hyper-parameters)
run_one models_eq/chol_E1_diag \
  --arch ln_resnet --epochs 81 \
  --bias_scale 0.25 --mse_anchor_weight 0.1 --mse_warmup_epochs 10

run_one models_eq/chol_E2_spec \
  --arch ln_resnet_cov --epochs 81 \
  --bias_scale 0.25 --mse_anchor_weight 0.1 --mse_warmup_epochs 10

echo "===== ALL 10 SWEEPS DONE $(date) ====="
