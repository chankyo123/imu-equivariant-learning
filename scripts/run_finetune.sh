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
    --batch_size 4096 -j 16 \
    --no-save_plot \
    --out_dir $out "$@" 2>&1 | tee logs/$(basename $out).log
  echo "===== END $(basename $out) at $(date) (rc=$?) ====="
}

# === Phase 1: fine-tune around the best run_sweep.sh configuration (5 runs) ===
# Best so far: bias=0.25, anchor=0, init=-1.897, lr=2e-4

# 1. bias 0.20
run_one models_eq/chol_FT_bs0.20 \
  --arch ln_resnet_chol --epochs 81 --seed 42 --lr 2e-4 \
  --bias_scale 0.20 --mse_anchor_weight 0.0 \
  --mse_warmup_epochs 10 --init_log_a_scale -1.897

# 2. bias 0.30
run_one models_eq/chol_FT_bs0.30 \
  --arch ln_resnet_chol --epochs 81 --seed 42 --lr 2e-4 \
  --bias_scale 0.30 --mse_anchor_weight 0.0 \
  --mse_warmup_epochs 10 --init_log_a_scale -1.897

# 3. anchor 0.05
run_one models_eq/chol_FT_anc0.05 \
  --arch ln_resnet_chol --epochs 81 --seed 42 --lr 2e-4 \
  --bias_scale 0.25 --mse_anchor_weight 0.05 \
  --mse_warmup_epochs 10 --init_log_a_scale -1.897

# 4. init -1.5
run_one models_eq/chol_FT_init-1.5 \
  --arch ln_resnet_chol --epochs 81 --seed 42 --lr 2e-4 \
  --bias_scale 0.25 --mse_anchor_weight 0.0 \
  --mse_warmup_epochs 10 --init_log_a_scale -1.5

# 5. lr 3e-4
run_one models_eq/chol_FT_lr3e-4 \
  --arch ln_resnet_chol --epochs 81 --seed 42 --lr 3e-4 \
  --bias_scale 0.25 --mse_anchor_weight 0.0 \
  --mse_warmup_epochs 10 --init_log_a_scale -1.897

echo "===== FINETUNE 5 DONE $(date) ====="

# === Phase 2: test every finished fine-tune run (plus the existing chol_B_anchor0.0) ===
for d in models_eq/chol_FT_*/; do
  name=$(basename $d)
  if [ -f $d/checkpoint_best.pt ] && [ ! -f $d/test/dataset_summary.json ]; then
    echo "===== Testing $name ====="
    $PY src/main_net.py --mode test \
      --root_dir local_data/tlio_golden/ \
      --out_dir $d/test/ \
      --model_path $d/checkpoint_best.pt \
      --arch ln_resnet_chol --body_frame False \
      --batch_size 2048 -j 16 \
      --no-save_plot \
      2>&1 | tee logs/$(basename $name)_test.log | tail -3
  fi
done

# === Phase 3: pick the best configuration automatically and train it for 200 epochs ===
echo "===== Computing best config $(date) ====="
BEST_LINE=$($PY << 'PYEOF'
import torch, json, numpy as np, os, glob
best_ate, best_args, best_name = 1e9, None, None
candidates = glob.glob("models_eq/chol_FT_*/") + ["models_eq/chol_B_anchor0.0/"]
for d in candidates:
    summ = os.path.join(d, "test/dataset_summary.json")
    metr = os.path.join(d, "test/metrics.json")
    ckp  = os.path.join(d, "checkpoint_best.pt")
    if not (os.path.exists(summ) and os.path.exists(metr) and os.path.exists(ckp)):
        continue
    m = json.load(open(metr))
    ate = np.mean([v["ronin"]["ate"] for v in m.values()])
    if ate < best_ate:
        best_ate = ate
        ck = torch.load(ckp, map_location="cpu", weights_only=False)
        best_args = ck["args"]
        best_name = os.path.basename(d.rstrip('/'))
if best_args:
    print(f"{best_name} {best_ate:.4f} "
          f"{best_args.get('bias_scale',0.25)} "
          f"{best_args.get('mse_anchor_weight',0.0)} "
          f"{best_args.get('mse_warmup_epochs',10)} "
          f"{best_args.get('init_log_a_scale',-1.897)} "
          f"{best_args.get('lr',2e-4)}")
PYEOF
)
echo "BEST: $BEST_LINE"
if [ -z "$BEST_LINE" ]; then echo "no finished fine-tune run found; skipping the long run"; exit 0; fi

BEST_NAME=$(echo $BEST_LINE | awk '{print $1}')
BEST_ATE=$(echo $BEST_LINE | awk '{print $2}')
BEST_BIAS=$(echo $BEST_LINE | awk '{print $3}')
BEST_ANC=$(echo $BEST_LINE | awk '{print $4}')
BEST_WU=$(echo $BEST_LINE | awk '{print $5}')
BEST_INIT=$(echo $BEST_LINE | awk '{print $6}')
BEST_LR=$(echo $BEST_LINE | awk '{print $7}')

echo "Long training with best: bias=$BEST_BIAS anc=$BEST_ANC wu=$BEST_WU init=$BEST_INIT lr=$BEST_LR"

# Long chol 200 epochs with the automatically selected configuration
run_one models_eq/chol_FT_LONG_best \
  --arch ln_resnet_chol --epochs 201 --seed 42 --lr $BEST_LR \
  --bias_scale $BEST_BIAS --mse_anchor_weight $BEST_ANC \
  --mse_warmup_epochs $BEST_WU --init_log_a_scale $BEST_INIT

echo "===== ALL DONE $(date) ====="
