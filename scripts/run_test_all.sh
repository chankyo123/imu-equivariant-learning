#!/bin/bash
set +e
cd "$(dirname "$0")/.."
MODELS_GLOB="${MODELS_GLOB:-models/*/}"      # e.g. MODELS_GLOB="models_eq/chol_*/"
DATA="${DATA:-local_data/tlio_golden/}"        # dataset root with test_list.txt
BODY_FRAME="${BODY_FRAME:-False}"              # "True" for sensor-frame (bodyframe) datasets
mkdir -p logs/test
for d in $MODELS_GLOB; do
  name=$(basename $d)
  ckpt=$d/checkpoint_best.pt
  [ -f "$ckpt" ] || { echo "skip $name (no best)"; continue; }
  [ -f $d/test/dataset_summary.json ] && { echo "skip $name (already)"; continue; }
  arch=$(python3 -c "import torch; print(torch.load('$ckpt', map_location='cpu', weights_only=False)['args']['arch'])" 2>/dev/null)
  echo "===== Testing $name (arch=$arch) at $(date) ====="
  python3 src/main_net.py --mode test \
    --root_dir "$DATA" \
    --out_dir $d/test/ \
    --model_path $ckpt --arch $arch --body_frame "$BODY_FRAME" \
    --batch_size 2048 -j 16 \
    2>&1 | tee logs/test/${name}.log | tail -3
done
echo "===== ALL TESTS DONE $(date) ====="
