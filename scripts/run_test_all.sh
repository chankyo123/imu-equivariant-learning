#!/bin/bash
set +e
for d in models_eq/chol_*/; do
  name=$(basename $d)
  ckpt=$d/checkpoint_best.pt
  [ -f "$ckpt" ] || { echo "skip $name (no best)"; continue; }
  [ -f $d/test/dataset_summary.json ] && { echo "skip $name (already)"; continue; }
  arch=$(python3 -c "import torch; print(torch.load('$ckpt', map_location='cpu')['args']['arch'])" 2>/dev/null)
  echo "===== Testing $name (arch=$arch) at $(date) ====="
  python3 src/main_net.py --mode test \
    --root_dir local_data/tlio_golden/ \
    --out_dir $d/test/ \
    --model_path $ckpt --arch $arch --body_frame False \
    --batch_size 2048 -j 16 \
    2>&1 | tee logs/test/${name}.log | tail -3
done
echo "===== ALL TESTS DONE $(date) ====="
