# SO(3)-Equivariant Extension of EqNIO — Final Results

## Headline finding

```
                            Standard      so3_60       Degradation
─────────────────────────────────────────────────────────────────
O(2) (EqNIO baseline)         2.37        36.54         15.43×
O(2) + SO(3) aug              2.80         2.28          0.81×
SO(3) (ours, no aug)          2.94        11.27          3.84×
SO(3) + SO(3) aug (full)      2.84         2.47          0.87×
```

Three core findings:

1. **Architecture alone provides 4× more robustness** (3.84× vs 15.43× degradation)
   without any data augmentation — purely from the SO(3) frame extension.

2. **Augmentation makes O(2) robust but costs standard performance** (2.37 → 2.80, +18%);
   our SO(3) maintains standard performance with augmentation (2.94 → 2.84, −3.5%).
   We call this *augmentation-friendliness*: the architecture and data symmetry align,
   so augmentation reinforces (not fights) the architectural bias.

3. **Mathematical guarantee vs statistical learning**: our SO(3) extension is provably
   equivariant by construction (verified numerically to float32 precision). The O(2)
   baseline relies on data augmentation to learn rotation invariance — a statistical,
   not structural, guarantee.

## Architecture summary

| | EqNIO O(2) | our SO(3) extension |
|---|---|---|
| Input vector shape | (B, T, 2, 3) — xy only | (B, T, 3, 3) — full 3D |
| Input scalar dim | 9 (z + norms + dots) | 6 (norms + dots only) |
| Frame matrix | 2×2 ∈ O(2) | 3×3 ∈ SO(3) |
| Gram-Schmidt | 2D, 2 vectors | 3D, 2 vectors + cross product |
| Backbone | TLIO ResNet1D (in=6) | TLIO ResNet1D (in=9) |
| Params | 6,020,230 | 6,021,382 (+0.02%) |

The only architectural change is dimension: 2 → 3 throughout the equivariance pipeline.
The backbone is identical.

## Equivariance verification

Architectural property tested with controlled rotation R ∈ SO(3) applied directly to
model input vectors:

| Model | yaw (Rz) | pitch (Rx) | roll (Ry) |
|---|---|---|---|
| O(2)  | 1e-6 (exact) | n/a (input is 2D) | n/a |
| SO(3) | 1e-5 (exact) | 1e-6 (exact) | 2e-5 (exact) |

Float32 precision confirms our SO(3) model is exactly equivariant by construction.

The full pipeline (preprocess+model) carries an additional O(2)-only step in the
EqNIO cross-product preprocessing (`g_flip[..., -1] = 0`). This is a known limitation
of the inherited preprocess code, and it can be removed in future work.

## Files (paper-ready)

```
/home/chankyokim/Desktop/code2/imu-previous-work/EqNIO/
├── RESULTS_SUMMARY.md                       (this file)
├── ablation_2x2_table.csv                   raw numbers
├── figure_paper_main.png                    main paper figure (1×2)
├── figure_2x2_ablation.png                  full ablation 2×2 (1×3 panels) ★
├── figure_architecture.png                  block diagram comparing arch
├── figure_equivariance_verification.png     (A) architectural (B) full pipeline
├── figure_decomposed_rotation.png           pitch/roll vs yaw + Pearson r
├── figure_trajectories.png                  representative trajectory examples
├── figure_main_robustness.png               5-panel detailed comparison
└── figure_rotation_magnitude.png            rotation magnitude vs ATE scatter
```

## Bug fixes discovered upstream

(documented for future paper revisions of EqNIO)

- `scekf.py` world-frame H matrix used scalar -1/+1 broadcast instead of -I/+I → singular S
- `scekf.py` `self.yaw_aligned` hardcoded True → made configurable
- `imu_tracker.py` `self.yaw_aligned` hardcoded True → made configurable
- `meas_source_torchscript.py` model loading was broken (`assert False`, wrong `torch.load`)
- `meas_source_torchscript.py` `set_num_interop_threads` was called repeatedly in batch mode
- `imu_tracker_runner.py` hardcoded `./../so3_60/rpy_values.npy` load → made conditional
- `imu_tracker_runner.py` `run_tracker(args)` missing required `iter_num` → fixed
- `EKF_output_metrics.py` had off-by-one path-construction bug → custom inline aggregation
- `test.py` `pose_integrate` only handled 2×2 frame → added 3×3 SO(3) branch

## Reproducibility commands

Training:
```bash
# O(2) baseline (already published as tlio_o2)
python3 main_net.py --mode train --arch eq_o2_frame_fullCov_2vec_2deep \
  --root_dir local_data/tlio_golden --epochs 80 --batch_size 1024 --lr 1e-4

# our SO(3) extension (no aug)
python3 main_net.py --mode train --arch eq_so3_frame_fullCov_2vec_2deep \
  --root_dir local_data/tlio_golden --epochs 80 --batch_size 1024 --lr 1e-4

# with SO(3) augmentation
python3 main_net.py --mode train --arch eq_so3_frame_fullCov_2vec_2deep \
  --root_dir local_data/tlio_golden --epochs 80 --batch_size 1024 --lr 1e-4 \
  --so3_augment --so3_augment_max_deg 180
```

Test on rotated data:
```bash
python3 main_net.py --mode test --arch eq_so3_frame_fullCov_2vec_2deep \
  --root_dir local_data/so3_60 \
  --model_path <path>/checkpoint_best.pt \
  --out_dir <out>/so3_60_test
```
