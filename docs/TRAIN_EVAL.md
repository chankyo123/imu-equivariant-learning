# Training and evaluation — full command sheets

The README keeps one minimal path per family; this page has the complete commands. Every command
assumes the data layout of [`DATA.md`](DATA.md) and the environment of [`INSTALL.md`](INSTALL.md).
What each released checkpoint reproduces is in [`../checkpoints/MANIFEST.md`](../checkpoints/MANIFEST.md);
the metric definitions are in [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md).

## 1. Lie-Neuron family — `src/` (GINIO-W / S / RD / RS / R)

Architectures (`--arch`, `src/network/model_factory.py`):

| key | class | head | used for |
|---|---|---|---|
| `ln_resnet` | `SO3EquivariantReluBracketLayers` (`ln_conv_2regress_nochannelmix_bias.py`, LeakyReLU slope 0.0) | diagonal log-std | GINIO-W, GINIO-S, GINIO-RD, GINIO-R |
| `ln_resnet_cov` | `SO3EquivariantReluBracketLayers_cov` | spectral `V diag(σ²) Vᵀ` | spectral covariance (TLIO frozen-mean study) |
| `ln_resnet_cov_diag` | same backbone, coordinate-frame diagonal head | diagonal | the matched diagonal arm (`--freeze_mean`) |
| `ln_resnet_chol` | `SO3EquivariantReluBracketLayers_chol` | `A Aᵀ + εI` (Cholesky-parametrised spectral) | GINIO-RS |
| `resnet` | `ResNet1D` (TLIO) | diagonal | ResNet / ResNet1D baselines |
| `ln_resnet_slope1_nobias` | `ln_conv_2regress_nochannelmix_slope_1.py` (LeakyReLU slope 0.1, **bias-free** FcBlocks) | diagonal log-std | the released **RIDI GINIO-W** checkpoint only; it does not load into `ln_resnet` |
| `imunet` | `IMUNet` reimplementation (`model_imunet.py`) | diagonal | IMUNet baseline (RIDI) |
| `ln189_speccov_s0`, `ln189_speccov_s0_diag` | `ln_resnet` backbone + spectral / coordinate-diagonal covariance head (`..._bias_speccov_slope0.py`) | spectral / diagonal | Fig. 5(b) frozen-mean covariance arms on TLIO (load the GINIO-W checkpoint with `--freeze_mean`) |
| `ln189_speccov`, `ln189_speccov_diag` | same heads on a slope-0.1 backbone | — | superseded arms kept only to reproduce old runs; the deployed backbone is slope 0.0 |

### Train

```bash
# GINIO-RD (Table 2, Optuna trial 119): NanoBench, sensor frame, no LFA, diagonal head.
# Hyper-parameters are the ones recorded in the released checkpoint's parameters.json.
python src/main_net.py --mode train \
  --root_dir local_data/nanobench_bodyframe_nolfa_2nd_version --out_dir models/ginio_rd \
  --arch ln_resnet --body_frame True --epochs 170 --batch_size 256 --lr 5.44e-4 --weight_decay 2.41e-4 \
  --imu_freq 200 --imu_base_freq 1000 --window_time 0.775 --sample_freq 50 \
  --do_bias_shift --accel_bias_range 0.3878 --gyro_bias_range 0.0189 \
  --input_dim 6 --output_dim 3 --seed 1 --workers 4 --dataset_style mmap --nll_weight 1.0

# GINIO-RS: Cholesky-parametrised spectral head, 1.0 s window (the spectral heads are wired for 200-sample windows)
python src/main_net.py --mode train --root_dir local_data/nanobench_bodyframe_nolfa_2nd_version --out_dir models/ginio_rs \
  --arch ln_resnet_chol --cov_type eq_chol --body_frame True --window_time 1.0 --sample_freq 20 --imu_freq 200 --imu_base_freq 1000 \
  --batch_size 16 --epochs 201 --lr 4.1e-4 --weight_decay 5.8e-3 --init_log_a_scale -1.897

# GINIO-W: TLIO golden, world frame
python src/main_net.py --mode train --root_dir local_data/tlio_golden --out_dir models/ginio_w \
  --arch ln_resnet --body_frame False --window_time 1.0 --sample_freq 20 --imu_freq 200 --imu_base_freq 1000 --batch_size 1024 --epochs 200

# GINIO-S: TLIO body-frame copy with LFA (path must contain "bodyframe" and must not contain "nolfa")
python src/main_net.py --mode train --root_dir local_data/tlio_bodyframe --out_dir models/ginio_s \
  --arch ln_resnet --body_frame True --window_time 1.0 --sample_freq 20 --imu_freq 200 --imu_base_freq 1000 --batch_size 1024 --epochs 200
```

Switches: `--so3_train_aug [--so3_train_aug_max_deg 180]` (random SO(3) rotation of each window and
its vector targets; off for all paper models), `--do_bias_shift --accel_bias_range 0.2 --gyro_bias_range 0.05`
(per-window constant bias shift; on for the Lie-Neuron paper models), `--cov_type {diag,eq_spec,eq_chol}`
(inferred from the architecture when omitted; pass it explicitly for the `ln189_speccov*` arms), `--freeze_mean --model_path <base.pt>`
(freeze backbone and mean head, train only the covariance head), `SAVE_ALL_EPOCHS=1` (environment; keep a checkpoint every epoch).

### Test (network only)

```bash
python src/main_net.py --mode test --root_dir local_data/nanobench_bodyframe_nolfa_2nd_version \
  --out_dir test_outputs/ginio_rd --model_path checkpoints/nanobench_ginio_rd_t119/checkpoint_best.pt --arch ln_resnet --body_frame True \
  --batch_size 1024 --imu_freq 200 --imu_base_freq 1000 --window_time 0.775 --sample_freq 50
# test-time perturbations (camera-ready Table 9): gravity-frame tilt / LFA relative-frame noise / SO(3)
python src/main_net.py --mode test ... --test_perturb_type gravity --test_perturb_deg 10
python src/main_net.py --mode test ... --test_perturb_type lfa     --test_perturb_deg 10
python src/main_net.py --mode test ... --test_perturb_type so3     --test_perturb_deg 90
```

Outputs per sequence: `trajectory.txt` (`ts, pos_pred, pos_gt, vel_body_pred`), `net_outputs.txt`
(`ts, pred_vel, targ_vel, sigmas`), and `metrics.json` with `ate_um` (Umeyama ATE, the Table-2
metric), `ate` (TLIO mean-L2), `rmse`, `rpe`, `drift`, NLL statistics. Use the `window_time` and
`sample_freq` recorded in the checkpoint's `parameters.json`.

### Filter-connected (Table 1)

```bash
# 1. export TorchScript
python src/convert_model_to_torchscript.py --model_path models/ginio_w/checkpoint_best.pt \
  --model_param_path models/ginio_w/parameters.json --out_dir models/ginio_w --input_dim 6
# 2. SCEKF, world frame (GINIO-W / ResNet+SCEKF rows)
python src/main_filter.py --root_dir local_data/tlio_golden/ --model_path models/ginio_w/model_torchscript.pt \
  --model_param_path models/ginio_w/parameters.json --out_dir filter_outputs/ginio_w --update_freq 20 --erase_old_log
#    InEKF, sensor frame (GINIO-S / ResNet-S rows)
python src/main_filter.py --root_dir local_data/tlio_bodyframe/ --model_path models/ginio_s/model_torchscript.pt \
  --model_param_path models/ginio_s/parameters.json --out_dir filter_outputs/ginio_s --update_freq 20 --erase_old_log \
  --body_frame True --use_riekf True
# 3. metrics (TLIO metric code: origin-aligned RMSE, 1 s yaw-aligned RTE)
python src/plot_filter_state.py --root_dir local_data/tlio_golden/ --log_dir filter_outputs/ginio_w --no-make_plots --no-save_fig
```

Attitude-error sweep (Fig. 5a): set `ATT_ERR_DEG=<deg>` and optionally `ATT_ERR_AXIS={x,y,z,random}`
in the environment of `main_filter.py` to inject a fixed attitude error into the measurement
frame. Full 3×3 covariance from the spectral heads is passed to the filter when the TorchScript
model returns it (`src/tracker/meas_source_torchscript.py`); export spectral models with
`scripts/export/export_fullcov_ts.py`. See [`FILTER_EVALUATION.md`](FILTER_EVALUATION.md) for the
batch runners and plots.

## 2. Air-IO family — `airio/` (AirIO baselines and GINIO-A)

Run from inside `airio/`; configs are HOCON with dataset roots relative to the repo
(`../local_data/...`).

```bash
cd airio && export WANDB_MODE=offline
python train_motion.py --config configs/Nanobench/motion_body_noatt_vn_small.conf --device cuda:0          # GINIO-A (no-att.)
python train_motion.py --config configs/Nanobench/motion_body_rot_vn_small.conf   --device cuda:0          # GINIO-A (with-att., Log(R_WS) channel as in the paper)
python train_motion.py --config configs/Nanobench/motion_body_a2_spec_small.conf  --device cuda:0 --seed 0 # A2 attitude rows + spectral head
python train_motion.py --config configs/Nanobench/motion_body_noatt.conf          --device cuda:0          # AirIO (no-att.) baseline
python inference_motion.py --config configs/Nanobench/motion_body_noatt_vn_small.conf --device cuda:0      # whole-sequence inference -> net_output.pickle
python -m evaluation.eval_idso3 --config configs/Nanobench/motion_body_noatt_vn_small.conf --n_rot 3        # ID/ID + ID/SO(3), Umeyama ATE, equivariance and cov-law errors
CUDA_VISIBLE_DEVICES="" python -m model.test_a2_spectral_equivariance                                         # 24 CPU checks
```

Registry keys (`airio/model/__init__.py`): `codenetmotion`, `codewithrot` (AirIO), `vn_codenetmotion_small`,
`vn_codewithrot_small` (GINIO-A, Table 2), `vn_codenetmotion`, `vn_codewithrot` (0.273M originals,
Supp. Table 4), `vn_noatt_spec_small`, `vn_a{0,1,2,3}_spec_small`, `vn_a2_diag_small` (attitude encoding
× covariance-head study). Attitude encodings are in `airio/model/attitude.py`; the spectral head is
`VNSpectralCov` in `airio/model/vn_codenet.py`. `best_model.ckpt` is selected on the config's `test`
block (see EVAL_PROTOCOL.md).

## 3. Canonicalised family — `eqnio/` (EqNIO O(2) baseline and GINIO-E)

```bash
cd eqnio/TLIO-master
# EqNIO O(2) baseline
python src/main_net.py --mode train --arch eq_o2_frame_fullCov_2vec_2deep  --root_dir local_data/tlio_golden --out_dir models/eqnio_o2 \
  --batch_size 1024 --epochs 40 --lr 1e-4 --window_time 1.0 --sample_freq 20 --imu_freq 200 --imu_base_freq 1000 --workers 4 --dataset_style mmap
# GINIO-E (full SO(3) canonicalisation); the released GINIO-E was trained with --so3_augment
python src/main_net.py --mode train --arch eq_so3_frame_fullCov_2vec_2deep --root_dir local_data/tlio_golden --out_dir models/ginio_e \
  --so3_augment --so3_augment_max_deg 180 \
  --batch_size 1024 --epochs 40 --lr 1e-4 --window_time 1.0 --sample_freq 20 --imu_freq 200 --imu_base_freq 1000 --workers 4 --dataset_style mmap
# filter chain
python src/convert_model_to_torchscript_eq.py --model_path models/ginio_e/checkpoint_best.pt --model_param_path models/ginio_e/parameters.json --out_dir models/ginio_e --cpu
python src/main_filter.py --root_dir local_data/tlio_golden/ --model_path models/ginio_e/model_torchscript.pt --model_param_path models/ginio_e/parameters.json \
  --out_dir models/ginio_e/filter --update_freq 20.0 --erase_old_log --no-visualize
python src/plot_filter_state.py --root_dir local_data/tlio_golden/ --log_dir models/ginio_e/filter --no-make_plots --no-save_fig
# body-frame / RIEKF variant of the filter
python src/main_filter_riekf.py --help
# SO(3) equivariance check of the GINIO-E model
python ../test_vn_eqnio_equivariance.py
```

`EQNIO_MSE_ONLY=1` keeps the MSE loss for the whole run (architecture-only comparison);
`--express_in_local_gravity_aligned False` when evaluating a Lie-Neuron checkpoint through this tree.

## 4. Paper tables and figures — `scripts/paper/`

`scripts/paper/README.md` lists, per script, which camera-ready table or figure it reproduces, its
inputs and the exact command (NanoBench cross-model Umeyama ATE, covariance ablation, ID/SO(3) and
frame-mismatch sweeps, final-table builder).
