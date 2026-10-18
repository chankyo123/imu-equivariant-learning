# GINIO: A Geometric SO(3)-Equivariant Interface for Neural Inertial Odometry

[![Paper](https://img.shields.io/badge/CoRL%202026-OpenReview-b31b1b.svg)](https://openreview.net/forum?id=nl8iBTm3Mt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Chankyo Kim, Minghan Zhu, Tzu-Yuan Lin, Avantika Rattan, Maani Ghaffari**
University of Michigan, [CURLY Lab](https://curly.engin.umich.edu/)

*Conference on Robot Learning (CoRL) 2026, Austin, TX.*
Paper and reviews: https://openreview.net/forum?id=nl8iBTm3Mt

> **Formerly `imu-equivariant-learning`.** This repository started in 2023 as a yaw-equivariant
> network-coupled inertial odometry code base. GINIO is the same project matured to full SO(3)
> equivariance with a covariance interface, and it is now named after the CoRL 2026 paper. The last
> pre-GINIO version is tagged [`imu-equivariant-learning-final`](../../releases/tag/imu-equivariant-learning-final);
> old links redirect here.

## Overview

Neural inertial odometry (NIO) uses a network as a learned measurement inside a filter. GINIO is an
SO(3)-equivariant *interface* for that measurement: given a calibrated IMU window, it predicts a
motion measurement whose **mean transforms as a vector** and whose **covariance transforms
congruently as a second-order tensor** under any rotation of the IMU measurement frame. The same
interface is instantiated on several backbones:

| Suffix | Benchmark | What it is | Code |
|---|---|---|---|
| GINIO-W / GINIO-S | TLIO, RIDI (filter-connected) | world-frame / sensor-frame Lie-Neuron ResNet (`ln_resnet`) + SCEKF / InEKF | [`src/`](src/) |
| GINIO-RD / GINIO-RS | NanoBench | `ln_resnet` with a diagonal / spectral covariance head | [`src/`](src/) |
| GINIO-R | Fetch remounting | `ln_resnet` | [`src/`](src/) |
| GINIO-A | NanoBench | Vector-Neuron version of the AirIO recurrent model (VN-CNN + VN-GRU) | [`airio/`](airio/) |
| GINIO-E | NanoBench | full-SO(3) canonicalised EqNIO-style model | [`eqnio/`](eqnio/) |

**Last-Frame Alignment (LFA)** is a deterministic preprocessing step that expresses a window in the
frame of its last IMU sample; for an SO(3)-equivariant predictor it is provably equivalent to
world-frame training (Appendix D of the paper). It is implemented in
[`src/dataloader/sequences_dataset.py`](src/dataloader/sequences_dataset.py) and switched by the
dataset path (`bodyframe` in the path selects the sensor-frame branch; `nolfa` turns LFA off).

## Installation

```bash
conda env create -f environment.yml      # python 3.12 + pip requirements
conda activate ginio
# install the torch build matching your CUDA driver, e.g.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

The released checkpoints were trained and evaluated with torch 2.12 (nightly, cu128), numpy 1.26,
scipy 1.17, pypose 0.9.5 and pyhocon 0.3.63; see [`requirements.txt`](requirements.txt).

## Data

All loaders expect the TLIO layout: one folder per sequence with `imu0_resampled.npy` (17 columns:
`ts_us | gyr(3) | acc(3) | q_xyzw_World_Device(4) | pos_World(3) | vel(3)`),
`imu0_resampled_description.json`, `calibration.json`, plus `train_list.txt / val_list.txt /
test_list.txt` at the dataset root. Put datasets under `local_data/` (git-ignored).

| Dataset | Source | Notes |
|---|---|---|
| TLIO golden | [CathIAS/TLIO](https://github.com/CathIAS/TLIO) release (`golden-new-format-cc-by-nc-with-imus-v1.5.zip`) | 284 / 36 / 36 sequences, world-frame IMU. `gdown 14YKW7PsozjHo_EdxivKvumsQB7JMw1eg && unzip ... -d local_data/` |
| RIDI | [RIDI data_publish_v2](https://github.com/higerra/ridi_imu) | Convert to the TLIO layout before use (converter not yet included, see *Status*). |
| NanoBench | [NanoBench](https://arxiv.org/abs/2603.09908) (Crazyflie 2.1 flights, Vicon ground truth) | Our processed world-frame split (74 / 16 / 17 sequences, 100 Hz, gravity removed) will be published with the checkpoints. Then derive the sensor-frame copy: `python scripts/derive_body_npy.py` and `ln -s nanobench_bodyframe_2nd_version local_data/nanobench_bodyframe_nolfa_2nd_version` (same data; the `nolfa` name disables LFA). |
| Fetch remounting | collected by us (Adafruit BNO055 in a Fetch mobile base; 23 nominal + 23 x-remount + 23 y-remount sequences) | Shipped as `fetch_dataset.zip` with the paper supplement; will be mirrored here. |

## Quick start

### Lie-Neuron models (GINIO-W/S/RD/RS/R) — `src/`

Train the NanoBench GINIO-RD configuration (sensor frame, no LFA, diagonal head; this is the
hyper-parameter set found by the Optuna sweep in the paper):

```bash
python src/main_net.py --mode train \
  --root_dir local_data/nanobench_bodyframe_nolfa_2nd_version --out_dir models/ginio_rd \
  --arch ln_resnet --body_frame True --epochs 201 --batch_size 512 --lr 4.1e-4 --weight_decay 5.8e-3 \
  --imu_freq 200 --imu_base_freq 1000 --window_time 0.675 --sample_freq 10 \
  --input_dim 6 --output_dim 3 --seed 1 --workers 4 --dataset_style mmap \
  --rpe_window 2.0 --mse_warmup_epochs 12 --nll_weight 1.0
```

Spectral (GINIO-RS) and world-frame (GINIO-W) variants:

```bash
# spectral covariance head, LFA on (path without "nolfa"), 1.0 s window (the spectral head is wired for window_time 1.0)
python src/main_net.py --mode train --root_dir local_data/nanobench_bodyframe_2nd_version --out_dir models/ginio_rs \
  --arch ln_resnet_chol --cov_type eq_chol --body_frame True --window_time 1.0 --sample_freq 20 --imu_freq 200 --imu_base_freq 1000 \
  --batch_size 16 --epochs 201 --lr 4.1e-4 --weight_decay 5.8e-3 --init_log_a_scale -1.897
# world-frame TLIO (GINIO-W)
python src/main_net.py --mode train --root_dir local_data/tlio_golden --out_dir models/ginio_w \
  --arch ln_resnet --body_frame False --window_time 1.0 --sample_freq 20 --imu_freq 200 --imu_base_freq 1000 --batch_size 1024 --epochs 120
```

Test (writes per-sequence `trajectory.txt`, `net_outputs.txt` and `metrics.json`; `ate_um` is the
Umeyama-aligned ATE used in Table 2, `ate` the TLIO-style mean position error):

```bash
python src/main_net.py --mode test --root_dir local_data/nanobench_bodyframe_nolfa_2nd_version \
  --out_dir test_outputs/ginio_rd --model_path models/ginio_rd/checkpoint_best.pt --arch ln_resnet --body_frame True \
  --batch_size 1024 --imu_freq 200 --imu_base_freq 1000 --window_time 0.675 --sample_freq 10
```

Filter-connected evaluation (TLIO/RIDI, Table 1): export TorchScript, run the tracker, then compute
metrics. `--use_riekf True --body_frame True` selects the InEKF sensor-frame configuration (GINIO-S).

```bash
python src/convert_model_to_torchscript.py --model_path models/ginio_w/checkpoint_best.pt \
  --model_param_path models/ginio_w/parameters.json --out_dir models/ginio_w --input_dim 6
python src/main_filter.py --root_dir local_data/tlio_golden/ --model_path models/ginio_w/model_torchscript.pt \
  --model_param_path models/ginio_w/parameters.json --out_dir filter_outputs/ginio_w --update_freq 20 --erase_old_log
python src/plot_filter_state.py --root_dir local_data/tlio_golden/ --log_dir filter_outputs/ginio_w --no-make_plots --no-save_fig
```

Optional training/evaluation switches (ported from the rebuttal study, off by default):
`--so3_train_aug [--so3_train_aug_max_deg 180]` rotates each training window and its vector targets by
a random rotation (world- and sensor-frame); `--freeze_mean --model_path <base.pt>` loads a trained
model, freezes backbone and mean head and trains only the covariance head (the `ln_resnet_cov_diag`
architecture is the diagonal-head twin of `ln_resnet_cov` for this matched-pair study);
`--test_perturb_type {gravity,lfa,so3} --test_perturb_deg D` applies the frame-estimation-mismatch
perturbations of Supplement Table 10 at test time.

Important flags: `--body_frame` is a string (`"True"`/`"False"`); always pass `--imu_freq 200`
(the window length is `int(200 * window_time)` samples, so on 100 Hz NanoBench `--window_time 1.0`
is a 2 s window); `--cov_type {diag, eq_spec, eq_chol}` is inferred from the architecture when omitted.

### AirIO-style recurrent models (GINIO-A) — `airio/`

Run from inside `airio/`. Configs are HOCON; dataset roots are relative (`../local_data/...`).

```bash
cd airio && export WANDB_MODE=offline
# GINIO-A (no attitude input), the 0.045M "small" model of Table 2
python train_motion.py --config configs/Nanobench/motion_body_noatt_vn_small.conf --device cuda:0
# GINIO-A with attitude channel; A2 ("rows of R_WS") attitude encoding + spectral covariance head
python train_motion.py --config configs/Nanobench/motion_body_a2_spec_small.conf --device cuda:0 --seed 0
# whole-sequence inference -> experiments/<exp>/<conf>/net_output.pickle
python inference_motion.py --config configs/Nanobench/motion_body_noatt_vn_small.conf --device cuda:0
# ID/ID and ID/SO(3): Umeyama ATE, mean-equivariance error and covariance-law error under
# (P) a physical remount R_WS -> R_WS Q^T and (G) the "rotate every vector input" protocol
python -m evaluation.eval_idso3 --config configs/Nanobench/motion_body_noatt_vn_small.conf --n_rot 3 --device cuda:0
# unit test of the A2 + spectral construction (24 checks, CPU)
CUDA_VISIBLE_DEVICES="" python -m model.test_a2_spectral_equivariance
```

Attitude encodings live in [`airio/model/attitude.py`](airio/model/attitude.py): `logmap`
(`Log(R_WS)`, the encoding used for the paper's with-attitude row), `rows` (the three rows of
`R_WS`, exactly equivariant under a remount and lossless), `up`, `rel`. The spectral head is
`VNSpectralCov` in [`airio/model/vn_codenet.py`](airio/model/vn_codenet.py).

### Canonicalised models (GINIO-E and the EqNIO O(2) baseline) — `eqnio/`

```bash
cd eqnio/TLIO-master
# EqNIO O(2) baseline / GINIO-E (full SO(3)); add --so3_augment --so3_augment_max_deg 180 for SO(3) augmentation
python src/main_net.py --mode train --arch eq_o2_frame_fullCov_2vec_2deep  --root_dir local_data/tlio_golden --out_dir models/eqnio_o2 \
  --batch_size 1024 --epochs 40 --lr 1e-4 --window_time 1.0 --sample_freq 20 --imu_freq 200 --imu_base_freq 1000 --workers 4 --dataset_style mmap
python src/main_net.py --mode train --arch eq_so3_frame_fullCov_2vec_2deep --root_dir local_data/tlio_golden --out_dir models/ginio_e --so3_augment --so3_augment_max_deg 180 \
  --batch_size 1024 --epochs 40 --lr 1e-4 --window_time 1.0 --sample_freq 20 --imu_freq 200 --imu_base_freq 1000 --workers 4 --dataset_style mmap
# filter chain
python src/convert_model_to_torchscript_eq.py --model_path models/ginio_e/checkpoint_best.pt --model_param_path models/ginio_e/parameters.json --out_dir models/ginio_e --cpu
python src/main_filter.py --root_dir local_data/tlio_golden/ --model_path models/ginio_e/model_torchscript.pt --model_param_path models/ginio_e/parameters.json --out_dir models/ginio_e/filter --update_freq 20.0 --erase_old_log --no-visualize
python ../test_vn_eqnio_equivariance.py     # numerical SO(3) equivariance check of the GINIO-E model
```

## Model zoo

See [`checkpoints/MANIFEST.md`](checkpoints/MANIFEST.md) for every released checkpoint with
SHA-256, parameter count and the Table-2 value it reproduces. Released: the four AirIO-style
NanoBench rows (AirIO ×2, GINIO-A ×2), GINIO-E, GINIO-RS, the four covariance-ablation models of
Supplement Table 9, and the closest archived GINIO-RD run (Optuna trial 55). Not released because
no checkpoint survives: ResNet1D 5.43M, the EqNIO O(2) NanoBench row (the EqNIO authors' public
TLIO model evaluated on NanoBench), the TLIO/RIDI filter models of Table 1 and the Fetch models of
Table 3 (see *Status*).

## Evaluation protocol notes

* **Table 2 ATE** is the Umeyama-aligned (rotation + translation, no scale) position RMSE over the
  17 NanoBench test sequences after integrating the predicted body velocity through ground-truth
  attitude (`ate_um` in `src/network/test.py`, `ate_for` in `airio/evaluation/eval_idso3.py`).
* **ID/SO(3)** rotates the accelerometer and gyroscope channels by a sampled `R`, rotates vector
  targets by the same `R`, transforms covariance targets by `R Σ Rᵀ`, and un-rotates predictions
  before scoring. `airio/evaluation/eval_idso3.py` implements this (and additionally reports the
  mean-equivariance and covariance-congruence errors). The imu-equivariant-learning test transform
  `--test_perturb_type so3` in older copies of this code does *not* un-rotate predictions and must
  not be used for this protocol.
* **Checkpoint selection** in `airio/train_motion.py` follows upstream AirIO: `best_model.ckpt` is
  the epoch with the lowest loss on the config's `test` block, which in
  `airio/configs/datasets/Nanobench/nanobench_body.conf` is the 17-sequence reported test split
  (the `eval` block, 16 disjoint sequences, is logged only). This applies identically to the AirIO
  baselines and the GINIO-A models.
* **GINIO-E** in Table 2 was trained on TLIO with full-SO(3) augmentation and evaluated zero-shot on
  NanoBench (no NanoBench training).
* No rotation augmentation is used in any released training configuration; ID/SO(3) numbers measure
  architectural equivariance. The Lie-Neuron family uses a per-window constant bias shift
  (accel ±0.2 m/s², gyro ±0.05 rad/s) at training time; the AirIO family uses no augmentation.

## Status

* [x] Code for all five instantiations, evaluation scripts, filter chains.
* [x] NanoBench and covariance-ablation checkpoints (GitHub Release).
* [ ] Processed NanoBench split (1.0 GB) and the Fetch remounting set (57 MB) as a data release.
* [ ] RIDI → TLIO-layout converter and the TLIO/RIDI/Fetch model checkpoints (being recovered from
      the original training machine).
* [ ] AquaticVision evaluation code (Appendix K).

## Citation

```bibtex
@inproceedings{kim2026ginio,
  title     = {{GINIO}: A Geometric {SO(3)}-Equivariant Interface for Neural Inertial Odometry},
  author    = {Kim, Chankyo and Zhu, Minghan and Lin, Tzu-Yuan and Rattan, Avantika and Ghaffari, Maani},
  booktitle = {Proceedings of The 10th Conference on Robot Learning (CoRL)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=nl8iBTm3Mt}
}
```

## Acknowledgements

This code builds on [TLIO](https://github.com/CathIAS/TLIO), [EqNIO](https://github.com/RoyinaJayanth/EqNIO),
[AirIO](https://github.com/Air-IO/Air-IO), [Vector Neurons](https://github.com/FlyingGiraffe/vnn) and
[Lie Neurons](https://github.com/UMich-CURLY/LieNeurons). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
