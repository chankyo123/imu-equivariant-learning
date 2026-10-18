# Evaluation protocol

This page states exactly how every number in the README and `checkpoints/MANIFEST.md` was
computed, so that a reproduction can be compared like for like.

## Metrics

| name | where | definition |
|---|---|---|
| `ate_um` (Table 2, NanoBench) | `src/network/test.py`, `airio/evaluation/eval_idso3.py:ate_for`, `scripts/paper/unified_eval.py` | integrate the predicted body velocity through **ground-truth** attitude, Umeyama-align (rotation + translation, **no scale**) to the ground-truth trajectory, RMSE of position error, mean over the 17 test sequences |
| `ate` (TLIO convention) | `src/network/test.py` | mean L2 position error without alignment; called `ATE_meanL2` in the paper's supplementary material. Not comparable to `ate_um` |
| filter ATE (Table 1, TLIO/RIDI) | `src/plot_filter_state.py` (TLIO metric) and `eqnio/TLIO-master/src/analysis/EKF_output_metrics.py` (EqNIO metric) | **two different codes exist**: the TLIO metric origin-aligns and uses a 1 s yaw-aligned RTE (the paper's stated definition); the EqNIO metric does not origin-align, skips the first 2000 samples and does not remove yaw in RTE. Table 1 mixes them; `checkpoints/MANIFEST.md` states which one each recomputed value used |
| velocity RMSE (Table 8/9 of the camera-ready, Fig. 4/8) | `scripts/paper/` covariance-ablation scripts | per-window RMSE of predicted vs ground-truth body velocity |

Network-only testing integrates through ground-truth orientation. As the TLIO authors put it,
these trajectories are a debugging tool, not a benchmark of a full system; the filter-connected
numbers are the system numbers.

## ID/SO(3)

Rotate the accelerometer and gyroscope channels of every window by a sampled `R ∈ SO(3)`, rotate
vector targets by the same `R`, transform covariance targets by `R Σ Rᵀ`, un-rotate the predictions
and score them against the original targets. Implementations:

* `airio/evaluation/eval_idso3.py` — NanoBench, Air-IO family. Reports ATE under (I) identity,
  (P) a physical remount `R_WS → R_WS Qᵀ` of the attitude input and (G) the "rotate every vector
  input" reading, plus the mean-equivariance error `‖Qᵀ f(Qx) − f(x)‖` and the covariance
  congruence error `‖Qᵀ Σ(Qx) Q − Σ(x)‖_F / ‖Σ‖_F`. Rotations: uniform random axis, angle
  uniform in [0.2, π], `numpy.random.default_rng(0)`.
* `src/main_net.py --mode test --test_perturb_type so3 --test_perturb_deg D` — Lie-Neuron
  family. `TransformTestSO3Rotation` rotates IMU channels **and** vector targets by the same
  rotation (equivalent to un-rotating predictions). An older copy of this transform rotated only
  the IMU and therefore penalised equivariant models; it is not in this repository.
* Filter-connected ID/SO(3) (Table 1) uses the EqNIO authors' rotated-test generator
  `eqnio/TLIO-master/src/dataprep_test_augmentation/dataprep_ekf.py` (a consistent change of
  basis of the raw IMU stream and calibration; four rotations per sequence), with the rotated
  ground truth and no filter-initialisation compensation.

The rotation distribution of the submitted Table 2 ID/SO(3) column for the non-equivariant rows
was not recorded and no generator for those four cells survives; the equivariant rows are
unchanged under any rotation by construction (measured ≤ 5e-9 relative for GINIO-A).

## Checkpoint selection

* `airio/train_motion.py` (AirIO baselines and GINIO-A) follows upstream AirIO: `best_model.ckpt`
  is the epoch with the lowest loss on the config's `test` block, which in
  `airio/configs/datasets/Nanobench/nanobench_body.conf` is the 17-sequence reported test split
  (the `eval` block is a disjoint 16-sequence set that is only logged). This applies identically
  to AirIO and GINIO-A rows.
* `src/network/train.py` (Lie-Neuron family) keeps `checkpoint_best.pt` at the lowest mean
  validation loss (`val_list.txt`), plus fixed-epoch snapshots.
* Validation loss and NanoBench ATE are anti-correlated across our runs (r ≈ −0.67): ATE
  integrates velocity **bias**, so lower-loss checkpoints can have worse ATE. Report both.

## Augmentation

No rotation augmentation is used in any released training configuration, so ID/SO(3) results
measure architecture, not learned invariance. The Lie-Neuron family applies a per-window constant
bias shift (accel ±0.2 m/s², gyro ±0.05 rad/s) during training; the Air-IO family applies none.
`--so3_train_aug` exists for the augmentation ablations only.

## Provenance notes (read before quoting a number)

* GINIO-E in Table 2 was trained on TLIO with full-SO(3) augmentation and evaluated zero-shot on
  NanoBench; there is no NanoBench-trained GINIO-E.
* GINIO-RS and GINIO-RD were trained on `nanobench_bodyframe_nolfa_2nd_version`, i.e. without
  LFA, although Table 2 labels GINIO-RS "body+LFA".
* The processed NanoBench `.npy` files have gravity removed from the accelerometer although the
  description JSON says otherwise. This does not affect the NN-only Table 2 but breaks any
  NanoBench filter run that expects gravity.
* The RIDI "EqNIO" row of Table 1 is the SO(3)-augmented EqNIO model.
* The last-pre-GINIO code and every provenance discrepancy we know of are listed in
  [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).
