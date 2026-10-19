<!-- Historical planning document of the covariance-head ablation (2026-05-25), copied verbatim from the
     authors' research box (nano_results/aws_results/cov_ablation/PLAN.md). Paths and checkpoint names in it
     refer to that machine; see README.md in this folder for the released layout and checkpoint names.
     The "main paper table" mock at the bottom was never filled by filter_connected_eval.py. -->

# Covariance Head Ablation — Implementation Plan

Scope: head-only and filter-connected ablations comparing diagonal-from-norms vs spectral
(and Cholesky in appendix) covariance heads on TLIO-golden and Nanobench.

All scripts: `/home/ubuntu/imu-work/nano_results/aws_results/cov_ablation/`
Outputs: `./outputs/<dataset>_<head>_<phase>/{summary.txt,results.json,*.csv}`

## File list

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| `cov_eq_error.py` | Σ-eq error + μ-eq error per head on a batch of test windows | model ckpt + arch tag (`vn_airio_diag`, `vn_eqnio_so3_frame`, `eqnio_o2_frame_full`, `chol`) | `outputs/eq_error_<arch>.json` |
| `nll_decompose.py` | Per-sample Mahalanobis `e^T Σ^-1 e`, log-det `log|Σ|`, total NLL | predictions+GT csv (or live forward on a loader) | `nll_decomp_<run>.csv`, `*.json` |
| `nll_temperature_calib.py` | Fit α (val) such that Σ̂_α=αΣ̂ minimizes val NLL, report calibrated test NLL | predictions+GT (val and test) | `temp_calib_<run>.json` |
| `nees_coverage.py` | Mean NEES, χ²₃ coverage at q∈{0.5,0.7,0.9,0.95}, calibration curve PNG | predictions+GT | `nees_<run>.json`, `coverage_<run>.png` |
| `head_only_ablation.py` | Loads two checkpoints sharing the mean head; differing only in cov head; runs eq_error + NLL on the same loader | ckpt list + arch list | `head_only_<dataset>.json` |
| `filter_connected_eval.py` | Drives TLIO `main_filter.py` (and Nanobench `filter_eval.py`) on clean vs SO(3)-rotated input, per cov head | torchscript model paths + dataset config | `filter_<dataset>_<head>_<clean|rotR>.json` + main paper table CSV |
| `_utils.py` | Shared helpers (load configs, random SO(3), Σ reconstruction matching `meas_source_torchscript.py`) | — | — |

## Dependencies / sys.path additions

Each script (re)inserts:
- `/home/ubuntu/imu-work/Air-IO` (for `model`, `model.vn_codenet`, `datasets`, `configs`)
- `/home/ubuntu/imu-work/EqNIO/TLIO-master/src` (for `network.model_factory.get_model`,
  `tracker.meas_source_torchscript.MeasSourceTorchScript`)
- `/home/ubuntu/imu-work/equivariant_models_handoff/equivariance_tests` (for
  `test_cov_equivariance` helpers — `rotate_vectors_btf3`, `assemble_cov`)

Pip deps: `numpy scipy torch pyhocon pandas matplotlib tqdm`.

## Σ reconstruction (must match TLIO filter exactly)

Verified against `/home/ubuntu/imu-work/EqNIO/TLIO-master/src/tracker/meas_source_torchscript.py`:

- VN-AirIO diag (`vn_airio_diag`): `cov` head outputs 3 log-variances (after `exp(cov_raw-5.0)`),
  Σ = diag(exp(cov_raw-5)). Applied in BODY frame. Not rotated post-hoc.
- VN-EqNIO SO(3) (`so3_frame`): model returns `(frame R^T, disp_canonical, cov_canonical)`.
  Disp_world = `frame^T @ disp_canonical`; Σ_world = `frame^T @ diag(exp(2·log_std)) @ frame`.
  Clamp `log_std<-4 → -4`. (lines 296–312)
- EqNIO O(2) `_fullCov` (`tlio_frame_fullCov`): `meas_cov[:-1]` are xy log-std (canonical);
  Σ_xy_world = `frame^T @ diag(exp(2·log_std_xy)) @ frame`; z-variance is exp(2·log_std_z) on the
  diagonal corner. (lines 314–333)
- Cholesky (`chol`): net returns `(mean, A)` shape [B,3,3]; Σ = `A Aᵀ + 1e-3 I`. (lines 347–355)

Tests must reuse these branches verbatim — script `cov_eq_error.py` imports `_utils.assemble_sigma`
that mirrors them.

## Checkpoints to use

- VN-AirIO Nanobench (diag): `/home/ubuntu/imu-work/Air-IO/experiments/nanobench_noatt_vn/motion_body_noatt_vn/ckpt/best_model.ckpt`
  and `/home/ubuntu/imu-work/Air-IO/experiments/nanobench_rot_vn/motion_body_rot_vn/ckpt/best_model.ckpt`
  (`strict=False` to tolerate VNLayerNorm/conv-channel drift — same as `inference_motion.py`).
- VN-EqNIO SO(3) torchscript: pick best filter ATE from leaderboard; e.g.
  `nano_results/aws_results/overnight/_existing_vn_so3_t0012/model_torchscript.pt` and
  `vn_so3_so3aug/model_torchscript.pt`.
- EqNIO O(2) torchscript: `overnight/o2_paper/model_torchscript.pt`, `overnight/_existing_o2_t0007/model_torchscript.pt`.

For head-only on AirIO we use the two existing VN-AirIO checkpoints that share backbone+mean
head (built the same way) — the diff is then strictly the cov head signal under R-rotation,
since both share the diag-from-norms reconstruction (so within-AirIO comparison degenerates).
For TRUE head-only Diag-vs-Spectral, the script also documents the retraining command
(see Compute estimate below).

## Compute estimate

| Task | GPU-hours | Wall on 4×RTX PRO 6000 | Data prep |
|---|---|---|---|
| `cov_eq_error.py` per arch (1 batch×8 trials) | 0 | <2 min CPU/GPU | none |
| `nll_decompose.py` over Nanobench test (17 seq) | 0.1 | ~5 min/arch | needs preds dump |
| `nll_temperature_calib.py` | 0 | <1 min | needs val+test preds |
| `nees_coverage.py` | 0 | <1 min/run | preds+GT |
| `head_only_ablation.py` (no retraining, eval only) | 0.2 | ~10 min/arch | none |
| `head_only_ablation.py` (full retrain frozen mean → swap cov head) | ~2 h/head/dataset GPU | ~2 h on 1 GPU | full train loader |
| `filter_connected_eval.py` TLIO 36 seqs clean | ~3 h CPU/arch | 3 h | torchscript convert: 30s |
| `filter_connected_eval.py` TLIO 36 seqs rotated | ~3 h CPU/arch | 3 h | random R per-seq fixed |
| Nanobench filter (17 seqs) clean+rot | ~2 h CPU/arch | 2 h | — |
| **Main table all rows** (Diag + Spec on 2 datasets, clean+rot) | ~20 h CPU | ~5 h wall (4 workers) | — |

Honest gaps:
- **Pure head-only** requires retraining VN-AirIO and VN-EqNIO with a frozen-mean phase where the
  cov head is swapped. Not currently in `train_motion.py`. ~2h/head once added.
- Filter on **Nanobench** requires an EqNIO-style filter for the drone domain. We have
  `aws_results/filter_eval.py` (TLIO-style with our gravity-aware tweaks). Re-using on Nanobench
  presumes the same EKF interface — verify before launching.

## Risks / unknowns

1. **VN-AirIO ckpt drift**: saved weights predate current `vn_codenet.py` (VNLayerNorm added,
   conv-channel layout shifted). Loading must use `strict=False`. Document missing/unexpected key
   counts in JSON output. Solve: stick with `best_model.ckpt` in `nanobench_*_vn`; do NOT
   re-train mid-ablation.
2. **TLIO filter expects torchscript-traced model**: rerun `convert_model_to_torchscript_eq.py`
   for any new cov-head variant. Cholesky and head-swapped models need new trace.
3. **Calibration-mismatch perturbation must be in body frame**: when running `--perturb-cal`,
   apply bias before the gravity removal (see fix in `scenarios_real_eval.py`).
4. **Σ-eq error has unit ambiguity**: report normalized `‖Σ̂(R·z)−R Σ̂(z) Rᵀ‖_F / (‖R Σ̂ Rᵀ‖_F + 1e-12)`
   averaged over 8 random R, dtype=float64 forward, log raw and normalized.
5. **NLL log-det sign**: log|Σ| = sum log eig; small Σ → very negative log-det → can dominate.
   `nll_decompose.py` reports terms separately so reviewers see what's pushing the number.
6. **α calibration scope**: α is fit on val NLL only; reported test NLL uses train-time Σ scaled
   by that α. Do NOT refit on test.

## Main paper table (mock — fill in from eval)

| Dataset | Cov head | Σ-eq? | Clean ATE (m) | Rotated ATE (m) | Δ_R NLL |
|---|---|---|---|---|---|
| TLIO-Golden | Diag (VN-AirIO style) | No (2e-3) | {TBD} | {TBD} | {TBD} |
| TLIO-Golden | Spectral (VN-EqNIO SO(3)) | Yes (6e-15) | {TBD} | {TBD} | {TBD} |
| Nanobench   | Diag (VN-AirIO) | No | {TBD} | {TBD} | {TBD} |
| Nanobench   | Spectral (VN-EqNIO SO(3)) | Yes | {TBD} | {TBD} | {TBD} |

Appendix table: + Cholesky row + per-head NLL decomposition (Mahalanobis / log-det / total),
calibrated vs raw NLL, NEES mean, χ²₃ coverage at 0.5/0.7/0.9/0.95.

## Execution order

1. Sanity: `python cov_eq_error.py --arch vn_airio_diag` (expect Σ-err ~2e-3) and
   `python cov_eq_error.py --arch vn_eqnio_so3_frame` (expect ~1e-14). Runs in seconds.
2. Dump predictions (val+test) for each (dataset, model) using a thin wrapper around
   `inference_motion.py` (Air-IO) / `EqNIO src/network/test.py` (EqNIO). Save `.npz` of
   `{e, mu, Sigma, gt}`.
3. Run `nll_decompose.py` + `nll_temperature_calib.py` + `nees_coverage.py` on each dump.
4. Convert any new-head models to torchscript, then `filter_connected_eval.py` on TLIO and
   Nanobench, clean and SO(3)-rotated.
5. Aggregate to main_table.csv via `filter_connected_eval.py --aggregate`.
