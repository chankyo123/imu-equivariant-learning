# Paper evaluation scripts (NanoBench, covariance ablation, frame-mismatch sweeps)

This folder keeps the "evaluation scripts will be released" promise of the paper. The scripts are
the authors' NanoBench evaluation code (May 2026, re-audited August 2026), ported mechanically:
numerical logic is unchanged; only machine paths, imports, argparse options and output locations
were edited, and the hand-off helpers were copied in (`_helpers.py`) so nothing depends on files
outside this repository. The first comment line of every script states which paper table or figure
it produced.

## Conventions

* All paths derive from the repository root (`GINIO_ROOT`, default: two levels above
  `scripts/paper/`). Overrides: `GINIO_CKPT_ROOT` (default `checkpoints/`), `GINIO_DATA_ROOT`
  (default `local_data/`), `GINIO_PAPER_OUT` (default `scripts/paper/outputs/`, git-ignored).
  Every script also exposes these as command-line options (`--ckpt_root`, `--data_root` /
  `--data_lfa` / `--data_nolfa`, `--out_dir`; the covariance-ablation tools use dashed names,
  `--out-dir`, as in the original).
* Checkpoints are referred to by their folder names in [`../../checkpoints/MANIFEST.md`](../../checkpoints/MANIFEST.md)
  (`ginio_rd_lfa_mse_nanobench`, ...). The flat names of the release bundle
  (`nanobench_ginio_rd_lfa_mse`, `tlio_eqnio_o2_ours`, ...) are accepted as aliases
  (`_paths.CKPT_ALIASES`); whichever folder exists under the checkpoint root is used.
* Data layout is the one of the main README: `local_data/nanobench_processed_2nd_version`
  (world frame, used for ground truth and by the Air-IO family),
  `local_data/nanobench_bodyframe_2nd_version` (body frame, LFA on by name) and the
  `local_data/nanobench_bodyframe_nolfa_2nd_version` symlink (same bytes, LFA off by name). The 17
  test sequences are listed in `nano_test_list.txt` (identical to `test_list.txt` of the dataset).
* Air-IO models are run through `airio/inference_motion.py`, which reads
  `airio/<exp_dir>/ckpt/best_model.ckpt`. `round1_eval*.py` and `scenarios_real_eval.py` create
  that `ckpt/` as a symlink to the released checkpoint folder when it is missing
  (`--no_link_ckpts` disables this).
* Runtime class: *CPU ok* means seconds to a few minutes on a laptop; *GPU recommended* means a
  full forward pass over the 17 test sequences per (model, perturbation) and takes hours on CPU.
* Python environment: the main `environment.yml` / `requirements.txt` (torch, numpy, scipy,
  matplotlib, pyhocon, pypose). No GPU is required for anything in this folder, only time.

## What reproduces what

| Paper item | Scripts | Notes |
|---|---|---|
| Table 2 metric (Umeyama-aligned ATE, unaligned RMSE, RPE@2s, drift) | `unified_eval.py` | the metric definitions; combines Lie-Neuron / EqNIO `trajectory.txt` trees and Air-IO `net_output.pickle` files |
| Table 2, Air-IO-family rows | `round1_eval.py` (AirIO no-att 5.579, with-att 1.205), `round1_eval_small.py` (GINIO-A no-att 1.430, with-att 1.045) | run `inference_motion.py`, integrate body velocity through ground-truth attitude, Umeyama ATE |
| Table 2 assembly: `canonical_v2.json`, `table_main.{md,tex}`; Supp. Table 9 ATE columns (`table_cov_ablation.md`); LFA ablation (`table_lfa_ablation.md`); figures 1-6 of the results bundle | `build_v2_paper_quality.py` | inputs shipped in `reference_inputs/`; regenerates the snapshot files byte-identically |
| Supp. Table 9 (covariance-head ablation, Diag/Spectral x MSE/NLL) | `run_phase_ab.py --phase A`, `sigma_eq_error.py`, `sigma_eq_error_lfa_aware.py`, `nll_decompose_calib.py`, `aggregate_results.py` (`TABLE_A_cov_ablation.csv`); per-dump tools `nll_decompose.py`, `nees_coverage.py`, `nll_temperature_calib.py`; other heads: `cov_eq_error.py`, `head_only_ablation.py` | the four Lie-Neuron models `ginio_rd_lfa_{mse,nll}_nanobench`, `ginio_rs_chol_{mse,nll}_nanobench` |
| Supp. Table 10 and Figures 6-7 (ID-SO(3) angle sweep 30-180 deg, mount-tilt 3-20 deg, LFA-mismatch, gravity-tilt) | `run_phase_ab.py --phase B`, `run_phase_ab_airio.py`, `run_phase_ab_eqnio.py`, then `compute_idso3_ate.py`, `aggregate_results.py` (`TABLE_B_*.csv`, `fig_id_so3_*`, `fig_calib_mismatch_*`), `cross_model_summary.py` (`TABLE_B_cross_model.csv`, `fig_eq_vs_noneq_perturbation.png`) | 8 models x 5 angles x 3 axes per scenario |
| Scenario evaluation of the Air-IO family (Standard / Aggressive / ID-SO(3) physical remount / calibration mismatch) | `scenarios_real_eval.py` | produced `scenarios_metrics.json`; **not** the generator of the Table-2 ID/SO(3) column (see provenance note 2) |
| none (planning stub) | `filter_connected_eval.py`, `PLAN.md` | command manifest + aggregation skeleton for a filter-connected ablation that was never run |

## Script reference

Commands are given from the repository root with the default locations; `python
scripts/paper/<script>.py --help` lists every option.

### `unified_eval.py` -- Table 2 metric and cross-model table (CPU ok)
Inputs: world-frame NanoBench (`--data_root`); per-model test outputs:
`--imueq_models name=subdir` under `--imueq_out` (default `test_outputs/`, i.e. what
`python src/main_net.py --mode test --out_dir test_outputs/<name> ...` writes: one
`trajectory.txt = [ts, pos_pred, pos_gt, vel_body_pred]` per sequence), `--airio_pickle
[name=]path` (`airio/inference_motion.py` output), `--eqnio_dir [name=]dir` (same `trajectory.txt`
format from `eqnio/TLIO-master/src/main_net.py --mode test`). Defaults name the released
checkpoints; models whose outputs are absent are skipped.
```bash
python scripts/paper/unified_eval.py \
  --imueq_models ginio_rs_chol_mse_nanobench=ginio_rs_chol_mse_nanobench ginio_rd_nolfa_t55_nanobench=ginio_rd_nolfa_t55_nanobench \
  --airio_pickle ginio_a_noatt_nanobench=airio/experiments/nanobench_noatt_vn_small/motion_body_noatt_vn_small/net_output.pickle
# -> scripts/paper/outputs/unified/{all_models_metrics.json, summary.txt, per_sequence.csv, plots/<seq>.png}
```

### `round1_eval.py`, `round1_eval_small.py` -- Air-IO-family Table-2 rows (GPU recommended for inference; metrics CPU ok)
Inputs: `airio/configs/Nanobench/motion_body_{noatt,rot}.conf` (+ `_vn_small` for GINIO-A) and the
released `best_model.ckpt` (`airio_noatt_nanobench`, `airio_withatt_nanobench`,
`ginio_a_noatt_nanobench`, `ginio_a_withatt_nanobench`), world-frame NanoBench.
```bash
python scripts/paper/round1_eval.py --device cuda:0          # AirIO no-att / with-att (+ the unreleased 0.273M VN originals, skipped)
python scripts/paper/round1_eval_small.py --device cuda:0    # GINIO-A no-att / with-att (Table 2)
# -> scripts/paper/outputs/round1_nanobench_metrics{,_small}.json, round1_nanobench_summary.txt
```
Add `--skip-inference` to re-score existing pickles. Whole-sequence inference of the VN-GRU is a
float64 Python time loop (several minutes per model on GPU, much longer on CPU).

### `build_v2_paper_quality.py` -- canonical_v2.json and the paper tables (CPU ok, seconds)
Inputs (defaults in `reference_inputs/`): `round1_nanobench_metrics.json` (output of
`round1_eval.py`), `nano_per_seq_dual_metric.csv` and `nano_aggregate_dual_metric.json` (the
authors' per-sequence `ATE_um` / `ATE_meanL2` and aggregate RMSE / RPE / drift for the ResNet1D,
EqNIO O(2), Lie-Neuron and native-AirIO rows; produced on the original training machine).
```bash
python scripts/paper/build_v2_paper_quality.py            # -> scripts/paper/outputs/nanobench_final_table/
python scripts/paper/build_v2_paper_quality.py --aws_metrics scripts/paper/outputs/round1_nanobench_metrics.json   # with your own Air-IO numbers
```
Verified: with the shipped inputs the output `canonical_v2.json`, `table_main.md`, `table_main.tex`,
`table_cov_ablation.md` and `table_lfa_ablation.md` are byte-identical to the authors' files
(`reference_inputs/canonical_v2.json` is that reference).

### `run_phase_ab.py` -- Phase A/B on the four Lie-Neuron models (GPU recommended)
Inputs: `checkpoints/ginio_rd_lfa_{mse,nll}_nanobench`, `checkpoints/ginio_rs_chol_{mse,nll}_nanobench`
(`checkpoint_best.pt` with its `args`), body-frame NanoBench (`--data_lfa` for the LFA models,
`--data_nolfa` for the spectral models), `scripts/lfa_preprocessing.py`, `src/network/model_factory.py`.
```bash
python scripts/paper/run_phase_ab.py --phase A                # clean: per-window dumps + NLL / NEES / coverage per model
python scripts/paper/run_phase_ab.py --phase B                # ID-SO(3) {30,60,90,120,180} deg, mount tilt {3,5,10,15,20} deg, LFA mismatch {1,3,5,10} deg; 3 axes each
# -> outputs/aggregate__<model>__<scenario>__<seed>.json, outputs/dumps/<model>__<scenario>__<seed>__<seq>.npz
```
Phase B is 4 models x 42 runs over the 17 test sequences.

### `run_phase_ab_airio.py`, `run_phase_ab_eqnio.py` -- the same sweeps for the other families (GPU recommended)
`run_phase_ab_airio.py`: `airio_noatt_nanobench`, `airio_withatt_nanobench` (AirIO native) and
`ginio_a_noatt_nanobench`, `ginio_a_withatt_nanobench` (released GINIO-A; see provenance note 4),
body-frame NanoBench, no LFA. `run_phase_ab_eqnio.py`: the TorchScript traces
`eqnio_o2_tlio_ours/model_torchscript.pt` (EqNIO O(2), our TLIO reference model) and
`ginio_e_so3aug_tlio/model_torchscript.pt` (GINIO-E); converts body-frame IMU to gravity-aligned
world frame as the TLIO filter does; scenarios `id_so3`, `mount_tilt`, `gravity_tilt`.
```bash
python scripts/paper/run_phase_ab_airio.py --phase all
python scripts/paper/run_phase_ab_eqnio.py --phase all
```

### `compute_idso3_ate.py`, `compute_eqnio_ate.py` -- trajectory ATE from the dumps (CPU ok)
```bash
python scripts/paper/compute_idso3_ate.py     # -> outputs/trajectory_ate_idso3.json (all 10 models, clean + 5 angles)
python scripts/paper/compute_eqnio_ate.py     # -> outputs/eqnio_trajectory_ate.json (EqNIO_O2, VN_EqNIO_SO3 clean)
```
Both expect the complete set of dumps (they integrate the predicted velocity at stride 20 and
Umeyama-align with a similarity transform, i.e. including scale; Table 2's `ATE_um` in
`unified_eval.py` / `round1_eval.py` is rotation + translation only).

### `sigma_eq_error.py`, `sigma_eq_error_lfa_aware.py` -- Sigma-eq. / mu-eq. error (CPU ok)
64 windows of `B11_random_waypoints_rep1`, 8 random rotations, the four Lie-Neuron models.
The naive test rotates the IMU only; the LFA-aware test rotates IMU and quaternions jointly.
```bash
python scripts/paper/sigma_eq_error.py            # -> outputs/sigma_eq_results.json
python scripts/paper/sigma_eq_error_lfa_aware.py  # -> outputs/sigma_eq_results_lfa_aware.json
```

### `nll_decompose_calib.py`, `nll_decompose.py`, `nees_coverage.py`, `nll_temperature_calib.py` (CPU ok)
`nll_decompose_calib.py` reads the Phase-A clean dumps of the four Lie-Neuron models, splits the 17
test sequences deterministically (sorted, even -> val, odd -> test), fits the temperature
`alpha* = mean_val(Mahalanobis)/3` and writes `outputs/TABLE_A_nll_decomp_calib.json`. The other
three work on any single `.npz` with `(e|err, Sigma)` or `(pred, gt, Sigma)`, e.g. a dump:
```bash
python scripts/paper/nll_decompose_calib.py
python scripts/paper/nll_decompose.py --pred-npz scripts/paper/outputs/dumps/ln_resnet_chol_MSE__clean__clean__B11_random_waypoints_rep1.npz --name chol_mse_B11
python scripts/paper/nees_coverage.py --pred-npz <same> --name chol_mse_B11
python scripts/paper/nll_temperature_calib.py --val-npz <val.npz> --test-npz <test.npz> --name chol_mse
```

### `aggregate_results.py`, `cross_model_summary.py` -- tables and figures (CPU ok)
```bash
python scripts/paper/aggregate_results.py     # TABLE_A_cov_ablation.csv, TABLE_B_id_so3.csv, TABLE_B_calib_mismatch.csv, fig_*_rmse_vel.png, fig_*_mean_nll.png, fig_coverage_curve.png
python scripts/paper/cross_model_summary.py   # TABLE_B_cross_model.csv, fig_eq_vs_noneq_perturbation.png (needs all 10 models' clean + id_so3 + tilt runs)
```
`aggregate_results.py` needs Phase A of all four Lie-Neuron models plus both `sigma_eq_*` files;
`cross_model_summary.py` needs the clean runs and sweeps of all ten models.

### `cov_eq_error.py`, `head_only_ablation.py` -- Sigma-equivariance of the other covariance heads (CPU ok, float64 by design)
```bash
python scripts/paper/cov_eq_error.py --arch vn_airio_diag          # released GINIO-A (VNCodeNetMotionSmall); expect Sigma-err ~1e-3 (diagonal-from-norms is not congruent)
python scripts/paper/cov_eq_error.py --arch vn_eqnio_so3_frame --seqlen 200   # GINIO-E torchscript; expect ~1e-14
python scripts/paper/cov_eq_error.py --arch chol --ckpt <torchscript of a Cholesky-head model>
python scripts/paper/head_only_ablation.py --mode eval --dataset tlio --seqlen 200
python scripts/paper/head_only_ablation.py --mode retrain-cmd      # prints the (never executed) head-only retraining recipe
```
`--vn-class VNCodeNetMotion` selects the 0.273M class for a self-trained original checkpoint. The
released TLIO-trained TorchScript traces have a fixed 200-sample window, so pass `--seqlen 200`
for `vn_eqnio_so3_frame` / `chol` / `--spec-ts` (the inherited default of 128 only fits the
length-agnostic VN-AirIO models).

### `scenarios_real_eval.py` -- Air-IO-family scenario evaluation (GPU recommended)
Writes perturbed copies of the 17 test sequences to `--work_dir` (default `/tmp/ginio_scenarios`),
re-runs `inference_motion.py` on them and scores with the Table-2 metric. ID-SO(3) here is a
physical remount (quaternion `R_WS -> R_WS R_mount^T`, one random 45-135 deg rotation per
sequence); calibration mismatch is a constant body-frame gyro/accel bias.
```bash
python scripts/paper/scenarios_real_eval.py --scenarios standard aggressive id_so3 calib_mismatch --device cuda:0
# -> outputs/scenarios_metrics.json, outputs/scenarios_summary.txt
```

### `filter_connected_eval.py` -- planning stub (CPU ok, emits commands only)
```bash
python scripts/paper/filter_connected_eval.py --dataset tlio --emit-cmds    # outputs/filter/launch_tlio.sh
python scripts/paper/filter_connected_eval.py --dataset tlio --aggregate    # reads outputs/filter/<tag>/metrics.json if you produced them
```

## Reproduction order

1. Lie-Neuron Table-2 rows: `python src/main_net.py --mode test ... --out_dir test_outputs/<name>` for
   `ginio_rs_chol_mse_nanobench` and `ginio_rd_nolfa_t55_nanobench` (commands in the main README;
   window 1.0 / sample_freq 20 for the spectral model, window 0.675 / sample_freq 10 for trial 55),
   then `unified_eval.py`.
2. Air-IO rows: `round1_eval.py`, `round1_eval_small.py` (GINIO-E: `eqnio/TLIO-master/src/main_net.py --mode test` on the
   NanoBench body-frame data with the released `ginio_e_so3aug_tlio` checkpoint, then `unified_eval.py --eqnio_dir`).
3. `build_v2_paper_quality.py` (default inputs reproduce the paper's canonical file; pass your own metrics to rebuild).
4. Covariance ablation: `run_phase_ab.py --phase A`, `sigma_eq_error.py`, `sigma_eq_error_lfa_aware.py`,
   `nll_decompose_calib.py`, `aggregate_results.py`.
5. Sweeps: `run_phase_ab.py --phase B`, `run_phase_ab_airio.py`, `run_phase_ab_eqnio.py`,
   `compute_idso3_ate.py`, `aggregate_results.py`, `cross_model_summary.py`.

## Known provenance notes

Taken from the authors' code audit (`CODE_MAP` sections 7.1 and 8, gotchas G2-G4, G7, G14, G16,
G19-G23) and `checkpoints/MANIFEST.md`; please read before quoting numbers.

1. **`canonical_v3.json` is synthetic and must not be used.** It was produced by
   `make_canonical_v3_synthetic.py` (every per-sequence ATE shifted by +0.1 m or redistributed,
   stamped `_SYNTHETIC`) for a plotting test. Neither that script nor the synthetic path of
   `plot_nominal_cdf.py` is ported. The authoritative file is `canonical_v2.json`
   (`reference_inputs/`), which `build_v2_paper_quality.py` regenerates.
2. **The Table-2 ID/SO(3) column has no reproducible generator in this release.** The two
   candidates, `scenarios_real_eval.py` (physical remount, random 45-135 deg per sequence) and the
   `run_phase_ab*.py` angle sweeps, produced numbers that differ from the paper's column (e.g. AirIO
   with-att 10.717 vs 1.302) and contain no GINIO-A entries; the rotation distribution used for the
   column is undocumented. The paper's protocol (rotate accel/gyro and vector targets, un-rotate
   predictions) is implemented for the Air-IO family in `airio/evaluation/eval_idso3.py` (rotation
   angles U[0.2, pi]). The imu-equivariant-learning test transform `--test_perturb_type so3` of
   older code copies does not un-rotate predictions and is not that protocol.
3. **GINIO-E numbers come from a TLIO-trained model.** `ginio_e_so3aug_tlio` was trained on TLIO
   golden with full-SO(3) augmentation (40 epochs) and evaluated zero-shot on NanoBench; it was
   never trained on NanoBench. The `EqNIO_O2` entry of `run_phase_ab_eqnio.py` is
   `eqnio_o2_tlio_ours`, our own TLIO-trained O(2) reference model; the paper's EqNIO O(2)
   NanoBench row (0.828) used the EqNIO authors' public model, for which no checkpoint exists here.
   Both released traces are identified by the `out_dir` recorded in their `parameters.json`
   (`overnight/vn_so3_so3aug`, `overnight/o2_paper`).
4. **The VN-AirIO rows of the sweeps were originally the unreleased 0.273M models.**
   `run_phase_ab_airio.py`, `round1_eval.py`, `scenarios_real_eval.py`, `cov_eq_error.py` and
   `head_only_ablation.py` loaded `experiments/nanobench_{noatt,rot}_vn` with `strict=False`; those
   checkpoints predate `VNLayerNorm`, load with 15 missing keys against the current
   `airio/model/vn_codenet.py` and are not released (MANIFEST note 5). The ported defaults use the
   released `_small` GINIO-A models of Table 2 instead, so re-running the sweeps gives different
   VN-AirIO rows than the May-2026 `TABLE_B_cross_model.csv`. Likewise `canonical_v2.json`'s
   `VN_AirIO_*` entries (1.502 / 1.062) are the 0.273M originals (Supplement Table 5); the Table-2
   GINIO-A values (1.430 / 1.045) are in `reference_inputs/round1_nanobench_metrics_small.json`.
5. **GINIO-RS and LFA.** The `parameters.json` of `ginio_rs_chol_mse_nanobench` records a `nolfa`
   training root while the paper labels the row body+LFA; MANIFEST note 2 reproduces 0.7887 with
   LFA applied at test time. `run_phase_ab.py` (and the Table-9 sweep) evaluates both spectral
   models **without** LFA (`lfa=False`, as in the original script); keep this in mind when comparing
   its `rmse_vel` with Table 2.
6. **GINIO-RD.** Optuna trial 119 (0.581) was not archived; `ginio_rd_nolfa_t55_nanobench` (trial
   55, 0.600, window 0.675 s) is the closest released model. The covariance-ablation scripts only
   use the four 1.0 s-window Table-9 models.
7. **Supplement Table 9 confound (G19).** The diagonal models (`ln_resnet`, FcBlock `bias=True`)
   and the spectral models (`ln_resnet_chol`, `bias=False`, +40 % parameters) differ in more than
   the head, so the `mu-eq. err` and RMSE columns are not a pure head comparison; the `Sigma-eq.
   err` column is unaffected. `--arch ln_resnet` loads `ln_conv_2regress_nochannelmix_bias.py`
   (`negative_slope=0.0`, MANIFEST note 6).
8. **Two ATE conventions (G20).** `ate` in `src/network/test.py` is the TLIO mean-L2 position error;
   Table 2, `canonical_v2.json`, `unified_eval.py` and `eval_idso3.py` use the Umeyama-aligned RMSE
   `ATE_um`. `compute_idso3_ate.py` / `compute_eqnio_ate.py` additionally estimate a scale factor.
9. **Rows that cannot be regenerated here**: ResNet1D 5.43M, EqNIO O(2) NanoBench, GINIO-RD trial
   119, all Table-1 (TLIO/RIDI filter) and Table-3 (Fetch) models have no checkpoint in the
   authors' archive; their numbers enter only through `reference_inputs/nano_per_seq_dual_metric.csv`.
10. **Checkpoint selection and augmentation (G4, G16).** Air-IO `best_model.ckpt` is the epoch with
    the lowest loss on the config's `test` block, which lists the 17 reported sequences (identical
    for the AirIO baselines and GINIO-A). No released training configuration uses rotation
    augmentation; the sweeps therefore measure architectural equivariance.
11. **Quaternion renormalisation (G7).** The sweeps feed the stored NanoBench quaternions (unit to
    ~5e-9) without renormalising; the recurrent VN-GRU amplifies this to ~1e-1 apparent
    equivariance error over a whole sequence, and the with-attitude GINIO-A model moves 1.0453 ->
    1.0412 when quaternions are renormalised.
12. **`filter_connected_eval.py` never produced paper numbers.** Its TLIO command template uses
    `--model_arch`, `--list_path`, `--so3_rotate_inputs`, which the released
    `eqnio/TLIO-master/src/main_filter.py` does not implement (it requires `--model_param_path`);
    the NanoBench branch references a `filter_eval.py` that is not part of this release; the
    "main paper table" of `PLAN.md` was never filled. The registry entry for the unreleased
    `overnight/_existing_vn_so3_t0012` trace was dropped.
13. **Temperature calibration uses half of the test set.** `nll_decompose_calib.py` fits `alpha*`
    on the even-indexed (sorted) test sequences and reports on the odd-indexed ones; there is no
    separate validation dump.
14. The hand-off helpers `rotate_vectors_btf3` and `assemble_cov` are copied verbatim into
    `_helpers.py` from `equivariant_models_handoff/equivariance_tests/test_cov_equivariance.py`
    (2026-05-23); the `Sigma` reconstruction formulas in `_utils.py` mirror
    `eqnio/TLIO-master/src/tracker/meas_source_torchscript.py`.

## Files

| File | Role |
|---|---|
| `_paths.py` | repository-relative paths, checkpoint-name aliases, Air-IO experiment/checkpoint helpers |
| `_helpers.py` | `rotate_vectors_btf3`, `assemble_cov` (hand-off copies) |
| `_utils.py` | shared Sigma reconstruction, NLL terms, SO(3) sampler, save helpers (covariance ablation) |
| `nano_test_list.txt` | the 17 NanoBench test sequences |
| `reference_inputs/` | `canonical_v2.json` (reference), `nano_per_seq_dual_metric.csv`, `nano_aggregate_dual_metric.json`, `round1_nanobench_metrics.json`, `round1_nanobench_metrics_small.json` |
| `PLAN.md` | the May-2026 planning note of the covariance ablation (historical; research-box paths) |
| `outputs/` | everything the scripts write (git-ignored) |
