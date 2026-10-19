# Known issues: paper text vs. surviving evidence

Erratum worklist for the authors of *GINIO: A Geometric SO(3)-Equivariant Interface for Neural
Inertial Odometry* (CoRL 2026, OpenReview `nl8iBTm3Mt`). Every entry states what the camera-ready
PDF prints, what the stored artifacts show, and where the evidence is. Table and figure numbers
are the **camera-ready** ones (camera-ready Table 8 = submission Table 9 covariance ablation;
camera-ready Table 9 = submission Table 10 frame-estimation mismatch; camera-ready Tables 10-12 =
Fetch). `cr.txt:N` refers to line N of the text dump of `GINIO_CoRL2026_camera_ready.pdf`;
`results/...`, `scripts/...`, `checkpoints/<bench>/...` paths are inside the Server B handoff bundle
(result files are shipped in `paper_results.tar.gz` under `serverB/`); `Server A ...` paths are from
the AWS box (`~/imu-work`), shipped under `serverA/` where they are result files.
`RECOMPUTED` = `results/recomputed_2026-10/RECOMPUTED_FILTER_METRICS.csv`;
`TABLE_A` = Server A `nano_results/aws_results/cov_ablation/outputs/TABLE_A_cov_ablation.csv`;
`MASTER` = `results/HANDOFF_paper_handoff_v2/MASTER_RESULTS.csv`.

Entries are ordered by how much of the paper they touch. "Measured" means a value read from a
stored output file; nothing below is inferred.

---

## 1. Table 8, Spectral/NLL row: NLL and NEES have no measured source

* Paper: Table 8 prints Spectral NLL = **-1.875**, NEES = **2.974** (`cr.txt:1321`); Sec. 5.2 says
  "the NLL-trained spectral model has NEES closer to the ideal value of 3" (`cr.txt:536-537`).
* Evidence: the only measurement of this model (`nanobench_ginio_rs_chol_nll`, sha `8a9f5c51b9b34ba9`)
  is `TABLE_A:5` (`ln_resnet_chol_NLL`): mean_NLL **-0.8747**, mean_mahal (NEES) **6.774**, cov_95
  0.8370. The diagonal row reproduces exactly (`TABLE_A:3`: -1.7411 / 2.9590 / 0.9146 vs printed
  -1.741 / 2.959 / 0.915), as do the Spectral/NLL velocity RMSE (0.26050), cov95 (0.8370),
  Sigma-eq. (4.337e-4) and mu-eq. (4.302e-4). No file on either machine contains -1.875 or 2.974.
  Server A's own audit already flagged the NEES/cov95 claim (`DO_NOT_USE_OR_CONDITIONAL.md` D2).
* Action: replace the two cells with the measured values or mark them unverified; the Sec. 5.2
  sentence about NEES is not supported by the measurement (6.774 vs 2.959).

## 2. Table 2, GINIO-RS: labelled body+LFA, trained without LFA

* Paper: Table 2 lists GINIO-RS with input "body+LFA" (`cr.txt:410`).
* Evidence: the checkpoint that produces the printed 0.789 (`nanobench_ginio_rs_chol_mse`,
  sha `0cbb3275e118a9f7`, Umeyama ATE 0.788745 in `results/nano_unified_metrics_v2/all_summary.txt:11`)
  records `root_dir = local_data/nanobench_bodyframe_nolfa_2nd_version`
  (`checkpoints/nanobench/ginio_rs_chol_mse/parameters.json:37`), i.e. **LFA off**. Server A's
  `canonical_v2.json` labels the same result `"frame": "body + LFA", "lfa": true` (lines 905-912),
  which is where the table label came from. The 88 LFA-trained `ln_resnet_chol` Optuna trial
  directories on Server B contain no weights (`results/server_B_rebuttal/INTEGRATION_CROSS_SERVER.md`,
  item A5).
* Action: change the input column of the GINIO-RS row to "body" (or retrain with LFA and re-report).

## 3. Table 2, GINIO-E: a TLIO-trained model evaluated zero-shot, not stated

* Paper: Table 2 presents GINIO-E (0.785, "gravity-aligned", 6.03M) alongside NanoBench-trained
  rows (`cr.txt:408`); Sec. 4.3 (`cr.txt:386-387`) does not mention a transfer.
* Evidence: the only model that produces 0.785 is `nanobench_ginio_e_so3aug_tlio` (sha
  `46ed9465a95d53ed`), trained on `local_data/tlio_golden` for 40 epochs with `so3_augment: true`
  (its `parameters.json` lines 10, 27, 31) and run on the NanoBench test split (Server A
  `cov_ablation/outputs/eqnio_trajectory_ate.json:26-46`, key `VN_EqNIO_SO3`, mean 0.7854). No
  NanoBench-trained `eq_so3_frame_fullCov_2vec_2deep` exists on either machine (Server A
  `MISSING_ARTIFACTS.md` A3; Server B `INTEGRATION_CROSS_SERVER.md` A3). The EqNIO O(2) row of the
  same table (0.828) *is* NanoBench-trained (`checkpoints/nanobench/eqnio_o2/parameters.json`,
  `root_dir ... nanobench_processed_2nd_version`), so the two rows differ in training data as well
  as architecture. Our TLIO-trained EqNIO O(2) transfers at 0.8044 (same json, key `EqNIO_O2`).
* Action: state the training set and the zero-shot protocol for GINIO-E in Table 2 / Sec. G.1.

## 4. Table 2, ID/SO(3) column of the non-equivariant rows: no generator

* Paper: 5.679, 1.302, 0.710, 1.267 (`cr.txt:401, 402, 406, 407`); Sec. 4.3 says these "are
  measured by rerunning inference under rotated test inputs" (`cr.txt:388-389`).
* Evidence: no script, notebook or output producing these four values exists on either machine
  (Server A `MISSING_ARTIFACTS.md` section D; `DO_NOT_USE_OR_CONDITIONAL.md` D12, line 77; Server B
  `MISSING.md`). Server A's two rotation-injection scripts give different numbers for the same cell:
  AirIO with-att. ID/SO(3) = 10.717 (project script, physical remount), 5.3877 (uniform SO(3),
  physical remount) and 1.7314 (Supplement G.1 reading), none equal to 1.302; the project script's
  AirIO no-att. ID/SO(3) row equals its ID/ID row on every metric. Server B's per-window sweeps
  (`MASTER`, `so3` rows) give ResNet1D 0.660 ... 1.562 and EqNIO O(2) 0.809 ... 0.876 depending on
  the angle.
* Action: either regenerate the column with a documented rotation distribution (and state it in
  Sec. G.1) or remove the four values.

## 5. Table 2 and App. C.3: GINIO-A with-attitude is not equivariant under a remount

* Paper: GINIO-A (with-att.) ID/ID = ID/SO(3) = 1.045 (`cr.txt:404`); App. C.3: "The full GINIO-A
  model accumulates to a mean-equivariance error below 5e-9" (`cr.txt:878`).
* Evidence: Server A `GATE_A__attitude_log_map.md` A.3-A.4.3 (lines 90-121): the attitude channel
  is `Log(R_WS)` fed as a vector channel; `Log(R Q^T) != Q Log(R)`, so under a physical remount /
  sensor-frame coordinate rotation the with-attitude model is not equivariant (ATE 1.0738, relative
  mean-equivariance error **0.90**). The equality 1.045 -> 1.045 holds only when the attitude channel
  itself is rotated like every other vector (the Supplement G.1 reading; error 5.0e-2, ATE 1.0461).
  The no-attitude model is exactly equivariant (4.4e-12, ATE 1.4302 under both readings), so the
  "< 5e-9" sentence is true for it alone. The with-attitude model is also chaotically conditioned
  over whole sequences (1e-16 input perturbation -> 32 % output change; A.4.4), and a retrain with
  the identical config gave 1.221 / 1.387 / 1.421 (`DO_NOT_USE_OR_CONDITIONAL.md` D3).
* Action: confine the exact-equivariance and ID/SO(3)-invariance claims to the no-attitude /
  direct / LFA variants and describe the with-attitude row as an AirIO-native attitude-aided
  comparison whose ID/SO(3) entry is protocol-defined.

## 6. Table 1 / Table 5, TLIO GINIO-W: the two cells of one row come from different weights, and 2.018 does not reproduce

* Paper: GINIO-W ID/ID 1.615, ID/SO(3) 2.018 (`cr.txt:347`, `cr.txt:1145`).
* Evidence: 1.615 is produced by the epoch-189 weights (`checkpoints/tlio/ginio_w_ln189/model_torchscript.pt`
  = `checkpoints/checkpoint_189.pt`; `RECOMPUTED:22`, 1.6146 TLIO-metric). The ID/SO(3) run used the
  epoch-86 `checkpoint_best.pt` (different weights; `checkpoints/CHECKPOINT_MANIFEST.csv:2-4`,
  `START_HERE.md:54-56`) and gives **2.2052** on the `so3_60` copy (`RECOMPUTED:36`); 2.018 is not
  reproduced by any stored output. Both weight sets are released in `tlio_ginio_w_ln189`.
* Action: re-run the ID/SO(3) cell with the epoch-189 weights (or re-run both cells with one weight
  set) and state which checkpoint the row uses.

## 7. Table 1 / Table 5, RIDI "EqNIO": the SO(3)-augmented EqNIO, and 3.460 does not reproduce

* Paper: RIDI EqNIO 2.408 / 3.460 under the same "EqNIO" label as the unaugmented TLIO EqNIO
  (`cr.txt:346`, `cr.txt:1153`).
* Evidence: 2.408 is `EqNIO/models_ridi/tlio_o2_so3aug` with `so3_augment: true` (180 deg)
  (`checkpoints/ridi/eqnio_o2_so3aug/parameters.json:33`; `results/RIDI_EqNIO_ATE_summary.md:22`,
  `RECOMPUTED:50`, 2.4082). The unaugmented RIDI EqNIO gives 2.160 (`RECOMPUTED:46`;
  `RIDI_EqNIO_ATE_summary.md:21`). Under the equivariance protocol the augmented model gives 2.4447
  (random SO(3)) / 2.4069 (60 deg tilt), not 3.460 (`RIDI_EqNIO_ATE_summary.md:22`).
* Action: label the row "EqNIO + SO(3) aug." (or report the unaugmented 2.160) and re-derive or
  remove 3.460.

## 8. Table 1 / Table 5 and Sec. 4.2, RIDI GINIO-W: filter cells do not reproduce; the NN-only 0.927 is a Sim(3)-aligned ATE

* Paper: RIDI GINIO-W 0.952 / 2.450 (`cr.txt:347`); Sec. 4.2 "GINIO achieves 0.927 m ATE"
  (`cr.txt:371`).
* Evidence: with the only RIDI GINIO-W checkpoint (`ridi_ginio_w`, sha `424a9a9319b09c3f`) the
  EqNIO-repo SCEKF gives 2.8357 (`RECOMPUTED:54`) and the Aug-2026 re-filter 1.417 Umeyama
  (`results/server_B_rebuttal/ridi_refilter/ridi_3arm_ate.json`, 22 common sequences); neither
  matches 0.952 or 2.450. The 0.927 is the mean **Sim(3)**-aligned ATE over the 23 test sequences
  (`results/nn_outputs/...batch_test_outputs_ridi__world_ln2regress_slope1/aligned_ate.json`, key
  `sim3`, mean 0.9274); the Umeyama SE(3) value is 1.2524 and the unaligned mean-L2 2.6276. The
  paper does not say that this number is scale-aligned. The checkpoint itself is the slope-0.1,
  no-output-bias Lie-Neuron class (`src/network/ln_conv_2regress_nochannelmix_slope_1.py`), not the
  `ln_resnet` registry class used for TLIO (strict load as `ln_resnet`: 6 missing keys; registry key
  `ln_resnet_slope1_nobias`).
* Action: state the alignment used for 0.927; re-run or mark the 0.952 / 2.450 cells.

## 9. Sec. 4.2, "1.43 m for IMUNet": no source

* Paper: `cr.txt:371-372`.
* Evidence: no output, aggregate or ledger on either machine contains 1.43. IMUNet was never run
  from its public code; Server B's from-scratch reimplementation (`ridi_imunet_reimpl`, arch
  `imunet`) gives Sim(3) ATE 0.9380 on the same 23 sequences
  (`checkpoints/ridi/imunet_reimpl/aligned_ate.json`). The handoff notes disagree about what the
  IMUNet paper itself reports (`results/server_B_rebuttal/NOMINAL_BASELINE_DISCUSSION.md`, "VERIFY
  BEFORE USE").
* Action: cite the IMUNet paper's own number with its split, or replace it with the measured
  reimplementation value and say so.

## 10. Table 9 / Fig. 4 / Sec. 5.3: TLIO and RIDI cells are hard-coded, EqNIO cells are projected, NanoBench cells differ from the measurements

* Paper: Table 9 (`cr.txt:1416-1436`), Fig. 4, and Sec. 5.3: "GINIO-R has the smallest 3 -> 20 deg
  ATE increase (+0.03 m), compared with EqNIO O(2) at +0.25 m and its augmented variant at +0.10 m"
  (`cr.txt:568-569`).
* Evidence: the figure generator `scripts/workspace_root/make_fig_paper_B_FINAL.py` (docstring lines
  9-13) and `results/HANDOFF_paper_handoff_v2/05_figure_B_handoff/INSIGHTS.md` (lines 62-73, 178-179)
  state that the TLIO and RIDI values for GINIO-R, ResNet and ResNet+SO(3) aug. are literals from a
  spreadsheet ("user Excel", "User-curated Excel (hardcoded literals)") and that all EqNIO O(2) and
  EqNIO O(2)+aug. cells for TLIO and RIDI are "projected" (measured network-only EqNIO on body-frame
  TLIO saturates at 23-30 m: `results/HANDOFF_paper_handoff_v2/04_raw_metrics/tlio/results_network.csv`).
  The NanoBench cells *were* measured, but the printed values are not the measured ones (`MASTER`
  lines 2-6, 18-22, 34-38, 50-54, 66-70): GINIO-R 0.5814 -> 0.5899 (+0.009; printed 0.56 -> 0.59),
  EqNIO O(2) 0.9295 -> 0.9047 (-0.024; printed 0.93 -> 1.18), EqNIO O(2)+aug. 0.9606 -> 0.9465
  (-0.015; printed 0.96 -> 1.06), ResNet 0.6459 -> 0.6679 (printed 0.66 -> 0.68), ResNet+aug.
  1.2056 -> 1.2012 (printed 1.20 -> 1.26). The main-text increments +0.03 / +0.25 / +0.10 are
  therefore not supported.
* Action: replace the NanoBench rows and the Sec. 5.3 sentence with the measured values; mark the
  TLIO/RIDI rows as taken from the earlier publication or re-measure them; remove or clearly label
  the projected EqNIO cells.

## 11. Fig. 5(b) and Sec. 5.3: the two quoted spectral numbers come from two different curves; the diagonal arm is not the best checkpoint; selection used test sequences

* Paper: "The coordinate-diagonal head is marginally better nominally (1.072 vs. 1.081 m), whereas
  the equivariant spectral-derived covariance is more robust at 20 deg (1.349 vs. 1.826 m)"
  (`cr.txt:575-576`, repeated `cr.txt:1340-1343`).
* Evidence: `results/server_B_rebuttal/full36/p4_fullcov_WIDE.json` has three curves: `spec_full`
  (full spectral covariance), `spec_diag` (spectral-derived diagonal marginal) and `diag_model`.
  1.081 is `spec_full` at 0 deg (1.0810) while 1.349 is `spec_diag` at 20 deg (1.3494); the same-curve
  pairs are `spec_full` 1.0810 -> 2.364 and `spec_diag` 1.0753 -> 1.3494 (`diag_model` 1.0724 ->
  1.8256). The diagonal arm plotted is epoch 20, not `checkpoint_best.pt` (epoch 9)
  (`results/server_B_rebuttal/full36/best_epochs.txt`, `p4a_epoch_select.json`;
  `checkpoints/CHECKPOINT_MANIFEST.csv:20`), and both arms' epochs were chosen on the 5 *test*
  sequences of `splits/tlio_golden/test_list1.txt`.
* Action: quote both numbers from one curve (spectral-derived diagonal: 1.075 vs 1.349) and state
  the epoch-selection protocol.

## 12. Fig. 9(c) plots post-hoc calibrated metrics while Table 8 and Sec. H.5 describe raw ones

* Paper: Fig. 9(c) "Distance to ideal": NEES gap 1.65 (Diag) vs 0.81 (Spectral), 95 % coverage gap
  0.13 vs 0.06 (`cr.txt:1225-1253`); Sec. H.5: "We report the raw NEES coverage of each covariance
  head, with no post-hoc scaling" (`cr.txt:1324-1333`); Table 8 NEES 2.959 / 2.974, cov95 0.915 /
  0.837 (`cr.txt:1319-1321`).
* Evidence: Server A `cov_ablation/outputs/TABLE_A_nll_decomp_calib.json` holds, for the two NLL
  models, `nees_cal_test` = 4.6465 (diag, line 38) and 3.8119 (spectral, line 88) and
  `coverage_cal_test["0.95"]` = 0.8169 (line 49) and 0.8895 (line 99) after fitting a scale
  `alpha_star` on the validation split. |4.6465 - 3| = 1.65, |3.8119 - 3| = 0.81,
  |0.8169 - 0.95| = 0.13, |0.8895 - 0.95| = 0.06: Fig. 9(c) shows the calibrated metrics. The raw
  metrics printed in Table 8 give gaps of 0.04 / (6.774 - 3 = 3.77 measured) and 0.035 / 0.113, with
  the diagonal head closer on coverage. Server B's note "(c) NEES/cov95 gaps inconsistent with Tab8"
  (`PAPER_ARTIFACT_MAP.csv`, Fig9 row) describes the same mismatch.
* Action: state in the Fig. 9 caption that panel (c) uses validation-calibrated covariances, or
  replot it with the raw metrics.

## 13. Table 8 is not a covariance-head-only ablation

* Paper: Table 8 caption "NanoBench covariance ablation" (`cr.txt:1315`); Sec. 5.2 attributes the
  velocity-RMSE and mu-equivariance differences to the head (`cr.txt:530-535`).
* Evidence: the diagonal rows (`ln_resnet`, sha `9f2c07d52c4c691c` / `845c4f63e82a96e9`) were trained
  on the LFA data set (`checkpoints/nanobench/cov_diag_mse_lfa/parameters.json:37`,
  `nanobench_bodyframe_2nd_version`), batch 1024, 81 epochs, with output-block biases; the spectral
  rows (`ln_resnet_chol`, `0cbb3275e118a9f7` / `8a9f5c51b9b34ba9`) on the no-LFA set
  (`ginio_rs_chol_mse/parameters.json:37`), batch 16, 201 epochs, without those biases and with 40 %
  more parameters (802,248 vs 572,581). Server A `DO_NOT_USE_OR_CONDITIONAL.md` D1: the mu-eq. gap
  (0.317 vs 4.30e-4) is the bias term, not the head. The Sigma-eq. column is unaffected. A diagonal
  NLL model trained with the spectral rows' recipe exists (`nanobench_ginio_rd_nolfa_nll_matched`,
  velocity RMSE 0.230, i.e. lower than both NLL rows).
* Action: say in the caption that the rows differ in training data, batch size, epochs and output
  biases, or re-run a matched pair.

## 14. Table 2, EqNIO O(2): scored against integrated targets, unlike the other rows

* Paper: EqNIO O(2) ATE 0.828 (`cr.txt:407`), caption "ATE follows the benchmark alignment protocol
  over 17 test sequences" (`cr.txt:395`).
* Evidence: `results/nano_unified_metrics_v2/VALIDATION_NOTES.md:160`: the 0.828 scores the
  frame-transformed `net_outputs.txt` against integrated velocity targets; scored against
  ground-truth positions the same outputs give 0.970. The other Table 2 rows are scored against
  ground-truth positions.
* Action: re-score EqNIO O(2) against ground-truth positions (0.970) or state the difference.

## 15. Table 7: duplicated rows, and two values with no source

* Paper: Table 7 (`cr.txt:1185-1192`) lists "GINIO InEKF (gravity comp.) 2.883 / 3.424" twice and
  "GINIO InEKF (no gravity comp.) 2.704 / 2.759" twice.
* Evidence: 4.191 and 2.883 survive only as comments (`results/ledger_comments/covTLIO_json_analysis.py:266`
  `#4.1910023`; `:276` `#2.883275086`; `:167` `# 2.883687` for a different path). 6.455 and 3.424
  appear in no file. The checkpoints and output directories are deleted.
* Action: remove the duplicate rows; mark 6.455 and 3.424 as unverifiable or re-run.

## 16. Table 6: the "GINIO-Sensor" (no LFA) number comes from an LFA model; three values have no source

* Paper: Table 6 (`cr.txt:1157-1163`): ResNet 1.817 / 108.235, ResNet+SO(3) aug. 1.966 / 3.189,
  GINIO-Sensor 2.293 / 2.293, GINIO-World and GINIO-Sensor+LFA 1.828 / 1.828.
* Evidence: 1.817 survives as `#1.817235` (`covTLIO_json_analysis.py:45`); 108.235, 1.966 and 3.189
  appear nowhere. The 2.293 path in the ledger is `ln_conv_last_align_notcomp_...` (`last_align` =
  LFA), while the row is labelled sensor frame without LFA (Server B `START_HERE.md:64-65`). 1.828 is
  recorded only in an unbundled `newTLIO/batch_test_outputs/...189/metrics.json` (1.8276); the
  bundled epoch-86 outputs give 2.0511 (ID) / 2.0422 (SO(3)) (`results/server_B_rebuttal/TABLE7_NETWORK_RECONSTRUCTION.csv`).
* Action: verify the GINIO-Sensor label; mark 108.235 / 1.966 / 3.189 as unverifiable or re-run.

## 17. Table 1 / Table 5, TLIO ResNet rows: original model lost; augmented row near but not exact; identical RTE

* Paper: ResNet+SCEKF 1.725 / 35.699, +SO(3) aug. 1.844 / 8.687 (`cr.txt:344-345`); Table 5 prints
  RTE 0.565 for both the ResNet and the ResNet+aug. ID/SO(3) cells (`cr.txt:1141-1142`).
* Evidence: 1.7251 survives only as a ledger comment (`imu-equivariant-learning_json_analysis.py:23`);
  the 2026-03 retrain gives 1.7459 / 1.7517 (`RECOMPUTED:32`). 35.699 has no source. The augmented
  model gives 1.8420 / 1.8499 and 8.4249 (`RECOMPUTED:11`, `:38`). Its `parameters.json`
  (`checkpoints/tlio/resnet_so3aug/parameters.json`) records no augmentation flag.
* Action: mark 1.725 / 35.699 as from the lost run; check the duplicated RTE 0.565.

## 18. Table 1 / Table 5 mix two filter-metric codes and RTE windows

* Paper: Sec. 4.1 defines RTE as the yaw-aligned 1 s relative translation RMSE (`cr.txt:329-331`).
* Evidence: the EqNIO rows reproduce only with the EqNIO-repo code (`EKF_output_metrics.py`: no
  origin alignment, 2000 samples skipped, RTE without yaw removal): 1.5545 vs 1.5651 TLIO-metric
  (`RECOMPUTED:33`); the GINIO-W row reproduces with the TLIO-repo code (`plot_filter_state.py`,
  origin-aligned RMSE, 1 s yaw-aligned RTE): 1.6146 (`RECOMPUTED:22`). Network RPE windows differ
  between pipelines (2.0 s in `src/network/test.py`, about 20 s in covTLIO/EqNIO) and filter RPE keys
  (1 s vs 10 s) (`results/server_B_rebuttal/OPEN_ISSUES.md`, item 2; `RTE_PROTOCOL.md`).
* Action: state per row which metric code produced it, or re-score every row with one code
  (`RECOMPUTED` has both for every surviving output).

## 19. Table 1 / Table 5, RIDI baselines: no model behind three rows

* Paper: RIDI ResNet+SCEKF 1.207 / 2.990, ResNet-S+InEKF 1.445 / 4.752, GINIO-S 1.319 / 1.457
  (`cr.txt:344, 349-350`).
* Evidence: the only RIDI ResNet checkpoint is epoch 1 (`checkpoints/CHECKPOINT_MANIFEST.csv:34`)
  and the filter diverges with it (38.28 m, `RECOMPUTED:44`); no RIDI sensor-frame model (ResNet-S or
  GINIO-S) exists on either machine (`MISSING.md`); RIDI ResNet+aug. (`ridi_resnet_so3aug`) has no
  filter output.
* Action: mark these cells as unverifiable from surviving artifacts.

## 20. Parameter counts: two conventions and one wrong value

* Paper: Table 1 "5.42M / 6.02M / 0.57M" (`cr.txt:344-347`); Table 2 "5.43M / 6.03M / 6.03M / 0.57M /
  0.80M" (`cr.txt:406-410`); Table 4 "0.273M" (`cr.txt:1018-1020`).
* Evidence (release `INDEX.json`): ResNet1D 5,424,646 excluding / 5,434,780 including BatchNorm
  statistics; EqNIO 6,020,998 / 6,031,132; GINIO-E 6,022,150 / 6,032,284; GINIO-RS 802,248 /
  804,551; `ln_resnet` (1.0 s window) 572,581 / 574,714. Table 1 uses the excluding convention,
  Table 2 the including one. GINIO-RD (trial 119, 0.775 s window) has **544,021 / 546,154**
  parameters, not 0.57M. The Table 4 "original" models have 271,939 / 272,163 parameters (Server A
  audit `CHECKPOINT_MANIFEST.csv`).
* Action: use one convention and correct GINIO-RD to 0.54M (or 0.55M).

## 21. Table 4, GINIO-A original (0.273M): checkpoints cannot be loaded strictly

* Paper: 1.502 / 1.062 (`cr.txt:1018, 1020`).
* Evidence: the values reproduce from Server A aggregates (`canonical_v2.json` keys `VN_AirIO_no_att`
  1.5019, `VN_AirIO_with_att` 1.0617) but the checkpoints (`Air-IO/experiments/nanobench_noatt_vn`,
  `nanobench_rot_vn`) fail strict loading against the current `airio/model/vn_codenet.py` (15 missing
  / 3 unexpected keys; `DO_NOT_USE_OR_CONDITIONAL.md` D11, line 69). They are not released.
* Action: note that the Table 4 "original" rows are not reproducible from released code.

## 22. Model selection on test sequences

* Evidence: `splits/ridi/val_list.txt` is byte-identical to `test_list.txt` (23 sequences), so every
  RIDI `checkpoint_best.pt` was selected on the test split. `airio/train_motion.py` selects
  `best_model.ckpt` on the config's `test` block, which is the 17-sequence reported NanoBench split
  (upstream AirIO behaviour; applies to AirIO and GINIO-A alike; already noted in the README). The
  Fig. 5(b) arms were selected on 5 test sequences (`splits/tlio_golden/test_list1.txt`,
  `full36/p4a_epoch_select.json`).
* Action: disclose in Sec. 4.1 / App. G.

## 23. TLIO split sizes

* Paper/README: 284 / 36 / 36 sequences. Evidence: `splits/tlio_golden/train_list.txt` and
  `val_list.txt` contain 283 and 35 sequence ids plus one blank line each; the test list has 36.
  Minor.

## 24. App. C.3, GINIO-E "within 1e-15"

* Evidence: Server A `GATE_C__spectral_covariance.md` C.2 measures the GINIO-E covariance-law error
  at 1.32e-14 (float64). Minor (order of magnitude).

## 25. Fetch (Tables 10-12, Figs. 13-14) and AquaticVision (Table 13): nothing reproducible

* Evidence: the only Fetch artifact is `fetch_dataset.zip` in the CoRL supplement (69 sequences:
  23 nominal + 23 x90 + 23 y90, whereas Sec. K.1 / Table 10 describe 23 training + 4 held-out nominal
  + 46 remount = 73; `cr.txt:1484, 1498-1499`); no loader, config, checkpoint, result or figure
  source on either machine. AquaticVision: no data, code or results anywhere.
* Action: recover from the authors' records or mark as not reproducible from the release.

## 26. Fig. 8: no generator; the "GINIO-RS" curve is the Spectral/NLL model

* Evidence: no script produces Fig. 8 (`cr.txt:1219`); the curve labelled GINIO-RS corresponds to
  the Spectral/NLL checkpoint (velocity RMSE 0.260), not to the Table 2 GINIO-RS (Spectral/MSE,
  0.234) (Server B `PAPER_ARTIFACT_MAP.csv`, Fig8 row). The angle-resolved measurements that exist
  (Server A `TABLE_B_cross_model.csv`) include two VN-AirIO rows produced by non-strict loading of the
  0.273M checkpoints (velocity RMSE 4.5 / 7.8 m/s, coverage 0) that must not be used (D11).

## 27. Table 1 ID/SO(3) for TLIO depends on one particular data copy

* Evidence: 76.389 and the GINIO-W 2.205 reproduce only against the `imu-previous-work/so3_60`
  rotated copy of the test set (rotated ground truth, no filter-init compensation); against
  `tlio_golden` ground truth the same runs give 77.43 / 18.55 (`RECOMPUTED:34-36` vs the `tlio_golden`
  rows), and four `so3_60` copies with different checksums existed on Server B (`START_HERE.md:71`).
  The rotation distribution of `so3_60` is not described in the paper.
* Action: describe the ID/SO(3) data generation (per-sequence rotation distribution) in Sec. 4.1.

## 28. Code-release gaps that affect reproducibility of released checkpoints

* Four registry keys had to be added to `src/network/model_factory.py` for this release so that
  every released checkpoint strict-loads: `ln_resnet_slope1_nobias` (`ridi_ginio_w`; bias-free,
  slope-0.1 class `ln_conv_2regress_nochannelmix_slope_1.py`), `ln189_speccov_s0` /
  `ln189_speccov_s0_diag` (Fig. 5(b) arms; class files `ln_conv_2regress_nochannelmix_bias_speccov*.py`)
  and `imunet` (`ridi_imunet_reimpl`; `model_imunet.py`). Server B's `START_HERE.md` trap 2 ("all
  `ln_resnet` checkpoints load into `..._bias.py`, slope 0.0") is wrong for the RIDI GINIO-W
  checkpoint (6 missing bias keys as `ln_resnet`). Bias-carrying `ln_resnet` checkpoints also load
  without error into the slope-0.1 `..._slope_1_bias.py` class (identical parameter names) and give
  wrong outputs there.
* `tlio_eqnio_o2/model_torchscript.pt` is missing from the handoff (listed MISSING in Server B's
  manifest); re-export with `eqnio/TLIO-master/src/convert_model_to_torchscript_eq.py`.
* The Fig. 5 attitude-error hook (`ATT_ERR_DEG` / `ATT_ERR_AXIS`, `src/tracker/imu_tracker.py`) and the
  full 3x3 covariance passthrough (`src/tracker/meas_source_torchscript.py`) are in this repository; the
  Fig. 5 runs themselves were made with Server B's TLIO working tree (`code/TLIO.tracked_changes.patch`).
* The gravity-removed processed NanoBench set (`splits/README.md`) makes any NanoBench filter run
  invalid until the data are rebuilt.

## 29. Covariance-equivariance numbers depend on the measurement protocol

* Evidence: Server A `TABLE_A` (printed in Table 8) measures Spectral/NLL Sigma-eq. = 4.34e-4 and
  Spectral/MSE mu-eq. = 7.24e-4; Server B's 50-rotation x 32-window protocol
  (`results/server_B_rebuttal/D5_cov_equivariance.json`) gives 8e-8 ... 1e-6 for the same spectral
  checkpoints and 0.52-0.61 for the diagonal ones (paper 5.52e-1). The qualitative conclusion is the
  same; the printed spectral values are protocol-specific.

## 30. Minor: Table 2 GINIO-A parameter ratio and abstract FLOPs

* "7.4x / 8.7x fewer parameters" (`cr.txt:382, 384`): 330,598 / 44,486 = 7.43 and 387,014 / 44,563 =
  8.68 - consistent. "11.6x fewer FLOPs" (`cr.txt:30`): 206.78 / 17.81 = 11.61 - consistent with
  Table 1; FLOPs themselves were not re-measured.
