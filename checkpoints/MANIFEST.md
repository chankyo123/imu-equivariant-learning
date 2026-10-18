# GINIO model zoo (released checkpoints)

This is the authoritative list of every checkpoint published with the GINIO code release
(CoRL 2026, OpenReview `nl8iBTm3Mt`). Download the per-benchmark archives from the GitHub Release
page (see `docs/RELEASE_LAYOUT.md`) and extract them here, so that each folder sits at
`checkpoints/<folder>/`.

**Naming.** Folders follow one flat scheme, `<benchmark>_<row>_<variant>`, where `benchmark` is the
paper table the model backs (`tlio`, `ridi`, `nanobench`), `row` is the paper's row label and
`variant` disambiguates. The benchmark prefix is the *paper* benchmark, not the training set:
`nanobench_ginio_e_so3aug_tlio` is the Table 2 GINIO-E row although it was trained on TLIO.

**Table numbering.** All table and figure numbers below use the **camera-ready** numbering
(`paper/GINIO_CoRL2026_camera_ready.pdf`, built 2026-10-13). The camera-ready renumbered the
supplement relative to the submission: camera-ready **Table 8** (covariance ablation) is the
submission's Table 9, camera-ready **Table 9** (frame-estimation mismatch) is the submission's
Table 10, and camera-ready **Tables 10-12** are the Fetch tables. Older internal notes that say
"Table 9 covariance ablation" or "Table 10 frame mismatch" refer to the submission numbering.

**Identity.** Every file is identified by its SHA-256 (full digests in the release `SHA256SUMS`;
16-hex prefixes below). Directory names, epoch numbers and `parameters.json` contents are *not*
evidence of identity: several archived directory names and metadata fields contradict the weights
they sit next to (see *Traps*). A row is marked PINNED only when the checkpoint loads strictly
(`strict=True`, zero missing / unexpected keys) into the code in this repository *and* the paper
value is reproduced from stored outputs to the printed precision.

**Parameter counts.** `params` is the number of values in `model_state_dict` excluding BatchNorm
running statistics (`running_mean`, `running_var`, `num_batches_tracked`); this is the convention
behind Table 1's 5.42M / 6.02M / 0.57M. The count *including* BatchNorm statistics (in
parentheses where it matters) is the convention behind Table 2's 5.43M / 6.03M. The Lie-Neuron
state dicts also hold 27 non-trainable Lie-algebra constants, so `module.parameters()` returns 27
fewer values (e.g. 572,554 vs 572,581).

**Status codes.**

| code | meaning |
|---|---|
| PINNED | checkpoint released, loads strictly, paper value reproduced from stored outputs |
| CKPT_ONLY | checkpoint released and loads, but the paper value does not reproduce from any stored output (the recomputed value is given) |
| RETRAIN | the model that produced the paper value is lost; a retrained model of the same recipe is released and its own value is given |
| NOT_PAPER_MODEL | released for completeness or as a control; it does not back any printed number |
| UNSUPPORTED | a printed number for which no measured source exists on either machine |

Two machines held the artifacts: **Server A** (AWS box `ip-10-150-0-126`, `~/imu-work`; NanoBench
AirIO-family runs, covariance ablation, GINIO-E) and **Server B** (the authors' Alienware
workstation, `~/Desktop/code2/imu-previous-work`; everything TLIO/RIDI, the Lie-Neuron NanoBench
runs, the EqNIO and ResNet NanoBench baselines). "Reproduced" values name the machine and the file.

---

## TLIO (Table 1 world/sensor-frame rows, Tables 5-7, Fig. 5, Fig. 12, Table 9 TLIO rows)

| folder | paper row(s) | arch / registry key and loader note | params | paper value | reproduced value and source | status | files (sha256 prefix) |
|---|---|---|---:|---|---|---|---|
| `tlio_ginio_w_ln189` | Table 1 / Table 5 TLIO GINIO-W: ID/ID 1.615, ID/SO(3) 2.018; Table 6 GINIO-World and GINIO-Sensor+LFA 1.828; Fig. 5(a) GINIO curve; Fig. 12 GINIO; Table 9 TLIO GINIO-R | `ln_resnet` in `src/network/model_factory.py` (class file `src/network/ln_conv_2regress_nochannelmix_bias.py`, `VNLeakyReLU(negative_slope=0.0)`). `parameters.json` says `"arch": "resnet"` - that field is stale, load as `ln_resnet`. `in_dim = 200//32+1 = 7` (window 1.0 s at 200 Hz). **Two weight sets** - see `WEIGHTS.txt` | 572,581 (574,714) | 1.615 / 2.018 | ID/ID: 1.6146 m (epoch-189 weights, TLIO-metric origin-aligned RMSE; Server B `results/recomputed_2026-10/RECOMPUTED_FILTER_METRICS.csv`, row `imu-equivariant-learning/batch_filter_outputs_uf20/...world_disp_mle_189`). ID/SO(3): 2.2052 m with the epoch-86 weights on the `imu-previous-work/so3_60` rotated copy (same file, row `EqNIO/results/ln_189_so3_60`); the printed 2.018 does not reproduce | PINNED (ID/ID) / CKPT_ONLY (ID/SO(3)) | `checkpoint_189.pt` 6c0c4a9e531f3aeb (epoch 189); `model_torchscript.pt` 2c4f171fc7ed1afa (epoch 189); `checkpoint_best.pt` 5785386fe4d5b58a (epoch 86, different weights); `parameters.json` 58ad9010a11b47aa; `WEIGHTS.txt` |
| `tlio_eqnio_o2` | Table 1 / Table 5 TLIO EqNIO: 1.554 / 76.389; Fig. 12 EqNIO | `eq_o2_frame_fullCov_2vec_2deep` in `eqnio/TLIO-master/src/network/model_factory.py`; `in_dim = 7`; loads with `eqnio/TLIO-master/src/network/test.py`. According to Server A's inventory this is the EqNIO authors' released TLIO O(2) model (not trained by us; check the EqNIO license before redistributing) | 6,020,998 (6,031,132) | 1.554 / 76.389 | 1.5545 m (EqNIO-metric RMSE, `results/eqnio_csv/TLIO_o2_ekf_results.csv` and RECOMPUTED row `EqNIO/models/tlio_o2/TLIO_o2/ekf_test`; the TLIO-metric value is 1.5651). 76.3895 m on the `imu-previous-work/so3_60` copy, rotated ground truth, no filter-init compensation (RECOMPUTED row `EqNIO/results/TLIO_o2_so3_60_nograv`) | PINNED | `checkpoint_best.pt` 26e6e3ad879f5cfb (epoch 24); `parameters.json` 7012d6021bfb9fcb. `model_torchscript.pt` was listed in Server B's manifest as MISSING and is not in the bundle - re-export with `eqnio/TLIO-master/src/convert_model_to_torchscript_eq.py` |
| `tlio_resnet_so3aug` | Table 1 / Table 5 TLIO ResNet+SCEKF+SO(3) aug.: 1.844 / 8.687; Fig. 12 ResNet+aug; Table 9 TLIO ResNet+SO(3) aug. | `resnet` (TLIO ResNet1D; identical class in `src/network/model_factory.py` and `eqnio/TLIO-master/src/network/model_factory.py`), `in_dim = 7`. Trained inside the EqNIO repository; its `parameters.json` records no augmentation flag, so "SO(3) aug." rests on the directory name | 5,424,646 (5,434,780) | 1.844 / 8.687 | 1.8420 (EqNIO-metric) / 1.8499 (TLIO-metric), RECOMPUTED row `EqNIO/models/resnet_so3aug_eqnio_filter_std`; ID/SO(3) 8.4249 on `so3_60` (row `EqNIO/results/resnet_so3aug_so3_60_no_grav`). Near, not exact | CKPT_ONLY | `checkpoint_best.pt` a70098a6f0a5c0a2 (epoch 190); `model_torchscript.pt` 2fbb207f07f4967c; `parameters.json` 0f200d7e67adb4ca |
| `tlio_resnet_scekf_retrain2026` | Table 1 / Table 5 TLIO ResNet+SCEKF (1.725 printed); Fig. 5(a) TLIO baseline curve (4.85 at 20 deg) | `resnet`, `in_dim = 7`; standard TLIO recipe (`TLIO/models/resnet`, retrained 2026-03-16 because the original checkpoint was overwritten) | 5,424,646 | 1.725 (Table 1); 4.85 (Fig. 5a) | Table 1: this retrain gives 1.7459 (EqNIO-metric) / 1.7517 (TLIO-metric), RECOMPUTED row `TLIO/batch_filter_outputs_uf20/models-resnet`; the original 1.7251 survives only as a comment in `results/ledger_comments/*json_analysis.py`. Fig. 5(a): 4.8502 m at 20 deg reproduces exactly (`results/server_B_rebuttal/full36/p3_raw_sweep_WIDE.json`, `ResNet.20.raw_mean`) | RETRAIN (Table 1) / PINNED (Fig. 5a) | `checkpoint_best.pt` 653f34e1c3a0e82e (epoch 60); `model_torchscript.pt` ed39d955462df276; `parameters.json` 25244208155148ac |
| `tlio_resnet_s_lfa_candidate` | Table 1 / Table 5 / Table 7 TLIO ResNet-S+InEKF: 2.814 / 36.593 | `resnet`, `in_dim = 7`, sensor frame with LFA (`body_frame True`, root `local_data_bodyframe/tlio_golden`). Only candidate found; no ledger entry ties it to the printed numbers | 5,424,646 | 2.814 / 36.593 | 2.8761 (EqNIO-metric) / 2.8734 (TLIO-metric) on 35 sequences, RECOMPUTED row `imu-equivariant-learning/batch_filter_outputs_body_riekf_uf20/models_body-lfa_resnet_gyrofix`; ID/SO(3) 16.15 (tlio_golden GT) or 12.76 (`local_data_bodyframe/so3_60` GT) - not 36.593 | CKPT_ONLY | `checkpoint_best.pt` 571f12e42c6d37ca (epoch 18); `model_torchscript.pt` 0e0123a07aae8b3d; `parameters.json` caf21c66aa100678 |
| `tlio_fig5b_spec_s0v2` | Fig. 5(b) spectral covariance arm (1.081 nominal, 1.349 at 20 deg); Sec. 5.3 text | `ln189_speccov_s0` in `src/network/model_factory.py` (class `SO3EquivariantReluBracketLayers_speccov_s0`, `src/network/ln_conv_2regress_nochannelmix_bias_speccov_slope0.py`, `cov_head='spectral'`): the frozen epoch-86 ln189 mean plus a spectral head trained with `--freeze_mean`; `in_dim = 7`; strict load verified (0 missing / 0 unexpected). Do not load it as `ln_resnet_cov` (45 missing / 87 unexpected keys) | 1,382,339 (1,385,067) | 1.081 / 1.349 | `results/server_B_rebuttal/full36/p4_fullcov_WIDE.json`: `spec_full.0.mean` = 1.0810, `spec_diag.20.mean` = 1.3494. Note the two printed numbers come from two different curves of the same model: the full spectral covariance at 20 deg is 2.364, the spectral-derived diagonal marginal at 0 deg is 1.0753 | PINNED (with that caveat) | `checkpoint_best.pt` 4b662c363fcc00b2 (epoch 19); `parameters.json` a2b857448efb4f41; `WEIGHTS.txt` |
| `tlio_fig5b_diag_s0v2` | Fig. 5(b) coordinate-diagonal arm (1.072 nominal, 1.826 at 20 deg) | `ln189_speccov_s0_diag` in `src/network/model_factory.py` (same class, `cov_head='diag'`), `in_dim = 7`; strict load verified. The plotted model is **epoch 20**, not `checkpoint_best.pt` (epoch 9); epoch selection used the 5 sequences of `splits/tlio_golden/test_list1.txt` | 1,382,339 | 1.072 / 1.826 | `p4_fullcov_WIDE.json`: `diag_model.0.mean` = 1.0724, `diag_model.20.mean` = 1.8256 | PINNED | `checkpoint_20.pt` b333f2c72543b18e (epoch 20, used); `checkpoint_best.pt` 75a0047f9580ec93 (epoch 9, not used); `parameters.json` e20e71b2c45e23a2; `WEIGHTS.txt` |
| `tlio_eqnio_o2_ours` | none (reference) | `eq_o2_frame_fullCov_2vec_2deep` trained by us on TLIO golden (50 epochs, Server A `overnight/o2_paper`) | 6,020,998 | - | zero-shot NanoBench ATE 0.8044 (Server A `cov_ablation/outputs/eqnio_trajectory_ate.json`, key `EqNIO_O2`) - this is *not* the Table 2 EqNIO O(2) row, which used the NanoBench-trained `nanobench_eqnio_o2` | NOT_PAPER_MODEL | `checkpoint_best.pt` 9de553e92ba36664 (epoch 47); `model_torchscript.pt` 03885ba331b46a69; `parameters.json` |

## RIDI (Table 1 RIDI columns, Table 5, main-text NN-only comparison)

| folder | paper row(s) | arch / registry key and loader note | params | paper value | reproduced value and source | status | files (sha256 prefix) |
|---|---|---|---:|---|---|---|---|
| `ridi_ginio_w` | Table 1 / Table 5 RIDI GINIO-W: 0.952 / 2.450; main text Sec. 4.2 NN-only "GINIO 0.927 m" | `ln_resnet_slope1_nobias` in `src/network/model_factory.py` (class file `src/network/ln_conv_2regress_nochannelmix_slope_1.py`: `VNLeakyReLU(negative_slope=0.1)`, bias-free output blocks), `in_dim = 7`; strict load verified. It does **not** load as `ln_resnet` (6 missing bias tensors). Being bias-free, its mean is equivariant to float32 precision (measured 9e-7). The TorchScript file needs no class | 571,899 (574,032) | 0.927 (NN-only); 0.952 / 2.450 (filter) | NN-only: 0.9274 m = mean Sim(3)-aligned ATE over the 23 test sequences (`results/nn_outputs/...batch_test_outputs_ridi__world_ln2regress_slope1/aligned_ate.json`, key `sim3`; the Umeyama SE(3) value is 1.2524, the unaligned mean-L2 is 2.6276). Filter: 2.8357 (RECOMPUTED row `EqNIO/models/ln_ridi_wf_filter_noaug`, EqNIO SCEKF, 22 seqs) and 1.417 Umeyama in the Aug-2026 re-filter (`results/server_B_rebuttal/ridi_refilter/ridi_3arm_ate.json`, 22 common seqs); neither matches 0.952 / 2.450 | PINNED (0.927) / CKPT_ONLY (Table 1 cells) | `checkpoint_best.pt` 424a9a9319b09c3f (epoch 233); `model_torchscript.pt` d8f5e3adbb09c5b2; `parameters.json` a4817316551ca9b1 |
| `ridi_eqnio_o2_so3aug` | Table 1 / Table 5 RIDI "EqNIO": 2.408 / 3.460 | `eq_o2_frame_fullCov_2vec_2deep` (eqnio), `in_dim = 7`, trained on RIDI with **full-SO(3) training augmentation** (`so3_augment: true, 180 deg`, 80 epochs). The RIDI "EqNIO" row is therefore the augmented model while the TLIO "EqNIO" row is unaugmented | 6,020,998 | 2.408 / 3.460 | 2.4082 m (EqNIO-metric RMSE, RECOMPUTED row `EqNIO/models/eqnio_o2_ridi_wf_filter_so3aug`, 22 seqs; `results/RIDI_EqNIO_ATE_summary.md`). ID/SO(3): 2.4447 (random SO(3), equivariance protocol) / 2.4069 (60 deg pure tilt) - the printed 3.460 does not reproduce | PINNED (ID/ID) / CKPT_ONLY (ID/SO(3)) | `checkpoint_best.pt` 280088754d97b1e3 (epoch 70); `model_torchscript.pt` 65bb4ee014d7dc9a; `parameters.json` bd37aa769c9ba0b5 |
| `ridi_eqnio_o2_noaug` | none (reference) | as above, no augmentation, 100 epochs | 6,020,998 | - | 2.1602 m (RECOMPUTED row `EqNIO/models/eqnio_o2_ridi_wf_filter_noaug`) | NOT_PAPER_MODEL | `checkpoint_best.pt` ae31dc19548cc699 (epoch 52); `model_torchscript.pt` 42e042c7c2c4d6a7; `parameters.json` f531c3e5d65bb560 |
| `ridi_resnet_so3aug` | Table 1 / Table 5 RIDI ResNet+SCEKF+SO(3) aug.: 1.401 / 3.011 | `resnet`, `in_dim = 7`, `so3_train_aug: true` (180 deg), 80 epochs | 5,424,646 | 1.401 / 3.011 | no filter output exists for this checkpoint on either machine | CKPT_ONLY | `checkpoint_best.pt` f778c926eecd693f (epoch 13); `parameters.json` 555a515455150b34 |
| `ridi_resnet_epoch1` | Table 1 RIDI ResNet+SCEKF (RoNIN+SCEKF): 1.207 / 2.990 | `resnet`, `in_dim = 7`. **Epoch-1 checkpoint**; the filter diverges with it (34-38 m, RECOMPUTED row `imu-equivariant-learning/batch_filter_outputs_ridi_uf200/resnet`) | 5,424,646 | 1.207 / 2.990 | the model behind the printed numbers was not found on either machine | NOT_PAPER_MODEL | `checkpoint_best.pt` 7f8af8d337f094e9 (epoch 1); `model_torchscript.pt` 6c1d4b9e6970d896; `parameters.json` ae52f4857ca21d04 |
| `ridi_imunet_reimpl` | main text Sec. 4.2 "1.43 m for IMUNet" | `imunet` in `src/network/model_factory.py` (class `IMUNet`, `src/network/model_imunet.py`): our from-scratch IMUNet reimplementation, `in_dim = 7`; strict load verified | 2,475,590 (2,492,010) | 1.43 | this reimplementation gives Sim(3) ATE 0.9380 m on the same 23 sequences (`aligned_ate.json` in the folder); no source for 1.43 was found anywhere | NOT_PAPER_MODEL | `checkpoint_best.pt` 9cb458cd76c49cd5 (epoch 35); `parameters.json` 85199e80b37c1ff3; `aligned_ate.json` 9f8f0df206129c06; `metrics.json` 8c887e5e998c7559 |

## NanoBench (Table 2, Table 4, Table 8, Table 9 NanoBench rows, Figs. 8-11)

| folder | paper row(s) | arch / registry key and loader note | params | paper value | reproduced value and source | status | files (sha256 prefix) |
|---|---|---|---:|---|---|---|---|
| `nanobench_airio_noatt` | Table 2 AirIO (no-att.): ATE 5.579, ID/SO(3) 5.679 | `codenetmotion` (`airio/model/code.py`), config `airio/configs/Nanobench/motion_body_noatt.conf`; `airio/evaluation/eval_idso3.py` loads `ckpt/best_model.ckpt` with `strict=True` | 330,598 (330,792) | 5.579 / 5.679 | 5.5785 m, 17 sequences (Server A `nanobench_final_table/canonical_v2.json`, key `AirIO_no_att`; Server A audit `CHECKPOINT_SELECTION.md`, SUBMITTED_EXACT). The ID/SO(3) value 5.679 has no measured source (see Traps) | PINNED (ATE) / UNSUPPORTED (ID/SO(3)) | `best_model.ckpt` d149f819dbdac4c3 (epoch 73); `parameters.yaml` |
| `nanobench_airio_withatt` | Table 2 AirIO (with-att.): 1.205 / 1.302 | `codewithrot`, config `motion_body_rot.conf` | 387,014 (387,982) | 1.205 / 1.302 | 1.2049 m (canonical_v2 key `AirIO_with_att`). ID/SO(3) 1.302 has no measured source; Server A measured 1.7314 (Supplement G.1 reading), 5.3877 (uniform SO(3), physical remount) and 10.717 (project script) for this cell | PINNED (ATE) / UNSUPPORTED (ID/SO(3)) | `best_model.ckpt` 66bcda3c09d09109 (epoch 67); `parameters.yaml` |
| `nanobench_airio_noatt_serverb` | none (Server B's own AirIO no-att. run) | `codenetmotion`, same config | 330,598 | - | 3.988 m (Server B) - a different training run from the Table 2 one | NOT_PAPER_MODEL | `best_model.ckpt` bad5dcbf1b81d5d2 (epoch 76) |
| `nanobench_airio_withatt_serverb` | none (Server B's own AirIO with-att. run) | `codewithrot`, same config | 387,014 | - | 1.1289 m (canonical_v2 key `AirIO`, label "AirIO (with-att, native)", source chankyo) | NOT_PAPER_MODEL | `best_model.ckpt` 8ce1e385662f647e (epoch 31) |
| `nanobench_ginio_a_noatt` | Table 2 GINIO-A (no-att.): 1.430 / 1.430; Table 4 GINIO-A reduced, body: 1.430 | `vn_codenetmotion_small` (`airio/model/vn_codenet.py`), config `motion_body_noatt_vn_small.conf` | 44,486 | 1.430 / 1.430 | 1.4302 m (Server A audit `CHECKPOINT_SELECTION.md`, SUBMITTED_EXACT). Mean-equivariance error 4.4e-12 under both the physical-remount and the Supplement G.1 protocol (Server A `GATE_A__attitude_log_map.md`, A.4.3), so the ID/SO(3) equality is a property of the model | PINNED | `best_model.ckpt` 7432b5751164d7bf (epoch 16); `parameters.yaml` |
| `nanobench_ginio_a_withatt` | Table 2 GINIO-A (with-att.): 1.045 / 1.045; Table 4 GINIO-A reduced, body+att.: 1.045 | `vn_codewithrot_small`, config `motion_body_rot_vn_small.conf`; attitude channel = `Log(R_WS)` as a vector channel | 44,563 | 1.045 / 1.045 | 1.0453 m (same source; 1.0479 after renormalising the dataset quaternions). The ID/SO(3) equality holds only under the Supplement G.1 protocol (rotate every vector input incl. the attitude channel); under a physical remount the model is not equivariant (ATE 1.0738, relative mean-equivariance error 0.90; Gate A, A.4.3) | PINNED (ATE) / UNSUPPORTED (ID/SO(3) as an equivariance claim) | `best_model.ckpt` f70ae327eccd0f33 (epoch 78); `parameters.yaml` |
| `nanobench_resnet1d` | Table 2 ResNet1D: 0.645 (ID/SO(3) 0.710); Fig. 11 right; Table 9 NanoBench ResNet | `resnet` (`src/`), `--body_frame True`, root `nanobench_bodyframe_nolfa_2nd_version`, `--window_time 1.0 --imu_freq 200` (200 samples; the processed NanoBench files are 100 Hz, so this spans about 2 s of flight - that is what the archived `w2sec` directory name means), `in_dim = 7`, MSE, 81 epochs | 5,424,646 (5,434,780 = "5.43M") | 0.645 / 0.710 | 0.645470 m Umeyama ATE, 17 seqs (Server B `results/nano_unified_metrics_v2/all_metrics.json`, key `nano\|ResNet_nano_NN\|NN`; re-verified 2026-08 in `results/server_B_rebuttal/table2_verify/A1_resnet_body_nolfa_w2sec_e80_mse.log`; Server A canonical_v2 key `ResNet` 0.6454706). ID/SO(3) 0.710 has no generator | PINNED (ATE) / UNSUPPORTED (ID/SO(3)) | `checkpoint_best.pt` 7fc638836d2964e0 (epoch 71); `parameters.json` e903ae00e7d7ad42 |
| `nanobench_resnet1d_so3aug` | Table 9 NanoBench ResNet+SO(3) aug. (1.20 ... 1.27) | `resnet`, `so3_train_aug: true` (180 deg), otherwise as above | 5,424,646 | 1.20 / 1.21 / 1.22 / 1.24 / 1.26 (gravity) | measured gravity sweep 1.2056, 1.2048, 1.2041, 1.2029, 1.2012 (3 ... 20 deg) and nominal 1.2068 (`results/HANDOFF_paper_handoff_v2/MASTER_RESULTS.csv`, model `ResNet_so3aug`); the printed increasing values are not the measured ones | CKPT_ONLY | `checkpoint_best.pt` a42f0302af7174cc (epoch 71); `parameters.json` 26ba7c3daa660828 |
| `nanobench_eqnio_o2` | Table 2 EqNIO O(2): 0.828 (ID/SO(3) 1.267); Fig. 11 right; Table 9 NanoBench EqNIO O(2) | `eq_o2_frame_fullCov_2vec_2deep` (eqnio), trained on `nanobench_processed_2nd_version` (gravity-aligned, MSE-only, 81 epochs), `in_dim = 7`. A NanoBench-trained model, not the EqNIO authors' TLIO model | 6,020,998 (6,031,132 = "6.03M") | 0.828 / 1.267 | 0.828205 m (all_metrics key `nano\|EqNIO_O2_nano_NN\|NN`; canonical_v2 key `EqNIO_O2`). Scored against integrated velocity targets after the O(2) frame transform of `net_outputs.txt`; scored against ground-truth positions the same outputs give 0.970 (`results/nano_unified_metrics_v2/VALIDATION_NOTES.md`) | PINNED (ATE) / UNSUPPORTED (ID/SO(3)) | `checkpoint_best.pt` 9a9d592f73f6ed46 (epoch 74); `parameters.json` f08b9404eef401a8 |
| `nanobench_eqnio_o2_so3aug` | Table 9 NanoBench EqNIO O(2)+SO(3) aug. (0.96 ... 1.06) | as above with `so3_augment: true` (180 deg), 80 epochs | 6,020,998 | 0.96 / 0.97 / 0.99 / 1.02 / 1.06 | measured gravity sweep 0.9606, 0.9563, 0.9572, 0.9438, 0.9465 (MASTER_RESULTS.csv, model `EqNIO_O2_so3aug`) | CKPT_ONLY | `checkpoint_best.pt` 3b5d43fd69da1fe7 (epoch 75); `parameters.json` 6bec8a617bfe32e9 |
| `nanobench_ginio_e_so3aug_tlio` | Table 2 GINIO-E: 0.785 / 0.785 | `eq_so3_frame_fullCov_2vec_2deep` (eqnio), **trained on TLIO golden** (40 epochs, `so3_augment: true`, 180 deg) and evaluated zero-shot on the NanoBench test split; `in_dim = 7`. No NanoBench-trained GINIO-E exists | 6,022,150 (6,032,284 = "6.03M") | 0.785 / 0.785 | 0.7854 m, 17 seqs (Server A `cov_ablation/outputs/eqnio_trajectory_ate.json`, key `VN_EqNIO_SO3`). Covariance-law error 1.32e-14 (Server A `GATE_C__spectral_covariance.md`, C.2) | PINNED (the TLIO-transfer protocol is not stated in the paper) | `checkpoint_best.pt` 46ed9465a95d53ed (epoch 31); `model_torchscript.pt` c749740fa8fe6799; `parameters.json` |
| `nanobench_ginio_rd_t119` | Table 2 GINIO-RD: 0.581 / 0.581; Fig. 11 right; Table 9 NanoBench GINIO-R (0.56 ... 0.59) | `ln_resnet` (`src/`), `--body_frame True`, root `nanobench_bodyframe_nolfa_2nd_version` (no LFA), `--window_time 0.775 --sample_freq 50 --imu_freq 200` (155 samples, `in_dim = 155//32+1 = 5`), NLL, Optuna trial 119, 170 epochs | 544,021 (546,154) - the paper's "0.57M" is the window-1.0 s configuration | 0.581 / 0.581 | 0.580530 m (all_metrics key `nano\|ln_resnet_t119_nano_NN...`; `table2_verify/A4.log`; canonical_v2 key `ln_resnet_nolfa_t119_ablation_only` 0.5805353). Table 9 gravity sweep measured 0.5814, 0.5821, 0.5841, 0.5867, 0.5899 (MASTER_RESULTS.csv, `GINIO_t119`) vs printed 0.56, 0.58, 0.58, 0.59, 0.59 | PINNED | `checkpoint_best.pt` 684665d0a6cf0b60 (epoch 135); `parameters.json` 27c84f4f82d557e6 |
| `nanobench_ginio_rd_t55` | none (closest archived sibling of GINIO-RD from the same Optuna sweep) | `ln_resnet`, no LFA, `--window_time 0.675 --sample_freq 10` (`in_dim = 5`), 201 epochs - this is the configuration shown in the README quick start | 544,021 | - | 0.600432 m (all_metrics key `nano\|ln_resnet_t55_nano_NN...`) | NOT_PAPER_MODEL | `checkpoint_best.pt` fe203747b800c650 (epoch 121); `parameters.json` |
| `nanobench_ginio_rs_chol_mse` | Table 2 GINIO-RS: 0.789 / 0.789; Table 8 Spectral / MSE (vel. RMSE 0.234, mu-eq. 7.24e-4) | `ln_resnet_chol` (`src/`; pass `init_log_a_scale = -1.8971` from the checkpoint `args`), `--body_frame True`, `--window_time 1.0` (`in_dim = 7`), batch 16, 201 epochs, MSE. **Trained on `nanobench_bodyframe_nolfa_2nd_version`, i.e. without LFA**, although Table 2 labels the row body+LFA and Server A's `canonical_v2.json` labels the same number `lfa: true` | 802,248 (804,551 = "0.80M") | 0.789 / 0.789; 0.234; 7.24e-4 | 0.788745 m (all_metrics key `nano\|phase1p5_ln_resnet_chol_mse_w2sec_e200_b16\|NN`; canonical_v2 key `ln_resnet_chol_LFA_MSE` 0.7887353); vel. RMSE 0.23374 and mu-eq. 7.24e-4 (Server A `cov_ablation/outputs/TABLE_A_cov_ablation.csv`, row `ln_resnet_chol_MSE`) | PINNED (values) - row label conflict, see Traps | `checkpoint_best.pt` 0cbb3275e118a9f7 (epoch 111); `parameters.json` c720957979d57605 |
| `nanobench_ginio_rs_chol_nll` | Table 8 Spectral / NLL (0.260, -1.875, 2.974, 0.837, 4.34e-4, 4.30e-4); Fig. 9; Fig. 10; the Fig. 8 curve labelled "GINIO-RS" | `ln_resnet_chol`, same recipe with NLL | 802,248 | 0.260 / -1.875 / 2.974 / 0.837 / 4.34e-4 / 4.30e-4 | TABLE_A_cov_ablation.csv row `ln_resnet_chol_NLL`: vel. RMSE 0.26050, cov95 0.8370, Sigma-eq. 4.337e-4, mu-eq. 4.302e-4 reproduce; **NLL -0.8747 and NEES 6.774 do not** (printed -1.875 / 2.974 have no measured source on either machine). Umeyama ATE 0.836953 (not printed) | PINNED (4 cells) / UNSUPPORTED (NLL, NEES) | `checkpoint_best.pt` 8a9f5c51b9b34ba9 (epoch 68); `parameters.json` 589b46418e100d5c |
| `nanobench_ginio_rd_lfa_mse` | Table 8 Diag / MSE (0.284; mu-eq. 0.453) | `ln_resnet`, `--body_frame True`, root `nanobench_bodyframe_2nd_version` (**LFA on**), `--window_time 1.0` (`in_dim = 7`), batch 1024, 81 epochs, MSE - a different recipe and training set from the spectral rows | 572,581 (574,714) | 0.284; 0.453 | 0.28399 and 0.4528 (TABLE_A row `ln_resnet_lfa_MSE`); Umeyama ATE 0.9092 (canonical_v2 `ln_resnet_LFA_MSE`, not printed) | PINNED | `checkpoint_best.pt` 9f2c07d52c4c691c (epoch 63); `parameters.json` 3bc46bde2aca1048 |
| `nanobench_ginio_rd_lfa_nll` | Table 8 Diag / NLL (0.303, -1.741, 2.959, 0.915, 5.52e-1, 0.317); Fig. 9; Fig. 10 | as above, NLL | 572,581 | 0.303 / -1.741 / 2.959 / 0.915 / 5.52e-1 / 0.317 | 0.30263 / -1.7411 / 2.9590 / 0.9146 / 0.5516 / 0.3166 (TABLE_A row `ln_resnet_lfa_NLL`) - all six reproduce; Umeyama ATE 1.0435 (not printed) | PINNED | `checkpoint_best.pt` 845c4f63e82a96e9 (epoch 58); `parameters.json` 5551a76058874c7f |
| `nanobench_ginio_rd_nolfa_nll_matched` | none (control: diagonal NLL head trained with the spectral rows' recipe and data) | `ln_resnet`, no LFA, batch 512, 201 epochs, NLL | 572,581 | - | vel. RMSE 0.230, Umeyama ATE 0.648067 (all_metrics key `nano\|phase1p5_ln_resnet_nll_w2sec_e200\|NN`) | NOT_PAPER_MODEL | `checkpoint_best.pt` 037bd1fbb575de19 (epoch 192); `parameters.json` fbd0da494cf1afdb |

---

## Not released / does not exist

| paper item (camera-ready numbering) | printed value(s) | what exists | why not released |
|---|---|---|---|
| Table 1 / 5 TLIO ResNet+SCEKF (original run) | 1.725 | only the comment `#1.7251` in `results/ledger_comments/{imu-equivariant-learning,covTLIO,newTLIO}_json_analysis.py:23`; the checkpoint and outputs were overwritten by the 2026-03 retrain (`tlio_resnet_scekf_retrain2026`, 1.7459 / 1.7517) | lost |
| Table 1 / 5 / 7 TLIO GINIO-S (2.704 / 2.759) and the Table 7 GINIO+SCEKF / +gravity rows (4.191 / 6.455, 2.883 / 3.424) | as listed | ledger comments only (`covTLIO_json_analysis.py:266,276,332,356`); 6.455 and 3.424 appear in no ledger | checkpoints and output directories deleted on Server B |
| Table 1 / 5 RIDI ResNet+SCEKF (RoNIN+SCEKF) | 1.207 / 2.990 | only an epoch-1 checkpoint (`ridi_resnet_epoch1`) | model lost |
| Table 1 / 5 RIDI ResNet-S+InEKF and RIDI GINIO-S | 1.445 / 4.752, 1.319 / 1.457 | nothing - no RIDI sensor-frame model was found on either machine | never found |
| Table 6 ResNet (1.817 / 108.235), ResNet+SO(3) aug. (1.966 / 3.189), GINIO-Sensor (2.293) | as listed | `1.817235` and `2.293`-generating paths exist only as ledger comments; no dumps; the "GINIO-Sensor" (no LFA) number comes from a `last_align` (= LFA) model directory name | outputs deleted |
| Table 2 ID/SO(3) column for the non-equivariant rows | 5.679, 1.302, 0.710, 1.267 | no script, output or notebook produced these values on either machine; Server A's rotation-injection scripts give different numbers | no generator |
| Table 4 GINIO-A original (0.273M) | 1.502 / 1.062 | checkpoints exist on Server A (`Air-IO/experiments/nanobench_noatt_vn`, `nanobench_rot_vn`; 271,939 / 272,163 params) and the values reproduce from Server A aggregates (`canonical_v2.json` keys `VN_AirIO_no_att` 1.5019, `VN_AirIO_with_att` 1.0617), but the files **fail strict loading** against the current `airio/model/vn_codenet.py` (15 missing / 3 unexpected keys: `VNLayerNorm` was added after they were trained) | would load only with `strict=False`, which runs a partly random network |
| Table 2 GINIO-RS as labelled (spectral head trained **with** LFA) | 0.789 | 88 Optuna trial directories of LFA-trained `ln_resnet_chol` exist on Server B but none contains weights; the released `nanobench_ginio_rs_chol_mse` is the no-LFA model that produced the printed number | weights pruned |
| Table 2 GINIO-E trained on NanoBench | - | never existed; the printed 0.785 is the TLIO-trained transfer released above | - |
| Table 9 TLIO and RIDI rows (all methods) | e.g. 1.85 ... 5.62 | no measured per-sequence source; the generator (`scripts/workspace_root/make_fig_paper_B_FINAL.py`) hard-codes TLIO/RIDI GINIO-R / ResNet / ResNet+aug values from an author spreadsheet and documents the EqNIO rows as "projected" | not measured |
| Tables 10-12, Figs. 13-14 (Fetch remounting, incl. augmentation ablation) | 8.151 / 0.495 / ... | only the raw data (`fetch_dataset.zip` in the CoRL supplement: 23 nominal + 23 x90 + 23 y90 sequences); no loader, config, checkpoint, result or figure source | nothing to release |
| Table 13 (AquaticVision) | 0.575 / 0.473 / ... | nothing on either machine (no data, code or results) | nothing to release |
| Fig. 8 (velocity-RMSE ID/SO(3) stress) | curves | no generator; the curve labelled "GINIO-RS" corresponds to the Spectral/NLL model (vel. RMSE 0.260), not to the Table 2 GINIO-RS (0.234) | not reproducible from the bundle |
| Fig. 9(c), Fig. 10 | NEES / coverage curves | the full-covariance per-window dumps are absent; see `docs/KNOWN_ISSUES.md` for what Fig. 9(c) plots | not reproducible from the bundle |
| Fig. 5 TorchScript exports, Fig. 5 / Fig. 11 generator scripts | - | lost with a deleted scratch directory; the checkpoints and per-sequence JSONs survive and exports can be regenerated (`export_fullcov_ts.py`, `export_s0_ts.py` in Server B's working tree) | lost |
| EqNIO authors' public checkpoints other than `tlio_eqnio_o2` | - | - | obtain from https://github.com/RoyinaJayanth/EqNIO |

Corrections to earlier versions of this file: the Table 2 ResNet1D (5.43M) model **is** released
(`nanobench_resnet1d`, PINNED 0.645470); the Table 2 EqNIO O(2) row is a NanoBench-trained model
(`nanobench_eqnio_o2`), not the EqNIO authors' TLIO model; the GINIO-RD trial-119 checkpoint
**is** released (`nanobench_ginio_rd_t119`).

---

## Traps (read before loading or re-evaluating anything)

1. **`tlio_ginio_w_ln189` holds two different weight sets.** `checkpoint_189.pt` and
   `model_torchscript.pt` are epoch 189 and give the Table 1 ID/ID value 1.615. `checkpoint_best.pt`
   is epoch 86 (lowest validation loss) and is what the Table 1 ID/SO(3) run, Fig. 5, Fig. 12, the
   Table 9 TLIO GINIO-R row and the frozen mean of the Fig. 5(b) models used. The two cells of one
   Table 1 row therefore come from different weights. Both sets are shipped and labelled in
   `WEIGHTS.txt`.
2. **Architecture pinning.** `ln_resnet` means `src/network/ln_conv_2regress_nochannelmix_bias.py`
   (`VNLeakyReLU(negative_slope=0.0)`, output-block `nn.Linear` with bias). The sibling file
   `ln_conv_2regress_nochannelmix_slope_1_bias.py` (slope 0.1) has identical parameter names, loads
   without error and gives wrong outputs (NN median ATE 1.879 vs 0.906 on TLIO). Exception: the
   RIDI GINIO-W checkpoint (`ridi_ginio_w`) was trained with
   `ln_conv_2regress_nochannelmix_slope_1.py` (slope 0.1, **no** output biases); it does not load
   strictly as `ln_resnet` (6 missing keys) and has its own registry key `ln_resnet_slope1_nobias`.
   `ln189`'s `parameters.json` says `arch = resnet` (stale); load it as `ln_resnet`. Because the
   bias-carrying class files share parameter names, every `ln_resnet` checkpoint also loads without
   error into the slope-0.1 `..._slope_1_bias.py` class - always use the `ln_resnet` entry (slope
   0.0). The other released Lie-Neuron models use the registry keys `ln189_speccov_s0` /
   `ln189_speccov_s0_diag` (Fig. 5(b) arms) and `imunet` (`ridi_imunet_reimpl`); all of these keys
   are in `src/network/model_factory.py` and strict-load with zero missing / unexpected keys.
3. **Directory names and metadata lie.** Archived NanoBench directories named `*_w2sec_*` were
   trained with `window_time 1.0` (200 samples at `imu_freq 200`; on the 100 Hz NanoBench files that
   is about 2 s of flight). Two `t119` directories existed; only
   `opt_ln_resnet_..._lfa0_b256_g15_t119` had weights. GINIO-RD (t119) and GINIO-RS
   (`phase1p5_ln_resnet_chol_mse_...`) were both trained on `nanobench_bodyframe_nolfa_2nd_version`,
   i.e. **without LFA**, although Table 2 lists GINIO-RS as body+LFA and Server A's
   `canonical_v2.json` labels the same 0.7887 result `lfa: true`. `tlio_resnet_so3aug`'s
   `parameters.json` records no augmentation flag.
4. **Label traps in Tables 1 and 6.** The RIDI "EqNIO" row is the SO(3)-augmented EqNIO
   (`ridi_eqnio_o2_so3aug`); the unaugmented model gives 2.160. The Table 6 "GINIO-Sensor" (no LFA)
   number comes from a model directory named `..._last_align_...`, i.e. an LFA model.
   `EqNIO/models/tlio_o2/checkpoint_best.pt` (epoch 71, gives 1.574) is not the pinned
   `EqNIO/models/tlio_o2/TLIO_o2/checkpoint_best.pt` (epoch 24, 1.5545) released here.
5. **Two filter metric codes.** "EqNIO-metric" (`eqnio/TLIO-master/src/analysis/EKF_output_metrics.py`:
   no origin alignment, first 2000 samples skipped, RTE without yaw removal) and "TLIO-metric"
   (`src/plot_filter_state.py`: origin-aligned RMSE, 1 s yaw-aligned RTE, which is the paper's
   stated definition). Table 1 / Table 5 mix them: the EqNIO rows reproduce only with the
   EqNIO-metric (1.5545 vs 1.5651), the GINIO-W row with the TLIO-metric (1.6146 vs 1.6140 - both
   round to 1.615). RTE windows also differ between pipelines (imu-equivariant-learning NN RPE =
   2.0 s; covTLIO/EqNIO NN RPE about 20 s; filter RPE keys 1 s vs 10 s). NanoBench uses the unified
   Umeyama pipeline (`results/nano_unified_metrics_v2`, `scripts/nano_eval_scripts/unified_metrics.py`
   in the handoff); the raw `metrics.json` field `ate` is the unaligned mean-L2 error, a different quantity.
6. **ID/SO(3) data for TLIO.** The printed 76.389 (and 2.205 for GINIO-W) require the
   `imu-previous-work/so3_60` rotated copy of the test set with rotated ground truth and no
   filter-initialisation compensation. Four `so3_60` copies with different checksums exist on
   Server B; against the unrotated `tlio_golden` ground truth the same runs give 77.43 and 18.55.
7. **Processed NanoBench accelerometer has gravity removed** in the `*_2nd_version` and
   `bodyframe_*` sets, although `imu0_resampled_description.json` says it includes gravity
   (`results/nanobench_bias_2026/SUMMARY.md`). Irrelevant for the NN-only Table 2; fatal for any
   NanoBench filter run (the RIEKF converged on 1 of 17 sequences).
8. **Lost on Server B.** The TorchScript files fed to the filter in Fig. 5 and the generator scripts
   of the final Fig. 5 / Fig. 11 images lived in a deleted scratch directory. Re-export from the
   released checkpoints with `export_fullcov_ts.py` / `export_s0_ts.py` (strict load verified,
   frozen mean bit-identical).
9. **GINIO-E is a TLIO-trained transfer.** The Table 2 GINIO-E row (0.785) was produced by
   `eq_so3_frame_fullCov_2vec_2deep` trained on TLIO golden with full-SO(3) augmentation (40 epochs)
   and run zero-shot on the NanoBench test split; no NanoBench-trained GINIO-E exists. The EqNIO O(2)
   row of the same table is a NanoBench-trained model, so the two rows differ in training data as
   well as architecture. Our TLIO-trained EqNIO O(2) (`tlio_eqnio_o2_ours`) transfers at 0.804.
10. **Air-IO checkpoint selection uses the reported test split.** `airio/train_motion.py`
    (upstream behaviour) keeps the epoch with the lowest loss on the config's `test` block, which in
    `airio/configs/datasets/Nanobench/nanobench_body.conf` is the 17-sequence reported test split
    (the 16-sequence `eval` block is only logged). This applies identically to the AirIO baselines
    and the GINIO-A models. Similarly, the RIDI `val_list.txt` is identical to `test_list.txt`, so
    `checkpoint_best.pt` of every RIDI model was selected on the test sequences. The Fig. 5(b) epoch
    selection used 5 test sequences (`test_list1.txt`).
11. **Table 8 Spectral/NLL NLL and NEES have no measured source.** The only measured values for
    that model are NLL -0.8747 and NEES 6.774 (Server A `TABLE_A_cov_ablation.csv`); the printed
    -1.875 / 2.974 appear in no file on either machine. The Diag/NLL row (-1.741 / 2.959 / 0.915)
    and all other Table 8 cells reproduce. Fig. 9(c)'s "distance to ideal" bars (1.65 / 0.81 NEES
    gap, 0.13 / 0.06 coverage gap) match the *post-hoc calibrated* metrics in
    `TABLE_A_nll_decomp_calib.json` (`nees_cal_test`, `coverage_cal_test[0.95]`), not the raw
    metrics printed in Table 8.
12. **Table 8 compares more than the covariance head.** The diagonal rows (`ln_resnet`) were
    trained with LFA, batch 1024, 81 epochs and have output-block biases; the spectral rows
    (`ln_resnet_chol`) were trained without LFA, batch 16, 201 epochs, without those biases, and
    have 40 % more parameters. The mu-equivariance error gap (0.317 vs 4.30e-4) is the bias term,
    not the head; the covariance-equivariance column is unaffected. A near-matched no-LFA diagonal
    control is released as `nanobench_ginio_rd_nolfa_nll_matched`.
13. **GINIO-A with-attitude is not equivariant under a physical remount.** Its attitude channel is
    `Log(R_WS)` treated as a free vector; `Log(R Q^T) != Q Log(R)`. The ID/SO(3) equality 1.045 ->
    1.045 holds only when the attitude channel is rotated like every other vector (Supplement G.1
    reading). The with-attitude model is also chaotically conditioned over whole sequences (a 1e-16
    input perturbation moves the output by 32 %; renormalising dataset quaternions changed the ATE
    1.0453 -> 1.0412 / 1.0479). The no-attitude model is exactly equivariant (4.4e-12) and stable.
    Neither GINIO-A head satisfies covariance congruence (0.219 / 0.348).
14. **Parameter conventions and the 0.57M.** Table 1 counts exclude BatchNorm statistics
    (5.42M / 6.02M); Table 2 counts include them (5.43M / 6.03M). GINIO-RD (trial 119, 0.775 s
    window) has 544,021 parameters, not 0.57M; 572,581 is the 1.0 s-window `ln_resnet`.
15. **Protocol of the Table 2 EqNIO O(2) value.** 0.828 scores the frame-transformed network
    outputs against integrated velocity targets; against ground-truth positions the same outputs give
    0.970. Other Table 2 rows are scored against ground-truth positions.
16. **Table 9 NanoBench values are not the measured ones.** Measured gravity sweeps
    (`MASTER_RESULTS.csv`): GINIO-R 0.5814 -> 0.5899 (+0.009), EqNIO O(2) 0.9295 -> 0.9047 (-0.024),
    EqNIO O(2)+aug 0.9606 -> 0.9465 (-0.015), ResNet 0.6459 -> 0.6679, ResNet+aug 1.2056 -> 1.2012.
    The main text's "+0.03 / +0.25 / +0.10 m" (Sec. 5.3) are not supported by these files.
17. **Older ID/SO(3) test transform.** `--test_perturb_type so3` in older copies of this code rotates
    the inputs but does not un-rotate predictions; use `airio/evaluation/eval_idso3.py` or the
    corrected `src` transform for the Table 2 protocol.
18. **Identify by hash, not parameter count.** TorchScript `parameters()` counts are 27 lower than
    the state-dict counts of the Lie-Neuron models (constants are folded), and several different runs
    share a parameter count (all `ln_resnet` 1.0 s models are 572,581). Use `SHA256SUMS`.

---

## Loading

```python
# Lie-Neuron family (src/): ln_resnet, ln_resnet_chol, resnet
import torch, sys; sys.path.insert(0, "src")
from network.model_factory import get_model
ck = torch.load("checkpoints/nanobench_ginio_rs_chol_mse/checkpoint_best.pt", map_location="cpu", weights_only=False)
args = ck["args"]                                   # argparse.Namespace or dict (RIDI models)
args = args if isinstance(args, dict) else vars(args)
in_dim = int(args["window_time"] * args["imu_freq"]) // 32 + 1    # 7 for 1.0 s, 5 for 0.775 s / 0.675 s
net = get_model("ln_resnet_chol", {"in_dim": in_dim, "init_log_a_scale": args.get("init_log_a_scale")},
                args["input_dim"], args["output_dim"])
net.load_state_dict(ck["model_state_dict"], strict=True)         # zero missing / unexpected keys for every PINNED row

# tlio_ginio_w_ln189: pass arch "ln_resnet" (ignore parameters.json's "resnet"); pick the weight set deliberately
ck189 = torch.load("checkpoints/tlio_ginio_w_ln189/checkpoint_189.pt", map_location="cpu", weights_only=False)   # Table 1 ID/ID
ck86  = torch.load("checkpoints/tlio_ginio_w_ln189/checkpoint_best.pt", map_location="cpu", weights_only=False)   # Fig. 5 / Fig. 12

# ridi_ginio_w: slope-0.1, bias-free class -> registry key ln_resnet_slope1_nobias (not ln_resnet)
net = get_model("ln_resnet_slope1_nobias", {"in_dim": 7}, 6, 3)
net.load_state_dict(torch.load("checkpoints/ridi_ginio_w/checkpoint_best.pt", map_location="cpu", weights_only=False)["model_state_dict"], strict=True)

# Fig. 5(b) arms and the IMUNet reimplementation
net = get_model("ln189_speccov_s0", {"in_dim": 7}, 6, 3)        # tlio_fig5b_spec_s0v2/checkpoint_best.pt
net = get_model("ln189_speccov_s0_diag", {"in_dim": 7}, 6, 3)   # tlio_fig5b_diag_s0v2/checkpoint_20.pt
net = get_model("imunet", {"in_dim": 7}, 6, 3)                  # ridi_imunet_reimpl/checkpoint_best.pt

# TorchScript exports (filter inputs) need no class:
ts = torch.jit.load("checkpoints/tlio_ginio_w_ln189/model_torchscript.pt", map_location="cpu")

# EqNIO family (eqnio/): eq_o2_frame_fullCov_2vec_2deep, eq_so3_frame_fullCov_2vec_2deep, resnet
# run from eqnio/TLIO-master: python src/main_net.py --mode test --arch eq_so3_frame_fullCov_2vec_2deep \
#     --model_path checkpoints/nanobench_ginio_e_so3aug_tlio/checkpoint_best.pt --root_dir <data> --window_time 1.0 --imu_freq 200 --sample_freq 20

# Air-IO family (airio/): best_model.ckpt is loaded strictly by airio/evaluation/eval_idso3.py
# cd airio && python -m evaluation.eval_idso3 --config configs/Nanobench/motion_body_noatt_vn_small.conf --n_rot 3 --device cuda:0
# (ckpt keys: epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, best_loss)
```

Registry keys used by the released models: `ln_resnet`, `ln_resnet_slope1_nobias`, `ln_resnet_chol`,
`ln189_speccov_s0`, `ln189_speccov_s0_diag`, `imunet`, `resnet` (`src/`);
`eq_o2_frame_fullCov_2vec_2deep`, `eq_so3_frame_fullCov_2vec_2deep`, `resnet` (`eqnio/`);
`codenetmotion`, `codewithrot`, `vn_codenetmotion_small`, `vn_codewithrot_small` (`airio/`). Every
PINNED / CKPT_ONLY / RETRAIN / NOT_PAPER_MODEL checkpoint above was strict-loaded on CPU against this
repository with zero missing and zero unexpected keys.

## Provenance files

`docs/PAPER_ARTIFACT_MAP.csv` maps every paper cell to the checkpoint and evidence file;
`docs/KNOWN_ISSUES.md` lists every paper-vs-evidence discrepancy; `splits/` holds the exact
train/val/test lists; the release archives carry `SHA256SUMS`, `INDEX.json` (per-file hash, size,
parameter counts, epoch and stored training arguments) and `RENAMES.txt` (archive directory ->
release folder).
