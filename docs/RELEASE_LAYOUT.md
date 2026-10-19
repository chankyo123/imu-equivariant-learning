# Release layout: how the checkpoints and result files are published

All binary assets go on the GitHub Release page of the `ginio` repository (tag `v1.0-corl2026`, one release); nothing binary is committed to git (`.gitignore` excludes `checkpoints/*/`, `*.pt`, `*.ckpt`). GitHub allows release assets of up to 2 GB each, so the 0.69 GB of released checkpoints are split per benchmark - three archives of 149-230 MiB - plus one small archive with the stored results behind the paper's numbers. The repository itself carries `checkpoints/MANIFEST.md` (model zoo), `docs/PAPER_ARTIFACT_MAP.csv|md` (cell-by-cell provenance), `docs/KNOWN_ISSUES.md` (erratum worklist) and `splits/` (train/val/test lists).

## Assets

| asset | contents | uncompressed | archive size | sha256 |
|---|---|---:|---:|---|
| `ginio_checkpoints_tlio.tar.gz` | 4 checkpoint folders `tlio_*` + `MANIFEST.md`, `INDEX.json`, `SHA256SUMS`, `RENAMES.txt` | 263.1 MB | 241.2 MB | `4388199e414ea1918f823bef22ebca65e91864472d0ef29be7e98b1f06a9825b` |
| `ginio_checkpoints_ridi.tar.gz` | 3 checkpoint folders `ridi_*` + `MANIFEST.md`, `INDEX.json`, `SHA256SUMS`, `RENAMES.txt` | 170.5 MB | 155.8 MB | `53d8e52ab93a573c1a8f7593e514e22bddabe349ebefddedb071d2b0cb062e84` |
| `ginio_checkpoints_nanobench.tar.gz` | 9 checkpoint folders `nanobench_*` + `MANIFEST.md`, `INDEX.json`, `SHA256SUMS`, `RENAMES.txt` | 254.5 MB | 233.3 MB | `28eda5d5de5f5876311f600631d74d923ce30155d4c6a8cae362e977fe057cee` |
| `paper_results.tar.gz` | 0 result files from both machines (see below) | 0.0 MB | 18.4 MB | `181563ad06a78bf95de3410331ee0fa292179bfd369cde9750676d8a15edb8e8` |
| `SHA256SUMS` | digests of the four archives | - | - | - |

Total download: 648.6 MB. Every archive is well under the 2 GB per-asset limit; the three checkpoint archives hold the 16 released folders of the 32 documented in `checkpoints/MANIFEST.md` (688.2 MB of files); the 16 folders marked `NOT_IN_V1.0` there are not in any asset.

Archives were created with `tar --sort=name --owner=0 --group=0 --numeric-owner -czf` from the staging directory, so member order and ownership are deterministic (file mtimes are the originals). Staging copies on the AWS box: `/home/ubuntu/ginio_release/checkpoints_release/` (checkpoints) and `/home/ubuntu/ginio_release/release_assets_v2/` (archives); `RENAMES.txt` maps every archived Server A / Server B directory to its release folder.

## Checkpoint archives

### `ginio_checkpoints_tlio.tar.gz`

| folder | size | weight files |
|---|---:|---|
| `tlio_eqnio_o2` | 72.4 MB | `checkpoint_best.pt` |
| `tlio_ginio_w_ln189` | 16.4 MB | `checkpoint_189.pt`, `checkpoint_best.pt`, `model_torchscript.pt` |
| `tlio_resnet_s_lfa_candidate` | 87.2 MB | `checkpoint_best.pt`, `model_torchscript.pt` |
| `tlio_resnet_so3aug` | 87.1 MB | `checkpoint_best.pt`, `model_torchscript.pt` |

### `ginio_checkpoints_ridi.tar.gz`

| folder | size | weight files |
|---|---:|---|
| `ridi_eqnio_o2_so3aug` | 96.8 MB | `checkpoint_best.pt`, `model_torchscript.pt` |
| `ridi_ginio_w` | 8.5 MB | `checkpoint_best.pt`, `model_torchscript.pt` |
| `ridi_resnet_so3aug` | 65.3 MB | `checkpoint_best.pt` |

### `ginio_checkpoints_nanobench.tar.gz`

| folder | size | weight files |
|---|---:|---|
| `nanobench_airio_noatt` | 8.0 MB | `best_model.ckpt` |
| `nanobench_airio_withatt` | 8.8 MB | `best_model.ckpt` |
| `nanobench_eqnio_o2` | 66.1 MB | `checkpoint_best.pt` |
| `nanobench_ginio_a_noatt` | 1.1 MB | `best_model.ckpt` |
| `nanobench_ginio_a_withatt` | 1.2 MB | `best_model.ckpt` |
| `nanobench_ginio_e_so3aug_tlio` | 96.8 MB | `checkpoint_best.pt`, `model_torchscript.pt` |
| `nanobench_ginio_rd_t119` | 6.6 MB | `checkpoint_best.pt` |
| `nanobench_ginio_rs_chol_mse` | 6.9 MB | `checkpoint_best.pt` |
| `nanobench_resnet1d` | 58.9 MB | `checkpoint_best.pt` |

Each checkpoint archive also contains the shared `INDEX.json` (per-file sha256, size, parameter counts, epoch and the training arguments stored in the checkpoint), the `SHA256SUMS` of the released folders' files, `RENAMES.txt`, and a copy of `checkpoints/MANIFEST.md` at packaging time. Extract with

```bash
cd <repo>/checkpoints
tar xzf ginio_checkpoints_nanobench.tar.gz   # and/or _tlio, _ridi
sha256sum -c SHA256SUMS --ignore-missing    # verifies the extracted files
```

## `paper_results.tar.gz`

Small per-sequence outputs and aggregates that let anyone check the "reproduced value" column of `checkpoints/MANIFEST.md` and `docs/PAPER_ARTIFACT_MAP.csv` without re-running inference. Layout (a `README.txt` inside repeats this):

| path inside the archive | origin | what it is | size |
|---|---|---|---:|
| `serverB/nn_outputs/<model dir>/` | Server B `results/nn_outputs/` | network-only test outputs: `metrics.json` (per sequence: `ate` = unaligned mean-L2, `rmse_vel`, `rpe`, ...), `dataset_summary.json`, and for the 8 NanoBench models and the RIDI GINIO-W model the per-sequence `trajectory.txt` / `net_outputs.txt` (17 and 23 sequences). The two TLIO ln189 directories (36 sequences, 113 MB each) contribute `metrics.json` + `dataset_summary.json` only; their trajectories (226 MB) are omitted | 45 MB |
| `serverB/recomputed_2026-10/` | Server B | `RECOMPUTED_FILTER_METRICS.csv` (every surviving filter output re-scored under both metric codes and both ground-truth copies); `best_ckpt_param_counts.tsv` | 0.1 MB |
| `serverB/nano_unified_metrics_v2/` | Server B | unified Umeyama-ATE pipeline output for NanoBench / TLIO (`all_metrics.json` per sequence, `all_summary.txt`, `VALIDATION_NOTES.md`) | 0.3 MB |
| `serverB/eqnio_csv/` | Server B | EqNIO-repo aggregate CSVs (Table 1 EqNIO and GINIO-W ID/SO(3) cells) | < 0.1 MB |
| `serverB/ledger_comments/` | Server B | the three `json_analysis.py` files whose comments are the only record of several Table 1 / 6 / 7 numbers | 0.1 MB |
| `serverB/nanobench_bias_2026/` | Server B | residual-bias extraction on the 17 NanoBench test sequences; gravity-removed note | < 0.1 MB |
| `serverB/server_B_rebuttal/full36/` | Server B | Fig. 5(a)/(b) sweeps on the 36 TLIO test sequences (`p3_raw_sweep_WIDE.json`, `p4_fullcov_WIDE.json`, epoch selection) | 0.3 MB |
| `serverB/server_B_rebuttal/ridi_refilter/`, `*.csv`, `D5_cov_equivariance.json`, `figures_csv/` | Server B | Aug-2026 RIDI re-filter, Table 6 network reconstruction, covariance-equivariance measurements, nominal metrics, result registry | 0.1 MB |
| `serverB/HANDOFF_paper_handoff_v2/` | Server B | frame-estimation-mismatch sweeps (`MASTER_RESULTS.csv`, `04_raw_metrics/` with 243 per-cell `metrics.json`) and the Fig. 4 generator notes (`05_figure_B_handoff/INSIGHTS.md`, `reproduce_figure_B.py`) | 4.4 MB |
| `serverB/*.md`, `*_serverB_original.csv` | Server B | `RIDI_EqNIO_ATE_summary.md`, `PHASE1_RESULTS.md`, the handoff's own artifact map and checkpoint manifest | < 0.1 MB |
| `serverA/nanobench_final_table/` | Server A | `canonical_v2.json` (per-sequence Umeyama ATE for 12 NanoBench models; Table 2 source) and its rendered tables | 0.1 MB |
| `serverA/cov_ablation_outputs/` | Server A | `TABLE_A_cov_ablation.csv` (the measured Table 8), `TABLE_A_nll_decomp_calib.json`, `TABLE_B_*.csv` (angle-resolved sweeps), `eqnio_trajectory_ate.json` (GINIO-E 0.7854) | < 0.1 MB |
| `serverA/audit_2026-08/` | Server A | checkpoint manifest of the AWS box (73 checkpoints with hashes and parameter counts), per-sequence NanoBench results | 0.1 MB |

Uncompressed 0.0 MB (0 files), 18.4 MB compressed.

## Not included (and where it lives)

| item | size | reason / source |
|---|---:|---|
| TLIO golden dataset | 8.5 GB | public: CathIAS/TLIO release (`golden-new-format-cc-by-nc-with-imus-v1.5.zip`) |
| RIDI raw data and our TLIO-layout conversion (`ridi_tlio`) | 5.3 GB raw | public RIDI `data_publish_v2`; conversion scripts are part of the code release |
| processed NanoBench (`nanobench_processed_2nd_version`, 100 Hz, gravity removed) and the two `bodyframe` derivatives | 1.1 GB + 59 MB | planned separate data release (README *Status*); note the gravity caveat in `splits/README.md` |
| Fetch remounting raw data (`fetch_dataset.zip`, 69 sequences) | 57 MB | shipped with the CoRL supplementary material; no loader or results exist |
| `so3_60` rotated TLIO test copies | - | four copies with different checksums existed on Server B; the one that reproduces Table 1 is `imu-previous-work/so3_60` (not in the handoff) |
| per-sequence TLIO trajectories of the two ln189 network-only runs | 226 MB | omitted from `paper_results.tar.gz`; `metrics.json` carries the per-sequence numbers |
| raw per-timestep filter states (`not_vio_state.txt.npy`), attitude-sweep outputs | 6.4 GB per 36-sequence batch, 8.8 GB | Server B only |
| intermediate epoch checkpoints and Optuna trial directories | ~4,000 files | Server B only |
| Table 4 GINIO-A original (0.273M) checkpoints | 2 x 1.2 MB | on Server A; fail strict loading against the current code (see `checkpoints/MANIFEST.md`) |
| EqNIO authors' checkpoints other than `tlio_eqnio_o2` | - | https://github.com/RoyinaJayanth/EqNIO |
| 16 surviving models documented in `checkpoints/MANIFEST.md` with status `NOT_IN_V1.0` (Fig. 5(b) arms, Table 8 ablation models other than GINIO-RS, Table 9 augmented NanoBench baselines, 2026 ResNet+SCEKF retrain, eight controls) | 638 MB | kept out of v1.0 by the authors' choice; staged on the AWS box, available on request |

## Suggested release notes

* 16 checkpoint folders (TLIO 4, RIDI 3, NanoBench 9): every Table 1 and Table 2 model and baseline that survives on the two machines that ran the paper's experiments; 16 further surviving models are documented in the MANIFEST as `NOT_IN_V1.0`; see `checkpoints/MANIFEST.md` for the status of every paper cell (PINNED / CKPT_ONLY / RETRAIN / NOT_PAPER_MODEL / UNSUPPORTED).
* Table 2 is fully backed by released checkpoints (AirIO x2, GINIO-A x2, ResNet1D, EqNIO O(2), GINIO-E, GINIO-RD, GINIO-RS); the ID/SO(3) column of the non-equivariant rows is not.
* Table 8 (covariance ablation): only the Spectral/MSE model (= Table 2 GINIO-RS) is in v1.0; the other three are documented, and the Spectral/NLL NLL and NEES cells are not reproduced (see `docs/KNOWN_ISSUES.md`).
* TLIO Table 1 models: GINIO-W (both weight sets), EqNIO, ResNet+SO(3) aug., a ResNet-S+InEKF candidate; GINIO-S is lost and the ResNet+SCEKF row points to the TLIO authors' model. RIDI: GINIO-W, the SO(3)-augmented EqNIO and ResNet+SO(3) aug.; the RIDI ResNet baseline is lost (epoch-1 checkpoint only).
* Fetch and AquaticVision models do not exist in any archive.
* Table and figure numbers everywhere follow the camera-ready PDF (Table 8 = covariance ablation, Table 9 = frame-estimation mismatch, Tables 10-12 = Fetch).
