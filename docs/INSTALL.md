# Installation

```bash
git clone https://github.com/chankyo123/GINIO.git && cd GINIO
conda env create -f environment.yml          # python 3.12 + pip requirements
conda activate ginio
```

`environment.yml` installs everything except PyTorch, because the right wheel depends on your
CUDA driver. Install one of:

```bash
# CUDA 12.x (Ampere / Ada / Hopper)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Blackwell (sm_120) needs a cu128 build; the paper's final runs used torch 2.12 nightly cu128
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
# CPU only (unit tests, loading checkpoints, metrics)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Known-good versions

| package | paper runs (box ip-10-150-0-126, 2026-05..08) | earlier runs (RTX 4090 workstation) |
|---|---|---|
| python | 3.12.13 | 3.12 |
| torch | 2.12.0.dev20260407+cu128 | 2.4.0+cu121 |
| numpy | 1.26.4 | 1.26 |
| scipy | 1.17.1 | — |
| pypose | 0.9.5 | 0.9.5 |
| pyhocon | 0.3.63 | 0.3.63 |

numpy must stay below 2.0: the released checkpoints and the `.npy` data were written with numpy 1.x.

## Optional extras

| package | needed for |
|---|---|
| `fvcore` | FLOP counting only (`FlopCountAnalysis`); imports are guarded |
| `open3d` | the `--visualize` window of `src/main_filter.py` |
| `casadi` | `airio/EKF/casADI_EKF` (upstream AirIO EKF; not used for the paper) |
| `wandb` | Air-IO training logs; use `WANDB_MODE=offline` or `--log` to disable |

## Check the install (CPU, ~1 min)

```bash
cd airio && CUDA_VISIBLE_DEVICES="" python -m model.test_a2_spectral_equivariance   # expects "24/24 checks passed"
```
This needs one NanoBench sequence under `local_data/` (or set `GINIO_DATA` to a folder that
contains `nanobench_processed_2nd_version/B11_random_waypoints_rep1/imu0_resampled.npy`).

## Gotchas

* `--body_frame` in `src/main_net.py` is a string: pass `"True"` or `"False"`.
* Always pass `--imu_freq 200` to `src/main_net.py`; the window length is `int(200 * window_time)`
  samples regardless of the data rate, so on 100 Hz NanoBench `--window_time 1.0` is a 2 s window.
* The spectral and Cholesky heads (`ln_resnet_cov`, `ln_resnet_chol`) are wired for a 200-sample
  window (`FcBlock(..., 7)`); other windows need the `in_dim` argument adjusted.
* `torch.load` of the released checkpoints requires `weights_only=False` (they store the training
  `args`); the test and filter scripts already do this.
