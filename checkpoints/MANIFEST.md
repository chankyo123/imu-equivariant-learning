# Released checkpoints

Download the archive from the GitHub Release page (link in the README) and extract it here so
that each folder below sits at `checkpoints/<name>/`. Every file is identified by its SHA-256
prefix; the parameter count and the recomputed Table-2 ATE are the identification criteria the
authors used (directory names alone are not evidence).

## NanoBench (Table 2 of the paper)

| Folder | Paper row | Arch / registry key | Params | Paper ATE | Recomputed | Files (sha256 prefix) |
|---|---|---|---|---:|---:|---|
| `airio_noatt_nanobench` | AirIO (no-att.) | `codenetmotion` (`airio/model/code.py`) | 330,598 | 5.579 | 5.5785 | `best_model.ckpt` d149f819dbdac4c3, `parameters.yaml` |
| `airio_withatt_nanobench` | AirIO (with-att.) | `codewithrot` | 387,014 | 1.205 | 1.2049 | `best_model.ckpt` 66bcda3c09d09109, `parameters.yaml` |
| `ginio_a_noatt_nanobench` | GINIO-A (no-att.) | `vn_codenetmotion_small` (`airio/model/vn_codenet.py`) | 44,486 | 1.430 | 1.4302 | `best_model.ckpt` 7432b5751164d7bf, `parameters.yaml` |
| `ginio_a_withatt_nanobench` | GINIO-A (with-att.) | `vn_codewithrot_small` | 44,563 | 1.045 | 1.0453 | `best_model.ckpt` f70ae327eccd0f33, `parameters.yaml` |
| `ginio_e_so3aug_tlio` | GINIO-E | `eq_so3_frame_fullCov_2vec_2deep` (`eqnio/`) | 6,032,284 | 0.785 | 0.7854 | `checkpoint_best.pt` 46ed9465a95d53ed, `model_torchscript.pt` c749740fa8fe6799, `parameters.json` |
| `ginio_rs_chol_mse_nanobench` | GINIO-RS (see note 2) | `ln_resnet_chol`, MSE | 804,551 | 0.789 | 0.7887 | `checkpoint_best.pt` 0cbb3275e118a9f7, `parameters.json` |
| `ginio_rd_nolfa_t55_nanobench` | closest available to GINIO-RD (see note 3) | `ln_resnet`, no LFA, NLL | 574,714 | 0.581 | 0.600 | `checkpoint_best.pt` fe203747b800c650, `parameters.json` |

Supplement Table 9 (covariance ablation; also the inputs of the frame-mismatch sweeps):

| Folder | Head / loss | Files |
|---|---|---|
| `ginio_rd_lfa_mse_nanobench` | diagonal, MSE (LFA on) | `checkpoint_best.pt` 9f2c07d52c4c691c |
| `ginio_rd_lfa_nll_nanobench` | diagonal, NLL (LFA on) | `checkpoint_best.pt` 845c4f63e82a96e9 |
| `ginio_rs_chol_mse_nanobench` | spectral (Cholesky-parametrised), MSE | `checkpoint_best.pt` 0cbb3275e118a9f7 |
| `ginio_rs_chol_nll_nanobench` | spectral, NLL | `checkpoint_best.pt` 8a9f5c51b9b34ba9 |

Reference model trained by us for the EqNIO O(2) comparison on TLIO (not a paper row):
`eqnio_o2_tlio_ours` (`checkpoint_best.pt` 9de553e92ba36664, `model_torchscript.pt` 03885ba331b46a69).

## Notes

1. **GINIO-E** was trained on TLIO (golden split) with full-SO(3) augmentation for 40 epochs
   and evaluated zero-shot on the NanoBench test split; it was not trained on NanoBench.
2. **GINIO-RS**: the checkpoint's own `parameters.json` records a `nolfa` training root while
   the paper labels the row body+LFA; the ATE reproduces (0.7887) with LFA applied at test time.
   Treat the training-time LFA flag of this checkpoint as unverified.
3. **GINIO-RD** (Optuna trial 119, ATE 0.581) was not archived. Trial 55 from the same sweep
   (ATE 0.600, same architecture and data) is released instead.
4. Not released because no checkpoint exists in the authors' archive: ResNet1D 5.43M and
   EqNIO O(2) NanoBench rows (the latter is the EqNIO authors' public TLIO+O(2) model,
   https://github.com/RoyinaJayanth/EqNIO, evaluated on NanoBench), all TLIO/RIDI filter models
   of Table 1, and all Fetch models of Table 3.
5. The two original 0.273M GINIO-A models of Supplement Table 5 predate `VNLayerNorm` and do
   not load against the current `airio/model/vn_codenet.py`; they are not released.
6. **Activation slope of the Lie-Neuron checkpoints.** The archive that shipped the four
   Table-9 models also contained a variant architecture file with `VNLeakyReLU(negative_slope=0.1)`
   (`src/network/ln_conv_2regress_nochannelmix_slope_1_bias.py`). The published Table-9 numbers were
   computed with the `ln_resnet` registry entry, i.e. `ln_conv_2regress_nochannelmix_bias.py`
   (`negative_slope=0.0`), and that is what `--arch ln_resnet` loads here. Both files accept the
   checkpoints without missing keys; on two test sequences the velocity RMSE differs by about 1 %
   (0.320 vs 0.325 m/s on B11_random_waypoints_rep1), so use the default.

## Loading

```python
# Lie-Neuron family (src/)
import torch, sys; sys.path.insert(0, "src")
from network.model_factory import get_model
ck = torch.load("checkpoints/ginio_rs_chol_mse_nanobench/checkpoint_best.pt", map_location="cpu", weights_only=False)
args = ck["args"]            # training-time config (arch, root_dir, window_time, ...)
net = get_model(args["arch"], {"in_dim": 7, "init_log_a_scale": args.get("init_log_a_scale")}, 6, 3)
net.load_state_dict(ck["model_state_dict"])

# Air-IO family (airio/): see airio/evaluation/eval_idso3.py, which loads <exp_dir>/ckpt/best_model.ckpt strictly.
```
