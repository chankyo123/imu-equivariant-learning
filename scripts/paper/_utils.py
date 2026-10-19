# Shared helpers of the covariance ablation (Supplement Table 9 / Figures 6-7); no table of its own.
"""Shared helpers for the covariance-head ablation scripts.

All Sigma-reconstruction helpers mirror
``eqnio/TLIO-master/src/tracker/meas_source_torchscript.py``.
If that file changes, update here too.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import AIRIO_ROOT, EQNIO_SRC, PAPER_DIR, OUTPUT_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
# Formerly: Air-IO root, EqNIO/TLIO-master/src and the hand-off `equivariance_tests` dir.
# The hand-off helpers (rotate_vectors_btf3, assemble_cov) now live in scripts/paper/_helpers.py.
HELPERS_DIR = PAPER_DIR


def insert_paths() -> None:
    for p in (AIRIO_ROOT, EQNIO_SRC, HELPERS_DIR):
        p = str(p)
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# SO(3) sampler
# ---------------------------------------------------------------------------

def random_rotation(dtype: torch.dtype = torch.float64, seed: int | None = None) -> torch.Tensor:
    if seed is not None:
        rng = np.random.default_rng(seed)
        R_np = Rotation.random(random_state=rng).as_matrix()
    else:
        R_np = Rotation.random().as_matrix()
    return torch.tensor(R_np, dtype=dtype)


# ---------------------------------------------------------------------------
# Sigma reconstruction -- mirrors meas_source_torchscript.py post-processing
# ---------------------------------------------------------------------------

def sigma_from_vn_airio_diag(cov_raw: torch.Tensor) -> torch.Tensor:
    """VN-AirIO 'diagonal-from-norms': cov_decoder output (B, T, 3) of raw log-vars
    -> Sigma = diag(exp(cov_raw - 5.0)) in body frame.

    Matches `vn_codenet.py` lines 165-167 / 211-213.
    """
    var = torch.exp(cov_raw - 5.0)
    return torch.diag_embed(var)


def sigma_from_so3_frame(frame_RT: torch.Tensor, log_std_can: torch.Tensor,
                         clamp_min: float = -4.0) -> torch.Tensor:
    """VN-EqNIO SO(3) spectral.

    Inputs match the torchscript outputs:
      frame_RT  : (B, 3, 3)  -- model emits R^T (called `frame` downstream)
      log_std_can : (B, 3)   -- canonical diagonal log-std

    Returns Sigma_world = R^T diag(exp(2*log_std)) R = frame^T . diag(sigma^2) . frame
    NOTE: this matches `meas_source_torchscript.py:306-310`.
    """
    log_std = torch.clamp(log_std_can, min=clamp_min)
    sigma2 = torch.exp(2.0 * log_std)
    Sigma_can = torch.diag_embed(sigma2)
    # frame.T @ Sigma_can @ frame
    return torch.einsum("bij,bjk,blk->bil",
                        frame_RT.transpose(-2, -1), Sigma_can, frame_RT.transpose(-2, -1))


def sigma_from_o2_fullCov(frame_2x2: torch.Tensor, log_std: torch.Tensor,
                          clamp_min: float = -4.0) -> torch.Tensor:
    """EqNIO O(2) `_fullCov` head -- reconstruct 3x3 Sigma from a 2x2 frame and (xy, z) log-stds.

    Mirrors `meas_source_torchscript.py:314-333`.

    Inputs:
      frame_2x2 : (B, 2, 2)
      log_std   : (B, 3)  -- interpreted as (log_std_xy_can[0], log_std_xy_can[1], log_std_z)
    """
    log_std = torch.clamp(log_std, min=clamp_min)
    var = torch.exp(2.0 * log_std)
    B = frame_2x2.shape[0]
    Sigma_xy = torch.einsum("bji,bjk,blk->bil", frame_2x2,
                            torch.diag_embed(var[:, :2]), frame_2x2)  # (B,2,2)
    Sigma = torch.zeros(B, 3, 3, dtype=var.dtype, device=var.device)
    Sigma[:, :2, :2] = Sigma_xy
    Sigma[:, 2, 2] = var[:, 2]
    return Sigma


def sigma_from_chol(A: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Cholesky head: Sigma = A A^T + eps I. Matches `meas_source_torchscript.py:347-355`."""
    return A @ A.transpose(-2, -1) + eps * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)


# ---------------------------------------------------------------------------
# NLL pieces
# ---------------------------------------------------------------------------

def nll_terms(e: np.ndarray, Sigma: np.ndarray) -> dict:
    """Per-sample Mahalanobis / log-det / NLL.

    e:     (N, 3)
    Sigma: (N, 3, 3)
    Returns dict of arrays (N,)
    """
    assert e.ndim == 2 and Sigma.ndim == 3 and e.shape[0] == Sigma.shape[0]
    N, d = e.shape
    # symmetrize for numerical safety
    Sigma_s = 0.5 * (Sigma + np.swapaxes(Sigma, -1, -2))
    try:
        L = np.linalg.cholesky(Sigma_s)
    except np.linalg.LinAlgError:
        # fallback: jitter
        Sigma_s = Sigma_s + 1e-6 * np.eye(d)[None]
        L = np.linalg.cholesky(Sigma_s)
    y = np.linalg.solve(L, e[..., None])[..., 0]
    mahal = (y ** 2).sum(-1)
    logdet = 2.0 * np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(-1)
    nll = 0.5 * (mahal + logdet + d * np.log(2.0 * np.pi))
    return {"mahal": mahal, "logdet": logdet, "nll": nll}


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def jsonsafe(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: jsonsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonsafe(v) for v in o]
    return o


def save_summary(out_dir: str | Path, name: str, payload: dict, text_summary: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(jsonsafe(payload), f, indent=2)
    with open(out_dir / f"{name}.txt", "w") as f:
        f.write(text_summary)
    print(f"[saved] {out_dir/(name+'.json')}")
    print(f"[saved] {out_dir/(name+'.txt')}")


def fail_if_missing(path: str | Path, what: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")
    return p


# ---------------------------------------------------------------------------
# OUTPUT_DIR convenience (formerly .../cov_ablation/outputs; now scripts/paper/outputs or $GINIO_PAPER_OUT)
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(OUTPUT_DIR)
