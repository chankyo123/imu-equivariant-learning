"""Smoke test for the new SVD/polar cov head (eq_chol_svd):
  (1) Σ-equivariance still holds (rotating input rotates Σ by R Σ Rᵀ).
  (2) Non-degeneracy regulariser keeps singular values away from collapse.
  (3) New loss numerically stable vs old loss on a fixed batch.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from network.losses import loss_distribution_eq_chol, loss_distribution_eq_chol_svd

torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def sigma_from_chol_gram(A, eps=1e-3):
    I = torch.eye(3, device=A.device).unsqueeze(0)
    return A @ A.transpose(-1, -2) + eps * I


def sigma_from_svd_clamp(A, log_sigma_min=float(np.log(0.02)), log_sigma_max=float(np.log(5.0))):
    eps_svd = 1e-6
    I = torch.eye(3, device=A.device).unsqueeze(0)
    A_safe = A + eps_svd * I
    U, S, _ = torch.linalg.svd(A_safe, full_matrices=False)
    log_S = torch.log(S.clamp(min=1e-8))
    log_S_clamped = log_S.clamp(min=log_sigma_min, max=log_sigma_max)
    S_clamped = torch.exp(log_S_clamped)
    return U @ torch.diag_embed(S_clamped ** 2) @ U.transpose(-1, -2)


def test_sigma_equivariance():
    B = 16
    A = torch.randn(B, 3, 3, device=DEV) * 0.3
    # Add a few near-singular rows to stress-test
    A[:4, :, 2] = 0.001 * A[:4, :, 2]
    R = torch.from_numpy(Rotation.from_rotvec(np.random.randn(3) * 1.5).as_matrix()).float().to(DEV)
    A_rot = R @ A

    Sigma_old   = sigma_from_chol_gram(A)
    Sigma_rotS  = sigma_from_chol_gram(A_rot)
    Sigma_pred  = R @ Sigma_old @ R.T
    err_old = (Sigma_rotS - Sigma_pred).abs().max().item()

    Sigma_svd       = sigma_from_svd_clamp(A)
    Sigma_svd_rotS  = sigma_from_svd_clamp(A_rot)
    Sigma_svd_pred  = R @ Sigma_svd @ R.T
    err_svd = (Sigma_svd_rotS - Sigma_svd_pred).abs().max().item()

    # Spectrum check: SVD-clamped should have eigenvalues >= clamp_min^2
    eigvals = torch.linalg.eigvalsh(Sigma_svd)
    min_eig = eigvals.min().item()
    floor = (0.02) ** 2

    print(f"[SO(3)-equivariance] old eq_chol Σ-err = {err_old:.3e}")
    print(f"[SO(3)-equivariance] new eq_chol_svd Σ-err = {err_svd:.3e}")
    print(f"[Non-degeneracy] min eigenvalue of SVD-clamped Σ = {min_eig:.5e} (floor {floor:.5e})")
    assert err_old < 1e-4, "Old gram cov should still be equivariant"
    assert err_svd < 1e-4, "New SVD/polar cov should still be equivariant"
    assert min_eig >= 0.9 * floor, "Non-degeneracy clamp not enforced"
    print("✓ equivariance + non-degeneracy verified")


def test_loss_numerics():
    B = 32
    A = torch.randn(B, 3, 3, device=DEV) * 0.3
    A[:8, :, 2] *= 0.01  # near-rank-deficient batch
    mean = torch.randn(B, 3, device=DEV)
    targ = mean + torch.randn(B, 3, device=DEV) * 0.1

    nll_old = loss_distribution_eq_chol(mean, A, targ)
    nll_new = loss_distribution_eq_chol_svd(mean, A, targ)
    print(f"\n[Loss numerics] old eq_chol mean NLL = {nll_old.sum(-1).mean().item():+.4f}, "
          f"max = {nll_old.sum(-1).max().item():+.4f}")
    print(f"[Loss numerics] new eq_chol_svd mean NLL = {nll_new.sum(-1).mean().item():+.4f}, "
          f"max = {nll_new.sum(-1).max().item():+.4f}")

    # Gradient flow check
    A.requires_grad_(True)
    nll = loss_distribution_eq_chol_svd(mean, A, targ).sum()
    nll.backward()
    g = A.grad
    print(f"[Grad] grad-norm = {g.norm().item():.4f}, any NaN = {torch.isnan(g).any().item()}")
    assert not torch.isnan(g).any(), "NaN gradient on SVD cov loss"
    print("✓ loss numerically stable, gradients clean")


if __name__ == "__main__":
    test_sigma_equivariance()
    test_loss_numerics()
