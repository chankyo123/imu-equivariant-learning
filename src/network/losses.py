import torch
import numpy as np
from network.covariance_parametrization import (
    DiagonalParam, CovarianceParam, PearsonParam, SinhParam,
)

MIN_LOG_STD = float(np.log(1e-3))
MAX_LOG_STD = float(np.log(10.0))


# ============================================================
# Basic losses
# ============================================================

def loss_mse(pred, targ):
    """
    Per-dim MSE (no covariance).
    Returns (B,3).
    """
    return (pred - targ).pow(2)


def loss_distribution_diag(pred, pred_logstd, targ):
    """
    Diagonal Gaussian NLL. Returns (B,3) per-dim.
    NLL_i = (r_i^2)/(2 sigma_i^2) + log(sigma_i)
    """
    pred_logstd = torch.clamp(pred_logstd, min=MIN_LOG_STD, max=MAX_LOG_STD)
    return ((pred - targ).pow(2)) / (2 * torch.exp(2 * pred_logstd)) + pred_logstd


def loss_distribution_covariance(pred_mean, pred_V, pred_inv, targ,
                                 min_log_std=MIN_LOG_STD,
                                 max_log_std=MAX_LOG_STD):
    """
    Equivariant spectral NLL.
    Sigma = V diag(exp(2*inv)) V^T.
    Equivalent NLL in the V-basis:
        per_dim = 0.5 * (V^T (targ - mean))^2 / sigma^2 + log(sigma)
    Returns (B,3) per-dim.
    """
    pred_logstd = torch.clamp(pred_inv, min=min_log_std, max=max_log_std)
    var = torch.exp(2.0 * pred_logstd)                          # (B,3)

    residual = (targ - pred_mean).unsqueeze(-1)                 # (B,3,1)
    r_proj = torch.matmul(pred_V.transpose(1, 2), residual).squeeze(-1)  # (B,3)

    per_dim = 0.5 * (r_proj ** 2) / var + pred_logstd           # (B,3)
    return per_dim


def loss_distribution_eq_chol(pred_mean, pred_A, targ, eps=1e-3):
    """
    Equivariant Cholesky-style NLL.
    Sigma = A A^T + eps*I  (SPD by construction).
    NLL = 0.5 * (err^T Sigma^{-1} err) + 0.5 * logdet(Sigma) + const
    Returns (B,3) per-dim for compatibility with the rest of the pipeline.
    """
    B, D = pred_mean.shape
    I = torch.eye(D, device=pred_A.device, dtype=pred_A.dtype).unsqueeze(0)
    Sigma = pred_A @ pred_A.transpose(-1, -2) + eps * I          # (B,3,3)

    err = (pred_mean - targ).unsqueeze(-1)                       # (B,3,1)
    L = torch.linalg.cholesky(Sigma)                             # (B,3,3)
    y = torch.linalg.solve_triangular(L, err, upper=False).squeeze(-1)  # (B,3)
    mahal = y ** 2                                                # (B,3)
    logdet_per_dim = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1))  # (B,3)

    per_dim = 0.5 * mahal + 0.5 * logdet_per_dim                 # (B,3)
    return per_dim


def loss_distribution_eq_chol_svd(pred_mean, pred_A, targ,
                                   log_sigma_min=float(np.log(0.005)),
                                   log_sigma_max=float(np.log(5.0)),
                                   reg_weight=0.0):
    """
    Equivariant SVD/polar-decomposition NLL with non-degeneracy regularizer.

    A = U @ diag(S) @ V_h  (SVD, A in R^{B x 3 x 3}, U & V_h orthogonal, S>=0)
    Sigma = A A^T = U @ diag(S^2) @ U^T (since V_h V_h^T = I)
    -> U gives the equivariant eigen-frame of Sigma, S^2 gives the eigenvalues.

    Differences from gram-style loss_distribution_eq_chol:
      (i)  Singular values S are explicitly clamped to [exp(log_sigma_min),
           exp(log_sigma_max)] BEFORE Sigma is rebuilt -> non-degeneracy is
           enforced by construction (no passive eps*I masking).
      (ii) A barrier regularizer penalises any S that falls below
           exp(log_sigma_min): softplus(log_sigma_min - log S). This pulls
           the decoder away from collapsing covariance directions during
           training instead of relying on the loss exploding once the matrix
           becomes ill-conditioned.
      (iii) Sigma is built from the *clamped* spectrum, so the loss is well
            defined even if the raw A is near-rank-deficient.

    Equivariance: if input rotates by R then A -> R A, so SVD gives
    U' = R U, S' = S, V_h' = V_h (up to sign convention applied here only on V_h).
    Hence Sigma' = U' diag(S^2) U'^T = R Sigma R^T -- still equivariant.

    Args:
        pred_mean: (B, 3) mean prediction.
        pred_A:    (B, 3, 3) equivariant cov factor.
        targ:      (B, 3) regression target.
        log_sigma_min/max: hard clamp on log singular values.
        reg_weight: barrier-regulariser weight (per-dim contribution distributed).

    Returns:
        per_dim (B, 3) — sum equals NLL + regulariser.
    """
    B, D = pred_mean.shape
    # Stable SVD: add tiny ridge so SVD never sees exactly rank-deficient A.
    eps_svd = 1e-6
    I = torch.eye(D, device=pred_A.device, dtype=pred_A.dtype).unsqueeze(0)
    A_safe = pred_A + eps_svd * I

    U, S, _Vh = torch.linalg.svd(A_safe, full_matrices=False)
    log_S = torch.log(S.clamp(min=1e-8))
    log_S_clamped = log_S.clamp(min=log_sigma_min, max=log_sigma_max)
    S_clamped = torch.exp(log_S_clamped)

    # Sigma = U diag(S_clamped^2) U^T  (SPD by construction)
    Sigma = U @ torch.diag_embed(S_clamped ** 2) @ U.transpose(-1, -2)

    err = (pred_mean - targ).unsqueeze(-1)                              # (B,3,1)
    L = torch.linalg.cholesky(Sigma)                                    # (B,3,3)
    y = torch.linalg.solve_triangular(L, err, upper=False).squeeze(-1)  # (B,3)
    mahal = y ** 2                                                      # (B,3)
    logdet_per_dim = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1))  # (B,3)

    per_dim = 0.5 * mahal + 0.5 * logdet_per_dim                        # (B,3)

    # Non-degeneracy barrier: penalise any raw log_S that wants to dip below the floor.
    if reg_weight > 0.0:
        violation = torch.nn.functional.softplus(log_sigma_min - log_S)  # (B,3)
        per_dim = per_dim + (reg_weight * violation)
    return per_dim


# ============================================================
# Legacy criterion (kept for backward compatibility)
# ============================================================

def criterion_distribution(pred, covariance, targ, use_DiagonalParam):
    if use_DiagonalParam == 1:
        return DiagonalParam.toMahalanobisDistance(targ, pred, covariance, clamp_covariance=False)
    elif use_DiagonalParam == 2:
        return PearsonParam.toMahalanobisDistance(targ, pred, covariance, clamp_covariance=False)
    elif use_DiagonalParam == 3:
        return CovarianceParam.toMahalanobisDistance(targ, pred, covariance, clamp_covariance=False)


# ============================================================
# Top-level dispatcher: get_loss
# ============================================================

def get_loss(pred, pred_cov_params, targ, epoch,
             body_frame_3regress=False,
             cov_type='diag',
             mse_warmup_epochs=10,
             nll_weight=1.0,
             mse_anchor_weight=0.0):
    """
    Top-level loss dispatcher.

    Args:
        pred (B,3): predicted mean velocity (or displacement).
        pred_cov_params: covariance parameters; shape depends on cov_type:
            - 'diag'    : (B,3) log-std vector
            - 'eq_spec' : tuple (V (B,3,3), inv (B,3))
            - 'eq_chol' : A (B,3,3) such that Sigma = A A^T + eps*I
        targ (B,3): regression target.
        epoch: current training epoch; used for MSE-warmup.
        body_frame_3regress: legacy flag.
        cov_type: 'diag' | 'eq_spec' | 'eq_chol'.
        mse_warmup_epochs: epochs of pure MSE before NLL is enabled.
        nll_weight: weight on the NLL term after warmup.
        mse_anchor_weight: weight on an MSE anchor term added to NLL.
            Setting this to ~0.1 keeps the mean predictor sharp (P4 hybrid loss).
    Returns:
        per-sample-per-dim loss (B,3) for compatibility with write_summary.
    """
    # MSE-only warmup (covariance head is frozen / not driving the loss)
    if epoch < mse_warmup_epochs:
        return loss_mse(pred, targ)

    # NLL dispatch
    if cov_type == 'diag':
        nll = loss_distribution_diag(pred, pred_cov_params, targ)
    elif cov_type == 'eq_spec':
        pred_V, pred_inv = pred_cov_params
        nll = loss_distribution_covariance(pred, pred_V, pred_inv, targ)
    elif cov_type == 'eq_chol':
        nll = loss_distribution_eq_chol(pred, pred_cov_params, targ)
    elif cov_type == 'eq_chol_svd':
        nll = loss_distribution_eq_chol_svd(pred, pred_cov_params, targ)
    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")

    if mse_anchor_weight > 0.0:
        return nll_weight * nll + mse_anchor_weight * loss_mse(pred, targ)
    return nll_weight * nll