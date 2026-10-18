# Helpers copied from the authors' equivariance-test hand-off (no paper table of their own).
"""Rotation / covariance-assembly helpers.

Copied verbatim from ``equivariant_models_handoff/equivariance_tests/test_cov_equivariance.py``
(2026-05-23) so that nothing in this folder imports from the hand-off directory.  They implement
the two conventions used throughout the covariance ablation:

  rotate_vectors_btf3  rotate every 3-vector of a (B,T,3) / (B,T,F,3) / (B,T,3,F) tensor by R
  assemble_cov         normalise a model's (mu, Sigma) output to an explicit 3x3 covariance
                       ("diag_3", "full_33" or the "spectral" frame form
                       Sigma = F^T diag(exp(2 log sigma)) F, mu = F^T mu_canonical)
"""
import torch


def rotate_vectors_btf3(x, R, vec_axis=-1):
    """Rotate vectors by R along the specified vec_axis.

    Examples:
      x: (B, T, 3)           vec_axis=-1   -> rotate last axis
      x: (B, T, F, 3)        vec_axis=-1   -> rotate last axis (each of F vectors)
      x: (B, T, 3, F)        vec_axis=-2   -> rotate -2 axis (each of F vectors stored along last)
    """
    if x.dim() == 3:
        return torch.einsum("ij,btj->bti", R, x)
    elif x.dim() == 4:
        if vec_axis == -1 or vec_axis == 3:
            return torch.einsum("ij,btfj->btfi", R, x)
        elif vec_axis == -2 or vec_axis == 2:
            return torch.einsum("ij,btjf->btif", R, x)
        else:
            raise ValueError(f"unsupported vec_axis {vec_axis}")
    else:
        raise ValueError(f"unsupported shape {x.shape}")


def assemble_cov(model_output, kind):
    """
    Normalize different cov representations to a (B, T, 3, 3) tensor.

    kind:
      "diag_3"   -- model returns a (B, T, 3) of variances -> diag-embed.
      "full_33"  -- model returns a (B, T, 3, 3) -- already a matrix.
      "spectral" -- model returns (F, mu_canonical, log_sigma_canonical) where F is (B,3,3).
                   Sigma = F^T @ diag(exp(2 log_sigma_can)) @ F, mu = F^T @ mu_canonical
    """
    if kind == "diag_3":
        mu, sigma2 = model_output
        Sigma = torch.diag_embed(sigma2)
        return mu, Sigma
    elif kind == "full_33":
        mu, Sigma = model_output
        return mu, Sigma
    elif kind == "spectral":
        F, mu_can, log_sigma_can = model_output
        # mu_world = F^T @ mu_canonical
        mu_world = torch.einsum("bji,bj->bi", F, mu_can)
        # Sigma_world = F^T @ diag(sigma^2) @ F
        var = torch.exp(2 * log_sigma_can)              # (B, 3)
        Diag = torch.diag_embed(var)                    # (B, 3, 3)
        Sigma_world = torch.einsum("bji,bjk,bkl->bil", F, Diag, F)
        return mu_world, Sigma_world
    else:
        raise ValueError(f"unknown kind {kind}")
