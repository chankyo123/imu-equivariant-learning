"""Numerical equivariance verification for VN-EqNIO `eq_so3_frame_fullCov_2vec_2deep`.

The model takes (vector, scalar, original_scalars) where:
  vector: (B, T, 3, 3)   — three 3D-vector channels (a, v1, v2)
  scalar: (B, T, 6)      — 3 norms + 3 dot products, SO(3)-invariant
  original_scalars: (B, T, 3) — kept as-is (these are time-positional encodings, not affected by R)

For any R ∈ SO(3):
  model(R·vec, scalar, orig_scalars) → (R·frame, R·disp, R·cov·R^T)
  (the frame is a 3x3 → so rotated; disp is 3-vec; cov is 3x3 covariance matrix)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TLIO-master', 'src'))

import torch
import numpy as np
from scipy.spatial.transform import Rotation
from network.model_factory import get_model


def random_R(dtype=torch.float64):
    return torch.tensor(Rotation.random().as_matrix(), dtype=dtype)


def rotate_vec_field(v, R):
    """v: (B, T, vec_dim=3, n_features). Rotate the vec_dim axis (-2). R: (3, 3)."""
    return torch.einsum("ij,btjf->btif", R, v)


def recompute_scalars(vector):
    """Reconstruct the 6 invariant scalars (3 norms + 3 dot products) from the 3 vector features.
    vector: (B, T, vec_dim=3, n_features=3)
    """
    a = vector[..., :, 0]   # (B, T, 3) — accel as 3D vector
    v1 = vector[..., :, 1]  # (B, T, 3)
    v2 = vector[..., :, 2]
    na = torch.norm(a, dim=-1, keepdim=True)
    nv1 = torch.norm(v1, dim=-1, keepdim=True)
    nv2 = torch.norm(v2, dim=-1, keepdim=True)
    av1 = (a * v1).sum(-1, keepdim=True)
    v1v2 = (v1 * v2).sum(-1, keepdim=True)
    av2 = (a * v2).sum(-1, keepdim=True)
    return torch.cat([na, nv1, nv2, av1, v1v2, av2], dim=-1)


def main():
    print("=" * 60)
    print(" VN-EqNIO (eq_so3_frame_fullCov_2vec_2deep) equivariance test")
    print("=" * 60)

    nc = {"in_dim": 6}
    model = get_model("eq_so3_frame_fullCov_2vec_2deep", nc).eval().double()

    B, T = 2, 200
    n_trials = 5
    errors_disp = []
    errors_cov = []
    errors_frame = []

    for t in range(n_trials):
        torch.manual_seed(t + 10)
        # vector: (B, T, vec_dim=3, n_features=3) — three 3D vectors per timestep
        vec = torch.randn(B, T, 3, 3, dtype=torch.float64)
        scalar = recompute_scalars(vec)
        R = random_R()

        with torch.no_grad():
            try:
                frame_o, disp_o, cov_o = model(vec, scalar)
            except Exception as e:
                print(f"  model call failed: {e}")
                return

            vec_R = rotate_vec_field(vec, R)
            scalar_R = recompute_scalars(vec_R)
            scalar_inv_err = (scalar_R - scalar).abs().max().item()

            frame_R, disp_R, cov_R = model(vec_R, scalar_R)

            # Expected: disp_R == R @ disp_o ; cov_R == R @ cov_o @ R^T; frame_R == R @ frame_o
            expected_disp = (R @ disp_o.transpose(-1, -2)).transpose(-1, -2) if disp_o.dim() == 3 else (R @ disp_o.T).T
            # Actually disp shape is (B*T, 3) — flat. Let me handle generically.
            R64 = R
            expected_disp = torch.einsum("ij,...j->...i", R64, disp_o)
            expected_cov = torch.einsum("ij,...jk,lk->...il", R64, cov_o, R64)
            expected_frame = torch.einsum("ij,...jk->...ik", R64, frame_o)

            err_d = (disp_R - expected_disp).abs().max().item()
            err_c = (cov_R - expected_cov).abs().max().item()
            err_f = (frame_R - expected_frame).abs().max().item()

            errors_disp.append(err_d)
            errors_cov.append(err_c)
            errors_frame.append(err_f)

        print(f"  trial {t}: disp_err={err_d:.2e}, cov_err={err_c:.2e}, frame_err={err_f:.2e}, scalar_inv_err={scalar_inv_err:.2e}")

    print()
    max_d = max(errors_disp); max_c = max(errors_cov); max_f = max(errors_frame)
    print(f"  max disp err:  {max_d:.2e}")
    print(f"  max cov err:   {max_c:.2e}")
    print(f"  max frame err: {max_f:.2e}")
    print()
    ok = (max_d < 1e-4 and max_c < 1e-4 and max_f < 1e-4)
    print(" ✓ PASS — VN-EqNIO is SO(3)-equivariant" if ok else " ✗ FAIL — not equivariant")


if __name__ == "__main__":
    main()
