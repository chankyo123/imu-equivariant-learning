"""
Numerical equivariance verification for VN ops + VN-GRU.

For each module, apply a random rotation R ∈ SO(3) to the input vectors,
forward through the module, then check that the output equals R applied to
the unrotated forward output (up to float32 numerical error ~1e-5).

Run:
    python test_vn_equivariance.py

Pass criterion: ALL tests show max_error < 1e-4 (typically 1e-6 to 1e-5).
"""

import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

# Local imports
from model.vn_gru import VNLinear, VNLeakyReLU, VNNorm, VNGRUCell, VNGRU


def random_rotation_matrix(device, dtype):
    """Generate a single random SO(3) matrix as a torch tensor."""
    R_np = Rotation.random().as_matrix()           # (3, 3) numpy
    return torch.tensor(R_np, device=device, dtype=dtype)


def rotate_vec_field(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Apply R to vector axis of x.
    x: (..., C, 3, ...)  — the 3 axis is the vector axis (typically at index -2 or -1)
    R: (3, 3)

    Rule: vector y[i] = R @ x[i], i.e. transform along the size-3 axis.
    """
    if x.dim() == 3:           # (B, C, 3)
        return torch.einsum("ij,bcj->bci", R, x)
    elif x.dim() == 4:
        # Could be (B, C, 3, T) or (B, T, C, 3) — detect by checking which axis is 3
        if x.shape[2] == 3:
            return torch.einsum("ij,bcjt->bcit", R, x)
        elif x.shape[3] == 3:
            return torch.einsum("ij,btcj->btci", R, x)
        else:
            raise ValueError(f"No vector axis of size 3 in {x.shape}")
    else:
        raise ValueError(f"Unsupported {x.shape}")


def max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def test_module(name: str, module: nn.Module, x_shape, n_trials: int = 5, atol: float = 1e-4):
    """Apply random R; check module(R x) == R module(x)."""
    print(f"\n>>> {name}")
    device = next(module.parameters()).device if any(True for _ in module.parameters()) else "cpu"
    dtype = torch.float64   # higher precision for testing
    module.to(device=device, dtype=dtype).eval()
    errors = []
    for trial in range(n_trials):
        torch.manual_seed(trial)
        x = torch.randn(*x_shape, device=device, dtype=dtype)
        R = random_rotation_matrix(device, dtype)
        with torch.no_grad():
            y = module(x)
            y_from_rot = module(rotate_vec_field(x, R))
            y_rot_then = rotate_vec_field(y, R)
        err = max_err(y_from_rot, y_rot_then)
        errors.append(err)
    max_e = max(errors)
    status = "✓ PASS" if max_e < atol else "✗ FAIL"
    print(f"    {status}  max_error over {n_trials} random R: {max_e:.2e}  (threshold {atol:.0e})")
    print(f"    individual: " + ", ".join(f"{e:.2e}" for e in errors))
    return max_e < atol


def test_vngru_cell(atol: float = 1e-4):
    """Test that VNGRUCell is equivariant w.r.t. simultaneous rotation of x and h."""
    print(f"\n>>> VNGRUCell")
    torch.manual_seed(0)
    B, C_in, C_h = 4, 6, 8
    cell = VNGRUCell(C_in, C_h, gate_norm_channels=16).double().eval()

    errors = []
    for trial in range(5):
        torch.manual_seed(trial + 100)
        x = torch.randn(B, C_in, 3, dtype=torch.float64)
        h = torch.randn(B, C_h, 3, dtype=torch.float64)
        R = random_rotation_matrix("cpu", torch.float64)
        with torch.no_grad():
            y_orig = cell(x, h)
            y_after_rot = cell(rotate_vec_field(x, R), rotate_vec_field(h, R))
            y_rot_then = rotate_vec_field(y_orig, R)
        err = max_err(y_after_rot, y_rot_then)
        errors.append(err)
    max_e = max(errors)
    status = "✓ PASS" if max_e < atol else "✗ FAIL"
    print(f"    {status}  max_error: {max_e:.2e}  (threshold {atol:.0e})")
    print(f"    individual: " + ", ".join(f"{e:.2e}" for e in errors))
    return max_e < atol


def test_vngru_seq(atol: float = 1e-4, bidirectional: bool = False):
    """Test that VNGRU sequence layer is equivariant w.r.t. rotation of input sequence."""
    print(f"\n>>> VNGRU (bidirectional={bidirectional})")
    torch.manual_seed(0)
    B, T, C_in, C_h = 2, 8, 6, 8
    gru = VNGRU(input_size=C_in, hidden_size=C_h, num_layers=2, bidirectional=bidirectional).double().eval()

    errors = []
    for trial in range(3):
        torch.manual_seed(trial + 200)
        # Input is (B, T, C, 3) — vector axis at -1
        x = torch.randn(B, T, C_in, 3, dtype=torch.float64)
        R = random_rotation_matrix("cpu", torch.float64)

        with torch.no_grad():
            y_orig = gru(x)                          # (B, T, C_out, 3)
            x_rot = rotate_vec_field(x, R)           # rotate vector axis
            y_after_rot = gru(x_rot)
            y_rot_then = rotate_vec_field(y_orig, R)
        err = max_err(y_after_rot, y_rot_then)
        errors.append(err)
    max_e = max(errors)
    status = "✓ PASS" if max_e < atol else "✗ FAIL"
    print(f"    {status}  max_error: {max_e:.2e}")
    print(f"    individual: " + ", ".join(f"{e:.2e}" for e in errors))
    return max_e < atol


if __name__ == "__main__":
    print("=" * 60)
    print(" VN-GRU equivariance verification")
    print("=" * 60)

    all_pass = True

    # Test basic building blocks first
    all_pass &= test_module("VNLinear (3D shape)", VNLinear(8, 16), (4, 8, 3))
    all_pass &= test_module("VNLinear (4D shape with T)", VNLinear(8, 16), (4, 8, 3, 10))
    all_pass &= test_module("VNLeakyReLU (3D)", VNLeakyReLU(8, negative_slope=0.0), (4, 8, 3))
    all_pass &= test_module("VNLeakyReLU (4D)", VNLeakyReLU(8, negative_slope=0.0), (4, 8, 3, 10))

    # VNNorm: should be invariant — output unchanged under rotation
    print("\n>>> VNNorm (invariance check, not equivariance)")
    torch.manual_seed(0)
    x = torch.randn(4, 8, 3, dtype=torch.float64)
    R = random_rotation_matrix("cpu", torch.float64)
    n1 = VNNorm()(x)
    n2 = VNNorm()(rotate_vec_field(x, R))
    err = max_err(n1, n2)
    print(f"    {'✓ PASS' if err < 1e-10 else '✗ FAIL'}  norm-invariance error: {err:.2e}")
    all_pass &= (err < 1e-10)

    # GRU cell + sequential
    all_pass &= test_vngru_cell()
    all_pass &= test_vngru_seq(bidirectional=False)
    all_pass &= test_vngru_seq(bidirectional=True)

    print()
    print("=" * 60)
    if all_pass:
        print(" 🎉 ALL EQUIVARIANCE TESTS PASSED")
    else:
        print(" ❌ SOME TESTS FAILED — review VN-GRU implementation")
    print("=" * 60)
