"""
Full-model equivariance test for VNCodeNetMotion (no attitude) and VNCodeNetMotionwithRot.

Rotates input acc/gyro (and rot log-map) by random R, checks output velocity rotates
correspondingly.

Pass criterion: max error < 1e-4 on float64.
"""

import torch
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
from vn_codenet import VNCodeNetMotion, VNCodeNetMotionwithRot


def random_R(dtype=torch.float64):
    return torch.tensor(Rotation.random().as_matrix(), dtype=dtype)


def test_vn_codenetmotion(propcov: bool = True, atol: float = 1e-4):
    """Test no-attitude VN-AirIO. Input: acc, gyro per timestep."""
    print(f"\n>>> VNCodeNetMotion (propcov={propcov})")
    conf = SimpleNamespace(propcov=propcov)
    model = VNCodeNetMotion(conf).double().eval()

    B, T = 2, 200  # nanobench window_size
    errors = []
    for trial in range(5):
        torch.manual_seed(trial + 50)
        acc = torch.randn(B, T, 3, dtype=torch.float64)
        gyro = torch.randn(B, T, 3, dtype=torch.float64)
        R = random_R()

        with torch.no_grad():
            out_orig = model({"acc": acc, "gyro": gyro})
            # Rotate inputs: each vector axis is the last dim of (B,T,3)
            acc_R = torch.einsum("ij,btj->bti", R, acc)
            gyro_R = torch.einsum("ij,btj->bti", R, gyro)
            out_rot = model({"acc": acc_R, "gyro": gyro_R})

            # Expected: net_vel rotated by R; cov unchanged (invariant)
            expected_vel = torch.einsum("ij,btj->bti", R, out_orig["net_vel"])
        vel_err = (out_rot["net_vel"] - expected_vel).abs().max().item()
        errors.append(vel_err)
        if propcov:
            cov_err = (out_rot["cov"] - out_orig["cov"]).abs().max().item()
            errors.append(cov_err)

    max_e = max(errors)
    status = "✓ PASS" if max_e < atol else "✗ FAIL"
    print(f"    {status}  max_error: {max_e:.2e}  (threshold {atol:.0e})")
    print(f"    individual: " + ", ".join(f"{e:.2e}" for e in errors))
    return max_e < atol


def test_vn_codenetmotionwithrot(propcov: bool = True, atol: float = 1e-4):
    """Test attitude-aware VN-AirIO. Input: acc, gyro, rot log-map per timestep."""
    print(f"\n>>> VNCodeNetMotionwithRot (propcov={propcov})")
    conf = SimpleNamespace(propcov=propcov)
    model = VNCodeNetMotionwithRot(conf).double().eval()

    B, T = 2, 200
    errors = []
    for trial in range(5):
        torch.manual_seed(trial + 60)
        acc = torch.randn(B, T, 3, dtype=torch.float64)
        gyro = torch.randn(B, T, 3, dtype=torch.float64)
        rot_log = torch.randn(B, T, 3, dtype=torch.float64)  # log-map of attitude
        R = random_R()

        with torch.no_grad():
            out_orig = model({"acc": acc, "gyro": gyro}, rot=rot_log)
            acc_R = torch.einsum("ij,btj->bti", R, acc)
            gyro_R = torch.einsum("ij,btj->bti", R, gyro)
            rot_R = torch.einsum("ij,btj->bti", R, rot_log)
            out_rot = model({"acc": acc_R, "gyro": gyro_R}, rot=rot_R)

            expected_vel = torch.einsum("ij,btj->bti", R, out_orig["net_vel"])
        vel_err = (out_rot["net_vel"] - expected_vel).abs().max().item()
        errors.append(vel_err)
        if propcov:
            cov_err = (out_rot["cov"] - out_orig["cov"]).abs().max().item()
            errors.append(cov_err)

    max_e = max(errors)
    status = "✓ PASS" if max_e < atol else "✗ FAIL"
    print(f"    {status}  max_error: {max_e:.2e}  (threshold {atol:.0e})")
    print(f"    individual: " + ", ".join(f"{e:.2e}" for e in errors))
    return max_e < atol


if __name__ == "__main__":
    print("=" * 60)
    print(" Full-model VN-AirIO equivariance verification")
    print("=" * 60)

    ok = True
    ok &= test_vn_codenetmotion(propcov=True)
    ok &= test_vn_codenetmotion(propcov=False)
    ok &= test_vn_codenetmotionwithrot(propcov=True)
    ok &= test_vn_codenetmotionwithrot(propcov=False)

    print()
    print("=" * 60)
    print(" 🎉 ALL FULL-MODEL EQUIVARIANCE TESTS PASSED" if ok else " ❌ FAILED")
    print("=" * 60)
