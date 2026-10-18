# Paper artefact: Sigma-equivariance / mu-equivariance error per covariance-head architecture (the "Sigma-eq. err"
# evidence of Supplement Table 9 for the VN-AirIO diagonal head and the SO(3) spectral / Cholesky heads).
"""Sigma-equivariance / mu-equivariance error per covariance-head architecture.

For each architecture we load a real trained model (or its torchscript trace),
draw N random SO(3) rotations R, run the model on (x, R.x), reconstruct Sigma using
the same code path the TLIO filter uses (see _utils.sigma_from_*), then compute:
  mu-eq error    =  ||mu(R.x) - R.mu(x)||_inf
  Sigma-eq error =  ||Sigma(R.x) - R.Sigma(x).R^T||_F / (||R.Sigma(x).R^T||_F + 1e-12)

Supported --arch values:
  vn_airio_diag       -- Air-IO VN-AirIO body-frame, diag-from-norms head
  vn_eqnio_so3_frame  -- VN-EqNIO SO(3) spectral, torchscript model
  eqnio_o2_fullcov    -- EqNIO O(2), only checked for the 2D analogue (FYI)
  chol                -- Cholesky head (loaded from torchscript)

Outputs: <out-dir>/eq_error_<arch>.json + .txt

Release defaults: vn_airio_diag -> checkpoints/ginio_a_noatt_nanobench/best_model.ckpt with
--vn-class VNCodeNetMotionSmall (the original run used the unreleased 0.273M VNCodeNetMotion
checkpoint); vn_eqnio_so3_frame -> checkpoints/ginio_e_so3aug_tlio/model_torchscript.pt (the original
run used the unreleased overnight/_existing_vn_so3_t0012 trace of the same architecture).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import CKPT_ROOT, resolve_ckpt_dir  # noqa: E402
from _utils import (  # noqa: E402
    OUTPUT_DIR,
    fail_if_missing,
    insert_paths,
    random_rotation,
    save_summary,
    sigma_from_chol,
    sigma_from_o2_fullCov,
    sigma_from_so3_frame,
    sigma_from_vn_airio_diag,
)

insert_paths()


# ---------------------------------------------------------------------------
# Per-arch forward wrappers
# ---------------------------------------------------------------------------

def forward_vn_airio_diag(ckpt_path: Path, x_vec: torch.Tensor,
                          vn_class: str = "VNCodeNetMotionSmall") -> tuple[torch.Tensor, torch.Tensor]:
    """x_vec: (B, T, 2, 3)  with acc=x_vec[...,0,:], gyro=x_vec[...,1,:].
    vn_class selects the model class in airio/model/vn_codenet.py matching the checkpoint
    (VNCodeNetMotionSmall for the released GINIO-A checkpoints)."""
    import model.vn_codenet as vn_codenet
    conf = SimpleNamespace(propcov=True)
    model = getattr(vn_codenet, vn_class)(conf).double().eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"[WARN] missing keys (fresh init): {len(missing)} e.g. {missing[:2]}")
    if unexpected:
        print(f"[WARN] unexpected keys (ignored): {len(unexpected)} e.g. {unexpected[:2]}")
    acc, gyro = x_vec[..., 0, :], x_vec[..., 1, :]
    out = model({"acc": acc, "gyro": gyro})
    mu = out["net_vel"]            # (B, T_out, 3)  in body frame (rotation-equivariant)
    cov_raw = out["cov"]           # (B, T_out, 3) -- already exp(cov_raw - 5)
    # In vn_codenet the "cov" key already returned exp(cov_raw - 5.0), so it's variance.
    Sigma = torch.diag_embed(cov_raw.clamp(min=1e-12))
    return mu, Sigma


def forward_so3_frame_torchscript(ts_path: Path, vector: torch.Tensor, scalar: torch.Tensor):
    """Spectral SO(3) head. Returns (mu_world, Sigma_world)."""
    net = torch.jit.load(str(ts_path), map_location="cpu").eval().double()
    with torch.no_grad():
        frame, mu_can, log_std_can = net(vector, scalar)
    # mu_world = frame^T @ mu_can
    mu_world = torch.einsum("bij,bj->bi", frame.transpose(-2, -1), mu_can)
    Sigma = sigma_from_so3_frame(frame, log_std_can)
    return mu_world, Sigma


def forward_chol_torchscript(ts_path: Path, vector: torch.Tensor, scalar: torch.Tensor):
    net = torch.jit.load(str(ts_path), map_location="cpu").eval().double()
    with torch.no_grad():
        mu, A = net(vector, scalar)
    Sigma = sigma_from_chol(A)
    return mu, Sigma


# ---------------------------------------------------------------------------
# Equivariance loop
# ---------------------------------------------------------------------------

def rotate_input(x: torch.Tensor, R: torch.Tensor, vec_axis: int) -> torch.Tensor:
    if vec_axis == -1:
        return torch.einsum("ij,...j->...i", R, x)
    if vec_axis == -2:
        # Move axis -2 to last, rotate, move back
        x_p = x.transpose(-1, -2)
        x_r = torch.einsum("ij,...j->...i", R, x_p)
        return x_r.transpose(-1, -2)
    raise ValueError(f"unsupported vec_axis {vec_axis}")


def run_eq_test(forward_fn, x: torch.Tensor, scalar: torch.Tensor | None,
                vec_axis: int, n_trials: int, seed: int) -> dict:
    rng_seeds = list(range(seed, seed + n_trials))
    mu_errs, cov_errs, cov_errs_norm = [], [], []

    # Anchor reference forward
    if scalar is None:
        mu0, Sigma0 = forward_fn(x)
    else:
        mu0, Sigma0 = forward_fn(x, scalar)

    for trial, s in enumerate(rng_seeds):
        R = random_rotation(dtype=x.dtype, seed=s)
        x_rot = rotate_input(x, R, vec_axis)
        if scalar is None:
            mu_r, Sigma_r = forward_fn(x_rot)
        else:
            # Scalars are R-invariant (norms, dots) -> identical input
            mu_r, Sigma_r = forward_fn(x_rot, scalar)
        # Expected
        mu_exp = torch.einsum("ij,...j->...i", R, mu0)
        if Sigma0.dim() == 3:    # (B, 3, 3)
            Sigma_exp = torch.einsum("ij,bjk,lk->bil", R, Sigma0, R)
        else:                    # (B, T, 3, 3)
            Sigma_exp = torch.einsum("ij,btjk,lk->btil", R, Sigma0, R)

        mu_e = (mu_r - mu_exp).abs().max().item()
        cov_e_abs = (Sigma_r - Sigma_exp).abs().max().item()
        denom = Sigma_exp.flatten(-2).norm(dim=-1).clamp(min=1e-12)
        cov_e_norm = ((Sigma_r - Sigma_exp).flatten(-2).norm(dim=-1) / denom).max().item()
        mu_errs.append(mu_e)
        cov_errs.append(cov_e_abs)
        cov_errs_norm.append(cov_e_norm)
        print(f"  trial {trial:>2d}  mu_err={mu_e:.2e}  Sigma_err={cov_e_abs:.2e}  Sigma_err_norm={cov_e_norm:.2e}")
    return {
        "mu_err_max": max(mu_errs),
        "cov_err_max": max(cov_errs),
        "cov_err_norm_max": max(cov_errs_norm),
        "mu_err_per_trial": mu_errs,
        "cov_err_per_trial": cov_errs,
        "cov_err_norm_per_trial": cov_errs_norm,
    }


# ---------------------------------------------------------------------------
# Inputs for each arch
# ---------------------------------------------------------------------------

# (released folder per checkpoints/MANIFEST.md, file); resolved under --ckpt-root at run time
DEFAULT_CKPTS = {
    "vn_airio_diag": ("ginio_a_noatt_nanobench", "best_model.ckpt"),
    "vn_eqnio_so3_frame": ("ginio_e_so3aug_tlio", "model_torchscript.pt"),
    "chol": None,  # user must supply
}


def make_so3_inputs(B: int, T: int, dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (vector, scalar) inputs for SO(3) torchscript.

    vector: (B, T, 3, 3) -- three 3-d vectors per timestep
    scalar: (B, T, 6) -- R-invariant scalars (norms + dot products)
    """
    torch.manual_seed(0)
    vec = torch.randn(B, T, 3, 3, dtype=dtype)
    a, v1, v2 = vec[..., 0], vec[..., 1], vec[..., 2]
    na = a.norm(dim=-1, keepdim=True)
    nv1 = v1.norm(dim=-1, keepdim=True)
    nv2 = v2.norm(dim=-1, keepdim=True)
    av1 = (a * v1).sum(-1, keepdim=True)
    v1v2 = (v1 * v2).sum(-1, keepdim=True)
    av2 = (a * v2).sum(-1, keepdim=True)
    scalar = torch.cat([na, nv1, nv2, av1, v1v2, av2], dim=-1)
    return vec, scalar


def make_airio_inputs(B: int, T: int, dtype=torch.float64) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T, 2, 3, dtype=dtype)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True,
                   choices=["vn_airio_diag", "vn_eqnio_so3_frame", "chol"])
    p.add_argument("--ckpt", type=str, default=None,
                   help="checkpoint / torchscript path (default: released checkpoint for the arch)")
    p.add_argument("--vn-class", type=str, default="VNCodeNetMotionSmall",
                   choices=["VNCodeNetMotion", "VNCodeNetMotionSmall"],
                   help="airio/model/vn_codenet.py class for --arch vn_airio_diag "
                        "(Small = released GINIO-A checkpoints)")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seqlen", type=int, default=128,
                   help="window length of the random test input; TorchScript traces have a fixed window, "
                        "use 200 for the released TLIO-trained traces (ginio_e_so3aug_tlio, eqnio_o2_tlio_ours)")
    p.add_argument("--n-trials", type=int, default=8)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--ckpt-root", type=str, default=str(CKPT_ROOT), help="released checkpoint folders")
    args = p.parse_args()

    ckpt = args.ckpt
    if ckpt is None and DEFAULT_CKPTS[args.arch] is not None:
        folder, fname = DEFAULT_CKPTS[args.arch]
        ckpt = str(resolve_ckpt_dir(folder, args.ckpt_root) / fname)
    if ckpt is None:
        raise SystemExit(f"--ckpt required for arch={args.arch}")
    ckpt = fail_if_missing(ckpt, f"checkpoint for {args.arch}")

    if args.arch == "vn_airio_diag":
        x = make_airio_inputs(args.batch, args.seqlen)
        # Wrap so signature matches
        def fwd(xx):
            return forward_vn_airio_diag(ckpt, xx, args.vn_class)
        result = run_eq_test(fwd, x, scalar=None, vec_axis=-1,
                             n_trials=args.n_trials, seed=args.seed)
    elif args.arch == "vn_eqnio_so3_frame":
        vec, scalar = make_so3_inputs(args.batch, args.seqlen)
        # Some traced SO(3) models accept (B, N, 3, 3) where N is concatenated over time;
        # flatten B*T into N inside a wrapper.
        B, T = vec.shape[:2]
        vec_in = vec.reshape(B, T, 3, 3)
        scl_in = scalar.reshape(B, T, 6)
        def fwd(xx, ss):
            return forward_so3_frame_torchscript(ckpt, xx, ss)
        result = run_eq_test(fwd, vec_in, scalar=scl_in, vec_axis=-2,
                             n_trials=args.n_trials, seed=args.seed)
    elif args.arch == "chol":
        vec, scalar = make_so3_inputs(args.batch, args.seqlen)
        def fwd(xx, ss):
            return forward_chol_torchscript(ckpt, xx, ss)
        result = run_eq_test(fwd, vec, scalar=scalar, vec_axis=-2,
                             n_trials=args.n_trials, seed=args.seed)
    else:
        raise SystemExit(f"unknown arch {args.arch}")

    text = (
        f"Sigma-equivariance check  arch={args.arch}\n"
        f"  ckpt: {ckpt}\n"
        f"  batch={args.batch} seqlen={args.seqlen} trials={args.n_trials} dtype=float64\n"
        f"  mu-eq max error         : {result['mu_err_max']:.3e}\n"
        f"  Sigma-eq max abs error  : {result['cov_err_max']:.3e}\n"
        f"  Sigma-eq max rel error  : {result['cov_err_norm_max']:.3e}\n"
    )
    print("\n" + text)
    save_summary(args.out_dir, f"eq_error_{args.arch}",
                 {"arch": args.arch, "ckpt": str(ckpt), **result}, text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
