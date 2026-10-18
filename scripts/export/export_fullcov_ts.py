#!/usr/bin/env python3
"""Export a slope-0 speccov checkpoint (arch ln189_speccov_s0 / ln189_speccov_s0_diag) as a
TorchScript module emitting the FULL 3x3 measurement covariance. Pairs with the dim()==3
passthrough in src/tracker/meas_source_torchscript.py (Fig. 5 spectral-vs-diagonal filter runs).

Usage:
  python scripts/export/export_fullcov_ts.py <ckpt> <out.pt> <spec|specdiag|diag>
    spec     -> Sigma = V diag(exp(2*inv)) V^T          (equivariant, WITH off-diagonals)
    specdiag -> Sigma = diag( diagonal of the above )   (SAME model, off-diagonals ZEROED: clean ablation)
    diag     -> Sigma = diag(exp(2*diag_logstd))        (separately-trained diagonal head)
The exported module returns (mean, cov3x3).
Ported from Server B (chankyokim-Alienware); the sys.path is now resolved from the repo root.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from network.ln_conv_2regress_nochannelmix_bias_speccov_slope0 import SO3EquivariantReluBracketLayers_speccov_s0  # noqa: E402
from network.ln_conv_2regress_nochannelmix_bias import LN_BasicBlock1D  # noqa: E402


class SpecFull(nn.Module):
    def __init__(s, n): super().__init__(); s.net = n
    def forward(s, x):
        mean, V, inv = s.net(x); var = torch.exp(2.0 * inv)
        cov = torch.einsum('bij,bj,bkj->bik', V, var, V)
        return mean, cov


class SpecDiagOnly(nn.Module):   # same spectral model, off-diagonals removed
    def __init__(s, n): super().__init__(); s.net = n
    def forward(s, x):
        mean, V, inv = s.net(x); var = torch.exp(2.0 * inv)
        diagv = torch.sum(V ** 2 * var.unsqueeze(1), dim=2)   # diag of V diag(var) V^T
        return mean, torch.diag_embed(diagv)


class DiagFull(nn.Module):
    def __init__(s, n): super().__init__(); s.net = n
    def forward(s, x):
        mean, logstd = s.net(x)
        return mean, torch.diag_embed(torch.exp(2.0 * logstd))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="checkpoint_best.pt of an ln189_speccov_s0(_diag) run")
    ap.add_argument("out", help="output TorchScript path (model_torchscript.pt)")
    ap.add_argument("kind", choices=["spec", "specdiag", "diag"])
    ap.add_argument("--window", type=int, default=200, help="trace window length (samples)")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = ap.parse_args()

    dev = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    head = 'spectral' if args.kind in ('spec', 'specdiag') else 'diag'
    net = SO3EquivariantReluBracketLayers_speccov_s0(LN_BasicBlock1D, 6, 3, [2, 2, 2, 2],
                                                     args.window // 32 + 1, cov_head=head)
    sd = torch.load(args.ckpt, map_location=dev, weights_only=False)
    net.load_state_dict(sd.get("model_state_dict", sd), strict=True)

    wrap = {'spec': SpecFull, 'specdiag': SpecDiagOnly, 'diag': DiagFull}[args.kind](net).eval().to(dev)
    tr = torch.jit.trace(wrap, torch.zeros(1, 6, args.window, device=dev))
    tr.save(args.out)
    with torch.no_grad():
        m, c = wrap(torch.randn(1, 6, args.window, device=dev))
        offdiag = float((c[0] - torch.diag(torch.diag(c[0]))).abs().max())
    print(f"wrote {args.out} ({args.kind}) cov{tuple(c.shape)} max_offdiag={offdiag:.3e}")


if __name__ == "__main__":
    main()
