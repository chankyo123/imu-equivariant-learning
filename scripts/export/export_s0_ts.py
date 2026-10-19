#!/usr/bin/env python3
"""Export a slope-0 speccov checkpoint (arch ln189_speccov_s0 / ln189_speccov_s0_diag) as a
filter-ready TorchScript module with the SCEKF's expected (mean, pred_log_std) signature.

Usage:
  python scripts/export/export_s0_ts.py <ckpt> <out.pt> <spec|diag>
    spec -> spectral head, diag-extracted (Sigma=V diag(e^2inv) V^T -> pred_log_std = 0.5 log diag)
    diag -> diagonal head, pred_log_std = clamped diag log-std
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


class SpecDiag(nn.Module):
    def __init__(s, n): super().__init__(); s.net = n
    def forward(s, x):
        mean, V, inv = s.net(x); var = torch.exp(2.0 * inv)
        ds = torch.sum(V ** 2 * var.unsqueeze(1), dim=2)
        return mean, 0.5 * torch.log(ds.clamp(min=1e-12))


class DiagW(nn.Module):
    def __init__(s, n): super().__init__(); s.net = n
    def forward(s, x): return s.net(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="checkpoint_best.pt of an ln189_speccov_s0(_diag) run")
    ap.add_argument("out", help="output TorchScript path (model_torchscript.pt)")
    ap.add_argument("kind", choices=["spec", "diag"])
    ap.add_argument("--window", type=int, default=200, help="trace window length (samples)")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = ap.parse_args()

    dev = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    head = 'spectral' if args.kind == 'spec' else 'diag'
    net = SO3EquivariantReluBracketLayers_speccov_s0(LN_BasicBlock1D, 6, 3, [2, 2, 2, 2],
                                                     args.window // 32 + 1, cov_head=head)
    sd = torch.load(args.ckpt, map_location=dev, weights_only=False)
    net.load_state_dict(sd.get("model_state_dict", sd), strict=True)

    wrap = (SpecDiag(net) if args.kind == 'spec' else DiagW(net)).eval().to(dev)
    tr = torch.jit.trace(wrap, torch.zeros(1, 6, args.window, device=dev))
    tr.save(args.out)
    print(f"wrote {args.out} ({args.kind})")


if __name__ == "__main__":
    main()
