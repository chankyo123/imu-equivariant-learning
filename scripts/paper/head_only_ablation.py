# Paper artefact: head-only Sigma-equivariance comparison (VN-AirIO diagonal head vs SO(3) spectral head) that
# backs the Sigma-eq. discussion of Supplement Table 9; --mode retrain-cmd only prints a recipe (no numbers).
"""Head-only covariance ablation: same backbone + same mean head, different cov head.

There are two execution modes:

  1. EVAL mode (--mode eval): load two existing checkpoints that share the mean head,
     run cov_eq_error + dump (e, Sigma) per checkpoint over a shared test loader.
     This is the fast path the user can run today.

  2. RETRAIN mode (--mode retrain-cmd): does NOT train -- prints the exact training
     command(s) needed to (a) freeze mean head and (b) retrain only the cov head with
     a chosen Sigma parameterization. The user launches these manually on the GPUs.

For mode=eval we use:
  - Diag (VN-AirIO body, propcov=True):
      Nanobench: checkpoints/ginio_a_noatt_nanobench/best_model.ckpt (released GINIO-A; the original
                 run used the unreleased 0.273M experiments/nanobench_noatt_vn checkpoint)
  - Spectral (VN-EqNIO SO(3)):
      Nanobench: TBD (no current Nanobench-trained SO(3) torchscript)
      TLIO:      checkpoints/ginio_e_so3aug_tlio/model_torchscript.pt (released GINIO-E; the original
                 run used the unreleased overnight/_existing_vn_so3_t0012 trace)

Outputs: <out-dir>/head_only_<dataset>_<head>.json (one per model), plus
         <out-dir>/head_only_<dataset>_summary.json combining them.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import AIRIO_ROOT, CKPT_ROOT, resolve_ckpt_dir  # noqa: E402
from _utils import OUTPUT_DIR, fail_if_missing, insert_paths, save_summary  # noqa: E402

insert_paths()


# ---------------------------------------------------------------------------
# Eval mode
# ---------------------------------------------------------------------------

def eval_one(arch: str, ckpt: Path, batch: int, seqlen: int, n_trials: int,
             vn_class: str = "VNCodeNetMotionSmall") -> dict:
    """Run cov_eq_error in-process and return the summary dict."""
    from cov_eq_error import (forward_so3_frame_torchscript,
                              forward_vn_airio_diag, make_airio_inputs,
                              make_so3_inputs, run_eq_test)
    if arch == "vn_airio_diag":
        x = make_airio_inputs(batch, seqlen)
        def fwd(xx):
            return forward_vn_airio_diag(ckpt, xx, vn_class)
        return run_eq_test(fwd, x, scalar=None, vec_axis=-1,
                           n_trials=n_trials, seed=1000)
    if arch == "vn_eqnio_so3_frame":
        vec, scalar = make_so3_inputs(batch, seqlen)
        def fwd(xx, ss):
            return forward_so3_frame_torchscript(ckpt, xx, ss)
        return run_eq_test(fwd, vec, scalar=scalar, vec_axis=-2,
                           n_trials=n_trials, seed=1000)
    raise SystemExit(f"unsupported arch {arch}")


# ---------------------------------------------------------------------------
# Retrain command emit
# ---------------------------------------------------------------------------

RETRAIN_RECIPE = """\
# Head-only retraining recipe (run on a single GPU node).
# Strategy: load a backbone-stable VN model, freeze every parameter except cov head,
# train for ~10 epochs on the same dataset using NLL loss with mean held fixed.
#
# For VN-AirIO (diag-from-norms head, body frame):
cd {airio_root} && \\
python train_motion.py --config {airio_conf} --resume {airio_ckpt} \\
  --freeze-mean --train-cov-only --epochs 10 --lr 1e-4
# NOTE: --freeze-mean is NOT yet implemented in train_motion.py. You need to add a flag
# that requires_grad_=False for net.encoder, net.gru, net.fc and leaves
# net.cov_norm_proj + net.cov_decoder trainable. ~30 lines.
#
# For spectral SO(3) (replace diag-from-norms with spectral head on the same backbone):
# 1) Patch model: in vn_codenet.VNCodeNetMotion(WithRot), add a path that returns
#    (R^T, mu_can, log_std_can) using a VN-Linear -> 3 vectors (R^T) and a VN-Linear -> 3 scalars.
# 2) Add to model/__init__.py: 'vn_codenetmotion_spectral': VNCodeNetMotionSpectral.
# 3) Train with the same dataset/conf but switch network: vn_codenetmotion_spectral.
# (The released airio/ already contains VNSpectralCov / 'vn_noatt_spec_small' for step 1-2.)
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["eval", "retrain-cmd"], required=True)
    p.add_argument("--dataset", choices=["nanobench", "tlio"], default="nanobench")
    p.add_argument("--diag-ckpt", default=None,
                   help="path to VN-AirIO checkpoint (diag head); default: released ginio_a_noatt_nanobench")
    p.add_argument("--vn-class", default="VNCodeNetMotionSmall",
                   choices=["VNCodeNetMotion", "VNCodeNetMotionSmall"],
                   help="vn_codenet class matching --diag-ckpt (Small = released GINIO-A)")
    p.add_argument("--spec-ts", default=None,
                   help="path to VN-EqNIO SO(3) torchscript (spectral head); default for --dataset tlio: released ginio_e_so3aug_tlio")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seqlen", type=int, default=128,
                   help="window length of the random test input; use 200 when --spec-ts is one of the released "
                        "TLIO-trained traces (fixed 200-sample window)")
    p.add_argument("--n-trials", type=int, default=8)
    p.add_argument("--out-dir", default=str(OUTPUT_DIR))
    p.add_argument("--ckpt-root", default=str(CKPT_ROOT))
    p.add_argument("--airio-root", default=str(AIRIO_ROOT))
    p.add_argument("--airio-conf",
                   default="configs/Nanobench/motion_body_noatt_vn_small.conf")
    p.add_argument("--airio-ckpt",
                   default="experiments/nanobench_noatt_vn_small/motion_body_noatt_vn_small/ckpt/best_model.ckpt")
    args = p.parse_args()

    if args.mode == "retrain-cmd":
        print(RETRAIN_RECIPE.format(airio_root=args.airio_root, airio_conf=args.airio_conf,
                                    airio_ckpt=args.airio_ckpt))
        return 0

    # Defaults
    ckpt_root = Path(args.ckpt_root)
    if args.diag_ckpt is None:
        args.diag_ckpt = str(resolve_ckpt_dir("ginio_a_noatt_nanobench", ckpt_root) / "best_model.ckpt")
    if args.spec_ts is None:
        if args.dataset == "tlio":
            args.spec_ts = str(resolve_ckpt_dir("ginio_e_so3aug_tlio", ckpt_root) / "model_torchscript.pt")
        else:
            print("[INFO] no Nanobench-trained SO(3) torchscript known; pass --spec-ts to compare on Nanobench")
            args.spec_ts = None

    runs = {}
    diag_ckpt = fail_if_missing(args.diag_ckpt, "diag ckpt")
    print("=== diag head ===")
    runs["vn_airio_diag"] = eval_one("vn_airio_diag", diag_ckpt,
                                     args.batch, args.seqlen, args.n_trials, args.vn_class)
    if args.spec_ts is not None:
        spec_ts = fail_if_missing(args.spec_ts, "spectral torchscript")
        print("=== spectral head ===")
        runs["vn_eqnio_so3_frame"] = eval_one("vn_eqnio_so3_frame", spec_ts,
                                              args.batch, args.seqlen, args.n_trials)

    text_lines = [f"Head-only ablation  dataset={args.dataset}"]
    for k, v in runs.items():
        text_lines.append(
            f"  {k}: mu_err={v['mu_err_max']:.2e}  "
            f"Sigma_err_rel={v['cov_err_norm_max']:.2e}"
        )
    text = "\n".join(text_lines) + "\n"
    print("\n" + text)
    save_summary(args.out_dir, f"head_only_{args.dataset}_summary",
                 {"runs": runs, "diag_ckpt": str(args.diag_ckpt),
                  "spec_ts": str(args.spec_ts)}, text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
