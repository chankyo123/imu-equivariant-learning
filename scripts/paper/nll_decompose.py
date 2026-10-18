# Paper artefact: Gaussian NLL decomposition (Mahalanobis / log-det / total) of a single (residual, Sigma) dump --
# the NLL columns of Supplement Table 9 for one model.
"""Decompose Gaussian NLL into Mahalanobis term + log-det term + constant.

Input: an .npz containing
    e     : (N, 3)  residual = pred - gt      (run_phase_ab*.py dumps store it as "err")
    Sigma : (N, 3, 3)  predicted covariance (world frame, post-reconstruction)
or alternatively (pred, gt, Sigma) -- we compute e ourselves.

Output:
  per-sample CSV with [idx, mahal, logdet, nll]
  summary JSON+txt with mean/median/percentiles of each term
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import OUTPUT_DIR, fail_if_missing, jsonsafe, nll_terms, save_summary  # noqa: E402


def load_pred_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    keys = set(d.files)
    if {"e", "Sigma"}.issubset(keys):
        return d["e"].astype(np.float64), d["Sigma"].astype(np.float64)
    if {"pred", "gt", "Sigma"}.issubset(keys):
        return (d["pred"] - d["gt"]).astype(np.float64), d["Sigma"].astype(np.float64)
    if {"err", "Sigma"}.issubset(keys):
        return d["err"].astype(np.float64), d["Sigma"].astype(np.float64)
    raise KeyError(f"need {{e,Sigma}}, {{err,Sigma}} or {{pred,gt,Sigma}} in {path}; got {sorted(keys)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-npz", required=True,
                   help="path to .npz with keys (e|err, Sigma) or (pred, gt, Sigma)")
    p.add_argument("--name", required=True, help="run name (used for output filename)")
    p.add_argument("--out-dir", default=str(OUTPUT_DIR))
    args = p.parse_args()

    e, Sigma = load_pred_npz(fail_if_missing(args.pred_npz, "pred-npz"))
    terms = nll_terms(e, Sigma)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"nll_decomp_{args.name}.csv"
    with open(csv_path, "w") as f:
        f.write("idx,mahal,logdet,nll\n")
        for i, (m, ld, nl) in enumerate(zip(terms["mahal"], terms["logdet"], terms["nll"])):
            f.write(f"{i},{m:.6e},{ld:.6e},{nl:.6e}\n")
    print(f"[saved] {csv_path}")

    def stats(a: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)),
        }
    summary = {k: stats(terms[k]) for k in ("mahal", "logdet", "nll")}
    summary["n"] = int(e.shape[0])
    summary["dim"] = int(e.shape[1])
    text = (
        f"NLL decomposition  {args.name}\n"
        f"  N = {summary['n']}\n"
        f"  mahal  mean={summary['mahal']['mean']:.4f}  median={summary['mahal']['median']:.4f}\n"
        f"  logdet mean={summary['logdet']['mean']:.4f}  median={summary['logdet']['median']:.4f}\n"
        f"  NLL    mean={summary['nll']['mean']:.4f}    median={summary['nll']['median']:.4f}\n"
    )
    print("\n" + text)
    save_summary(args.out_dir, f"nll_decomp_{args.name}",
                 {"pred_npz": str(args.pred_npz), **jsonsafe(summary)}, text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
