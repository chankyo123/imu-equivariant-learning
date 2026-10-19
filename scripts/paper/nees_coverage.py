# Paper artefact: NEES mean and chi-square(3) coverage of a (residual, Sigma) dump -- the calibration columns of
# Supplement Table 9 and the coverage curve of Figure 7 for a single model dump.
"""NEES + chi2_3 coverage calibration.

For each sample:
  NEES_i = e_i^T Sigma_i^{-1} e_i

Reports:
  - mean NEES (should be 3 for perfectly calibrated 3-D Gaussian)
  - empirical coverage at q in {0.5, 0.7, 0.9, 0.95}
  - calibration curve: fraction of samples below chi2_3 quantile vs target

Outputs:
  <out-dir>/nees_<name>.json/.txt
  <out-dir>/coverage_<name>.png  (matplotlib)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import OUTPUT_DIR, fail_if_missing, jsonsafe, nll_terms, save_summary  # noqa: E402


def load_pred(path: Path):
    d = np.load(path)
    if {"e", "Sigma"}.issubset(set(d.files)):
        return d["e"].astype(np.float64), d["Sigma"].astype(np.float64)
    if {"pred", "gt", "Sigma"}.issubset(set(d.files)):
        return (d["pred"] - d["gt"]).astype(np.float64), d["Sigma"].astype(np.float64)
    if {"err", "Sigma"}.issubset(set(d.files)):   # run_phase_ab*.py dumps store the residual as "err"
        return d["err"].astype(np.float64), d["Sigma"].astype(np.float64)
    raise KeyError(f"need (e,Sigma), (err,Sigma) or (pred,gt,Sigma) in {path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-npz", required=True,
                   help="npz with (e|err, Sigma) or (pred, gt, Sigma); e.g. an outputs/dumps/*.npz file")
    p.add_argument("--name", required=True)
    p.add_argument("--quantiles", nargs="+", type=float,
                   default=[0.5, 0.7, 0.9, 0.95])
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--out-dir", default=str(OUTPUT_DIR))
    args = p.parse_args()

    e, Sigma = load_pred(fail_if_missing(args.pred_npz, "pred-npz"))
    terms = nll_terms(e, Sigma)
    nees = terms["mahal"]  # x^T Sigma^-1 x === mahal term
    dof = e.shape[1]

    coverage = {}
    for q in args.quantiles:
        thr = chi2.ppf(q, df=dof)
        coverage[f"target={q:.2f}"] = {
            "target": q,
            "chi2_threshold": float(thr),
            "empirical": float(np.mean(nees <= thr)),
        }

    summary = {
        "n": int(nees.size),
        "dof": int(dof),
        "nees_mean": float(np.mean(nees)),
        "nees_median": float(np.median(nees)),
        "ideal_mean": float(dof),
        "coverage": coverage,
    }

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            qs = np.linspace(0.01, 0.99, 99)
            thrs = chi2.ppf(qs, df=dof)
            emp = np.array([np.mean(nees <= t) for t in thrs])
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
            ax.plot(qs, emp, label=args.name)
            ax.set_xlabel("target coverage  (chi2_3 quantile)")
            ax.set_ylabel("empirical coverage")
            ax.set_title(f"NEES coverage  {args.name}")
            ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
            png_path = Path(args.out_dir) / f"coverage_{args.name}.png"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout(); fig.savefig(png_path, dpi=120); plt.close(fig)
            print(f"[saved] {png_path}")
        except Exception as ex:
            print(f"[WARN] coverage plot skipped: {ex}")

    text = (
        f"NEES / coverage  {args.name}\n"
        f"  N={summary['n']}  dof={summary['dof']}\n"
        f"  mean NEES = {summary['nees_mean']:.4f}  (ideal {summary['ideal_mean']:.1f})\n"
        + "".join(
            f"  cov@{c['target']:.2f}: empirical={c['empirical']:.3f} (thr={c['chi2_threshold']:.2f})\n"
            for c in coverage.values()
        )
    )
    print("\n" + text)
    save_summary(args.out_dir, f"nees_{args.name}", jsonsafe(summary), text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
