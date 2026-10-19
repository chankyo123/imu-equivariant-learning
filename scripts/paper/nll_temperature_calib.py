# Paper artefact: single-scalar temperature calibration of a covariance head from a val/test (residual, Sigma) pair
# (the alpha-calibrated NLL of Supplement Table 9; nll_decompose_calib.py is the batch version on Phase-A dumps).
"""Single-scalar covariance temperature calibration.

Fit alpha > 0 on a VAL set minimizing val NLL with Sigma_alpha = alpha Sigma_hat.
Then report calibrated TEST NLL using that alpha.

Why: defends against models that game NLL via init scale (a small constant Sigma_hat
multiplier can swing NLL meaningfully without changing geometry). With alpha absorbed,
two heads are compared on shape only.

Inputs:
  --val-npz   .npz with (e, Sigma) for the validation set
  --test-npz  .npz with (e, Sigma) for the test set
  --name      run name
Outputs:
  temp_calib_<name>.json/.txt  with alpha*, raw NLL, calibrated NLL
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import OUTPUT_DIR, fail_if_missing, jsonsafe, nll_terms, save_summary  # noqa: E402


def load_pred(path: Path):
    d = np.load(path)
    if {"e", "Sigma"}.issubset(set(d.files)):
        return d["e"].astype(np.float64), d["Sigma"].astype(np.float64)
    if {"pred", "gt", "Sigma"}.issubset(set(d.files)):
        return (d["pred"] - d["gt"]).astype(np.float64), d["Sigma"].astype(np.float64)
    if {"err", "Sigma"}.issubset(set(d.files)):   # run_phase_ab*.py dumps
        return d["err"].astype(np.float64), d["Sigma"].astype(np.float64)
    raise KeyError(f"need (e,Sigma), (err,Sigma) or (pred,gt,Sigma) in {path}")


def mean_nll(alpha: float, e: np.ndarray, Sigma: np.ndarray) -> float:
    if alpha <= 0:
        return np.inf
    terms = nll_terms(e, alpha * Sigma)
    return float(np.mean(terms["nll"]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-npz", required=True)
    p.add_argument("--test-npz", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--alpha-bounds", nargs=2, type=float, default=(1e-3, 1e3))
    p.add_argument("--out-dir", default=str(OUTPUT_DIR))
    args = p.parse_args()

    e_v, S_v = load_pred(fail_if_missing(args.val_npz, "val-npz"))
    e_t, S_t = load_pred(fail_if_missing(args.test_npz, "test-npz"))

    # Optimize over log alpha
    res = minimize_scalar(
        lambda log_alpha: mean_nll(float(np.exp(log_alpha)), e_v, S_v),
        bounds=(float(np.log(args.alpha_bounds[0])), float(np.log(args.alpha_bounds[1]))),
        method="bounded",
        options={"xatol": 1e-5},
    )
    alpha_star = float(np.exp(res.x))
    val_nll_raw = mean_nll(1.0, e_v, S_v)
    val_nll_cal = mean_nll(alpha_star, e_v, S_v)
    test_nll_raw = mean_nll(1.0, e_t, S_t)
    test_nll_cal = mean_nll(alpha_star, e_t, S_t)

    summary = {
        "alpha_star": alpha_star,
        "val_nll_raw": val_nll_raw,
        "val_nll_calibrated": val_nll_cal,
        "test_nll_raw": test_nll_raw,
        "test_nll_calibrated": test_nll_cal,
        "n_val": int(e_v.shape[0]),
        "n_test": int(e_t.shape[0]),
    }
    text = (
        f"Temperature calibration  {args.name}\n"
        f"  alpha*  = {alpha_star:.4f}\n"
        f"  VAL  NLL raw -> cal: {val_nll_raw:.4f} -> {val_nll_cal:.4f}\n"
        f"  TEST NLL raw -> cal: {test_nll_raw:.4f} -> {test_nll_cal:.4f}\n"
    )
    print("\n" + text)
    save_summary(args.out_dir, f"temp_calib_{args.name}", jsonsafe(summary), text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
