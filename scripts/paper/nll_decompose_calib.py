# Paper artefact: TABLE_A_nll_decomp_calib.json -- NLL decomposition, alpha-temperature calibration, NEES and
# coverage of the four Lie-Neuron models on the Phase-A clean dumps (calibration columns of Supplement Table 9).
"""
NLL decomposition + alpha-temperature calibration on Phase A clean dumps.

For each (model) we have per-window dumps with (mu, Sigma, e=mu-gt). Compute:

  M_w   = e^T Sigma^-1 e                            (Mahalanobis per window)
  D_w   = log|Sigma|                                (log-det per window)
  NLL_w = 0.5 * (M_w + D_w + 3 log 2pi)             (Gaussian neg log-likelihood)

Then split test seqs 50/50 into val and test (deterministically by name).
On val: fit alpha* = argmin_alpha  NLL(alpha*Sigma) = argmin  0.5*(M/alpha + D + 3 log alpha + 3 log 2pi)
                                  -> alpha* = mean_val(M) / 3   (analytic)
Report:
  raw test NLL,
  test NLL with Sigma <- alpha* * Sigma (temperature-calibrated),
  test NEES (= M) mean, coverage at p in {0.5,0.7,0.9,0.95} before/after calibration.

This isolates "is the Sigma shape good?" (M) from "is the Sigma magnitude good?" (alpha).
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import OUTPUT_DIR, TEST_LIST_FILE, read_test_list  # noqa: E402

MODELS = ['ln_resnet_lfa_MSE', 'ln_resnet_lfa_NLL', 'ln_resnet_chol_MSE', 'ln_resnet_chol_NLL']
SCENARIO = 'clean'; SEED = 'clean'


def split_val_test(test_list_file=TEST_LIST_FILE):
    # Deterministic val/test split: every other seq goes to val
    SEQS = read_test_list(test_list_file)
    SEQS_SORTED = sorted(SEQS)
    return SEQS_SORTED[0::2], SEQS_SORTED[1::2]


def load_dump(model, seq, dumps):
    p = dumps / f'{model}__{SCENARIO}__{SEED}__{seq}.npz'
    if not p.exists(): return None
    z = np.load(p)
    return dict(mu=z['mu'], Sigma=z['Sigma'], gt=z['gt'], err=z['err'])


def compute_stats(err, Sig):
    """err: (W,3), Sig: (W,3,3) -> return (M, D, NLL_per_window) numpy arrays."""
    inv = np.linalg.inv(Sig + 1e-12 * np.eye(3))
    M = np.einsum('wi,wij,wj->w', err, inv, err)
    sign, logabsdet = np.linalg.slogdet(Sig + 1e-12 * np.eye(3))
    D = logabsdet
    NLL = 0.5 * (M + D + 3 * np.log(2 * np.pi))
    return M, D, NLL


def coverage(M, q_levels=[0.5,0.7,0.9,0.95]):
    return {q: float(np.mean(M <= chi2.ppf(q, df=3))) for q in q_levels}


def main(out_dir=OUTPUT_DIR, test_list_file=TEST_LIST_FILE):
    OUT = Path(out_dir)
    DUMPS = OUT / 'dumps'
    VAL_SEQS, TEST_SEQS = split_val_test(test_list_file)
    results = {}
    for m in MODELS:
        val_M_list, val_D_list = [], []
        test_M_list, test_D_list = [], []
        for s in VAL_SEQS:
            d = load_dump(m, s, DUMPS);  M, D, _ = compute_stats(d['err'], d['Sigma'])
            val_M_list.append(M); val_D_list.append(D)
        for s in TEST_SEQS:
            d = load_dump(m, s, DUMPS);  M, D, _ = compute_stats(d['err'], d['Sigma'])
            test_M_list.append(M); test_D_list.append(D)
        val_M  = np.concatenate(val_M_list);  val_D  = np.concatenate(val_D_list)
        test_M = np.concatenate(test_M_list); test_D = np.concatenate(test_D_list)

        # alpha* = mean_val(M) / 3 (analytic minimizer for NLL w.r.t. alpha since dNLL/dalpha = -M/(2alpha^2)+3/(2alpha)=0)
        alpha = float(val_M.mean() / 3.0)
        # Raw NLL
        NLL_raw_test  = 0.5 * (test_M + test_D + 3*np.log(2*np.pi))
        # Calibrated: Sigma_alpha = alpha*Sigma -> M_alpha = M/alpha, D_alpha = D + 3 log alpha
        NLL_cal_test  = 0.5 * (test_M / alpha + test_D + 3 * np.log(alpha) + 3*np.log(2*np.pi))
        # NEES: mean M
        nees_raw_test = float(test_M.mean())
        nees_cal_test = float((test_M / alpha).mean())
        cov_raw = coverage(test_M)
        cov_cal = coverage(test_M / alpha)

        results[m] = dict(
            n_val_windows=int(len(val_M)),
            n_test_windows=int(len(test_M)),
            alpha_star=alpha,
            mean_NLL_raw_val=float(0.5*(val_M+val_D+3*np.log(2*np.pi)).mean()),
            mean_NLL_raw_test=float(NLL_raw_test.mean()),
            mean_NLL_cal_test=float(NLL_cal_test.mean()),
            mean_Mahal_raw_test=float(test_M.mean()),
            median_Mahal_raw_test=float(np.median(test_M)),
            mean_logdet_test=float(test_D.mean()),
            nees_raw_test=nees_raw_test,
            nees_cal_test=nees_cal_test,
            coverage_raw_test=cov_raw,
            coverage_cal_test=cov_cal,
        )
        print(f'{m:25s} | alpha*={alpha:.3f} | NLL raw={NLL_raw_test.mean():+.3f} -> cal={NLL_cal_test.mean():+.3f}'
              f' | NEES raw={nees_raw_test:.2f} (ideal 3) -> cal={nees_cal_test:.2f}'
              f' | cov95 raw={cov_raw[0.95]:.3f} -> cal={cov_cal[0.95]:.3f}')

    out_path = OUT / 'TABLE_A_nll_decomp_calib.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nWrote', out_path)
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out_dir', default=str(OUTPUT_DIR), help='folder with dumps/ from run_phase_ab.py --phase A; output goes here')
    ap.add_argument('--test_list', default=str(TEST_LIST_FILE))
    a = ap.parse_args()
    main(a.out_dir, a.test_list)
