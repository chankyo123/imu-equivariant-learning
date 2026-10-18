# Paper artefact: Sigma-eq. / mu-eq. error columns of Supplement Table 9 for the four Lie-Neuron models
# (head-only test: IMU rotated, quaternions untouched); writes sigma_eq_results.json read by aggregate_results.py.
"""
Sigma-equivariance error (head-only) on the released LN-ResNet / LN-ResNet-Chol ckpts.

For a random batch of windows from NanoBench, applies a random SO(3) rotation R
to the IMU input in body frame, runs forward, and measures:

    Sigma_err  = ||Sigma(R.z) - R.Sigma(z).R^T||_F / (||R.Sigma(z).R^T||_F + 1e-12)
    mu_err     = ||mu(R.z) - R.mu(z)||_2   / (||R.mu(z)||_2 + 1e-12)

Averaged over 8 random R and a batch of 64 windows from B11_random_waypoints_rep1.

Compares diagonal-from-norms (`ln_resnet`) vs Cholesky (`ln_resnet_chol`).
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import CKPT_ROOT, DATA_LFA, DATA_NOLFA, OUTPUT_DIR  # noqa: E402
from run_phase_ab import MODELS, load_model, sigma_from_output  # noqa: E402  (also puts src/ on sys.path)
from network.model_factory import get_model  # noqa: E402,F401

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def random_so3(seed):
    rng = np.random.default_rng(seed)
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = rng.uniform(0, np.pi)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def main(seq='B11_random_waypoints_rep1', n_R=8, batch=64, *, ckpt_root=CKPT_ROOT,
         data_lfa=DATA_LFA, data_nolfa=DATA_NOLFA, out_dir=OUTPUT_DIR, device=DEVICE):
    OUT = Path(out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for mname, info in MODELS.items():
        net, args, win = load_model(mname, ckpt_root, device)
        data_root = Path(data_lfa) if info['lfa'] else Path(data_nolfa)
        a = np.load(data_root / seq / 'imu0_resampled.npy')
        # Take `batch` windows striding evenly
        starts = np.linspace(0, a.shape[0]-win-1, batch).astype(int)
        feat = np.empty((batch, 6, win), dtype=np.float64)
        for i, s in enumerate(starts):
            feat[i, :3] = a[s:s+win, 1:4].T
            feat[i, 3:] = a[s:s+win, 4:7].T

        sigma_errs, mu_errs = [], []
        for r_seed in range(n_R):
            R = random_so3(r_seed)
            feat_r = np.einsum('ij,bjt->bit', R, feat[:, :3])
            feat_a = np.einsum('ij,bjt->bit', R, feat[:, 3:])
            feat_R = np.concatenate([feat_r, feat_a], axis=1).astype(np.float32)
            with torch.no_grad():
                x_orig = torch.from_numpy(feat.astype(np.float32)).to(device)
                x_rot  = torch.from_numpy(feat_R).to(device)
                mu0, S0 = sigma_from_output(info['head'], net(x_orig))
                muR, SR = sigma_from_output(info['head'], net(x_rot))
            mu0 = mu0.cpu().numpy(); S0 = S0.cpu().numpy()
            muR = muR.cpu().numpy(); SR = SR.cpu().numpy()
            # Predicted-equivariant: muR ~ R.mu0,  SR ~ R.S0.R^T
            mu_pred  = (R @ mu0.T).T
            Sig_pred = np.einsum('ij,bjk,lk->bil', R, S0, R)
            mu_err = np.linalg.norm(muR - mu_pred, axis=1) / (np.linalg.norm(mu_pred, axis=1) + 1e-12)
            Sig_err = np.linalg.norm(SR - Sig_pred, axis=(1,2)) / (np.linalg.norm(Sig_pred, axis=(1,2)) + 1e-12)
            sigma_errs.append(Sig_err.mean()); mu_errs.append(mu_err.mean())
        results[mname] = dict(
            head=info['head'], lfa=info['lfa'],
            sigma_eq_err_mean=float(np.mean(sigma_errs)),
            sigma_eq_err_std=float(np.std(sigma_errs)),
            mu_eq_err_mean=float(np.mean(mu_errs)),
            mu_eq_err_std=float(np.std(mu_errs)),
            n_R=n_R, batch=batch, seq=seq,
        )
        print(f'{mname:25s} | head={info["head"]:4s} lfa={info["lfa"]} | '
              f'Sigma-err={results[mname]["sigma_eq_err_mean"]:.3e}+-{results[mname]["sigma_eq_err_std"]:.1e}  '
              f'mu-err={results[mname]["mu_eq_err_mean"]:.3e}+-{results[mname]["mu_eq_err_std"]:.1e}')

    with open(OUT / 'sigma_eq_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seq', default='B11_random_waypoints_rep1')
    ap.add_argument('--n_R', type=int, default=8)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--ckpt_root', default=str(CKPT_ROOT))
    ap.add_argument('--data_lfa', default=str(DATA_LFA))
    ap.add_argument('--data_nolfa', default=str(DATA_NOLFA))
    ap.add_argument('--out_dir', default=str(OUTPUT_DIR))
    ap.add_argument('--device', default=DEVICE)
    a = ap.parse_args()
    main(a.seq, a.n_R, a.batch, ckpt_root=a.ckpt_root, data_lfa=a.data_lfa, data_nolfa=a.data_nolfa,
         out_dir=a.out_dir, device=a.device)
