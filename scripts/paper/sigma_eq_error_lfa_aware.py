# Paper artefact: LFA-aware Sigma-eq. / mu-eq. error columns of Supplement Table 9 (IMU and quaternions rotated
# jointly so LFA stays valid); writes sigma_eq_results_lfa_aware.json read by aggregate_results.py.
"""
LFA-aware Sigma/mu-equivariance test.

For LFA-trained models the head-only test must include LFA preprocessing. We
rotate IMU + quaternions jointly by R (so LFA still produces a valid input that
the network was trained on), and check that the (R, prediction) pair satisfies
the equivariance condition.

If LFA preserves equivariance under joint rotation, then the LFA-models should
show low Sigma-err and low mu-err on this test. If LFA breaks equivariance, we'll
see large errors regardless.
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import CKPT_ROOT, DATA_LFA, DATA_NOLFA, OUTPUT_DIR  # noqa: E402
from run_phase_ab import MODELS, load_model, sigma_from_output  # noqa: E402  (also puts src/ and scripts/ on sys.path)
from network.model_factory import get_model  # noqa: E402,F401
from lfa_preprocessing import apply_lfa  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def random_so3(seed):
    rng = np.random.default_rng(seed)
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = rng.uniform(0, np.pi)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def forward_with_optional_lfa(net, feat6_T, qts_TW4, use_lfa, head, device=DEVICE):
    """feat6_T: (B,6,T) numpy. qts_TW4: (B,T,4) numpy xyzw. Returns (mu, Sigma) numpy."""
    if use_lfa:
        out_feats = np.empty_like(feat6_T)
        for w in range(feat6_T.shape[0]):
            out_feats[w] = apply_lfa(feat6_T[w], qts_TW4[w])
        feat6_T = out_feats
    x = torch.from_numpy(feat6_T.astype(np.float32)).to(device)
    with torch.no_grad():
        out = net(x)
        mu, S = sigma_from_output(head, out)
    return mu.cpu().numpy(), S.cpu().numpy()


def main(seq='B11_random_waypoints_rep1', n_R=8, batch=64, *, ckpt_root=CKPT_ROOT,
         data_lfa=DATA_LFA, data_nolfa=DATA_NOLFA, out_dir=OUTPUT_DIR, device=DEVICE):
    OUT = Path(out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for mname, info in MODELS.items():
        net, args, win = load_model(mname, ckpt_root, device)
        data_root = Path(data_lfa) if info['lfa'] else Path(data_nolfa)
        a = np.load(data_root / seq / 'imu0_resampled.npy')
        starts = np.linspace(0, a.shape[0]-win-1, batch).astype(int)
        feat = np.empty((batch, 6, win), dtype=np.float64)
        qts  = np.empty((batch, win, 4), dtype=np.float64)
        for i, s in enumerate(starts):
            feat[i, :3] = a[s:s+win, 1:4].T
            feat[i, 3:] = a[s:s+win, 4:7].T
            qts[i]     = a[s:s+win, 7:11]

        sig_errs, mu_errs = [], []
        for r_seed in range(n_R):
            R = random_so3(r_seed)
            # joint rotation: gyr -> R.gyr, acc -> R.acc, quat -> q(R).q  (left action)
            feat_r = np.einsum('ij,bjt->bit', R, feat[:, :3])
            feat_a = np.einsum('ij,bjt->bit', R, feat[:, 3:])
            feat_R = np.concatenate([feat_r, feat_a], axis=1)
            q_R = Rotation.from_matrix(R).as_quat()  # xyzw
            # quaternion product q_R (x) q_old, applied per-sample
            qts_R = np.empty_like(qts)
            for b in range(batch):
                qts_R[b] = (Rotation.from_quat(q_R[None, :]) * Rotation.from_quat(qts[b])).as_quat()

            mu0, S0 = forward_with_optional_lfa(net, feat,   qts,   info['lfa'], info['head'], device)
            muR, SR = forward_with_optional_lfa(net, feat_R, qts_R, info['lfa'], info['head'], device)
            mu_pred = (R @ mu0.T).T
            Sig_pred = np.einsum('ij,bjk,lk->bil', R, S0, R)
            mu_err = np.linalg.norm(muR - mu_pred, axis=1) / (np.linalg.norm(mu_pred, axis=1) + 1e-12)
            Sig_err = np.linalg.norm(SR - Sig_pred, axis=(1,2)) / (np.linalg.norm(Sig_pred, axis=(1,2)) + 1e-12)
            sig_errs.append(Sig_err.mean()); mu_errs.append(mu_err.mean())

        results[mname] = dict(
            head=info['head'], lfa=info['lfa'],
            sigma_eq_err_mean=float(np.mean(sig_errs)),
            sigma_eq_err_std=float(np.std(sig_errs)),
            mu_eq_err_mean=float(np.mean(mu_errs)),
            mu_eq_err_std=float(np.std(mu_errs)),
            n_R=n_R, batch=batch, seq=seq, test='LFA_aware_joint_rotation',
        )
        print(f'{mname:25s} | head={info["head"]:4s} lfa={info["lfa"]} | '
              f'Sigma-err={results[mname]["sigma_eq_err_mean"]:.3e}+-{results[mname]["sigma_eq_err_std"]:.1e}  '
              f'mu-err={results[mname]["mu_eq_err_mean"]:.3e}+-{results[mname]["mu_eq_err_std"]:.1e}')

    with open(OUT / 'sigma_eq_results_lfa_aware.json', 'w') as f:
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
