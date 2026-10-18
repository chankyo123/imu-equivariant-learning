# Paper artefact: Supplement Table 9 (covariance-head ablation, Phase A) and the Lie-Neuron rows of the
# ID-SO(3) / mount-tilt sweeps of Supplement Table 10 and Figures 6-7 (Phase B); writes the per-window dumps
# consumed by compute_idso3_ate.py, nll_decompose_calib.py, aggregate_results.py and cross_model_summary.py.
"""
Phase A (cov-ablation: Sigma-eq, NLL decomp, NEES, coverage, alpha-calib) +
Phase B (perturbations: ID-SO(3), calib-mismatch / mount tilt) on NanoBench NN-only.

Self-contained: reads imu0_resampled.npy directly, applies LFA when needed,
runs forward through the released LN-ResNet / LN-ResNet-Chol ckpts, dumps per-window
(mu, Sigma_3x3, gt_vel_Body, error) plus aggregate metrics per (model, scenario).

Outputs:
  <out_dir>/dumps/<model>__<scenario>__<seed>__<seq>.npz   # per-seq per-window
  <out_dir>/aggregate__<model>__<scenario>__<seed>.json
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Resolve project paths (repo-relative; see _paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (CKPT_ROOT, DATA_LFA, DATA_NOLFA, OUTPUT_DIR, SCRIPTS_DIR, SRC_DIR,  # noqa: E402
                    TEST_LIST_FILE, insert_paths, read_test_list, resolve_ckpt_dir)
insert_paths(SRC_DIR, SCRIPTS_DIR)   # src/network/model_factory.py, scripts/lfa_preprocessing.py

from network.model_factory import get_model  # noqa: E402
from lfa_preprocessing import apply_lfa  # noqa: E402

# 4 released ckpts (formerly Archive/checkpoints/ln_resnet_{lfa_MSE,lfa_NLL,chol_MSE,chol_NLL}):
# (released folder under --ckpt_root, data_root_uses_lfa, head_kind)
MODELS = {
    'ln_resnet_lfa_MSE':   dict(ckpt='nanobench_ginio_rd_lfa_mse',   lfa=True,  head='diag'),
    'ln_resnet_lfa_NLL':   dict(ckpt='nanobench_ginio_rd_lfa_nll',   lfa=True,  head='diag'),
    'ln_resnet_chol_MSE':  dict(ckpt='nanobench_ginio_rs_chol_mse',  lfa=False, head='chol'),
    'ln_resnet_chol_NLL':  dict(ckpt='nanobench_ginio_rs_chol_nll',  lfa=False, head='chol'),
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_name, ckpt_root=CKPT_ROOT, device=DEVICE):
    info = MODELS[model_name]
    ckpt_path = resolve_ckpt_dir(info['ckpt'], ckpt_root) / 'checkpoint_best.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt['args'] if isinstance(ckpt['args'], dict) else vars(ckpt['args'])
    win = int(args['window_time'] * args['imu_freq'])
    in_dim = win // 32 + 1
    nc = {'in_dim': in_dim, 'init_log_a_scale': args.get('init_log_a_scale')}
    net = get_model(args['arch'], nc, input_dim=args['input_dim'], output_dim=args['output_dim'])
    net.load_state_dict(ckpt['model_state_dict'], strict=True)
    net.eval().to(device)
    return net, args, win


def sigma_from_output(head, out):
    """Returns Sigma (B,3,3) given network output tuple.
       diag: out=(mu, log_std) -> Sigma = diag(exp(2*log_std))
       chol: out=(mu, A)       -> Sigma = A.A^T + 1e-3.I  (matches meas_source_torchscript)
    """
    mu = out[0]
    if head == 'diag':
        log_std = out[1]
        var = torch.exp(2.0 * log_std)
        Sigma = torch.diag_embed(var)
    elif head == 'chol':
        A = out[1]  # (B,3,3)
        Sigma = torch.einsum('bij,bkj->bik', A, A) + 1e-3 * torch.eye(3, device=A.device)
    else:
        raise ValueError(head)
    return mu, Sigma


def build_windows(seq_dir, win, stride):
    """Read imu0_resampled.npy, build sliding windows.
       Returns: gyr (W,3,T), acc (W,3,T), quats (W,T,4), gt_vel_Body_last (W,3),
                ts_us_last (W,)
    """
    a = np.load(seq_dir / 'imu0_resampled.npy')  # (N,17)
    N = a.shape[0]
    starts = np.arange(0, N - win + 1, stride)
    W = len(starts)
    gyr = np.empty((W, 3, win))
    acc = np.empty((W, 3, win))
    qts = np.empty((W, win, 4))
    gt_vel = np.empty((W, 3))
    ts_last = np.empty((W,))
    for i, s in enumerate(starts):
        chunk = a[s:s+win]
        gyr[i] = chunk[:, 1:4].T
        acc[i] = chunk[:, 4:7].T
        qts[i] = chunk[:, 7:11]
        gt_vel[i] = chunk[-1, 14:17]
        ts_last[i] = chunk[-1, 0]
    return gyr, acc, qts, gt_vel, ts_last


def apply_perturb(scenario, gyr, acc, qts, gt_vel, R):
    """gyr,acc: (W,3,T); qts: (W,T,4) xyzw; gt_vel: (W,3); R: (3,3) numpy.
       Returns (gyr', acc', qts', gt_vel_target, unrotate_pred)
         - clean: passthrough
         - id_so3: apply R to body frame globally (IMU + GT + quats) -> predict in rotated frame
                   target_for_eval = R.gt_vel_Body  (since both rotated); we'll un-rotate prediction
                   for ATE comparison against original gt_vel_Body.
         - calib_mismatch: apply R only to IMU (gyr, acc). GT, quats unchanged.
    """
    if scenario == 'clean':
        return gyr, acc, qts, gt_vel, False
    if scenario == 'id_so3':
        gyr_r = np.einsum('ij,wjt->wit', R, gyr)
        acc_r = np.einsum('ij,wjt->wit', R, acc)
        gt_r  = (R @ gt_vel.T).T
        # rotate orientation quaternions: R_new = R . R_old -> q_new = q_R * q_old (left action on world)
        # For LFA, what matters is R_t_last; this is unchanged because both R_old and R_last are
        # left-multiplied by the same R. So quats can be left as-is for the LFA preprocessing
        # to produce equivalent rotation differences. We'll keep qts unchanged.
        return gyr_r, acc_r, qts, gt_r, True
    if scenario == 'calib_mismatch':
        gyr_r = np.einsum('ij,wjt->wit', R, gyr)
        acc_r = np.einsum('ij,wjt->wit', R, acc)
        return gyr_r, acc_r, qts, gt_vel, False
    if scenario == 'lfa_mismatch':
        # Inject independent per-sample rotation noise into the quaternion window before
        # LFA preprocessing. Magnitude = angle of R in axis-angle representation; axis is
        # random per-sample (so the noise does NOT cancel under LFA's left-anchor).
        # This models time-varying AHRS attitude error of comparable angular magnitude.
        rng = np.random.default_rng(int(np.abs(R).sum() * 1e6) % (2**31))
        angle_rad = np.linalg.norm(Rotation.from_matrix(R).as_rotvec())
        W, T = qts.shape[:2]
        # Per-(window, sample) random axis
        axes = rng.standard_normal((W, T, 3))
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True) + 1e-12
        rotvecs = axes * angle_rad  # (W, T, 3)
        qts_noisy = np.empty_like(qts)
        for w in range(W):
            R_noise = Rotation.from_rotvec(rotvecs[w])  # (T,)
            noisy_rots = R_noise * Rotation.from_quat(qts[w])
            qts_noisy[w] = noisy_rots.as_quat()
        # IMU, gt_vel unchanged -- only LFA preprocessing input (quat) is corrupted
        return gyr, acc, qts_noisy, gt_vel, False
    raise ValueError(scenario)


def run_one(model_name, scenario, R, seed_tag, batch=128, *, ckpt_root=CKPT_ROOT,
            data_lfa=DATA_LFA, data_nolfa=DATA_NOLFA, test_list=None, out_dir=OUTPUT_DIR,
            device=DEVICE):
    info = MODELS[model_name]
    net, args, win = load_model(model_name, ckpt_root, device)
    head = info['head']
    use_lfa = info['lfa']
    stride = max(1, int(args['imu_freq'] / args['sample_freq']))  # e.g., 200/10=20

    data_root = Path(data_lfa) if use_lfa else Path(data_nolfa)
    test_list = read_test_list() if test_list is None else test_list
    out_dir = Path(out_dir)
    (out_dir / 'dumps').mkdir(parents=True, exist_ok=True)

    per_seq = {}
    all_dumps = []
    t0 = time.time()
    for seq in test_list:
        seq_dir = data_root / seq
        if not seq_dir.exists():
            print(f'  skip missing seq: {seq}')
            continue
        gyr, acc, qts, gt_vel, ts_last = build_windows(seq_dir, win, stride)
        gyr, acc, qts, gt_target, unrotate_pred = apply_perturb(scenario, gyr, acc, qts, gt_vel, R)

        # LFA: rotate IMU into last-frame body axes (per-window)
        if use_lfa:
            for w in range(gyr.shape[0]):
                imu_w = np.vstack([gyr[w], acc[w]])  # (6,T)
                imu_lfa = apply_lfa(imu_w, qts[w])
                gyr[w] = imu_lfa[:3]
                acc[w] = imu_lfa[3:]

        # Network expects [B, 6, T] float64? Check forward earlier worked with float32 cuda.
        feat_np = np.concatenate([gyr, acc], axis=1).astype(np.float32)  # (W,6,T)
        W = feat_np.shape[0]
        mu_all = np.empty((W, 3))
        sig_all = np.empty((W, 3, 3))
        with torch.no_grad():
            for s in range(0, W, batch):
                x = torch.from_numpy(feat_np[s:s+batch]).to(device)
                out = net(x)
                mu, Sig = sigma_from_output(head, out)
                mu_all[s:s+batch] = mu.cpu().numpy()
                sig_all[s:s+batch] = Sig.cpu().numpy()

        # If id_so3, un-rotate prediction back so comparison is in original frame
        if unrotate_pred:
            mu_all = (R.T @ mu_all.T).T
            sig_all = np.einsum('ij,wjk,lk->wil', R.T, sig_all, R.T)
            gt_target_use = gt_vel  # original, unrotated
        else:
            gt_target_use = gt_target

        err = mu_all - gt_target_use  # (W,3)
        per_window = dict(mu=mu_all, Sigma=sig_all, gt=gt_target_use, err=err, ts=ts_last)

        # Per-seq metrics
        mse = np.mean(np.sum(err**2, axis=1))
        rmse_vel = np.sqrt(mse)
        # Mahalanobis & log|Sigma|
        inv_S = np.linalg.inv(sig_all + 1e-12 * np.eye(3))
        mahal = np.einsum('wi,wij,wj->w', err, inv_S, err)
        sign, logabsdet = np.linalg.slogdet(sig_all + 1e-12 * np.eye(3))
        log_det = logabsdet
        # NLL per window = 0.5*(M + log|Sigma| + 3*log(2pi))
        nll = 0.5 * (mahal + log_det + 3 * np.log(2 * np.pi))
        per_seq[seq] = dict(
            n_windows=int(W),
            rmse_vel=float(rmse_vel),
            mean_mahal=float(np.mean(mahal)),
            median_mahal=float(np.median(mahal)),
            mean_logdet=float(np.mean(log_det)),
            mean_nll=float(np.mean(nll)),
            coverage_50=float(np.mean(mahal <= 2.366)),  # chi2_3 p=0.5
            coverage_70=float(np.mean(mahal <= 3.665)),  # chi2_3 p=0.7
            coverage_90=float(np.mean(mahal <= 6.251)),  # chi2_3 p=0.9
            coverage_95=float(np.mean(mahal <= 7.815)),  # chi2_3 p=0.95
        )
        # Save dump
        npz_path = out_dir / 'dumps' / f'{model_name}__{scenario}__{seed_tag}__{seq}.npz'
        np.savez_compressed(npz_path, **per_window)
        all_dumps.append(npz_path.name)

    # Aggregate across seqs
    keys_to_agg = ['rmse_vel','mean_mahal','median_mahal','mean_logdet','mean_nll',
                   'coverage_50','coverage_70','coverage_90','coverage_95']
    agg = {k: float(np.mean([per_seq[s][k] for s in per_seq])) for k in keys_to_agg}
    n_seq = len(per_seq)
    elapsed = time.time() - t0
    print(f'  [{model_name} | {scenario} | {seed_tag}] {n_seq} seqs in {elapsed:.1f}s | '
          f'rmse_vel={agg["rmse_vel"]:.4f}  mean_NLL={agg["mean_nll"]:.3f}  '
          f'cov95={agg["coverage_95"]:.3f}')
    result = dict(
        model=model_name, scenario=scenario, seed_tag=seed_tag,
        n_seqs=n_seq, agg=agg, per_seq=per_seq, elapsed_s=elapsed,
        R_matrix=R.tolist(),
    )
    out_path = out_dir / f'aggregate__{model_name}__{scenario}__{seed_tag}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    return result


def random_so3(seed, deg=None):
    """If deg given, returns rotation by exactly that angle around a random axis (seeded)."""
    rng = np.random.default_rng(seed)
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = np.deg2rad(deg) if deg is not None else rng.uniform(0, np.pi)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def add_path_args(parser):
    parser.add_argument('--ckpt_root', default=str(CKPT_ROOT),
                        help='folder with the released checkpoint folders (checkpoints/MANIFEST.md)')
    parser.add_argument('--data_lfa', default=str(DATA_LFA),
                        help='body-frame NanoBench root used by the LFA models (ln_resnet_lfa_*)')
    parser.add_argument('--data_nolfa', default=str(DATA_NOLFA),
                        help='body-frame NanoBench root used by the spectral/chol models (same bytes, "nolfa" name)')
    parser.add_argument('--test_list', default=str(TEST_LIST_FILE), help='17 NanoBench test sequences')
    parser.add_argument('--out_dir', default=str(OUTPUT_DIR), help='where aggregate__*.json and dumps/ go')
    parser.add_argument('--device', default=DEVICE)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', choices=['A','B','sigma_eq','all'], default='all')
    parser.add_argument('--model', default=None, choices=list(MODELS), help='specific model name (default: all four)')
    add_path_args(parser)
    args = parser.parse_args()

    targets = [args.model] if args.model else list(MODELS.keys())
    kw = dict(ckpt_root=args.ckpt_root, data_lfa=args.data_lfa, data_nolfa=args.data_nolfa,
              test_list=read_test_list(args.test_list), out_dir=args.out_dir, device=args.device)

    if args.phase in ('A', 'all'):
        # Phase A: clean baseline (gives per-window dump for NLL decomp + NEES + coverage)
        I3 = np.eye(3)
        for m in targets:
            run_one(m, 'clean', I3, seed_tag='clean', **kw)

    if args.phase in ('B', 'all'):
        # Phase B1: ID-SO(3) at {30,60,90,120,180} x 3 random axes
        for m in targets:
            for deg in [30, 60, 90, 120, 180]:
                for ax_seed in range(3):
                    R = random_so3(seed=1000 + ax_seed, deg=deg)
                    run_one(m, 'id_so3', R, seed_tag=f'deg{deg}_seed{ax_seed}', **kw)

        # Phase B2: calib-mismatch (mount tilt) at {3,5,10,15,20} x 3 random axes
        for m in targets:
            for deg in [3, 5, 10, 15, 20]:
                for ax_seed in range(3):
                    R = random_so3(seed=2000 + ax_seed, deg=deg)
                    run_one(m, 'calib_mismatch', R, seed_tag=f'deg{deg}_seed{ax_seed}', **kw)

        # Phase B3: LFA-mismatch (GT quaternion noise) at {1,3,5,10} deg x 3 random axes
        # Only LFA models (lfa_*) are interesting here; for chol models LFA is bypassed
        # but we run all 4 for completeness (chol should be insensitive to quat noise).
        for m in targets:
            for deg in [1, 3, 5, 10]:
                for ax_seed in range(3):
                    R = random_so3(seed=4000 + ax_seed, deg=deg)
                    run_one(m, 'lfa_mismatch', R, seed_tag=f'deg{deg}_seed{ax_seed}', **kw)


if __name__ == '__main__':
    main()
